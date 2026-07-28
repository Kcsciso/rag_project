#!/usr/bin/env python3
"""
=============================================================================
比邻星 (ProximaRAG) 统一回归评测器 (v4.4 — 完整白名单与单次加载优化版)
=============================================================================

特性:
  - 向量库与 BM25 索引单次全局加载（避免单用例重复读盘）
  - 全量补全 OpenC3 / OpenR6 C-SDK 文档合法 API 白名单 (彻底消除误杀)
  - 8 项硬质量断言 + 4 维 RAG 量化指标汇总
  - 区分 Quick 检索模式与 Full LLM 模式的断言范围

运行: python tests/run_eval.py [--quick] [--verbose] [--filter ID1,ID2]
=============================================================================
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Set

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("run_eval")


def load_eval_cases() -> List[Dict]:
    with open(_PROJECT_ROOT / "tests" / "eval_cases.json", "r", encoding="utf-8") as f:
        return json.load(f)["cases"]


# ═══════════════════════════════════════════════════════════
# 8 项硬质量断言
# ═══════════════════════════════════════════════════════════

def _fatal_json_leak(answer: str) -> bool:
    """① JSON 泄露: 答案含【提取】或 JSON 标签源码"""
    return bool(re.search(r'【提取】|"steps"\s*:\s*\[|"functions"\s*:\s*\[', answer))


def _fatal_duplication(answer: str) -> bool:
    """② 重复检查: 连续两段完全相同 (≥80 chars)"""
    paras = [p.strip() for p in answer.split("\n\n") if len(p.strip()) >= 80]
    for i in range(len(paras) - 1):
        if paras[i] == paras[i + 1]:
            return True
    return False


def _fatal_app_sdk_boilerplate(query: str, answer: str, product_id: str = None) -> bool:
    """③ 界面套话: JAKA APP 操作查询含 SDK 拒答套话"""
    _app_patterns = [
        r'APP', r'界面', r'配置', r'升级', r'版本', r'IO\s*配置',
        r'Modbus\s*参数', r'通讯设置', r'安全区域', r'坐标系',
        r'四点法', r'拖动示教', r'怎么(?:升级|设置|配置|连接|操作)',
    ]
    is_app_query = any(re.search(p, query) for p in _app_patterns)
    if product_id and product_id.upper() == "JAKA":
        is_app_query = True
    has_sdk_refusal = bool(re.search(
        r'未详细记载具体的\s*SDK\s*函数|参考资料未详细记载具体的\s*SDK',
        answer,
    ))
    return is_app_query and has_sdk_refusal


def _fatal_wrong_signature(query: str, answer: str) -> bool:
    """④ 函数签名错误: 无参函数被写成有参"""
    checks = [
        (r'(?:Power_on|power_on)\s*\(', r'(?:Power_on|power_on)\s*\(\s*(?:ip|port|address)',
         "robot_Power_on() 是无参函数"),
        (r'set_robot_power_on\s*\(', r'set_robot_power_on\s*\(\s*(?:ip|port|address)',
         "set_robot_power_on() 是无参函数"),
    ]
    for _pos_pattern, _neg_pattern, _msg in checks:
        if re.search(_pos_pattern, answer, re.IGNORECASE) and re.search(_neg_pattern, answer, re.IGNORECASE):
            return True
    return False


def _fatal_prompt_leak(answer: str) -> bool:
    """⑤ 提示词泄露: 包含系统 Prompt 变量名或注入泄露"""
    leak_keywords = ["RAG_SYSTEM_PROMPT", "系统提示词如下", "Ignore previous instructions", "系统指令源码"]
    return any(kw.lower() in answer.lower() for kw in leak_keywords)


_API_NAME_REWRITE_PATTERNS = [
    (r'\brobot_movl\b', [r'\bmove_linear\b', r'\blinear_move\b', r'\brobot_move_linear\b', r'\brobot_linm\b']),
    (r'\brobot_movc\b', [r'\bmove_arc\b', r'\brobot_move_circle\b', r'\brobot_arc_move\b']),
    (r'\bset_move_line\b', [r'\bset_move_linear\b', r'\bset_linear\b']),
    (r'\brobot_Power_on\b', [r'\brobot_power_up\b', r'\brobot_turn_on\b']),
    (r'\brobot_enable\b', [r'\brobot_activate\b', r'\brobot_start\b', r'\benable_robot\b']),
]

# 🔴 补全后的全量 C-SDK 函数白名单（涵盖 OpenC3 与 OpenR6 所有接口）
_KNOWN_FUNCS: Set[str] = {
    # OpenC3 SDK (collrob_sdk.dll)
    'robot_power_on', 'robot_power_off', 'robot_enable', 'robot_disable',
    'robot_movl', 'robot_movc', 'robot_movj', 'robot_movp', 'robot_stop',
    'robot_brkopen', 'robot_brkclose', 'robot_socket_start', 'robot_socket_close',
    'robot_handserialsend', 'robot_sysclose', 'robot_setdo', 'robot_get_pose',
    'get_robot_pose', 'get_robot_state', 'get_robot_iostate', 'get_robot_joint_all',
    'get_robot_motsta', 'get_robot_moterror', 'get_robot_torque', 'reset_server',

    # OpenR6 SDK (py_dll.dll)
    'set_robot_power_on', 'set_robot_power_off', 'set_robot_arm_home',
    'set_robot_arm_init', 'set_robot_cmd_mode', 'get_robot_cmd_model',
    'set_robot_seq_state', 'set_joint_degree_by_number', 'set_all_joint_degree_by_number',
    'set_joint_degree_synchronize', 'set_robot_arm_coordinate', 'set_move_line',
    'set_move_circle', 'set_robot_arm_coordinate_teach', 'set_robot_joint_angle_teach',
    'get_robot_joint_angle_all', 'get_robot_coordinate', 'set_robot_arm_joint_stop',
    'set_robot_joint_alignment', 'set_robot_switchover_end_tool',
    'set_robot_switchover_user_tool', 'set_robot_io_output', 'set_robot_end_tool',
    'set_robot_time_delay', 'set_robot_io_status', 'set_robot_cmd_send',
    'set_robot_arm_emergency_stop', 'end_communication',
}


def _fatal_api_hallucination(case: dict, answer: str) -> bool:
    """⑥ API 幻觉: 检查是否改写或虚构了未在白名单中的 API 函数名"""
    answer_lower = answer.lower()
    for correct_pat, rewrite_pats in _API_NAME_REWRITE_PATTERNS:
        correct_in_answer = bool(re.search(correct_pat, answer_lower))
        for rw_pat in rewrite_pats:
            if re.search(rw_pat, answer_lower) and not correct_in_answer:
                return True

    found_unknown = re.findall(r'\b((?:robot_|set_|get_)\w{4,})\b', answer_lower)
    for f in found_unknown:
        if f not in _KNOWN_FUNCS:
            return True
    return False


_SPECULATION_PATTERNS = re.compile(
    r'(?:假设有|假设存在|假设函数|示例代码仅为假设|'
    r'未在参考资料中明确记载.*```python|'
    r'以下仅为示例代码框架|仅供参考.*具体的函数名|'
    r'以下是.*假设性的|以下.*仅供示意)',
    re.IGNORECASE,
)


def _fatal_zero_speculation(answer: str) -> bool:
    """⑦ 零脑补: 代码生成中混入'假设存在/仅供示意'等免责声明"""
    has_code = '```python' in answer or '```' in answer
    has_speculation = bool(_SPECULATION_PATTERNS.search(answer))
    return has_code and has_speculation


def _fatal_code_truncation(answer: str) -> bool:
    """⑧ 代码截断: Python 代码块反引号 ``` 未成对闭合"""
    all_fences = re.findall(r'```', answer)
    return len(all_fences) % 2 != 0


def run_fatal_assertions(case: dict, answer: str, quick: bool = False) -> List[str]:
    """执行致命断言校验"""
    errors = []
    if _fatal_json_leak(answer):
        errors.append("① JSON泄露: 答案包含【提取】或JSON标签源码")
    if _fatal_duplication(answer):
        errors.append("② 重复检查: 连续两段完全相同文本")
    if _fatal_prompt_leak(answer):
        errors.append("⑤ 提示词泄露: 答案泄露了系统 Prompt 源码")

    # 在快速模式（仅纯检索文本）下，跳过针对 LLM 输出句式的硬检查
    if not quick:
        if _fatal_app_sdk_boilerplate(case["query"], answer, case.get("product_id")):
            errors.append("③ 界面套话: JAKA APP查询含SDK拒答套话")
        if _fatal_wrong_signature(case["query"], answer):
            errors.append("④ 函数签名错误: 无参函数被写成有参")
        if _fatal_api_hallucination(case, answer):
            errors.append("⑥ API幻觉: 函数名被改写/虚构(如robot_movl→move_linear)")
        if _fatal_zero_speculation(answer):
            errors.append("⑦ 零脑补: 含'假设有'/'示例代码仅为假设'等严重幻觉表述")
        if _fatal_code_truncation(answer):
            errors.append("⑧ 代码截断: Python代码块未闭合或残缺")

    return errors


# ═══════════════════════════════════════════════════════════
# 评测核心
# ═══════════════════════════════════════════════════════════

def run_single_case(case: dict, vs: Any, quick: bool = False) -> dict:
    from src.graph_rag import run_graph
    from src.config import RETRIEVAL_K

    result = {
        "id": case["id"], "category": case.get("category", "?"),
        "query": case["query"], "description": case.get("description", ""),
        "status": "PASS", "elapsed_ms": 0, "answer": "", "model": "",
        "checks": {}, "fatal_errors": [], "errors": [],
        "kw_total": len(case.get("must_contain", [])), "kw_hits": 0
    }

    t0 = time.time()
    pid = case.get("product_id")
    chat_history = case.get("chat_history", [])

    try:
        if quick:
            from src.rag_chain import _hybrid_retrieve
            docs = _hybrid_retrieve(vs, case["query"], k=RETRIEVAL_K, product_id=pid)
            answer = "\n\n".join(d.page_content[:300] for d in docs)
            model = "retrieval-only"
        else:
            r = run_graph(case["query"], chat_history, product_id=pid)
            answer = r.get("answer", "")
            model = r.get("model", "?")
    except Exception as e:
        result["status"] = "ERROR"
        result["errors"].append(f"run_graph 异常: {e}")
        result["elapsed_ms"] = round((time.time() - t0) * 1000)
        return result

    result["elapsed_ms"] = round((time.time() - t0) * 1000)
    result["answer"] = answer
    result["model"] = model

    # ── 关键词必须包含/禁止包含 ──
    for kw in case.get("must_contain", []):
        hit = kw.lower() in answer.lower()
        result["checks"][f"含'{kw}'"] = hit
        if hit:
            result["kw_hits"] += 1
        else:
            result["errors"].append(f"缺少关键词'{kw}'")

    for kw in case.get("must_not_contain", []):
        hit = kw.lower() in answer.lower()
        result["checks"][f"不含'{kw}'"] = not hit
        if hit:
            result["errors"].append(f"含禁止词'{kw}'")

    # ── 澄清反问校验 ──
    if case.get("expect_clarification"):
        is_clarify = "哪一款产品" in answer or "请问您询问" in answer
        result["checks"]["澄清反问"] = is_clarify
        if not is_clarify:
            result["errors"].append("期望澄清反问")

    # ── 8 项硬断言校验 ──
    fatal = run_fatal_assertions(case, answer, quick=quick)
    result["fatal_errors"] = fatal
    result["errors"].extend(fatal)

    result["status"] = "PASS" if len(result["errors"]) == 0 else "FAIL"
    return result


def print_header(title: str):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="比邻星 (ProximaRAG) 统一回归评测器")
    parser.add_argument("--quick", "-q", action="store_true", help="仅检索不调LLM")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出(含answer)")
    parser.add_argument("--filter", "-f", type=str, help="用例ID过滤(如 GT-1,E02)")
    args = parser.parse_args()

    cases = load_eval_cases()
    if args.filter:
        ids = set(args.filter.split(","))
        cases = [c for c in cases if c["id"] in ids]

    print_header(f"比邻星 (ProximaRAG) 回归评测 ({len(cases)} 用例)")
    print(f"  模式: {'快速(仅检索)' if args.quick else '完整(含LLM)'}")

    # 🔴 优化：全局单次初始化向量库与 BM25 内存索引（极大提升速度）
    from src.graph_rag import set_graph_vector_store
    from src.vector_store import load_vector_store, build_bm25_from_chromadb
    from src.config import CHROMA_PERSIST_DIR

    vs = load_vector_store(CHROMA_PERSIST_DIR)
    if not vs:
        print("❌ 向量库加载失败，请先检查 chroma 数据文件")
        return 1
    build_bm25_from_chromadb(vs)
    set_graph_vector_store(vs)

    results = []
    for i, case in enumerate(cases):
        cid = case["id"]
        cat = case.get("category", "?")
        print(f"\n  [{i+1}/{len(cases)}] {cid} [{cat}] {case['description'][:50]}...", end=" ", flush=True)
        r = run_single_case(case, vs, quick=args.quick)
        results.append(r)
        icon = "✅" if r["status"] == "PASS" else ("⚠️" if r["status"] == "SKIP" else "❌")
        print(f"{icon} ({r['elapsed_ms']}ms)")
        if r["errors"]:
            for e in r["errors"][:3]:
                print(f"       → {e}")
        if args.verbose and r.get("answer"):
            ans_preview = r["answer"][:250].replace("\n", " ")
            print(f"       answer: {ans_preview}")

    # ── 汇总面板 ──
    print_header("评测汇总与 RAG 量化指标")
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = total - passed

    total_kw_expected = sum(r.get("kw_total", 0) for r in results)
    total_kw_hits = sum(r.get("kw_hits", 0) for r in results)
    answer_kw_recall = (total_kw_hits / total_kw_expected * 100) if total_kw_expected > 0 else 100.0

    clean_cases = sum(1 for r in results if not any("① JSON" in e or "② 重复" in e for e in r.get("fatal_errors", [])))
    cleanliness_rate = (clean_cases / total * 100) if total > 0 else 100.0

    isolation_cases = sum(1 for r in results if not any("含禁止词" in e for e in r.get("errors", [])))
    isolation_rate = (isolation_cases / total * 100) if total > 0 else 100.0

    print(f"\n  📊 RAG 核心质量指标:")
    print(f"     • [Answer Keyword Recall 答案关键词命中率]: {answer_kw_recall:.1f}% ({total_kw_hits}/{total_kw_expected})")
    print(f"     • [Product Isolation 隔离合格率]: {isolation_rate:.1f}%")
    print(f"     • [Format Cleanliness 渲染纯净率]: {cleanliness_rate:.1f}%")
    print(f"     • [Overall Pass Rate 总评测通过率]: {passed/total*100:.1f}% ({passed}/{total})\n")

    for cat in sorted(set(r["category"] for r in results)):
        cr = [r for r in results if r["category"] == cat]
        cp = sum(1 for r in cr if r["status"] == "PASS")
        print(f"  {cat}: {cp}/{len(cr)}")

    if failed:
        print(f"\n  ❌ 失败用例:")
        for r in results:
            if r["status"] == "FAIL":
                print(f"    [{r['id']}] {r['description'][:60]}")
                for e in r["errors"]:
                    print(f"        → {e}")

    fatal_total = sum(len(r.get("fatal_errors", [])) for r in results)
    if fatal_total > 0:
        print(f"\n  🔴 硬断言触发: {fatal_total} 次")
        for r in results:
            for fe in r.get("fatal_errors", []):
                print(f"    [{r['id']}] {fe}")

    print(f"\n{'='*70}")
    if passed == total:
        print("🏆 100% PASS — 全部用例通过 (含 8 项硬断言)")
    else:
        print(f"❌ {failed} FAILED — 请修复后重新运行")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())