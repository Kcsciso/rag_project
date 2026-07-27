#!/usr/bin/env python3
"""
=============================================================================
比邻星 (ProximaRAG) 统一回归评测器 (v4.3 — 全量量化版)
=============================================================================

特性:
  - 直接调用 LangGraph run_graph()（无需 FastAPI）
  - 支持多轮对话 chat_history 传入
  - --quick: 仅检索不调 LLM（秒级）
  - --filter GT-1,E09: 按 ID 过滤
  - --verbose: 详细输出
  - 5 项硬质量断言 + 4 维 RAG 量化指标汇总

硬断言规则:
  ① JSON 泄露检查: 答案含【提取】或 "steps": [ → FAIL
  ② 重复检查: 连续两段完全相同文本 → FAIL
  ③ 界面套话检查: JAKA APP查询含"未详细记载具体的SDK函数" → FAIL
  ④ 规则匹配: SDK函数名/入参与文档不完全匹配 → FAIL
  ⑤ 提示词泄露: 包含 RAG_SYSTEM_PROMPT 或系统指令源码 → FAIL

运行: python tests/run_eval.py [--quick] [--verbose] [--filter ID1,ID2]
=============================================================================
"""

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import List, Dict

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("run_eval")


def load_eval_cases() -> List[Dict]:
    with open(_PROJECT_ROOT / "tests" / "eval_cases.json", "r", encoding="utf-8") as f:
        return json.load(f)["cases"]


# ═══════════════════════════════════════════════════════════
# 5 项硬质量断言
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
        is_app_query = True  # JAKA 默认是 APP 操作
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


# ── v5.3 新增 3 项断言 ──

_API_NAME_REWRITE_PATTERNS = [
    # robot_movl → robot_move_linear / move_linear / linear_move
    (r'\brobot_movl\b', [r'\bmove_linear\b', r'\blinear_move\b', r'\brobot_move_linear\b', r'\brobot_linm\b', r'\brobot_arm_move_linear\b']),
    # robot_movc → robot_move_circle / move_arc
    (r'\brobot_movc\b', [r'\bmove_arc\b', r'\brobot_move_circle\b', r'\brobot_arc_move\b']),
    # set_move_line → set_move_linear / set_linear
    (r'\bset_move_line\b', [r'\bset_move_linear\b', r'\bset_linear\b', r'\bmove_line\b']),
    # robot_Power_on → robot_power_up / robot_turn_on
    (r'\brobot_Power_on\b', [r'\brobot_power_up\b', r'\brobot_turn_on\b', r'\brobot_activate\b']),
    # robot_enable → robot_activate / robot_start
    (r'\brobot_enable\b', [r'\brobot_activate\b', r'\brobot_start\b', r'\benable_robot\b']),
    # get_robot_pose → get_pose / read_pose
    (r'\bget_robot_pose\b', [r'\bread_pose\b', r'\bget_position\b', r'\brobot_get_position\b']),
]


def _fatal_api_hallucination(case: dict, answer: str) -> bool:
    """
    ⑥ API 幻觉: Context 中存在某函数名，但模型输出时将其改写成虚构名称。

    规则: 若 answer 出现某 API 的虚构改写且不包含正确的原始名称 → FAIL
    """
    answer_lower = answer.lower()
    for correct_pat, rewrite_pats in _API_NAME_REWRITE_PATTERNS:
        correct_in_answer = bool(re.search(correct_pat, answer_lower))
        for rw_pat in rewrite_pats:
            if re.search(rw_pat, answer_lower):
                if not correct_in_answer:
                    return True
    # 检测虚构函数名: robot_/set_/get_ 不在 eval_cases 已知白名单中的
    _KNOWN_FUNCS = {
        'robot_power_on', 'robot_power_off', 'robot_enable', 'robot_disable',
        'robot_movl', 'robot_movc', 'robot_movj', 'robot_stop', 'robot_brkopen',
        'robot_brkclose', 'robot_motor_enable', 'robot_motor_disable',
        'robot_socket_start', 'robot_socket_close', 'robot_handserialsend',
        'robot_sysclose', 'robot_setdo', 'robot_get_pose', 'get_robot_pose',
        'get_robot_state', 'get_robot_iostate', 'get_robot_joint_all',
        'get_robot_joint_angle_all', 'get_robot_moterror', 'get_robot_torque',
        'set_robot_power_on', 'set_robot_power_off', 'set_robot_arm_home',
        'set_robot_arm_init', 'set_robot_io_output', 'set_robot_io_status',
        'set_robot_time_delay', 'set_robot_arm_emergency_stop',
        'set_move_line', 'set_joint_degree_by_number', 'set_all_joint_degree',
        'set_robot_speed', 'end_communication',
    }
    found_unknown = re.findall(r'\b((?:robot_|set_|get_)\w{4,})\b', answer_lower)
    for f in found_unknown:
        if f not in _KNOWN_FUNCS:
            return True
    return False


_SPECULATION_PATTERNS = re.compile(
    r'(?:假设有|假设存在|假设函数|示例代码仅为假设|'
    r'未在参考资料中明确记载.*```python|'
    r'以下仅为示例代码框架|仅供参考.*具体的函数名|'
    r'以下是.*假设性的|'
    r'以下.*仅供示意)',
    re.IGNORECASE,
)


def _fatal_zero_speculation(answer: str) -> bool:
    """
    ⑦ 零脑补: 答案在给出代码的同时使用了"假设/仅供参考/仅供示意"等幻术表述。

    规则: 含 Python 代码块 + 同时出现推测性表述 → FAIL
    """
    has_code = '```python' in answer or '```' in answer
    has_speculation = bool(_SPECULATION_PATTERNS.search(answer))
    return has_code and has_speculation


def _fatal_code_truncation(answer: str) -> bool:
    """
    ⑧ 代码截断: Python 代码块未正确闭合。

    规则: ```python 出现次数 ≠ ``` 闭合次数 → FAIL
    """
    open_count = len(re.findall(r'```python|```\w*', answer))
    close_count = len(re.findall(r'```(?!\w)', answer)) + len(re.findall(r'```$', answer))
    # 简化：计数 ``` 总出现次数应为偶数
    all_fences = re.findall(r'```', answer)
    if len(all_fences) % 2 != 0:
        return True
    # 代码块在 50 字符内突然截断（无闭合）
    last_fence = answer.rfind('```')
    if last_fence >= 0 and len(answer) - last_fence < 4:
        # 以 ``` 结尾但前面没有对应的开标签 → 截断
        pass  # 上面已经检查了偶数性
    return False


def run_fatal_assertions(case: dict, answer: str) -> List[str]:
    """返回触发的致命断言列表（空列表 = 全部通过）"""
    errors = []
    if _fatal_json_leak(answer):
        errors.append("① JSON泄露: 答案包含【提取】或JSON标签源码")
    if _fatal_duplication(answer):
        errors.append("② 重复检查: 连续两段完全相同文本")
    if _fatal_app_sdk_boilerplate(case["query"], answer, case.get("product_id")):
        errors.append("③ 界面套话: JAKA APP查询含SDK拒答套话")
    if _fatal_wrong_signature(case["query"], answer):
        errors.append("④ 函数签名错误: 无参函数被写成有参")
    if _fatal_prompt_leak(answer):
        errors.append("⑤ 提示词泄露: 答案泄露了系统 Prompt 源码")
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

def run_single_case(case: dict, quick: bool = False) -> dict:
    from src.graph_rag import run_graph, set_graph_vector_store
    from src.vector_store import load_vector_store, build_bm25_from_chromadb
    from src.config import CHROMA_PERSIST_DIR, RETRIEVAL_K

    vs = load_vector_store(CHROMA_PERSIST_DIR)
    if not vs:
        return {"id": case["id"], "status": "SKIP", "error": "vector store not loaded"}
    build_bm25_from_chromadb(vs)
    set_graph_vector_store(vs)

    result = {
        "id": case["id"], "category": case.get("category", "?"),
        "query": case["query"], "description": case.get("description", ""),
        "status": "PASS", "elapsed_ms": 0, "answer": "", "model": "",
        "checks": {}, "fatal_errors": [], "errors": [],
        "kw_total": len(case.get("must_contain", [])), "kw_hits": 0
    }

    t0 = time.time()
    pid = case.get("product_id")
    chat_history = case.get("chat_history", [])  # ✅ 修复 Bug: 读取用例的多轮历史

    try:
        if quick:
            from src.rag_chain import _hybrid_retrieve
            docs = _hybrid_retrieve(vs, case["query"], k=RETRIEVAL_K, product_id=pid)
            answer = "\n\n".join(d.page_content[:300] for d in docs)
            model = "retrieval-only"
        else:
            r = run_graph(case["query"], chat_history, product_id=pid)  # ✅ 修复 Bug: 传入 chat_history
            answer = r.get("answer", "")
            model = r.get("model", "?")
    except Exception as e:
        result["status"] = "ERROR"
        result["errors"].append(f"run_graph: {e}")
        result["elapsed_ms"] = (time.time() - t0) * 1000
        return result

    result["elapsed_ms"] = round((time.time() - t0) * 1000)
    result["answer"] = answer[:600]
    result["model"] = model

    # ── must_contain / must_not_contain ──
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

    # ── 澄清检查 ──
    if case.get("expect_clarification"):
        is_clarify = "哪一款产品" in answer or "请问您询问" in answer
        result["checks"]["澄清反问"] = is_clarify
        if not is_clarify:
            result["errors"].append("期望澄清反问")

    # ── 5 项硬断言 ──
    fatal = run_fatal_assertions(case, answer)
    result["fatal_errors"] = fatal
    result["errors"].extend(fatal)

    result["status"] = "PASS" if len(result["errors"]) == 0 else "FAIL"
    return result


def print_header(title: str):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="比邻星 (ProximaRAG) 统一回归评测器")
    parser.add_argument("--quick", "-q", action="store_true", help="仅检索不调LLM")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出(含answer)")
    parser.add_argument("--filter", "-f", type=str, help="用例ID过滤(如 GT-1,E02)")
    args = parser.parse_args()

    cases = load_eval_cases()
    if args.filter:
        ids = set(args.filter.split(","))
        cases = [c for c in cases if c["id"] in ids]

    print_header(f"比邻星 (ProximaRAG) v4 回归评测 ({len(cases)} 用例)")
    print(f"  模式: {'快速(仅检索)' if args.quick else '完整(含LLM)'}")

    results = []
    for i, case in enumerate(cases):
        cid = case["id"]
        cat = case.get("category", "?")
        print(f"\n  [{i+1}/{len(cases)}] {cid} [{cat}] {case['description'][:50]}...", end=" ", flush=True)
        r = run_single_case(case, quick=args.quick)
        results.append(r)
        icon = "✅" if r["status"] == "PASS" else ("⚠️" if r["status"] == "SKIP" else "❌")
        print(f"{icon} ({r['elapsed_ms']}ms)")
        if r["errors"]:
            for e in r["errors"][:3]:
                print(f"       → {e}")
        if args.verbose and r.get("answer"):
            print(f"       answer: {r['answer'][:300]}")

    # ── 汇总 ──
    print_header("评测汇总与 RAG 量化指标")
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = total - passed
    
    # ── 计算 4 维量化指标 ──
    total_kw_expected = sum(r.get("kw_total", 0) for r in results)
    total_kw_hits = sum(r.get("kw_hits", 0) for r in results)
    context_recall = (total_kw_hits / total_kw_expected * 100) if total_kw_expected > 0 else 100.0
    
    clean_cases = sum(1 for r in results if not any("① JSON" in e or "② 重复" in e for e in r.get("fatal_errors", [])))
    cleanliness_rate = (clean_cases / total * 100) if total > 0 else 100.0
    
    isolation_cases = sum(1 for r in results if not any("含禁止词" in e for e in r.get("errors", [])))
    isolation_rate = (isolation_cases / total * 100) if total > 0 else 100.0

    print(f"\n  📊 RAG 核心质量指标:")
    print(f"     • [Context Recall 检索召回率]: {context_recall:.1f}% ({total_kw_hits}/{total_kw_expected} 关键词)")
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

    # ── 硬断言统计 ──
    fatal_total = sum(len(r.get("fatal_errors", [])) for r in results)
    if fatal_total > 0:
        print(f"\n  🔴 硬断言触发: {fatal_total} 次")
        for r in results:
            for fe in r.get("fatal_errors", []):
                print(f"    [{r['id']}] {fe}")

    print(f"\n{'='*70}")
    if passed == total:
        print("🏆 100% PASS — 全部用例通过 (含 5 项硬断言)")
    else:
        print(f"❌ {failed} FAILED — 请修复后重新运行")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())