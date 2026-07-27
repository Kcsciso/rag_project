#!/usr/bin/env python3
"""
=============================================================================
RAG v4 统一回归评测器 (v4.2 — 全量整合版)
=============================================================================

特性:
  - 直接调用 LangGraph run_graph()（无需 FastAPI）
  - --quick: 仅检索不调 LLM（秒级）
  - --filter GT-1,E09: 按 ID 过滤
  - --verbose: 详细输出
  - 4 项硬质量断言（触发任一 → FAIL）

硬断言规则:
  ① JSON 泄露检查: 答案含【提取】或 "steps": [ → FAIL
  ② 重复检查: 连续两段完全相同文本 → FAIL
  ③ 界面套话检查: JAKA APP查询含"未详细记载具体的SDK函数" → FAIL
  ④ 规则匹配: SDK函数名/入参与文档不完全匹配 → FAIL

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
# 4 项硬质量断言
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
        # OpenC3 robot_Power_on() 是无参函数
        (r'(?:Power_on|power_on)\s*\(', r'(?:Power_on|power_on)\s*\(\s*(?:ip|port|address)',
         "robot_Power_on() 是无参函数"),
        # OpenR6 set_robot_power_on() 是无参函数
        (r'set_robot_power_on\s*\(', r'set_robot_power_on\s*\(\s*(?:ip|port|address)',
         "set_robot_power_on() 是无参函数"),
    ]
    for _pos_pattern, _neg_pattern, _msg in checks:
        if re.search(_pos_pattern, answer, re.IGNORECASE) and re.search(_neg_pattern, answer, re.IGNORECASE):
            return True
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
    return errors


# ═══════════════════════════════════════════════════════════
# 评测核心
# ═══════════════════════════════════════════════════════════

def run_single_case(case: dict, quick: bool = False) -> dict:
    from src.graph_rag import run_graph, set_graph_vector_store
    from src.vector_store import (
        load_vector_store, build_bm25_from_chromadb,
        search_similar_with_threshold, _match_function_names, _extract_query_code_entities,
    )
    from src.config import CHROMA_PERSIST_DIR, RETRIEVAL_K
    from src.rag_chain import _is_noise_chunk

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
    }

    t0 = time.time()
    pid = case.get("product_id")

    try:
        if quick:
            from src.rag_chain import _hybrid_retrieve
            docs = _hybrid_retrieve(vs, case["query"], k=RETRIEVAL_K, product_id=pid)
            answer = "\n\n".join(d.page_content[:300] for d in docs)
            model = "retrieval-only"
        else:
            r = run_graph(case["query"], [], product_id=pid)
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
        if not hit:
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

    # ── 4 项硬断言 ──
    fatal = run_fatal_assertions(case, answer)
    result["fatal_errors"] = fatal
    result["errors"].extend(fatal)

    result["status"] = "PASS" if len(result["errors"]) == 0 else "FAIL"
    return result


def print_header(title: str):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAG v4 统一回归评测器")
    parser.add_argument("--quick", "-q", action="store_true", help="仅检索不调LLM")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出(含answer)")
    parser.add_argument("--filter", "-f", type=str, help="用例ID过滤(如 GT-1,E02)")
    args = parser.parse_args()

    cases = load_eval_cases()
    if args.filter:
        ids = set(args.filter.split(","))
        cases = [c for c in cases if c["id"] in ids]

    print_header(f"RAG v4 统一回归评测 ({len(cases)} 用例)")
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
    print_header("评测汇总")
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = total - passed
    print(f"\n  总计: {total}  通过: {passed}  失败: {failed}  通过率: {passed/total*100:.0f}%")

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
        print("🏆 100% PASS — 全部用例通过 (含 4 项硬断言)")
    else:
        print(f"❌ {failed} FAILED — 请修复后重新运行")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
