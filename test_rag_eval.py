#!/usr/bin/env python3
"""
=============================================================================
RAG 防过拟合自动化评测脚本 — test_rag_eval.py
=============================================================================

评估目标：
  1. 核心场景 5 项：验证产品路由、物理隔离、SDK 函数精确匹配
  2. 防过拟合泛化 3 项：验证系统能从文档中自行检索到未硬编码的函数名

运行方式：
  conda activate rag_agent
  python test_rag_eval.py

输出：
  - 每个用例的通过/失败状态
  - 失败用例的详细根因分析
  - 汇总统计
=============================================================================
"""

import json
import re
import sys
import time
from typing import Dict, List, Tuple, Optional

import requests

# ============================================================
# 配置
# ============================================================
BASE_URL = "http://localhost:7860"
CHAT_ENDPOINT = f"{BASE_URL}/api/chat"
TIMEOUT = 60  # 单次请求最长等待时间（秒）

# ============================================================
# 评测用例定义
# ============================================================
# 每个用例: (id, 类别, query, product_id, 必须包含的关键词列表, 必须不包含的关键词列表)
#   - product_id=None 表示不传，交由后端 Product Router 自动识别
#   - must_contain: 答案中必须出现的字符串（任一匹配即通过该项）
#   - must_not_contain: 答案中绝对不能出现的字符串（如跨库串词）

EVAL_CASES = [
    # ═══════════════════════════════════════════════════════════
    # 核心场景 5 项
    # ═══════════════════════════════════════════════════════════
    {
        "id": "CORE-1",
        "category": "核心场景",
        "query": "请问上电函数怎么写？",
        "product_id": None,  # 未指定产品 → 应触发澄清反问
        "must_contain": ["哪一款产品", "OpenR6", "OpenC3"],
        "must_not_contain": [],
        "expect_clarification": True,
        "description": "未指定产品时应触发澄清反问，绝不跨库盲猜",
    },
    {
        "id": "CORE-2",
        "category": "核心场景",
        "query": "OpenR6 SDK 里的上电和回零函数怎么写？",
        "product_id": None,  # 让 Product Router 自动识别 OpenR6
        "must_contain": ["set_robot_power_on", "set_robot_arm_home"],
        "must_not_contain": ["collrob_sdk", "collrob.dll"],  # 不能跨库输出 OpenC3 的动态库
        "expect_clarification": False,
        "description": "OpenR6 场景必须精准包含 py_dll 系函数",
    },
    {
        "id": "CORE-3",
        "category": "核心场景",
        "query": "在 py_dll 库里，机械臂末端怎么画直线？",
        "product_id": None,  # py_dll → Product Router → OpenR6
        "must_contain": ["set_move_line", "POSE"],
        "must_not_contain": ["robot_movl", "collrob"],  # 不能串到 OpenC3 的 movl
        "expect_clarification": False,
        "description": "py_dll/OpenR6 直线运动必须包含 set_move_line + POSE",
    },
    {
        "id": "CORE-4",
        "category": "核心场景",
        "query": "OpenC3 六轴机械臂打开抱闸和使能的函数是什么？",
        "product_id": None,
        "must_contain": ["robot_brkopen", "robot_enable"],
        "must_not_contain": ["set_robot_power_on", "py_dll"],  # 不能串到 OpenR6
        "expect_clarification": False,
        "description": "OpenC3 抱闸+使能必须包含 collrob 系函数名",
    },
    {
        "id": "CORE-5",
        "category": "核心场景",
        "query": "collrob_sdk 里面是怎么控制机械臂走直线 movl 的？",
        "product_id": None,
        "must_contain": ["robot_movl", "POSE"],
        "must_not_contain": ["set_move_line", "py_dll"],  # 不能串到 OpenR6
        "expect_clarification": False,
        "description": "OpenC3 movl 必须包含 robot_movl + POSE 结构体",
    },

    # ═══════════════════════════════════════════════════════════
    # 防过拟合泛化场景 3 项（验证系统检索泛化能力）
    # ═══════════════════════════════════════════════════════════
    {
        "id": "GEN-1",
        "category": "泛化场景",
        "query": "OpenR6 获取所有关节角度",
        "product_id": None,
        "must_contain": ["get_robot_joint_angle_all"],
        "must_not_contain": [],
        "expect_clarification": False,
        "description": "泛化：OpenR6 关节角度 → get_robot_joint_angle_all()",
    },
    {
        "id": "GEN-2",
        "category": "泛化场景",
        "query": "OpenC3 怎么读取各个关节的角度数值",
        "product_id": None,
        "must_contain": ["get_robot_joint_all"],
        "must_not_contain": [],
        "expect_clarification": False,
        "description": "泛化：OpenC3 关节角度 → get_robot_joint_all()",
    },
    {
        "id": "GEN-3",
        "category": "泛化场景",
        "query": "OpenR6 设置 1 号 IO 输出高电平",
        "product_id": None,
        "must_contain": ["set_robot_io_output"],
        "must_not_contain": [],
        "expect_clarification": False,
        "description": "泛化：OpenR6 IO 输出 → set_robot_io_output(1, 1)",
    },
]


# ============================================================
# 评测引擎
# ============================================================

def run_single_case(case: dict) -> dict:
    """
    执行单个评测用例，返回结果字典。
    """
    result = {
        "id": case["id"],
        "category": case["category"],
        "query": case["query"],
        "description": case["description"],
        "passed": False,
        "model": "?",
        "layer": "?",
        "elapsed_s": 0.0,
        "answer": "",
        "checks": {},
        "errors": [],
    }

    fd = {"query": case["query"], "stream": "false"}
    if case.get("product_id"):
        fd["product_id"] = case["product_id"]

    t0 = time.time()
    try:
        resp = requests.post(CHAT_ENDPOINT, data=fd, timeout=TIMEOUT)
        result["elapsed_s"] = round(time.time() - t0, 2)
        data = resp.json()
    except requests.Timeout:
        result["elapsed_s"] = round(time.time() - t0, 2)
        result["errors"].append(f"请求超时 (>{TIMEOUT}s)")
        return result
    except Exception as e:
        result["elapsed_s"] = round(time.time() - t0, 2)
        result["errors"].append(f"请求异常: {e}")
        return result

    answer = data.get("answer", "")
    result["answer"] = answer
    result["model"] = data.get("model", "?")
    result["needs_clarification"] = data.get("needs_clarification", False)

    # 判断容灾层级
    if "direct-retrieval" in str(data.get("model", "")):
        result["layer"] = "L3 (纯检索直出)"
    elif "product-clarification" in str(data.get("model", "")):
        result["layer"] = "L0 (产品澄清)"
    else:
        result["layer"] = "L1/L2 (LLM 生成)"

    # ── 检查期望的澄清行为 ──
    if case.get("expect_clarification"):
        result["checks"]["澄清反问"] = result["needs_clarification"]
        if not result["needs_clarification"]:
            result["errors"].append("期望触发澄清反问但未触发")

    # ── 检查必须包含的关键词 ──
    for kw in case.get("must_contain", []):
        found = kw.lower() in answer.lower()
        result["checks"][f"含 '{kw}'"] = found
        if not found:
            result["errors"].append(f"答案中缺少关键词: '{kw}'")

    # ── 检查必须不包含的关键词（跨库串词） ──
    for kw in case.get("must_not_contain", []):
        found = kw.lower() in answer.lower()
        result["checks"][f"不含 '{kw}'"] = not found
        if found:
            result["errors"].append(f"答案中包含禁止关键词（跨库串词）: '{kw}'")

    # ── 判定是否通过 ──
    result["passed"] = len(result["errors"]) == 0

    return result


def print_header(title: str):
    """打印分隔标题"""
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def print_result(result: dict, verbose: bool = True):
    """打印单个用例的评测结果"""
    status = "✅ PASS" if result["passed"] else "❌ FAIL"
    print(f"\n{'─' * 60}")
    print(f"  [{result['id']}] {result['category']} | {status} | {result['layer']} | {result['elapsed_s']}s")
    print(f"  Query: {result['query']}")
    print(f"  Desc:  {result['description']}")
    print(f"  Model: {result['model']}")

    if result["checks"]:
        for check_name, check_ok in result["checks"].items():
            icon = "✅" if check_ok else "❌"
            print(f"    {icon} {check_name}")

    if result["errors"]:
        print(f"  ⚠️  错误详情:")
        for err in result["errors"]:
            print(f"      • {err}")

    if verbose and result["answer"]:
        # 截取答案前 500 字符
        preview = result["answer"][:500].replace("\n", "\n      ")
        print(f"  📝 答案预览:\n      {preview}")
        if len(result["answer"]) > 500:
            print(f"      ... (共 {len(result['answer'])} 字符)")


def print_summary(results: List[dict]):
    """打印汇总统计"""
    print_header("📊 评测汇总")

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    print(f"\n  总计: {total}  通过: {passed}  失败: {failed}  通过率: {passed/total*100:.0f}%")

    # 按类别统计
    for cat in ["核心场景", "泛化场景"]:
        cat_results = [r for r in results if r["category"] == cat]
        cat_passed = sum(1 for r in cat_results if r["passed"])
        print(f"  {cat}: {cat_passed}/{len(cat_results)} 通过")

    # 失败用例列表
    if failed > 0:
        print(f"\n  ❌ 失败用例:")
        for r in results:
            if not r["passed"]:
                print(f"    [{r['id']}] {r['description']}")
                for err in r["errors"]:
                    print(f"        → {err}")

    # 容灾层级分布
    layers = {}
    for r in results:
        l = r.get("layer", "?")
        layers[l] = layers.get(l, 0) + 1
    print(f"\n  容灾层级分布: {layers}")

    # 平均耗时
    avg_time = sum(r["elapsed_s"] for r in results) / total
    print(f"  平均耗时: {avg_time:.1f}s")


# ============================================================
# 主流程
# ============================================================
def main():
    print_header("🔬 RAG 防过拟合自动化评测")
    print(f"  端点: {CHAT_ENDPOINT}")
    print(f"  用例数: {len(EVAL_CASES)} (核心场景 5 + 泛化场景 3)")
    print(f"  超时: {TIMEOUT}s/用例")

    # 前置检查
    try:
        status = requests.get(f"{BASE_URL}/api/status", timeout=10).json()
        if not status.get("ready"):
            print("\n  ❌ 向量库未就绪！请先上传 PDF 文档。")
            sys.exit(1)
        print(f"  ✅ 向量库就绪: {status['document_count']} 个片段")
    except Exception as e:
        print(f"\n  ❌ 无法连接 FastAPI: {e}")
        sys.exit(1)

    try:
        products = requests.get(f"{BASE_URL}/api/products", timeout=5).json()
        print(f"  ✅ 已注册产品: {products['products']}")
    except Exception:
        print(f"  ⚠️  无法获取产品列表")

    # 执行评测
    results = []
    for case in EVAL_CASES:
        print(f"\n  ⏳ 执行 [{case['id']}] {case['description'][:50]}...", end=" ", flush=True)
        result = run_single_case(case)
        results.append(result)
        status = "✅" if result["passed"] else "❌"
        print(f"{status} ({result['elapsed_s']}s)")

    # 详细输出
    print_header("📋 详细结果")
    for result in results:
        print_result(result, verbose=True)

    # 汇总
    print_summary(results)

    # 返回退出码（CI 友好）
    failed_count = sum(1 for r in results if not r["passed"])
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
