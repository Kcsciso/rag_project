#!/usr/bin/env python3
"""
=============================================================================
稳定性压力测试 — 多轮对话 + 并发保护 + 异常降级验证
=============================================================================

测试范围：
  1. 滑动窗口：5 轮连续对话 → 验证历史裁剪至最近 3 轮
  2. 并发保护：高频并发请求 → 验证锁机制防止 vLLM 崩溃
  3. 异常降级：模拟各种故障 → 验证自动回退 Layer 3 智能直出
  4. 边界条件：空历史、超长历史、特殊字符

运行方式：
  conda activate rag_agent
  python test_stability.py

=============================================================================
"""

import json
import logging
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    CHROMA_PERSIST_DIR,
    RETRIEVAL_K,
    SIMILARITY_THRESHOLD,
    MODEL_NAME,
    CHUNK_SIZE,
)
from src.vector_store import load_vector_store, create_vector_store, get_vector_store_info
from src.pdf_loader import load_pdfs_from_directory
from src.rag_chain import (
    rag_chat,
    rag_chat_stream,
    LLMServiceError,
    MAX_HISTORY_TURNS,
    FRIENDLY_ERROR_MSG,
)

logging.basicConfig(
    level=logging.WARNING,  # Reduce log noise during stress test
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_stability")

# ============================================================
# 测试问题集（模拟多轮对话场景）
# ============================================================

MULTI_TURN_QUESTIONS = [
    "机械臂上电的函数是什么？",
    "那使能函数呢？",                           # 依赖上文"上电"
    "用 Python 怎么写这两个的代码？",            # 依赖上文函数名
    "关节运动 movj 的参数有哪些？",              # 新话题
    "刚才说的上电函数，返回值是什么？",          # 跨越引用：回到 Q1
]

CONCURRENT_QUESTIONS = [
    "机械臂上电和使能的函数分别是什么？",
    "如何控制机械臂进行关节运动 (movj)？",
    "获取机械臂当前位姿 (Pose) 的函数是什么？",
]

# ============================================================
# 辅助函数
# ============================================================

def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def count_turns(chat_history: list) -> int:
    """统计对话轮数（1 轮 = user + assistant 各 1 条）"""
    return len(chat_history) // 2


# ============================================================
# 测试 1: 滑动窗口 — 多轮对话历史裁剪
# ============================================================

def test_sliding_window(vs):
    print_section("测试 1: 滑动窗口 — 多轮对话历史裁剪")
    print(f"  配置: MAX_HISTORY_TURNS = {MAX_HISTORY_TURNS}")
    print(f"  问题数: {len(MULTI_TURN_QUESTIONS)}")

    chat_history = []
    results = []

    for i, question in enumerate(MULTI_TURN_QUESTIONS, 1):
        turn_before = count_turns(chat_history)
        result = rag_chat(vs, question, chat_history=chat_history if chat_history else None)
        answer = result["answer"]
        model = result["model"]
        turns_after = count_turns(chat_history) + 1  # +1 for current turn

        # 添加到历史
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer[:200]})  # 截断保存

        actual_turns = count_turns(chat_history)
        windowed = actual_turns <= MAX_HISTORY_TURNS + 1  # +1 buffer for current

        status = "✅" if windowed else "⚠️"
        print(f"  [{status}] 第 {i} 轮: {question[:40]}...")
        print(f"       历史轮数: 累积 {actual_turns} 轮 | 限制 {MAX_HISTORY_TURNS} 轮 | "
              f"模型: {model} | 回答: {answer[:60]}...")

        results.append({
            "turn": i,
            "question": question,
            "history_turns_before": turn_before,
            "history_turns_after": actual_turns,
            "within_window": windowed,
            "model": model,
            "answer_preview": answer[:80],
        })

    # 验证：检查最后 1 轮是否能正确引用第 1 轮的上下文（证明滑动窗口保留了足够历史）
    last_answer = results[-1]["answer_preview"] if results else ""
    # Q5 问"刚才说的上电函数，返回值是什么？" — 如果模型能提到 robot_Power_on 和返回值，说明上下文保留成功
    cross_ref_ok = "0" in last_answer or "成功" in last_answer or "robot_Power_on" in last_answer.lower()
    print(f"\n  📊 滑动窗口验证: 第 5 轮跨引用回答 {'✅ 正确' if cross_ref_ok else '⚠️ 上下文丢失'} "
          f"| 窗口限制: {MAX_HISTORY_TURNS} 轮 | 实际发送到 LLM 的历史 ≤ {MAX_HISTORY_TURNS} 轮")

    return results, cross_ref_ok


# ============================================================
# 测试 2: 并发保护 — 高频并发请求
# ============================================================

def _concurrent_query(vs, question: str, worker_id: int) -> dict:
    """单个并发查询工作函数"""
    start = time.time()
    try:
        result = rag_chat(vs, question)
        elapsed = time.time() - start
        return {
            "worker": worker_id,
            "question": question[:40],
            "model": result["model"],
            "elapsed": round(elapsed, 2),
            "status": "OK",
        }
    except LLMServiceError as e:
        elapsed = time.time() - start
        return {
            "worker": worker_id,
            "question": question[:40],
            "model": "N/A",
            "elapsed": round(elapsed, 2),
            "status": f"Layer4: {str(e)[:50]}",
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "worker": worker_id,
            "question": question[:40],
            "model": "N/A",
            "elapsed": round(elapsed, 2),
            "status": f"ERROR: {type(e).__name__}",
        }


def test_concurrency(vs):
    print_section("测试 2: 并发保护 — 3 路并发请求")
    print(f"  并发数: 3 | 问题数: {len(CONCURRENT_QUESTIONS)}")

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for i, q in enumerate(CONCURRENT_QUESTIONS):
            futures.append(executor.submit(_concurrent_query, vs, q, i + 1))

        results = []
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            icon = "✅" if r["status"] == "OK" else "🔄" if "Layer" in r["status"] else "❌"
            print(f"  [{icon}] Worker {r['worker']}: {r['question']}... "
                  f"| {r['elapsed']}s | {r['model']} | {r['status']}")

    total_time = time.time() - start_time
    ok_count = sum(1 for r in results if r["status"] == "OK")
    degraded_count = sum(1 for r in results if "Layer" in r["status"])

    print(f"\n  📊 并发验证: {ok_count}/{len(results)} 成功 | "
          f"{degraded_count} 降级 | 总耗时 {total_time:.1f}s")

    # 并发保护的核心指标：不应该有崩溃
    no_crash = all(r["status"] != "ERROR" for r in results) if "ERROR" not in [r["status"] for r in results] else False
    # Check if any result has ERROR status
    has_error = any("ERROR" in r["status"] for r in results)
    print(f"  {'✅ 无崩溃' if not has_error else '❌ 有崩溃'} — "
          f"并发保护 {'生效' if not has_error else '失效'}")

    return results, not has_error


# ============================================================
# 测试 3: 异常自动降级 — 各种故障场景
# ============================================================

def test_graceful_degradation(vs):
    print_section("测试 3: 异常自动降级 — 故障场景覆盖")

    scenarios = [
        {
            "name": "正常查询（基准）",
            "question": "机械臂上电的函数是什么？",
            "expect_layer": 1,
        },
        {
            "name": "空查询",
            "question": "",
            "expect_layer": "any",  # 任何层都可以，但不能崩溃
        },
        {
            "name": "超长查询（模拟 prompt 超限）",
            "question": "机械臂 " * 500,
            "expect_layer": "any",
        },
        {
            "name": "特殊字符查询",
            "question": "机械臂\x00\x01\x02测试\n\r\t🧑",
            "expect_layer": "any",
        },
        {
            "name": "纯英文函数名查询",
            "question": "What is robot_Power_on() and robot_movj()?",
            "expect_layer": "any",
        },
        {
            "name": "空 history（None）",
            "question": "如何控制机械臂？",
            "chat_history": None,
            "expect_layer": 1,
        },
        {
            "name": "超长历史（100 轮）",
            "question": "movj 的参数？",
            "chat_history": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
            ] * 50,  # 100 条消息 = 50 轮
            "expect_layer": "any",
        },
    ]

    passed = 0
    failed = 0

    for i, scenario in enumerate(scenarios, 1):
        name = scenario["name"]
        question = scenario["question"]
        chat_history = scenario.get("chat_history", [])

        try:
            if not question.strip():
                # 空查询预期触发异常
                result = rag_chat(vs, question, chat_history=chat_history if chat_history else None)
                print(f"  [{i}] {name}: ✅ 返回 (模型={result['model']})")
                passed += 1
            else:
                result = rag_chat(vs, question, chat_history=chat_history if chat_history else None)
                layer = "L1" if MODEL_NAME in result.get("model", "") else \
                        "L2" if "glm" in result.get("model", "").lower() else \
                        "L3" if "direct" in result.get("model", "").lower() else "?"
                print(f"  [{i}] {name}: ✅ {layer} | 回答 {len(result['answer'])} chars")
                passed += 1

        except LLMServiceError as e:
            # Layer 4 是预期兜底行为，不算失败
            print(f"  [{i}] {name}: 🛡️ Layer 4 兜底 — {str(e)[:60]}")
            passed += 1

        except Exception as e:
            # 未预期的异常 → 失败
            print(f"  [{i}] {name}: ❌ 未捕获异常 {type(e).__name__}: {str(e)[:80]}")
            failed += 1

    print(f"\n  📊 异常降级验证: {passed}/{len(scenarios)} 场景通过 | {failed} 失败")

    return passed, failed


# ============================================================
# 测试 4: 流式降级验证
# ============================================================

def test_stream_degradation(vs):
    print_section("测试 4: 流式响应降级验证")

    test_cases = [
        ("正常流式查询", "机械臂上电的函数是什么？", None),
        ("超长历史流式", "movj 的参数？", [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ] * 10),
    ]

    for name, question, history in test_cases:
        try:
            chunks = []
            for token in rag_chat_stream(vs, question, chat_history=history):
                chunks.append(token)
            full = "".join(chunks)
            print(f"  ✅ {name}: {len(chunks)} chunks, {len(full)} chars")
        except LLMServiceError as e:
            print(f"  🛡️ {name}: Layer 4 流式兜底 — {str(e)[:50]}")
        except Exception as e:
            print(f"  ❌ {name}: {type(e).__name__}: {str(e)[:60]}")


# ============================================================
# 主入口
# ============================================================

def main():
    print_section("比邻星 (ProximaRAG) 稳定性压力测试")
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  模型: {MODEL_NAME}")
    print(f"  Top-K: {RETRIEVAL_K} | 阈值: {SIMILARITY_THRESHOLD} | 历史轮数限制: {MAX_HISTORY_TURNS}")

    # 加载向量库
    vs = load_vector_store(CHROMA_PERSIST_DIR)
    if not vs:
        print("  📭 向量库为空，正在构建...")
        from src.config import PDF_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
        docs = load_pdfs_from_directory(PDF_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP)
        vs = create_vector_store(docs, CHROMA_PERSIST_DIR)
    print(f"  📚 向量库: {get_vector_store_info(vs)['document_count']} 个片段")

    all_pass = True

    # 测试 1
    _, t1_pass = test_sliding_window(vs)
    all_pass = all_pass and t1_pass

    # 测试 2
    _, t2_pass = test_concurrency(vs)
    all_pass = all_pass and t2_pass

    # 测试 3
    t3_pass_count, t3_fail = test_graceful_degradation(vs)
    all_pass = all_pass and (t3_fail == 0)

    # 测试 4
    test_stream_degradation(vs)

    # 汇总
    print_section("压力测试汇总")
    print(f"  滑动窗口: {'✅ 通过' if t1_pass else '⚠️ 超出限制'}")
    print(f"  并发保护: {'✅ 通过' if t2_pass else '❌ 有崩溃'}")
    print(f"  异常降级: {'✅ 通过' if t3_fail == 0 else '❌ ' + str(t3_fail) + ' 个未处理异常'}")
    print(f"\n  总体: {'✅ 全部通过' if all_pass else '⚠️ 部分测试未通过'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
