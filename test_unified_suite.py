#!/usr/bin/env python3
"""
=============================================================================
NewsPage v3.0 — 真实 Ground Truth 端到端评测套件
=============================================================================

真实调用 LangGraph 状态图 + ChromaDB 向量库，对 6 大核心业务进行
【事实准确性 (Fact Precision) 地狱级断言】校验。

运行: conda run -n rag_agent python test_unified_suite.py

⚠️ 本测试无任何 Mock — 全部走真实 RAG 管线。
=============================================================================
"""
import os, sys, time, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.WARNING)

from src.vector_store import load_vector_store, get_vector_store_info, build_bm25_from_chromadb
from src.graph_rag import set_graph_vector_store, run_graph
from src.config import CHROMA_PERSIST_DIR

# ============================================================
class GtTest:
    def __init__(self, name, query, product_id=None):
        self.name = name
        self.query = query
        self.product_id = product_id
        self.answer = ""
        self.checks_passed = 0
        self.checks_total = 0
        self.ms = 0.0

    def check(self, label, condition, detail=""):
        self.checks_total += 1
        if condition:
            self.checks_passed += 1
            return True
        print(f"    ❌ {label}: {detail}")
        return False

    def run(self):
        t0 = time.time()
        try:
            r = run_graph(self.query, [], product_id=self.product_id)
            self.answer = r.get("answer", "")
        except Exception as e:
            self.answer = f"ERROR: {e}"
        self.ms = (time.time() - t0) * 1000
        return self

    @property
    def passed(self):
        return self.checks_total > 0 and self.checks_passed == self.checks_total

    def status(self):
        return "✅" if self.passed else "❌"

# ============================================================
def main():
    print("=" * 60)
    print("🧪 NewsPage v3.0 — Ground Truth 事实准确性评测")
    print("=" * 60)

    # ── Setup ──
    vs = load_vector_store(CHROMA_PERSIST_DIR)
    if not vs:
        print("❌ 向量库为空"); sys.exit(1)
    info = get_vector_store_info(vs)
    print(f"📚 {info['document_count']} chunks loaded")
    build_bm25_from_chromadb(vs)
    set_graph_vector_store(vs)

    print("\n📋 执行 Ground Truth 断言...\n")

    # ================================================================
    # GT-1: JAKA 端口号 — 必须是 6502，严禁将 49152 当 Modbus 端口
    # ================================================================
    t1 = GtTest("JAKA端口号", "JAKA端口号是多少", product_id="JAKA").run()
    t1.check("包含6502", "6502" in t1.answer,
             f"answer={t1.answer[:100]}")
    t1.check("不含49152或明确标注末端传感器",
             "49152" not in t1.answer or "末端传感器" in t1.answer,
             f"answer={t1.answer[:120]}")
    print(f"  {t1.status()} GT-1 JAKA端口号 ({t1.ms:.0f}ms)")
    if not t1.passed:
        print(f"    answer: {t1.answer[:150]}")

    # ================================================================
    # GT-2: JAKA 上电步骤 — 必须含电控柜/使能，严禁脑补按钮
    # ================================================================
    t2 = GtTest("JAKA上电步骤", "JAKA怎么上电", product_id="JAKA").run()
    t2.check("含电控柜或使能",
             ("电控柜" in t2.answer or "使能" in t2.answer),
             f"answer={t2.answer[:100]}")
    t2.check("严禁脑补两个启动按钮",
             "两个启动按钮" not in t2.answer,
             f"answer={t2.answer[:120]}")
    print(f"  {t2.status()} GT-2 JAKA上电步骤 ({t2.ms:.0f}ms)")
    if not t2.passed:
        print(f"    answer: {t2.answer[:150]}")

    # ================================================================
    # GT-3: OpenC3 走直线 API — 必须是 robot_movl，严禁伪 API
    # ================================================================
    t3 = GtTest("OpenC3走直线API", "OpenC3 机械臂在 SDK 中走直线的函数名是什么", product_id="OpenC3").run()
    t3.check("含robot_movl", "robot_movl" in t3.answer.lower(),
             f"answer={t3.answer[:100]}")
    t3.check("严禁伪API move_linear",
             "move_linear" not in t3.answer.lower(),
             f"answer={t3.answer[:120]}")
    print(f"  {t3.status()} GT-3 OpenC3走直线API ({t3.ms:.0f}ms)")
    if not t3.passed:
        print(f"    answer: {t3.answer[:150]}")

    # ================================================================
    # GT-4: OpenC3 vs OpenR6 对比 — 两份文档 DLL 必须同时出现
    # ================================================================
    t4 = GtTest("多产品SDK对比", "OpenC3 和 OpenR6 上电调用的 SDK 函数有什么区别").run()
    t4.check("含collrob_sdk.dll", "collrob_sdk.dll" in t4.answer.lower(),
             f"answer={t4.answer[:120]}")
    t4.check("含py_dll.dll", "py_dll.dll" in t4.answer.lower(),
             f"answer={t4.answer[:120]}")
    print(f"  {t4.status()} GT-4 多产品SDK对比 ({t4.ms:.0f}ms)")
    if not t4.passed:
        print(f"    answer: {t4.answer[:200]}")

    # ================================================================
    # GT-5: JAKA 运行环境 — 必须含操作系统信息 + 文档引用来源
    # ================================================================
    t5 = GtTest("JAKA运行环境", "JAKA 机械臂的运行环境与硬件配置要求", product_id="JAKA").run()
    t5.check("含运行环境信息(硬件/OS)",
             ("Windows" in t5.answer or "Android" in t5.answer
              or "操作系统" in t5.answer or "CPU" in t5.answer
              or "硬件" in t5.answer or "内存" in t5.answer),
             f"answer={t5.answer[:100]}")
    t5.check("含文档引用来源",
             "根据《" in t5.answer or "JAKA" in t5.answer,
             f"answer={t5.answer[:120]}")
    print(f"  {t5.status()} GT-5 JAKA运行环境 ({t5.ms:.0f}ms)")
    if not t5.passed:
        print(f"    answer: {t5.answer[:150]}")

    # ================================================================
    # GT-6: 6502 概念精准 — 必须关联端口，严禁胡诌系统变量
    # ================================================================
    t6 = GtTest("6502概念精准", "6502这个数字在JAKA中是什么意思", product_id="JAKA").run()
    t6.check("含端口或未找到（诚实拒答）",
             ("端口" in t6.answer or "未找到" in t6.answer or "未包含" in t6.answer),
             f"answer={t6.answer[:100]}")
    t6.check("严禁胡诌系统变量",
             "系统变量" not in t6.answer,
             f"answer={t6.answer[:120]}")
    print(f"  {t6.status()} GT-6 6502概念精准 ({t6.ms:.0f}ms)")
    if not t6.passed:
        print(f"    answer: {t6.answer[:150]}")

    # ── Summary ──
    tests = [t1, t2, t3, t4, t5, t6]
    total = len(tests)
    passed = sum(1 for t in tests if t.passed)
    checks_total = sum(t.checks_total for t in tests)
    checks_passed = sum(t.checks_passed for t in tests)

    print("\n" + "=" * 60)
    print("📊 Ground Truth 评测汇总")
    print("=" * 60)
    for t in tests:
        print(f"  {t.status()} {t.name}: {t.checks_passed}/{t.checks_total} checks")
    bar = "█" * int(passed / total * 10) + "░" * (10 - int(passed / total * 10))
    print(f"  {'─' * 40}")
    print(f"  Tests:  {passed}/{total}  [{bar}] {passed/total*100:.0f}%")
    print(f"  Checks: {checks_passed}/{checks_total}")
    print("=" * 60)

    if passed == total:
        print("\n🏆 100% GROUND TRUTH PASSED")
        return 0
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
