#!/usr/bin/env python3
"""
=============================================================================
机械臂 SDK 文档 RAG 自动化测试脚本
=============================================================================

测试范围：
  1. PDF 加载 → 向量库构建
  2. ChromaDB 语义检索（3 个典型机械臂控制问题）
  3. RAG 四层容灾全链路验证（Layer 1 vLLM → Layer 2 智谱 GLM-4.7-Flash → Layer 3 纯检索直出）
  4. 回答质量检查（ctypes 类型转换、函数名、参数匹配）

运行方式：
  conda activate rag_agent
  python test_robot_rag.py

=============================================================================
"""

import json
import logging
import os
import sys
import time

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.documents import Document

from src.config import (
    PDF_DATA_DIR,
    CHROMA_PERSIST_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVAL_K,
    MODEL_NAME,
    DEEPSEEK_MODEL,
)
from src.pdf_loader import load_pdfs_from_directory
from src.vector_store import (
    create_vector_store,
    load_vector_store,
    get_vector_store_info,
)
from src.rag_chain import (
    rag_chat,
    rag_chat_stream,
    _format_direct_retrieval_answer,
    DIRECT_RETRIEVAL_K,
    FRIENDLY_ERROR_MSG,
)
from src.vector_store import search_similar_with_threshold
from src.config import SIMILARITY_THRESHOLD

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_robot_rag")

# ============================================================
# 测试问题集（3 个典型机械臂控制场景）
# ============================================================

TEST_QUESTIONS = [
    {
        "id": "Q1",
        "question": "机械臂上电和使能的函数分别是什么？请给出 Python 示例代码。",
        "keywords": ["robot_Power_on", "robot_motor_enable", "ctypes", "上电", "使能"],
        "description": "验证：上电/使能函数名 + ctypes 调用示例",
    },
    {
        "id": "Q2",
        "question": "如何控制机械臂进行关节运动 (movj)？参数有哪些？",
        "keywords": ["movj", "关节运动", "joint", "速度", "加速度", "RobJoint"],
        "description": "验证：movj 函数名 + 关节参数结构体",
    },
    {
        "id": "Q3",
        "question": "获取机械臂当前位姿 (Pose) 的函数是什么？",
        "keywords": ["pose", "位姿", "robot_get_pose", "px", "py", "pz", "Rx", "Ry", "Rz"],
        "description": "验证：位姿获取函数 + Pose 结构体字段",
    },
    {
        "id": "Q4",
        "question": "摄像头支持哪些分辨率和帧率？",
        "keywords": [],  # 期望：所有文档切片被阈值过滤，无相关结果
        "description": "验证：无关问题经相似度阈值过滤后 Layer 3 返回空结果提示",
        "expect_empty": True,  # 标记：期望检索结果为空
    },
]

# ============================================================
# 辅助函数
# ============================================================

def count_keyword_hits(text: str, keywords: list) -> int:
    """统计回答中命中的关键词数量"""
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits


def print_separator(title: str, char: str = "=", width: int = 70):
    """打印格式化的分隔标题"""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")


# ============================================================
# 主测试流程
# ============================================================

def main():
    print_separator("六轴机械臂 SDK 文档 — RAG 自动化测试", "█")
    print(f"  测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  主 LLM 通道: {MODEL_NAME} (local vLLM)")
    print(f"  降级通道: {DEEPSEEK_MODEL} (智谱 GLM-4.7-Flash)")
    print(f"  检索参数: Top-K={RETRIEVAL_K}, chunk_size={CHUNK_SIZE}, threshold={SIMILARITY_THRESHOLD}")
    print(f"  测试问题数: {len(TEST_QUESTIONS)}")

    # ================================================================
    # 阶段一：PDF 加载与向量库构建
    # ================================================================
    print_separator("阶段一: PDF 加载与向量库构建", "-")

    # 检查是否有已有向量库
    vs = load_vector_store(CHROMA_PERSIST_DIR)
    if vs:
        info = get_vector_store_info(vs)
        print(f"  📚 已加载已有向量库: {info['document_count']} 个文档片段")
    else:
        print("  📭 向量库为空，正在从 PDF 构建...")
        documents = load_pdfs_from_directory(PDF_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP)

        if not documents:
            print("  ❌ 未找到任何 PDF 文件或提取到有效文本！")
            sys.exit(1)

        total_chars = sum(len(d.page_content) for d in documents)
        print(f"  📄 PDF 提取完成: {len(documents)} 个切片, {total_chars} 字符")
        print(f"  📄 平均每片: {total_chars // len(documents)} 字符")

        # 打印文档结构概览
        sources = set(d.metadata.get("source", "?") for d in documents)
        print(f"  📄 来源文件: {sources}")

        vs = create_vector_store(documents, CHROMA_PERSIST_DIR)
        info = get_vector_store_info(vs)
        print(f"  ✅ 向量库构建完成: {info['document_count']} 个片段已索引")

    # ================================================================
    # 阶段二：ChromaDB 语义检索验证
    # ================================================================
    print_separator("阶段二: ChromaDB 语义检索验证", "-")

    for tq in TEST_QUESTIONS:
        qid = tq["id"]
        question = tq["question"]
        keywords = tq["keywords"]

        print(f"  [{qid}] {question}")
        print(f"      预期关键词: {keywords}")

        # 检索 Top-K 文档（带相似度阈值过滤）
        context_docs = search_similar_with_threshold(
            vs, question, k=RETRIEVAL_K, threshold=SIMILARITY_THRESHOLD
        )
        print(f"      检索到 {len(context_docs)} 个相关片段（阈值过滤后）:")

        for i, doc in enumerate(context_docs, 1):
            source = doc.metadata.get("source", "?")
            content_preview = doc.page_content[:120].replace("\n", " ")
            print(f"        [{i}] {source} | {content_preview}...")

        # 检查关键词覆盖
        all_content = " ".join(d.page_content for d in context_docs)
        hits = count_keyword_hits(all_content, keywords)
        print(f"      📊 关键词命中: {hits}/{len(keywords)}")
        print()

    # ================================================================
    # 阶段三：RAG 全链路测试（容灾层级验证）
    # ================================================================
    print_separator("阶段三: RAG 全链路测试 — 四层容灾验证", "-")

    results = []

    for tq in TEST_QUESTIONS:
        qid = tq["id"]
        question = tq["question"]
        keywords = tq["keywords"]
        description = tq["description"]

        print(f"  ┌─ [{qid}] {question}")
        print(f"  │  验证目标: {description}")
        print(f"  │  预期关键词: {keywords}")

        # ---- 3a. 打印向量检索原文 ----
        print(f"  │")
        print(f"  │  📖 ChromaDB 检索到的原始文档切片 (Context Chunks, threshold={SIMILARITY_THRESHOLD}):")
        context_docs = search_similar_with_threshold(
            vs, question, k=RETRIEVAL_K, threshold=SIMILARITY_THRESHOLD
        )
        for i, doc in enumerate(context_docs, 1):
            source = doc.metadata.get("source", "?")
            content = doc.page_content.strip()
            print(f"  │  ┌─ [切片 {i}] 来源: {source} ({len(content)} chars)")
            # 缩进打印内容
            for line in content.split("\n")[:8]:  # 最多显示 8 行
                print(f"  │  │ {line[:100]}")
            if len(content.split("\n")) > 8:
                print(f"  │  │ ... (共 {len(content.split(chr(10)))} 行)")
            print(f"  │  └─")

        # ---- 3b. 执行 RAG 对话 ----
        print(f"  │")
        print(f"  │  🤖 正在调用 RAG 管线...")
        try:
            result = rag_chat(vs, question, k=RETRIEVAL_K)
            answer = result["answer"]
            model_used = result["model"]
            sources = result["sources"]

            # 判断容灾层级
            if "direct-retrieval" in model_used:
                layer = "Layer 3 — 纯向量检索直出模式 (CPU-only)"
                layer_num = 3
            elif DEEPSEEK_MODEL in model_used or "glm" in model_used.lower():
                layer = f"Layer 2 — 智谱 GLM-4.7-Flash API 降级"
                layer_num = 2
            elif MODEL_NAME in model_used:
                layer = f"Layer 1 — 本地 vLLM 推理"
                layer_num = 1
            else:
                layer = f"Unknown — model={model_used}"
                layer_num = 0

            print(f"  │  ✅ 调用成功")
            print(f"  │  🏷️  模型: {model_used}")
            print(f"  │  🛡️  容灾层级: {layer}")
            print(f"  │  📄 来源文件: {sources}")

            # ---- 3c. 回答质量检查 ----
            hits = count_keyword_hits(answer, keywords)
            if len(keywords) > 0:
                hit_rate = hits / len(keywords) * 100
            else:
                hit_rate = 100.0  # 无关键词时（如 Q4 期望空结果），默认满分
            print(f"  │  📊 关键词命中: {hits}/{len(keywords)} ({hit_rate:.0f}%)")

            # 判断状态：expect_empty 标记的问题使用特殊逻辑
            is_empty_result = (
                "未在现有文档中检索到" in answer
                or len(context_docs) == 0
            )
            if tq.get("expect_empty"):
                test_status = "PASS" if is_empty_result else "WARN"
            elif hit_rate >= 30:
                test_status = "PASS"
            else:
                test_status = "WARN"

            # 打印完整回答（截断显示）
            print(f"  │  ┌─ 完整回答 ({len(answer)} chars) ─")
            for line in answer.split("\n")[:20]:
                print(f"  │  │ {line[:120]}")
            if len(answer.split("\n")) > 20:
                print(f"  │  │ ... (共 {len(answer.split(chr(10)))} 行)")
            print(f"  │  └─")

            # ---- 3d. 流式输出测试 ----
            print(f"  │")
            print(f"  │  🌊 流式输出测试 (前 200 chars): ", end="", flush=True)
            streamed = ""
            try:
                for token in rag_chat_stream(vs, question, k=RETRIEVAL_K):
                    streamed += token
                    if len(streamed) <= 200:
                        print(token, end="", flush=True)
                print()
                print(f"  │  🌊 流式总长度: {len(streamed)} chars")
            except Exception as e:
                print(f"\n  │  ⚠️  流式失败: {e}")

            results.append({
                "id": qid,
                "question": question,
                "model": model_used,
                "layer": layer_num,
                "layer_name": layer,
                "keyword_hits": hits,
                "keyword_total": len(keywords),
                "answer_length": len(answer),
                "stream_length": len(streamed) if streamed else 0,
                "sources": sources,
                "status": test_status,
            })

        except Exception as e:
            print(f"  │  ❌ RAG 调用失败: {e}")
            results.append({
                "id": qid,
                "question": question,
                "model": "N/A",
                "layer": 4,
                "layer_name": f"Layer 4 — 友好错误: {FRIENDLY_ERROR_MSG}",
                "keyword_hits": 0,
                "keyword_total": len(keywords),
                "answer_length": 0,
                "stream_length": 0,
                "sources": [],
                "status": "FAIL",
            })

        print(f"  └─")

    # ================================================================
    # 阶段四：测试结果汇总
    # ================================================================
    print_separator("阶段四: 测试结果汇总", "█")

    print(f"  {'ID':<4} {'状态':<6} {'层级':<6} {'模型':<45} {'命中':<8} {'回答长度':<10}")
    print(f"  {'-'*4} {'-'*6} {'-'*6} {'-'*45} {'-'*8} {'-'*10}")

    for r in results:
        status_icon = "✅" if r["status"] == "PASS" else "⚠️" if r["status"] == "WARN" else "❌"
        layer_str = f"L{r['layer']}"
        hits_str = f"{r['keyword_hits']}/{r['keyword_total']}"
        print(f"  {r['id']:<4} {status_icon:<6} {layer_str:<6} {r['model']:<45} {hits_str:<8} {r['answer_length']:<10}")

    print()

    # 统计
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    warn_count = sum(1 for r in results if r["status"] == "WARN")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    layer_counts = {}
    for r in results:
        layer_counts[r["layer"]] = layer_counts.get(r["layer"], 0) + 1

    print(f"  📊 通过: {pass_count} | 警告: {warn_count} | 失败: {fail_count}")
    print(f"  🛡️  容灾层级分布: {layer_counts}")
    print(f"  📄 向量库状态: {get_vector_store_info(vs)['document_count']} 个文档片段")
    print()

    # 输出 JSON 结果（便于 CI 集成）
    print(f"  📋 JSON 结果:")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    print_separator("测试完成", "█")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
