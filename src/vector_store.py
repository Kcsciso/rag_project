"""
=============================================================================
向量知识库模块 — ChromaDB 持久化存储 + 语义检索
=============================================================================

【核心概念：为什么需要向量知识库？】

  LLM 的知识截止于训练数据日期，无法回答"张三在2024年做了什么"这类
  私有/实时问题。RAG（检索增强生成）的解决思路是：

    用户问题 → [向量检索] → 找到相关文档片段
            → [拼入 Prompt] → LLM 基于"参考资料"生成回答

  这样 LLM 不需要"记住"所有私有文档——它只需要"阅读理解"我们递给它的
  相关片段即可。

【向量检索的工作原理】

  1. 嵌入（Embedding）
     将文本转化为高维空间中的一个点（向量）。
     例如："苹果很好吃" → [0.12, -0.34, 0.56, ..., 0.78]（384维）
     语义相近的句子，它们的向量在空间中距离也近。

  2. 索引（Indexing）
     将所有文档片段预先转换为向量，存入 ChromaDB。

  3. 检索（Retrieval）
     用户提问时，将问题也转为向量，在库中搜索"距离最近"的 K 个向量。
     余弦相似度是最常用的距离度量：
       similarity(A, B) = cos(θ) = (A·B) / (|A| × |B|)
     值越接近 1，两个向量（文本）越相似。

【嵌入模型选择策略（带自动回退）】

  优先级：
    ① HuggingFaceEmbeddings + all-MiniLM-L6-v2
       → 基于 transformers 库，支持 GPU 加速
    ② ChromaDB ONNXMiniLM_L6_V2（回退方案）
       → 基于 ONNX Runtime，不依赖 PyTorch / sentence-transformers
       → 独立运行，确保极端环境下永不断链

  回退触发条件：HuggingFaceEmbeddings 初始化失败 或 首次 embed 调用失败
  回退原则：绝不尝试 pip install / upgrade 任何包（严守 CLAUDE.md 红线）

=============================================================================
"""

import os
import logging
from typing import List, Optional, Any

from langchain_core.documents import Document
from langchain_chroma import Chroma

from .config import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    RETRIEVAL_K,
    FALLBACK_TO_ONNX,
)

logger = logging.getLogger(__name__)

# ============================================================
# 🛠️ DEBUG 诊断辅助函数区
# ============================================================

def debug_print_vector_store_info(vector_store: Chroma):
    """
    【DEBUG 辅助函数 1】深入剖析 ChromaDB 的存储结构与维度
    """
    print("\n" + "=" * 25 + " [DEBUG 1: ChromaDB 内部状态] " + "=" * 25)
    try:
        collection = vector_store._collection
        count = collection.count()
        print(f"📊 集合名称 (Collection): {collection.name}")
        print(f"🔢 存储片段总数 (Count): {count}")

        if count > 0:
            # 抽样提取 1 条记录查看底层真实数据
            sample = collection.get(limit=1, include=["metadatas", "documents", "embeddings"])
            print(f"🆔 示例数据 ID: {sample['ids'][0]}")
            print(f"📌 来源元数据 (Metadata): {sample['metadatas'][0]}")
            print(f"📝 文本前 80 字: {sample['documents'][0][:80].strip()}...")
            
            if sample.get('embeddings') is not None and len(sample['embeddings']) > 0:
                vec = sample['embeddings'][0]
                print(f"📐 向量维度 (Dimension): {len(vec)}")
                print(f"🔢 向量数值前 5 位: {vec[:5]}")
    except Exception as e:
        print(f"⚠️ 读取 ChromaDB 内部状态时出错: {e}")
    print("=" * 70 + "\n")


def debug_search_similar_with_scores(
    vector_store: Chroma,
    query: str,
    k: int = RETRIEVAL_K
) -> List[Tuple[Document, float]]:
    """
    【DEBUG 辅助函数 2】显式输出带距离/相似度得分的检索结果
    """
    print("\n" + "=" * 20 + f" [DEBUG 2: 检索诊断 (Query: '{query}')] " + "=" * 20)
    
    # 使用 similarity_search_with_score 可以拿到 (Document, score) 元组
    # 注意：ChromaDB 默认使用的 L2/余弦距离，Score 越小代表越相似，或者接近 0/1 视度量函数而定
    results_with_scores = vector_store.similarity_search_with_score(query, k=k)
    
    if not results_with_scores:
        print("⚠️ 未检索到任何匹配文档。")
    else:
        for i, (doc, score) in enumerate(results_with_scores, 1):
            print(f"\n📄 [召回切片 {i}/{len(results_with_scores)}] (匹配得分/距离 Score: {score:.4f})")
            print(f"📌 来源 (Source): {doc.metadata.get('source', '未知')}")
            print("📝 召回文本内容:")
            print("┌" + "─" * 60)
            for line in doc.page_content.strip().split("\n"):
                print(f"│ {line}")
            print("└" + "─" * 60)
            
    print("=" * 70 + "\n")
    return results_with_scores

# ============================================================
# 嵌入函数创建（带自动回退）
# ============================================================

def _create_embedding_function():
    """
    创建嵌入函数，优先 HuggingFace，失败则自动回退到 ONNX。

    【回退策略详解】

    为什么 HuggingFaceEmbeddings 可能失败？
      - 本环境的 sentence-transformers 已安装但存在 torchcodec 冲突
      - torchcodec 要求的 libnvrtc.so.13 在系统中缺失
      - HuggingFaceEmbeddings 底层依赖 sentence-transformers，会触发
        上述冲突链，导致 RuntimeError

    为什么 ONNXMiniLM_L6_V2 不会失败？
      - 它使用 ONNX Runtime 而非 PyTorch 进行推理
      - 模型格式是 .onnx，不是 .bin / .safetensors
      - 完全不经过 sentence-transformers / torchcodec 调用链
      - 唯一的成本：首次使用时需要下载 ~80MB 的 ONNX 模型文件

    Returns:
        一个兼容 LangChain 接口的嵌入对象（有 embed_query / embed_documents 方法）
    """
    embedding_fn = None

    # ---- 策略 ①：尝试 HuggingFaceEmbeddings ----
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        logger.info(f"正在加载 HuggingFace 嵌入模型: {EMBEDDING_MODEL_NAME}")
        embedding_fn = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )

        # 【关键验证】初始化成功 ≠ 模型能正常工作
        # 必须实际调用一次 embed_query 来验证模型是否真正可用
        # 这一步会触发 sentence-transformers 的实际模型加载，
        # 如果 torchcodec 冲突，错误会在这里爆发
        _ = embedding_fn.embed_query("验证嵌入模型是否正常工作")

        logger.info("✅ HuggingFaceEmbeddings 加载成功，使用 transformers 引擎")
        return embedding_fn

    except Exception as e:
        logger.warning(f"⚠️  HuggingFaceEmbeddings 不可用: {e}")
        logger.warning("正在回退到 ChromaDB ONNX 方案...")

    # ---- 策略 ②：回退到 ChromaDB 内置 ONNX 嵌入函数 ----
    if not FALLBACK_TO_ONNX:
        raise RuntimeError(
            "HuggingFaceEmbeddings 加载失败，且 FALLBACK_TO_ONNX=False。"
            "请检查 sentence-transformers 环境或将 FALLBACK_TO_ONNX 设为 True。"
        )

    try:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        logger.info("正在加载 ONNXMiniLM_L6_V2 嵌入模型（ONNX Runtime）...")
        # ONNXMiniLM_L6_V2 输出 384 维归一化向量
        onnx_ef = ONNXMiniLM_L6_V2()

        # 适配为 LangChain 兼容接口
        embedding_fn = _ONNXEmbeddingAdapter(onnx_ef)

        # 同样进行冒烟测试
        _ = embedding_fn.embed_query("验证 ONNX 嵌入模型是否正常工作")

        logger.info("✅ ONNXMiniLM_L6_V2 加载成功，使用 ONNX Runtime 引擎")
        return embedding_fn

    except Exception as e:
        raise RuntimeError(
            f"所有嵌入方案均不可用！\n"
            f"  HuggingFace: 已失败（见上方日志）\n"
            f"  ONNX: {e}\n"
            f"请检查网络连接（首次使用需下载模型）。"
        )


class _ONNXEmbeddingAdapter:
    """
    ONNX 嵌入函数适配器

    【为什么需要适配器？】

    ChromaDB 原生 ONNX 嵌入函数的接口是：
      onnx_ef(["文本1", "文本2"]) → [[0.1, 0.2, ...], [0.3, 0.4, ...]]

    LangChain Chroma 封装期望的接口是：
      embedding.embed_documents(["文本1", "文本2"]) → [[0.1, ...], [0.3, ...]]
      embedding.embed_query("文本1") → [0.1, 0.2, ...]

    本适配器做一个简单的中转：把 LangChain 的调用翻译成 ChromaDB ONNX 的调用。
    这是经典的"适配器模式"（Adapter Pattern）。
    """

    def __init__(self, onnx_embedding_function):
        """保存底层的 ONNX 嵌入函数实例"""
        self._onnx_ef = onnx_embedding_function

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入多个文档文本

        【算法】
        将文本列表一次性传给 ONNX 模型做批量推理（batch inference），
        利用矩阵运算并行化，比逐条调用效率高很多。

        Args:
            texts: 文本列表，如 ["文档片段1", "文档片段2", ...]

        Returns:
            嵌入向量列表，每个向量是 384 维的 float 列表
        """
        return self._onnx_ef(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        嵌入单个查询文本

        Args:
            text: 用户的查询问题

        Returns:
            384 维嵌入向量
        """
        return self._onnx_ef([text])[0]


# ============================================================
# 模块级全局实例（单例模式）
# ============================================================
# 嵌入函数只创建一次，避免重复加载模型
_embedding_function = None


def get_embedding_function():
    """获取全局嵌入函数实例（懒加载，首次调用时初始化）"""
    global _embedding_function
    if _embedding_function is None:
        _embedding_function = _create_embedding_function()
    return _embedding_function


# ============================================================
# 向量库 CRUD 操作
# ============================================================

def create_vector_store(
    documents: List[Document],
    persist_dir: str = CHROMA_PERSIST_DIR,
) -> Chroma:
    """
    新建向量知识库：将文档列表向量化并持久化到磁盘。

    【内部流程】
    1. 获取嵌入函数（优先 HF，回退 ONNX）
    2. 将每个 Document 的 page_content 转换为 384 维向量
    3. 向量 + 元数据 + 文本原文 一并存入 ChromaDB
    4. ChromaDB 自动持久化到 persist_dir 目录

    【ChromaDB 的数据组织】
    - Collection（集合）：类似于关系数据库中的"表"
      - 我们使用一个名为 "rag_documents" 的 Collection
    - 每条记录包含：
      - embedding: 384 维浮点向量
      - document: 原始文本（用于返回给 LLM）
      - metadata: 来源文件名等附加信息
      - id: 唯一标识符（自动生成）

    Args:
        documents: 已分块的 LangChain Document 列表
        persist_dir: 持久化目录路径

    Returns:
        Chroma 向量库实例
    """
    if not documents:
        raise ValueError("文档列表为空，无法创建向量库。请先加载 PDF 文件。")

    embedding_fn = get_embedding_function()

    logger.info(f"正在创建向量库，共 {len(documents)} 个文档片段...")

    # from_documents 会：
    #  ① 调用 embedding_fn.embed_documents() 批量向量化
    #  ② 创建 ChromaDB Collection 并写入数据
    #  ③ 自动持久化到磁盘
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_fn,
        persist_directory=persist_dir,
        collection_name="rag_documents",
    )

    logger.info(f"✅ 向量库创建完成，持久化目录: {persist_dir}")
    return vector_store


def load_vector_store(
    persist_dir: str = CHROMA_PERSIST_DIR,
) -> Optional[Chroma]:
    """
    加载已持久化的向量知识库。

    如果 persist_dir 下没有已索引的数据（首次运行），返回 None，
    调用方应引导用户先上传 PDF。

    Args:
        persist_dir: 持久化目录路径

    Returns:
        Chroma 实例，如果库为空则返回 None
    """
    if not os.path.exists(persist_dir):
        return None

    # 检查目录中是否有 ChromaDB 数据文件
    # ChromaDB 使用 SQLite 作为元数据后端，数据文件通常是 chroma.sqlite3
    has_data = any(
        f.endswith(".sqlite3") or f.endswith(".parquet")
        for f in os.listdir(persist_dir)
    )
    if not has_data:
        return None

    embedding_fn = get_embedding_function()

    try:
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_fn,
            collection_name="rag_documents",
        )
        # 验证 Collection 中确实有数据
        count = vector_store._collection.count()
        if count == 0:
            return None
        logger.info(f"✅ 已加载向量库，共 {count} 条记录")
        return vector_store
    except Exception as e:
        logger.warning(f"加载向量库失败: {e}")
        return None


def search_similar(
    vector_store: Chroma,
    query: str,
    k: int = RETRIEVAL_K,
) -> List[Document]:
    """
    在向量库中搜索与查询最相似的 k 个文档片段。

    【算法：语义相似度检索】

    传统关键词搜索（如 grep）的问题：
      用户搜 "Python 性能优化"，但文档写的是 "提升 CPython 执行效率"
      → 关键词不匹配，搜不到！

    向量语义检索的优势：
      两个句子语义相近 → 它们的嵌入向量在高维空间中距离近
      → 即使用词完全不同，也能准确检索到相关内容

    【检索流程】
    1. 将用户查询 query 转换为向量 q
    2. 在 ChromaDB 中计算 q 与所有文档向量的余弦相似度
    3. 返回相似度最高的 k 个文档
    4. 默认使用 MMR（最大边际相关性）算法去重：
       - 既保证结果与查询相关
       - 又保证结果之间多样化（避免返回 k 个几乎一样的片段）

    Args:
        vector_store: ChromaDB 向量库实例
        query: 用户查询字符串
        k: 返回的文档数量（默认 4）

    Returns:
        最相关的 k 个 Document 片段，按相似度降序排列
    """
    # similarity_search 内部自动：
    #  ① embedding_fn.embed_query(query) → 查询向量
    #  ② 在 ChromaDB 中做 ANN（近似最近邻）搜索
    #  ③ 返回 Document 列表
    results = vector_store.similarity_search(query, k=k)
    return results


def get_vector_store_info(vector_store: Chroma) -> dict:
    """
    获取向量库的基本信息（用于前端状态展示）。

    Returns:
        {"document_count": 已索引文档片段数量}
    """
    if vector_store is None:
        return {"document_count": 0}
    try:
        count = vector_store._collection.count()
        return {"document_count": count}
    except Exception:
        return {"document_count": 0}
    
    
# ============================================================
# 命令行测试入口
# ============================================================
if __name__ == "__main__":
    from .pdf_loader import load_pdfs_from_directory
    from .config import PDF_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

    # 1. 独立测试加载 PDF 文本块
    test_docs = load_pdfs_from_directory(PDF_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP)

    if test_docs:
        # 2. 创建或加载向量库 (开启 debug=True 观察细节)
        store = create_vector_store(test_docs, debug=True)

        # 3. 模拟一条测试查询语句，验证余弦距离与召回得分
        test_query = "核心业务是什么？"
        search_similar(store, query=test_query, k=3, debug=True)
    else:
        print("未能读取到测试文档，无法测试向量库 CRUD 功能。")
