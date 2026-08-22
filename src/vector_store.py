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
import re
import logging
from typing import List, Optional, Any, Tuple, Dict

from langchain_core.documents import Document
from langchain_chroma import Chroma

from .config import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    RETRIEVAL_K,
    SIMILARITY_THRESHOLD,
    FALLBACK_TO_ONNX,
    PRODUCT_MAPPING_RULES,
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
            model_kwargs={"device": "cpu"},  # CPU 模式，规避 libnvrtc.so.13 冲突
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
    #
    # 【重要】collection_metadata={"hnsw:space": "cosine"} 强制使用余弦距离
    # 而非默认 L2 距离，确保 similarity_search_with_score 返回 0~2 的余弦距离值，
    # 从而 SIMILARITY_THRESHOLD=0.65 能正确过滤不相关切片。
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_fn,
        persist_directory=persist_dir,
        collection_name="rag_documents",
        collection_metadata={"hnsw:space": "cosine"},
    )

    logger.info(f"✅ 向量库创建完成，持久化目录: {persist_dir}")

    # 🔴 同步构建 BM25 索引（零显存开销，精确关键词匹配）
    build_bm25_index(documents, persist_dir)

    return vector_store


def clear_vector_store(persist_dir: str = CHROMA_PERSIST_DIR) -> bool:
    """
    彻底清空 ChromaDB 向量数据库。

    操作步骤：
      1. 加载现有 Collection
      2. 删除 Collection 中所有记录（保留 Collection 元数据/索引结构）
      3. 返回是否成功

    注意：此操作不可逆！所有已索引的文档片段将被永久删除。

    Args:
        persist_dir: ChromaDB 持久化目录路径

    Returns:
        True 如果清空成功，False 如果向量库不存在或清空失败
    """
    import shutil

    logger.warning("🧹 正在清空 ChromaDB 向量数据库...")

    try:
        # 方案 A：如果向量库已加载，直接通过 Collection API 删除所有记录
        embedding_fn = get_embedding_function()
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_fn,
            collection_name="rag_documents",
        )
        count = vector_store._collection.count()
        if count > 0:
            # 获取所有记录 ID 并批量删除
            all_ids = vector_store._collection.get()["ids"]
            if all_ids:
                vector_store._collection.delete(ids=all_ids)
                logger.info(f"✅ 已通过 Collection API 删除 {len(all_ids)} 条记录")
        else:
            logger.info("📭 向量库已为空，无需删除")

        logger.info("✅ ChromaDB 向量数据库已清空")
        return True

    except Exception as e:
        logger.warning(f"Collection API 删除失败，尝试物理删除目录: {e}")
        # 方案 B：物理删除持久化目录（暴力但可靠）
        try:
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)
                os.makedirs(persist_dir, exist_ok=True)
                logger.info("✅ 已通过物理删除清空向量库目录")
                return True
        except Exception as e2:
            logger.error(f"❌ 物理删除向量库目录失败: {e2}")
            return False

    return True


def resolve_product_id(filename: str) -> str:
    """
    根据文件名解析对应的产品标识。

    使用 PRODUCT_MAPPING_RULES 中的 filename_patterns 进行匹配，
    不区分大小写，任一模式命中即返回对应 product_id。

    若多个规则同时匹配，返回第一个匹配的（规则列表顺序即优先级）。

    Args:
        filename: 上传的 PDF 文件名（已通过 sanitize_filename 清洗）

    Returns:
        product_id 字符串，如 "OpenR6"、"OpenC3"
        若无法识别则返回 "unknown"
    """
    filename_lower = filename.lower()
    for rule in PRODUCT_MAPPING_RULES:
        for pattern in rule["filename_patterns"]:
            if pattern.lower() in filename_lower:
                logger.info(
                    f"🏷️  产品识别: '{filename}' → product_id='{rule['product_id']}' "
                    f"(命中模式: '{pattern}')"
                )
                return rule["product_id"]

    logger.warning(f"⚠️  无法识别产品: '{filename}'，标记为 'unknown'")
    return "unknown"


def get_registered_products(
    persist_dir: str = CHROMA_PERSIST_DIR,
) -> List[str]:
    """
    获取当前向量库中已注册（已入库）的产品 ID 列表。

    通过查询 ChromaDB Collection 中所有文档的 metadata，
    提取去重后的 product_id 值。

    Args:
        persist_dir: ChromaDB 持久化目录路径

    Returns:
        已注册的产品 ID 列表（如 ["OpenR6", "OpenC3"]），
        若向量库为空或不存在则返回空列表
    """
    if not os.path.exists(persist_dir):
        return []

    has_data = any(
        f.endswith(".sqlite3") or f.endswith(".parquet")
        for f in os.listdir(persist_dir)
    )
    if not has_data:
        return []

    try:
        embedding_fn = get_embedding_function()
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_fn,
            collection_name="rag_documents",
        )

        # 获取所有文档的 metadata
        collection_data = vector_store._collection.get(include=["metadatas"])
        metadatas = collection_data.get("metadatas", [])

        if not metadatas:
            return []

        # 提取去重的 product_id（set 去重 + 过滤无效值）
        products = set()
        for meta in metadatas:
            pid = (meta.get("product_id") or "").strip()
            # 🔴 严格过滤：排除空值、unknown、纯空白
            if pid and pid.lower() != "unknown":
                products.add(pid)

        # 🔴 保序去重：set 去重后按字母排序，确保每次返回结果一致
        product_list = sorted(products)
        logger.info(f"📋 已注册产品列表: {product_list}")
        return product_list

    except Exception as e:
        logger.warning(f"获取已注册产品列表失败: {e}")
        return []


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

    # ── v4 优先: 尝试加载 rag_v4_child（函数级精粒度索引）──
    try:
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_fn,
            collection_name="rag_v4_child",
        )
        count = vector_store._collection.count()
        if count > 0:
            logger.info(f"✅ 已加载 v4 向量库 (rag_v4_child)，共 {count} 条记录")
            return vector_store
    except Exception:
        logger.debug("v4 向量库未找到，回退 v3")

    # ── v3 回退: 加载旧 rag_documents ──
    try:
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_fn,
            collection_name="rag_documents",
        )
        count = vector_store._collection.count()
        if count == 0:
            return None
        logger.info(f"✅ 已加载 v3 向量库 (rag_documents)，共 {count} 条记录")
        # ── v4 MD5 记录自动恢复 (ADR-16) ──
        _init_md5_store_from_chroma(persist_dir)

        return vector_store
    except Exception as e:
        logger.warning(f"加载向量库失败: {e}")
        return None


def load_vector_store_from_name(
    collection_name: str,
    persist_dir: str = CHROMA_PERSIST_DIR,
) -> Optional[Chroma]:
    """
    按 Collection 名称加载指定向量库（用于 v4 Parent/Child 分离访问）。
    """
    embedding_fn = _create_embedding_function()
    try:
        vs = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_fn,
            persist_directory=persist_dir,
        )
        count = vs._collection.count()
        logger.info(f"✅ 已加载向量库 '{collection_name}': {count} 条")
        return vs
    except Exception as e:
        logger.warning(f"加载 '{collection_name}' 失败: {e}")
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


def search_similar_with_threshold(
    vector_store: Chroma,
    query: str,
    k: int = RETRIEVAL_K,
    threshold: Optional[float] = SIMILARITY_THRESHOLD,
    product_id: Optional[str] = None,
) -> List[Document]:
    """
    带相似度阈值过滤的向量检索（支持产品级物理隔离）。

    【为什么需要阈值过滤？】

    传统 Top-K 检索的问题：无论检索到的文档与问题是否真正相关，
    都会强行返回 k 个"最近"的文档。当用户问及文档库中完全不存在的话题时
    （如机械臂文档库中问"摄像头"），Top-K 仍会返回 k 个不相关的机械臂切片，
    导致 LLM 基于错误上下文产生幻觉。

    阈值过滤的解决思路：
      1. 使用 similarity_search_with_score 获取 (Document, distance) 元组
      2. 只保留 distance <= threshold 的切片
      3. 如果没有任何切片通过阈值，返回空列表 → 上层优雅处理

    【产品级物理隔离】

    当指定 product_id 时，ChromaDB 查询会添加 where 过滤条件，
    确保只检索该产品的切片，实现 100% 物理隔离（绝不跨库召回）。

    【ChromaDB 距离度量说明】

    使用 HuggingFaceEmbeddings (normalize_embeddings=True) 时，
    ChromaDB 默认使用余弦距离 (cosine distance)：
      - distance=0: 向量完全相同（语义完全一致）
      - distance=1: 向量正交（语义无关）
      - distance=2: 向量方向完全相反（语义对立）

    推荐阈值:
      - 0.50: 严格模式，仅保留高度相关的内容
      - 0.65: 平衡模式（默认），兼顾召回率与精确率
      - 0.80: 宽松模式，允许边缘相关内容通过
      - None: 禁用阈值过滤，等同于 search_similar()

    Args:
        vector_store: ChromaDB 向量库实例
        query: 用户查询字符串
        k: 检索候选数量
        threshold: 距离阈值，None 表示不过滤
        product_id: 产品标识（如 "OpenR6"），None 表示不过滤产品

    Returns:
        通过阈值过滤的 Document 列表（可能为空列表）
    """
    # 构建 ChromaDB where 过滤条件
    chroma_filter = None
    if product_id:
        chroma_filter = {"product_id": product_id}
    else:
        # Fix 4: product_id 为空时尝试从 query 推断，否则按产品分组召回
        _inferred = _infer_product_from_query(query)
        if _inferred:
            chroma_filter = {"product_id": _inferred}
            logger.info(f"🔍 自动推断 product_id='{_inferred}' (from query)")
        else:
            logger.warning(
                "⚠️  search_similar_with_threshold 未指定 product_id 且无法推断，"
                "将进行跨产品混合检索（结果将按产品分组标注）"
            )

    # 使用 similarity_search_with_score 获取带距离分数的检索结果
    # langchain-chroma 支持 filter 参数，底层转换为 ChromaDB 的 where 条件
    try:
        if chroma_filter:
            results_with_scores = vector_store.similarity_search_with_score(
                query, k=k, filter=chroma_filter
            )
        else:
            results_with_scores = vector_store.similarity_search_with_score(query, k=k)
    except Exception as e:
        logger.warning(
            f"⚠️  similarity_search_with_score 不支持 filter 参数，"
            f"降级为后置过滤: {e}"
        )
        # 回退：不使用 filter，在结果中手动过滤
        results_with_scores = vector_store.similarity_search_with_score(query, k=k)
        if product_id:
            results_with_scores = [
                (doc, score) for doc, score in results_with_scores
                if doc.metadata.get("product_id") == product_id
            ]

    if threshold is None:
        # 禁用阈值过滤，退化为普通 search_similar 行为
        return [doc for doc, _ in results_with_scores]

    filtered_docs = []
    filtered_count = 0
    for doc, score in results_with_scores:
        if score <= threshold:
            filtered_docs.append(doc)
            logger.debug(
                f"✅ 切片通过阈值 (distance={score:.4f} ≤ {threshold}): "
                f"{doc.metadata.get('source', '?')[:50]}"
            )
        else:
            filtered_count += 1
            logger.debug(
                f"❌ 切片被过滤 (distance={score:.4f} > {threshold}): "
                f"{doc.metadata.get('source', '?')[:50]}"
            )

    # 🔴 关键修复：当阈值过陡挤掉所有切片时，若候选池中有精准匹配 API 名字的切片，触发特例拉升保底
    if not filtered_docs and results_with_scores:
        all_docs = [doc for doc, _ in results_with_scores]
        boosted = _boost_api_chunks(query, all_docs)
        if boosted and boosted[0] != all_docs[0]:
            logger.info("🚀 阈值超限但命中 API 实体，触发特例强拉升保底！")
            return boosted[:k]

    if filtered_count > 0:
        logger.info(
            f"🔍 相似度阈值过滤: {len(filtered_docs)}/{len(results_with_scores)} "
            f"个切片通过 (threshold={threshold})"
        )

    # 🔴 返回前统一执行 API 强拉升
    return _boost_api_chunks(query, filtered_docs)


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
# v4 双 Collection 管理 — Parent-Child Dual Indexing (ADR-15)
# ============================================================

def _match_function_names(metadata_fn_str: str, query_entities: List[str]) -> bool:
    """
    Fix 1: 模糊匹配 function_names 元数据字符串与 query 代码实体。

    消除空格/大小写差异，支持子串匹配（如 query "movl" 匹配 "robot_movl"）。
    """
    if not metadata_fn_str or not query_entities:
        return False
    stored = [s.strip().lower() for s in metadata_fn_str.split(",") if s.strip()]
    query_lower = [q.strip().lower() for q in query_entities]
    for qe in query_lower:
        for sf in stored:
            if qe == sf or qe in sf or sf in qe:
                return True
    return False


def _infer_product_from_query(query: str) -> Optional[str]:
    """Fix 4: 从 query 中推断 product_id（简单关键词匹配）。"""
    q = query.lower()
    if any(kw in q for kw in ("openr6", "r6", "py_dll", "windows系统")):
        return "OpenR6"
    if any(kw in q for kw in ("openc3", "六轴", "collrob")):
        return "OpenC3"
    if any(kw in q for kw in ("jaka", "zuju", "modbus", "minicab", "vbrake")):
        return "JAKA"
    return None


def _extract_query_code_entities(query: str) -> List[str]:
    """从 query 中提取代码实体（复用 CodeEntityAnchor 模式）。"""
    import re
    patterns = [
        re.compile(r'\b(?:robot_|set_|get_)\w+\b', re.IGNORECASE),
        re.compile(r'\b(?:movl|movc|movj|movp|movb)\b', re.IGNORECASE),
        re.compile(r'\b(?:py_dll|collrob_sdk|ctypes\.CDLL)\b', re.IGNORECASE),
        re.compile(r'\b(?:power_on|enable|brkopen|home|joint_angle|io_output)\b', re.IGNORECASE),
    ]
    entities = []
    seen = set()
    for pat in patterns:
        for m in pat.finditer(query):
            e = m.group(0).lower()
            if e not in seen:
                seen.add(e)
                entities.append(e)
    # 🔴 v26: 复合词锚点 —— Ethernet/IP、Modbus-RTU 等斜杠/连字符专有名词
    # 由 _COMPOUND_RE 通用提取（doc 正文字面命中 → _boost_api_chunks Dense 侧强拉升）
    for _m in _COMPOUND_RE.finditer(query):
        _e = _m.group(0).lower().strip('-_')
        if len(_e) >= 3 and _e not in seen:
            seen.add(_e)
            entities.append(_e)
    return entities

def _boost_api_chunks(query: str, docs: List[Document]) -> List[Document]:
    """
    若 Query 中包含 SDK 函数名/代码实体，强行提升 metadata['function_names']
    或正文中命中该 API 的切片至头部（Hard Boost），解决 Dense Vector 对纯代码名召回靠后或被过滤的问题。
    """
    if not docs or not query:
        return docs

    entities = _extract_query_code_entities(query)
    if not entities:
        return docs

    boosted = []
    normal = []

    for doc in docs:
        fn_meta = doc.metadata.get("function_names", "")
        # 1. 优先检查元数据中记录的函数名
        is_hit = _match_function_names(fn_meta, entities)

        # 2. 若元数据未写全，后置扫描正文中的 [Functions: ...] 标头或代码实体
        if not is_hit:
            # 🔴 v26: 与 tokenizer 同构的空格归一化，保证 "Ethernet / IP" 也能字面命中
            content_lower = _SPACE_SEP_RE.sub(r'\1\2\3', doc.page_content.lower())
            for ent in entities:
                if len(ent) >= 3 and ent.lower() in content_lower:
                    is_hit = True
                    break

        if is_hit:
            boosted.append(doc)
        else:
            normal.append(doc)

    if boosted:
        logger.info(f"🚀 API 强拉升生效: 优先排序 {len(boosted)} 个命中 API 实体 {entities} 的切片")
        return boosted + normal

    return docs

def _embed_batched(
    texts: List[str],
    embedding_fn,
    batch_size: int = None,
) -> List[List[float]]:
    """
    GPU/CPU 自适应批量嵌入计算 — 手动分批调用 HF embed_documents()。

    修正 ①: CPU 模式 batch_size=16，GPU 模式 64，防 CPU 耗尽。
    """
    if batch_size is None:
        batch_size = 16 if EMBEDDING_DEVICE == "cpu" else 64
    embeddings = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        batch_emb = embedding_fn.embed_documents(batch)
        embeddings.extend(batch_emb)
        if total > batch_size and i % (batch_size * 4) == 0:
            logger.info(f"  嵌入进度: {min(i + batch_size, total)}/{total}")
    return embeddings


def _sanitize_metadata(meta: dict) -> dict:
    """
    修正 ②: 将 metadata dict 转为 ChromaDB 兼容格式（仅 str/int/float/bool）。

    ChromaDB 的 MetadataValue 不支持 list/dict/None 类型。
    """
    clean = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, list):
            clean[k] = ",".join(str(x) for x in v)
        elif v is None:
            clean[k] = ""
        elif isinstance(v, dict):
            import json
            clean[k] = json.dumps(v, ensure_ascii=False)
        else:
            clean[k] = str(v)
    return clean


def _add_to_existing_collection(
    collection_name: str,
    docs: list,
    texts: List[str],
    embeddings: List[List[float]],
    persist_dir: str,
    embedding_fn,
) -> Chroma:
    """
    向已有 ChromaDB Collection 增量追加文档（网络免疫版）。

    使用 collection.add(embeddings=precomputed) — ChromaDB 收到预计算向量后不调用嵌入函数。
    """
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    coll = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    ids = [f"{collection_name[4]}_{d.metadata.get('product_id','?')}_{hash(d.page_content[:80])}" for d in docs]
    metadatas = [_sanitize_metadata(d.metadata) for d in docs]
    coll.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return Chroma(
        client=client, collection_name=collection_name,
        embedding_function=embedding_fn,
    )


def create_dual_collections(
    parent_docs: list,
    child_docs: list,
    persist_dir: str = CHROMA_PERSIST_DIR,
    embedding_fn=None,
) -> Tuple[Chroma, Chroma]:
    """
    创建 Parent-Child 双层 ChromaDB Collection 并同步构建 BM25 索引。
    统一使用 LangChain Chroma 包装器复用客户端连接，避免底层 Settings 冲突。
    """
    if embedding_fn is None:
        embedding_fn = get_embedding_function()

    # 1. 清洗 Metadata，防止 None 导致主键生成异常
    for d in parent_docs:
        d.metadata = _sanitize_metadata(d.metadata)
    for d in child_docs:
        d.metadata = _sanitize_metadata(d.metadata)

    # 2. 初始化并重置 Parent Collection
    parent_vs = Chroma(
        persist_directory=persist_dir,
        collection_name="rag_v4_parent",
        embedding_function=embedding_fn,
        collection_metadata={"hnsw:space": "cosine"}
    )
    try:
        existing_p = parent_vs._collection.get()
        if existing_p and existing_p.get("ids"):
            parent_vs._collection.delete(ids=existing_p["ids"])
            logger.info(f"🗑️ 已清空旧 Parent 数据: {len(existing_p['ids'])} 条")
    except Exception as e:
        logger.debug(f"Parent 重置跳过: {e}")

    if parent_docs:
        p_ids = [
            doc.metadata.get("parent_id") or doc.metadata.get("chunk_id") or f"p_{doc.metadata.get('product_id', 'General')}_{i}"
            for i, doc in enumerate(parent_docs)
        ]
        parent_vs.add_documents(documents=parent_docs, ids=p_ids)
        logger.info(f"✅ v4 Parent Collection 已入库: {len(parent_docs)} 个切片")

    # 3. 初始化并重置 Child Collection
    child_vs = Chroma(
        persist_directory=persist_dir,
        collection_name="rag_v4_child",
        embedding_function=embedding_fn,
        collection_metadata={"hnsw:space": "cosine"}
    )
    try:
        existing_c = child_vs._collection.get()
        if existing_c and existing_c.get("ids"):
            child_vs._collection.delete(ids=existing_c["ids"])
            logger.info(f"🗑️ 已清空旧 Child 数据: {len(existing_c['ids'])} 条")
    except Exception as e:
        logger.debug(f"Child 重置跳过: {e}")

    if child_docs:
        c_ids = [
            doc.metadata.get("chunk_id") or f"c_{doc.metadata.get('product_id', 'General')}_{i}"
            for i, doc in enumerate(child_docs)
        ]
        child_vs.add_documents(documents=child_docs, ids=c_ids)
        logger.info(f"✅ v4 Child Collection 已入库: {len(child_docs)} 个切片")

    # 4. 同步构建 BM25 内存索引（全产品分词索引）
    build_bm25_index(child_docs, persist_dir)

    return parent_vs, child_vs


def search_dual_index(
    parent_vs: Chroma,
    child_vs: Chroma,
    query: str,
    k: int = 5,
    threshold: float = 0.55,
    product_id: Optional[str] = None,
) -> List[Any]:
    """
    v4 双索引检索 — Child 优先 + Parent 批量反查。

    策略（高效单次查询模式）:
      1. 在 Child Collection 中向量检索 Top-K → 得到精粒度候选
      2. 收集候选的 parent_id → 批量反查 Parent Collection
      3. 合并 Parent 概览 + Child 细节 → 最终结果列表
      4. 按相似度排序，Parent 排在对应 Child 前面（提供章节上下文）

    这是"Child 匹配 + parent_id 批量反查"的高效模式：
      - 只需 2 次 Collection 查询（Child 向量 + Parent get）
      - 不会为每个 Child 单独查 Parent
    """
    from langchain_core.documents import Document as LCDocument

    # ── Step 1: Child 向量检索 ──
    child_filter = {"product_id": product_id} if product_id else None
    try:
        child_results = child_vs.similarity_search_with_relevance_scores(
            query, k=k * 2, filter=child_filter,
        )
    except Exception:
        child_results = []

    # ── Step 2: 收集唯一 parent_id ──
    child_docs = []
    parent_ids_to_fetch = set()
    for doc, score in child_results:
        if score < threshold and len(child_docs) >= k:
            continue
        pid = doc.metadata.get("parent_id") if hasattr(doc, "metadata") else None
        if pid:
            parent_ids_to_fetch.add(pid)
        child_docs.append((doc, score))

    # ── Step 3: 批量反查 Parent ──
    parent_docs_by_id = {}
    if parent_ids_to_fetch:
        try:
            parent_data = parent_vs._collection.get(
                ids=list(parent_ids_to_fetch),
                include=["documents", "metadatas"],
            )
            for i, pid in enumerate(parent_data["ids"]):
                parent_docs_by_id[pid] = LCDocument(
                    page_content=parent_data["documents"][i],
                    metadata=parent_data["metadatas"][i],
                )
        except Exception:
            pass

    # ── Step 4: 合并结果 ──
    merged = []
    seen_parents = set()
    for child_doc, score in child_docs[:k]:
        pid = child_doc.metadata.get("parent_id") if hasattr(child_doc, "metadata") else None
        # Parent 先插入（提供章节上下文）
        if pid and pid in parent_docs_by_id and pid not in seen_parents:
            seen_parents.add(pid)
            parent_doc = parent_docs_by_id[pid]
            merged.append(LCDocument(
                page_content=parent_doc.page_content,
                metadata={**parent_doc.metadata, "source_type": "parent_overview"},
            ))
        # Child 紧随其后
        merged.append(LCDocument(
            page_content=child_doc.page_content,
            metadata={**child_doc.metadata, "source_type": "child_detail"},
        ))

    return _boost_api_chunks(query, merged)


# ============================================================
# v4 增量更新引擎 — Upsert + MD5 去重 (ADR-16)
# ============================================================

_product_md5_store: Dict[str, str] = {}  # {product_id: md5_hex}


def _init_md5_store_from_chroma(persist_dir: str = CHROMA_PERSIST_DIR):
    """从 ChromaDB Collection metadata 恢复 MD5 记录（系统重启后自动初始化）。"""
    global _product_md5_store
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        for coll_name in ["rag_v4_parent", "rag_v4_child"]:
            try:
                coll = client.get_collection(coll_name)
                md5_data = coll.metadata.get("product_md5", "{}")
                if md5_data:
                    import json
                    stored = json.loads(md5_data)
                    _product_md5_store.update(stored)
            except Exception:
                pass
        if _product_md5_store:
            logger.info(f"📋 MD5 记录已恢复: {len(_product_md5_store)} 产品")
    except Exception as e:
        logger.debug(f"MD5 恢复跳过: {e}")


def _persist_md5_store(persist_dir: str = CHROMA_PERSIST_DIR):
    """将 MD5 记录写入 ChromaDB Collection metadata（持久化）。"""
    import json
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        md5_json = json.dumps(_product_md5_store, ensure_ascii=False)
        for coll_name in ["rag_v4_parent", "rag_v4_child"]:
            try:
                coll = client.get_collection(coll_name)
                coll.modify(metadata={"product_md5": md5_json})
            except Exception:
                pass
    except Exception:
        pass


def delete_product_chunks(
    product_id: str,
    persist_dir: str = CHROMA_PERSIST_DIR,
) -> int:
    """
    级联删除指定产品的所有 Parent + Child 切片。

    Returns:
        删除的切片总数
    """
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    total_deleted = 0
    for coll_name in ["rag_v4_parent", "rag_v4_child"]:
        try:
            coll = client.get_collection(coll_name)
            result = coll.delete(where={"product_id": product_id})
            count = len(result) if isinstance(result, list) else 0
            total_deleted += count
            if count:
                logger.info(f"🗑️  {coll_name}: 删除 {count} 条 (product={product_id})")
        except Exception:
            pass
    return total_deleted


def bm25_upsert_product(
    product_id: str,
    new_docs: list,
    vector_store=None,
):
    """
    BM25 增量同步 — 对新增文档分词并更新内存索引。

    仅重建受影响产品的 BM25 索引（O(n), n=新增文档数）。
    """
    global _bm25_indexes, _bm25_corpus
    from rank_bm25 import BM25Okapi

    try:
        # 分词新文档
        new_tokens = [_tokenize_for_bm25(
            d.page_content if hasattr(d, "page_content") else str(d)
        ) for d in new_docs]

        if product_id in _bm25_corpus and product_id in _bm25_indexes:
            # 追加模式
            _bm25_corpus[product_id].extend(
                d.page_content if hasattr(d, "page_content") else str(d)
                for d in new_docs
            )
            all_tokens = _bm25_indexes[product_id].corpus + new_tokens
            # 重新计算 IDF（仅该产品）
            _bm25_indexes[product_id] = BM25Okapi(all_tokens)
        else:
            # 新产品模式
            _bm25_corpus[product_id] = [
                d.page_content if hasattr(d, "page_content") else str(d)
                for d in new_docs
            ]
            _bm25_indexes[product_id] = BM25Okapi(new_tokens)

        logger.info(f"📊 BM25 增量同步: product={product_id}, +{len(new_docs)} docs")
    except Exception as e:
        logger.warning(f"BM25 增量同步失败 ({product_id}): {e}")


def bm25_remove_product(product_id: str):
    """BM25 级联删除 — 清除指定产品的索引。"""
    global _bm25_indexes, _bm25_corpus
    _bm25_indexes.pop(product_id, None)
    _bm25_corpus.pop(product_id, None)
    logger.info(f"📊 BM25 已移除: product={product_id}")


def upsert_product_documents(
    file_path: str,
    product_id: Optional[str] = None,
    child_chunk_size: int = 500,
    parent_chunk_size: int = 1500,
) -> Dict[str, Any]:
    """
    增量摄入单个产品文档（PDF 或 Markdown），自动路由至 Stage 1 双轨解析引擎并持久化。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    from langchain_chroma import Chroma
    from .pdf_loader import (
        load_single_sdk_pdf,
        load_jaka_mineru_dual,
        _resolve_product_id_from_filename,
    )

    filename = os.path.basename(file_path)
    if not product_id:
        product_id = _resolve_product_id_from_filename(filename)

    logger.info(f"📥 开始增量解析文档: {filename} (目标产品线: {product_id})")

    # 1. 双轨解析路由
    parent_docs = []
    child_docs = []

    if file_path.lower().endswith(".md"):
        parent_docs, child_docs = load_jaka_mineru_dual(file_path)
    elif file_path.lower().endswith(".pdf"):
        parent_docs, child_docs = load_single_sdk_pdf(file_path, product_id=product_id)
    else:
        raise ValueError(f"不支持的文件格式: {filename}，仅支持 .pdf 与 .md")

    if not child_docs:
        logger.warning(f"⚠️ 文档未解析出任何有效子切片: {filename}")
        return {
            "status": "warning",
            "message": "文档解析为空",
            "parent_chunks": 0,
            "child_chunks": 0,
        }

    # 2. 清洗 Metadata (防止包含 list/dict 引发 Chroma 写入异常)
    for doc in parent_docs:
        doc.metadata = _sanitize_metadata(doc.metadata)
    for doc in child_docs:
        doc.metadata = _sanitize_metadata(doc.metadata)

    # 3. 使用 LangChain Chroma 包装器安全复用单例客户端进行写入
    embedding_fn = get_embedding_function()

    # 增量写入 Parent Collection
    if parent_docs:
        parent_vs = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name="rag_v4_parent",
            embedding_function=embedding_fn,
            collection_metadata={"hnsw:space": "cosine"}
        )
        # 🔴 关键防御：使用 or 替代 get(key, default)，空字符串 "" 会被 or 识别为 False 从而触发兜底 ID
        p_ids = [doc.metadata.get("parent_id") or doc.metadata.get("chunk_id") or f"p_{product_id}_{i}" for i, doc in enumerate(parent_docs)]
        parent_vs.add_documents(documents=parent_docs, ids=p_ids)
        logger.info(f"✅ 父切片增量入库: {len(parent_docs)} 个")

    # 增量写入 Child Collection
    if child_docs:
        child_vs = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name="rag_v4_child",
            embedding_function=embedding_fn,
            collection_metadata={"hnsw:space": "cosine"}
        )
        # 🔴 关键防御：同理，拦截空字符串
        c_ids = [doc.metadata.get("chunk_id") or f"c_{product_id}_{i}" for i, doc in enumerate(child_docs)]
        child_vs.add_documents(documents=child_docs, ids=c_ids)
        logger.info(f"✅ 子切片增量入库: {len(child_docs)} 个")

    # 4. BM25 内存索引增量同步
    bm25_upsert_product(product_id, child_docs)

    return {
        "status": "success",
        "product_id": product_id,
        "parent_chunks": len(parent_docs),
        "child_chunks": len(child_docs),
        "source": filename,
    }


# ============================================================
# BM25 稀疏检索索引（精确关键词匹配，零显存开销）
# ============================================================

import jieba

# 全局 BM25 索引：{product_id: BM25Okapi}
_bm25_indexes: dict = {}

# BM25 索引对应的文档文本列表：{product_id: [text1, text2, ...]}
_bm25_corpus: dict = {}


# ── jieba 自定义词典引导种子（少量关键 SDK 函数名，防冷启动漏词）──
# 🟢 主要术语注册已由 _auto_extract_and_register_terms() 在 BM25 构建时自动完成。
#    以下仅为冷启动引导种子——即使文档尚未入库，也能确保常见 SDK 函数不被切碎。
_SDK_FUNCTION_NAMES = [
    # OpenR6
    "set_robot_power_on", "set_robot_power_off", "set_robot_arm_home",
    "set_move_line", "set_robot_io_output", "set_robot_io_status",
    "get_robot_joint_angle_all", "get_robot_pose", "robot_socket_start",
    "set_joint_degree_by_number",
    # OpenC3
    "robot_Power_on", "robot_enable", "robot_disable",
    "robot_movj", "robot_movl", "robot_movec", "robot_stop",
    "robot_brkopen", "robot_brkclose", "robot_motor_enable", "robot_motor_disable",
    "get_robot_joint_all", "get_robot_iostate", "robot_get_pose",
]
for _func in _SDK_FUNCTION_NAMES:
    jieba.add_word(_func, freq=999, tag='eng')
    if _func.startswith("robot_"):
        short = _func[6:]
        if len(short) >= 3:
            jieba.add_word(short, freq=500, tag='eng')
jieba.add_word("py_dll", freq=999, tag='eng')
jieba.add_word("collrob_sdk", freq=999, tag='eng')

# ── 正则预分器：保留下划线连接的标识符作为原子 token ──
_IDENTIFIER_RE = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*')

# 🔴 v26: 复合词原子化 —— 斜杠/连字符连接的专有名词（Ethernet/IP、Modbus-RTU、TCP-IP）
# 整体作为原子 token 追加（绝不从原文删除，子段 token 保留 → 纯增量，零回归面）。
# 排除 '.' 分隔符：防止 robot.set_move_line 被吞成 robot.set + move_line（snake_case 保护立身之本）
_COMPOUND_RE = re.compile(r'[A-Za-z][A-Za-z0-9]*(?:[/\-][A-Za-z0-9]+)+')

# 🔴 v26: 分隔符空格归一化（doc/query 双侧对称，共用 _tokenize_for_bm25）
# 仅含 / 与 -，排除 '.'（同上回归理由）；两侧均须为字母，避免误伤自然写法
_SPACE_SEP_RE = re.compile(r'([A-Za-z])\s*([/\-])\s*([A-Za-z])')

# ── 已自动注册的术语集合（防重复注册）──
_auto_registered_terms: set = set()


def _auto_extract_and_register_terms(documents: list):
    """
    从文档全集中自动提取章节标题、表格表头和技术术语，
    动态注册为 jieba 整词，确保 BM25 检索时不会被切碎。

    提取源：
      1. 章节标题行（编号型 / 章型 / 中文序号型 / Markdown # 型）
      2. Markdown 表格表头（| 列1 | 列2 | 行）
      3. SDK 函数名（snake_case 标识符，如 robot_movj）
      4. 英文大写缩写（≥3 字符，如 TCP、JOG、IO、SDK）

    所有术语在模块级 _auto_registered_terms 集合中去重，
    避免重复调用 jieba.add_word()。
    """
    global _auto_registered_terms

    # ── 标题提取正则（与 pdf_loader / multimodal_loader 保持一致）──
    _HEADING_RES = [
        re.compile(r'^(\d+(?:\.\d+)+)\s+(.+?)(?:\r?\n|$)', re.MULTILINE),
        re.compile(r'^(第[一二三四五六七八九十\d]+[章节])\s*(.+?)(?:\r?\n|$)', re.MULTILINE),
        re.compile(r'^([（(]?[一二三四五六七八九十]+[）)]?[\s、,，])\s*(.+?)(?:\r?\n|$)', re.MULTILINE),
        re.compile(r'^(#{1,4})\s+(.+?)(?:\r?\n|$)', re.MULTILINE),
    ]
    # ── 表格表头正则：| col1 | col2 | ... |
    _TABLE_HEADER_RE = re.compile(r'^\|\s*(.+?)\s*\|\s*(.+?)\s*\|', re.MULTILINE)

    new_terms = set()

    for doc in documents:
        text = doc.page_content if hasattr(doc, 'page_content') else str(doc)

        # 1. 提取章节标题
        for pat in _HEADING_RES:
            for m in pat.finditer(text):
                title = m.group(0).strip()
                if 2 <= len(title) <= 40:
                    new_terms.add(title)
                # 也尝试单独提取标题中的中文关键词部分
                if m.lastindex and m.lastindex >= 2:
                    kw = m.group(m.lastindex).strip()
                    if 2 <= len(kw) <= 20 and re.search(r'[一-鿿]', kw):
                        new_terms.add(kw)

        # 2. 提取表格表头中的列名
        for m in _TABLE_HEADER_RE.finditer(text):
            header_line = m.group(0)
            cells = re.findall(r'\|\s*(.+?)\s*(?=\||$)', header_line)
            for cell in cells:
                cell = cell.strip()
                # 过滤掉分隔线行（---）和空单元格
                if cell and not re.match(r'^[-:]+$', cell) and 2 <= len(cell) <= 30:
                    new_terms.add(cell)

        # 3. 提取英文大写缩写（≥3 字符）
        for m in re.finditer(r'\b([A-Z][A-Z0-9]{2,}(?:-[A-Z][A-Z0-9]{2,})?)\b', text):
            term = m.group(1)
            if 2 <= len(term) <= 10:
                new_terms.add(term)

        # 4. 提取 snake_case SDK 函数名片段（独立关键词）
        for m in re.finditer(r'\b([a-z]+)_([a-z]+)_([a-z]+)\b', text, re.IGNORECASE):
            # 例如 "set_robot_power_on" → 注册整词 + 关键子词
            full = m.group(0).lower()
            if 6 <= len(full) <= 30:
                new_terms.add(full)

    # ── 批量注册到 jieba ──
    registered_count = 0
    for term in new_terms:
        if term not in _auto_registered_terms and 2 <= len(term) <= 40:
            # 🔴 允许 3-5 位数字通过（端口号 6502、密码等屏幕截图 OCR 提取值）
            if re.match(r'^[\d\.\s\-—|]+$', term):
                if re.match(r'^\d{3,5}$', term):
                    pass  # 放行 3-5 位数字
                else:
                    continue
            _auto_registered_terms.add(term)
            jieba.add_word(term, freq=300, tag='auto')
            registered_count += 1

    if registered_count > 0:
        logger.info(
            f"📖 自动术语注册: {registered_count} 个新词 → jieba 词典 "
            f"(累计 {len(_auto_registered_terms)} 词)"
        )
    return registered_count


def _tokenize_for_bm25(text: str) -> List[str]:
    """
    BM25 分词器：jieba 中文分词 + 英文标识符保护 + CodeEntityAnchor 标签。

    v3 增强 (ADR-14):
      1. [CODE:entity_name] 标签 → 提取实体名作为强保护 token（双倍写入）
      2. 先用正则提取所有英文标识符（如 set_move_line），防止 jieba 将
         下划线连接的函数名切成碎片（set, _, move, _, line）。
      3. 再从文本中移除已提取的标识符，剩余中文部分交给 jieba 分词。
      4. 最终返回小写的 token 列表。
    """
    tokens = []

    # 🔴 v26: 分隔符空格归一化（先于一切处理）—— "Ethernet / IP" → "Ethernet/IP"
    text = _SPACE_SEP_RE.sub(r'\1\2\3', text)

    # 🔴 v3 Step 0: 提取 [CODE:...] 强保护标签 — 三倍写入实现 Boost=3.0 效果
    code_entities = re.findall(r'\[CODE:([a-zA-Z_][a-zA-Z0-9_]*)\]', text)
    for entity in code_entities:
        entity_clean = entity.lower().strip('_')
        if len(entity_clean) >= 2:
            # 三倍写入 = 3× BM25 IDF 权重（对抗 PDF 原文标题错误等噪声）
            tokens.append(entity_clean)
            tokens.append(entity_clean)
            tokens.append(entity_clean)
            # 对 robot_xxx 函数名，同时也写不带前缀的短名（如 movl）
            if '_' in entity_clean:
                short_name = entity_clean.split('_', 1)[-1] if entity_clean.startswith('robot_') else None
                if not short_name:
                    parts = entity_clean.rsplit('_', 1)
                    short_name = parts[-1] if len(parts) > 1 else None
                if short_name and len(short_name) >= 3:
                    tokens.append(short_name)  # 额外 boost 短名匹配
    # 移除 [CODE:...] 标签避免污染后续分词
    text_clean = re.sub(r'\[CODE:[a-zA-Z_][a-zA-Z0-9_]*\]', ' ', text)

    # 🔴 v26 Step 0.5: 复合词原子化（只追加不删除）—— Ethernet/IP → "ethernet/ip"
    # 子段（ethernet、ip）由 Step 1 保留 → token 全集 = 原全集 ∪ 复合集，零删除零回归
    for _m in _COMPOUND_RE.finditer(text_clean):
        _comp = _m.group(0).lower().strip('-_')
        if len(_comp) >= 3 and re.match(r'^[a-z0-9]+(?:[/\-][a-z0-9]+)+$', _comp):
            tokens.append(_comp)

    # 🔴 Step 1: 提取英文标识符作为不可分割的整词
    identifiers = _IDENTIFIER_RE.findall(text_clean)
    for ident in identifiers:
        ident_lower = ident.lower().strip('_')
        if len(ident_lower) >= 2 and not ident_lower.startswith('0x'):
            tokens.append(ident_lower)

    # 🔴 Step 2: 移除标识符后再 jieba 分词中文部分
    text_no_ident = _IDENTIFIER_RE.sub(' ', text_clean)
    # 🔴 Step 2b: 提取 3-5 位数字作为原子 token（端口号 6502、密码等）
    _numeric_tokens = re.findall(r'\b(\d{3,5})\b', text_no_ident)
    for num in _numeric_tokens:
        tokens.append(num)
    text_no_ident = re.sub(r'\b\d{3,5}\b', ' ', text_no_ident)
    # jieba 分词
    for word in jieba.cut(text_no_ident):
        word = word.strip()
        if not word or word in (' ', '_', '.', ',', ':', ';'):
            continue
        tokens.append(word.lower())

    return tokens


def build_bm25_from_chromadb(vector_store: Chroma):
    """
    从 ChromaDB 中提取所有文档，重建 BM25 内存索引。

    用于 FastAPI 启动时恢复 BM25 索引（BM25 为纯内存索引，无持久化）。
    """
    global _bm25_indexes, _bm25_corpus
    from rank_bm25 import BM25Okapi

    from langchain_core.documents import Document

    try:
        collection_data = vector_store._collection.get(include=["metadatas", "documents"])
        if not collection_data or not collection_data.get("ids"):
            logger.warning("⚠️  ChromaDB 为空，BM25 索引跳过")
            return

        ids = collection_data["ids"]
        docs = collection_data["documents"]
        metas = collection_data["metadatas"]

        # 按 product_id 分组
        product_docs: Dict[str, list] = {}
        for i, doc_id in enumerate(ids):
            pid = metas[i].get("product_id", "unknown") if i < len(metas) else "unknown"
            if pid not in product_docs:
                product_docs[pid] = []
            doc = Document(page_content=docs[i], metadata=metas[i])
            product_docs[pid].append(doc)

        # 🔴 自动术语提取：从 ChromaDB 恢复的所有文档中扫描术语并注册到 jieba
        all_docs = [doc for docs_list in product_docs.values() for doc in docs_list]
        _auto_extract_and_register_terms(all_docs)

        for pid, p_docs in product_docs.items():
            tokenized = [_tokenize_for_bm25(doc.page_content) for doc in p_docs]
            if not tokenized:
                continue
            _bm25_indexes[pid] = BM25Okapi(tokenized)
            _bm25_corpus[pid] = p_docs
            logger.info(f"📊 BM25 索引已恢复: product_id='{pid}', {len(p_docs)} 个片段")

        total = sum(len(v) for v in _bm25_corpus.values())
        logger.info(f"✅ BM25 混合检索就绪: {len(_bm25_indexes)} 个产品索引, 共 {total} 个片段")
    except Exception as e:
        logger.error(f"❌ BM25 索引恢复失败: {e}")


def build_bm25_index(documents: list, persist_dir: str = CHROMA_PERSIST_DIR):
    """
    为每个产品构建独立的 BM25 索引。

    遍历所有 Document，按 product_id 分组后分别构建 BM25Okapi 索引，
    存入全局 `_bm25_indexes` 和 `_bm25_corpus` 字典。

    Args:
        documents: LangChain Document 列表（每个 doc.metadata 含 product_id）
        persist_dir: 暂不使用（BM25 为内存索引，无需持久化）
    """
    global _bm25_indexes, _bm25_corpus
    from rank_bm25 import BM25Okapi

    # 🔴 自动术语提取：从文档中扫描章节标题/表头/缩写并注册到 jieba
    _auto_extract_and_register_terms(documents)

    # 按 product_id 分组
    product_docs: Dict[str, list] = {}
    for doc in documents:
        pid = doc.metadata.get("product_id", "unknown")
        if pid not in product_docs:
            product_docs[pid] = []
        product_docs[pid].append(doc)

    for pid, docs in product_docs.items():
        # 分词
        tokenized = [_tokenize_for_bm25(doc.page_content) for doc in docs]
        if not tokenized:
            continue
        _bm25_indexes[pid] = BM25Okapi(tokenized)
        _bm25_corpus[pid] = docs
        logger.info(f"📊 BM25 索引已构建: product_id='{pid}', {len(docs)} 个文档片段")

    total = sum(len(v) for v in _bm25_corpus.values())
    logger.info(f"✅ BM25 混合检索就绪: {len(_bm25_indexes)} 个产品索引, 共 {total} 个片段")


def bm25_search(
    query: str,
    product_id: str,
    k: int = 5,
) -> list:
    """
    对指定产品的 BM25 索引执行关键词检索。

    Args:
        query: 查询字符串
        product_id: 产品标识
        k: 返回的文档数量

    Returns:
        [(Document, bm25_score), ...] 按 BM25 得分降序排列
    """
    if product_id not in _bm25_indexes:
        return []

    bm25 = _bm25_indexes[product_id]
    corpus = _bm25_corpus[product_id]
    tokenized_query = _tokenize_for_bm25(query)

    scores = bm25.get_scores(tokenized_query)
    # 配对 (index, score) 并排序
    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in indexed_scores[:k]:
        if score > 0:  # 只保留有正得分的文档
            results.append((corpus[idx], float(score)))

    return results


def cleanup_vector_store():
    """
    释放嵌入模型和 ChromaDB 客户端持有的资源。

    应在 FastAPI 的 shutdown 事件中调用。
    ChromaDB 的 SQLite 连接在其 Python 对象被 GC 回收时自动关闭，
    此函数确保显式释放并及时。
    """
    global _embedding_function, _bm25_indexes, _bm25_corpus
    logger.info("🧹 正在清理向量库资源...")
    if _embedding_function is not None:
        _embedding_function = None
        logger.info("✅ 嵌入函数引用已释放")
    # 清空 BM25 索引
    _bm25_indexes.clear()
    _bm25_corpus.clear()
    logger.info("✅ BM25 索引已清空")
    logger.info("✅ 向量库资源清理完成")


# ============================================================
# 命令行测试入口
# ============================================================
if __name__ == "__main__":
    from .pdf_loader import load_all_documents_v4_dual
    from .config import PDF_DATA_DIR, JAKA_MARKDOWN_PATH

    print("=== 正在测试 Stage 1 双轨数据加载与向量化 ===")
    p_docs, c_docs = load_all_documents_v4_dual(PDF_DATA_DIR, JAKA_MARKDOWN_PATH)
    print(f"✅ 加载完成: {len(p_docs)} 个父切片, {len(c_docs)} 个子切片")