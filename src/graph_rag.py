"""
=============================================================================
LangGraph RAG 状态图引擎 — 第一阶段架构重构
=============================================================================

将传统的平铺 RAG 管线（rag_chat / rag_chat_stream）重构为基于
LangGraph StateGraph 的四节点状态图引擎。

【图结构】

  START
    │
    ▼
  Node 1: query_fusion        ← QueryFusionNode（多轮融合 + 短词补全）
    │
    ▼
  Node 2: product_routing     ← ProductRoutingNode（产品识别 + 反问判断）
    │
    ├── route_status ∈ {clarify, chitchat, refuse}
    │       → 跳过检索，直接 build_direct_response → END
    │
    └── route_status ∈ {generate, fallback}
            │
            ▼
          Node 3: hybrid_retrieval  ← HybridRetrievalNode（章节注入混合检索）
            │
            ▼
          Node 4: llm_generation   ← LLMGenerationNode（四层容灾 + 章节溯源）
            │
            ▼
           END

【API 兼容】

  - run_graph(query, history, product_id)          → Dict（非流式）
  - run_graph_stream(query, history, product_id)   → Generator[str]（流式 SSE）

  上游 app.py 的 /api/chat 路由无需任何修改即可切换为 Graph 引擎。

=============================================================================
"""
import logging
import threading
from typing import List, Dict, Optional, Generator, Any

from langgraph.graph import StateGraph, END

from .agent_state import RAGState
from . import config as _cfg

logger = logging.getLogger(__name__)

# ============================================================
# 复用现有模块中的核心函数（零重复实现）
# ============================================================
from .rag_chain import (
    _preprocess_query,
    _fuse_short_query,
    _resolve_clarification_followup,
    _resolve_product_from_query,
    _is_chitchat,
    _is_impossible_query,
    _has_business_intent,
    _is_product_name_only,
    _build_messages,
    _expand_parent_sections,
    _last_numeric_context_missing,
    _hybrid_retrieve,
    _call_llm,
    _stream_llm,
    _check_vllm_health,
    _acquire_vllm_lock,
    _release_vllm_lock,
    _get_client,
    _get_deepseek_client,
    _resolve_vllm_model,
    _direct_retrieval_response,
    _direct_retrieval_response_stream,
    _hard_refusal_response,
    _HARD_REFUSAL,
    _FALLBACK_EXCEPTIONS,
    _FALLBACK_ENABLED,
    _chitchat_response,
    _chitchat_response_stream,
    _hard_refusal_stream,
    _build_clarification_response,
    _build_clarification_response_stream,
)
from .vector_store import get_registered_products, search_similar_with_threshold


# ============================================================
# Node 1: QueryFusionNode — 多轮对话融合 + 短词补全
# ============================================================

def query_fusion_node(state: RAGState) -> dict:
    """
    对原始 query 进行：口语噪音剥离 → 澄清补全 → 短词融合。

    输入：state["query"], state["chat_history"], state["product_id"]
    输出：fused_query, product_id（可能从澄清补全中解析）
    """
    query = state.get("query", "")
    chat_history = state.get("chat_history")
    product_id = state.get("product_id")

    logger.info(f"🟢 [Node 1] QueryFusion: raw='{query[:60]}'")

    # 第 1 步：澄清补全（检测上一轮是否为澄清反问 + 用户仅输入产品名）
    if not product_id:
        fused, resolved_pid = _resolve_clarification_followup(query, chat_history)
        if resolved_pid:
            product_id = resolved_pid
            query = fused
            logger.info(f"  ↳ 澄清补全: product_id='{product_id}', fused_query='{query[:80]}'")

    # 第 2 步：短词融合（< 8 字符 → 从历史拼接语义）
    fused_query = _fuse_short_query(query, chat_history, product_id)
    if fused_query != query:
        query = fused_query
        logger.info(f"  ↳ 短词融合: '{query[:80]}'")

    # 第 3 步：口语噪音剥离
    cleaned = _preprocess_query(query)

    return {
        "fused_query": cleaned,
        "query": query,  # 保留融合后的原始版本（含产品名关键词）供下游使用
        "product_id": product_id,
    }


# ============================================================
# Node 2: ProductRoutingNode — 产品识别 + 意图分类
# ============================================================

def product_routing_node(state: RAGState) -> dict:
    """
    确定 route_status：
      - "chitchat"  → 闲聊/身份询问
      - "refuse"    → 不可能组合（如 JAKA+NumPy）
      - "clarify"   → 产品未识别，需反问
      - "generate"  → 正常检索+生成
      - "fallback"  → 产品已识别但检索词较弱

    同时生成对应的直接回答（如澄清反问/身份回复/硬拒答），
    存入 state["final_answer"]。
    """
    query = state.get("fused_query") or state.get("query", "")
    product_id = state.get("product_id")

    logger.info(f"🟡 [Node 2] ProductRouting: query='{query[:60]}', product_id='{product_id}'")

    # ── 意图 1: 闲聊/身份询问 ──
    if _is_chitchat(query):
        logger.info("  ↳ route_status='chitchat'")
        resp = _chitchat_response()
        return {
            "route_status": "chitchat",
            "final_answer": resp["answer"],
            "sources": resp.get("sources", []),
            "model": resp.get("model", "identity-router"),
        }

    # ── 意图 2: 不可能组合 ──
    if _is_impossible_query(query):
        logger.info("  ↳ route_status='refuse'")
        resp = _hard_refusal_response()
        return {
            "route_status": "refuse",
            "final_answer": resp["answer"],
            "sources": [],
            "model": "hard-refusal",
        }

    # ── 意图 3: 产品未识别 → 反问澄清 ──
    if not product_id:
        from .rag_chain import _resolve_product_from_query as _resolve
        product_id = _resolve(query)
        if not product_id:
            registered = get_registered_products()
            resp = _build_clarification_response(registered)
            logger.info("  ↳ route_status='clarify'")
            return {
                "route_status": "clarify",
                "final_answer": resp["answer"],
                "sources": [],
                "model": "product-clarification",
                "product_id": None,
            }

    # ── 意图 4: 正常生成 ──
    logger.info(f"  ↳ route_status='generate', product_id='{product_id}'")
    return {
        "route_status": "generate",
        "product_id": product_id,
    }


# ============================================================
# 条件路由函数
# ============================================================

def _route_after_product_routing(state: RAGState) -> str:
    """
    根据 route_status 决定下一个节点：
      - 需要直接回复 → "build_direct_response"（跳过检索和 LLM）
      - 需要检索生成 → "hybrid_retrieval"
    """
    status = state.get("route_status", "generate")
    if status in ("clarify", "chitchat", "refuse"):
        return "build_direct_response"
    return "hybrid_retrieval"


# ============================================================
# Node 2b: BuildDirectResponse — 转发预生成的回答文本
# ============================================================

def build_direct_response_node(state: RAGState) -> dict:
    """
    对于澄清/闲聊/拒答场景，final_answer 已在 ProductRoutingNode 中填充完毕，
    此节点仅设置 route_status="complete" 让流式输出知道管线已结束。
    """
    logger.info(f"🟠 [Node 2b] BuildDirectResponse: route_status='{state.get('route_status')}'")
    return {"route_status": "complete"}


# ============================================================
# Node 3: HybridRetrievalNode — 章节注入混合检索 + 第二机会保底
# ============================================================

def hybrid_retrieval_node(state: RAGState) -> dict:
    """
    使用 fused_query + product_id 执行混合检索（向量 + BM25 + RRF + Autocut）。

    检索策略：
      1. 主检索：阈值过滤 + 产品隔离
      2. 若主检索为空 → 第二机会：无阈值原始向量 Top-3 兜底
      3. 若仍为空 → route_status="fallback"（仍尝试 LLM 生成）

    需要外部注入 vector_store 实例。通过模块级 _graph_vector_store 变量传递。
    """
    query = state.get("fused_query") or state.get("query", "")
    product_id = state.get("product_id")

    logger.info(f"🔵 [Node 3] HybridRetrieval: query='{query[:60]}', product_id='{product_id}'")

    vector_store = _get_graph_vector_store()
    if vector_store is None:
        logger.error("❌ vector_store 未注入到 Graph 引擎")
        return {
            "retrieved_docs": [],
            "route_status": "fallback",
        }

    # 主检索
    context_docs = _hybrid_retrieve(
        vector_store, query,
        k=_cfg.RETRIEVAL_K,
        threshold=_cfg.SIMILARITY_THRESHOLD,
        fetch_factor=5,
        product_id=product_id,
    )

    # 第二机会检索
    if not context_docs:
        logger.warning("⚠️  主检索为空，触发第二机会（无阈值 Top-3）")
        context_docs = search_similar_with_threshold(
            vector_store, query, k=3, threshold=None, product_id=product_id,
        )

    route_status = "fallback" if not context_docs else "generate"
    logger.info(
        f"  ↳ 检索完成: {len(context_docs)} chunks, route_status='{route_status}'"
    )

    return {
        "retrieved_docs": context_docs,
        "route_status": route_status,
    }


# ============================================================
# Node 4: LLMGenerationNode — 四层容灾 + 章节溯源约束
# ============================================================

def llm_generation_node(state: RAGState) -> dict:
    """
    使用检索到的文档片段，调用四层金字塔容灾 LLM 生成最终回答。

    容灾链路：本地 vLLM → 云端智谱 API → 纯检索直出 → 硬拒答兜底
    """
    query = state.get("fused_query") or state.get("query", "")
    context_docs = state.get("retrieved_docs", [])
    chat_history = state.get("chat_history")

    logger.info(f"🟣 [Node 4] LLMGeneration: {len(context_docs)} docs → LLM")

    # ── 父子切片扩展 + 构建消息 ──
    if context_docs:
        context_docs = _expand_parent_sections(
            context_docs, _get_graph_vector_store(),
            product_id=state.get("product_id"), max_siblings=2,
        )
    try:
        messages = _build_messages(query, context_docs, chat_history)
    except Exception as e:
        logger.error(f"❌ Prompt 构建失败: {e}")
        try:
            result = _direct_retrieval_response(context_docs, query)
            return {
                "final_answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "model": result.get("model", "direct-retrieval-fallback"),
                "route_status": "complete",
            }
        except Exception:
            return {
                "final_answer": _HARD_REFUSAL,
                "sources": [],
                "model": "fatal-fallback",
                "route_status": "complete",
            }

    # 🔴 数字请求无上下文硬防护
    if _last_numeric_context_missing:
        logger.info("🚫 [Graph] 数字请求无上下文 → 直接返回硬拒答")
        return {
            "final_answer": _HARD_REFUSAL,
            "sources": [],
            "model": "numeric-guard",
            "route_status": "complete",
        }

    # ── Layer 1: 本地 vLLM ──
    vllm_healthy = _check_vllm_health()
    if vllm_healthy:
        lock_acquired = _acquire_vllm_lock()
        try:
            if lock_acquired:
                model = _resolve_vllm_model()
                answer = _call_llm(_get_client(), model, messages)
                if answer and answer.strip():
                    logger.info("✅ Layer 1 (vLLM) 成功")
                    return _build_final_answer(answer, context_docs, model)
        except _FALLBACK_EXCEPTIONS as e:
            logger.warning(f"⚠️  Layer 1 不可用: {e}")
        except Exception as e:
            logger.warning(f"⚠️  Layer 1 异常: {type(e).__name__}: {e}")
        finally:
            if lock_acquired:
                _release_vllm_lock()

    # ── Layer 2: 云端智谱 API ──
    if _FALLBACK_ENABLED:
        logger.info("🔄 降级 Layer 2 (智谱 API)...")
        try:
            from .config import DEEPSEEK_MODEL
            answer = _call_llm(_get_deepseek_client(), DEEPSEEK_MODEL, messages)
            if answer and answer.strip():
                logger.info("✅ Layer 2 (智谱 API) 成功")
                return _build_final_answer(answer, context_docs, DEEPSEEK_MODEL)
        except _FALLBACK_EXCEPTIONS as e:
            logger.warning(f"⚠️  Layer 2 不可用: {e}")
        except Exception as e:
            logger.warning(f"⚠️  Layer 2 异常: {type(e).__name__}: {e}")

    # ── Layer 3: 纯检索直出 ──
    logger.info("🔄 降级 Layer 3 (纯检索直出)...")
    try:
        result = _direct_retrieval_response(context_docs, query)
        answer = result.get("answer", "")
        if answer.strip():
            return {
                "final_answer": answer,
                "sources": result.get("sources", []),
                "model": result.get("model", "direct-retrieval"),
                "route_status": "complete",
            }
    except Exception as e:
        logger.error(f"❌ Layer 3 失败: {e}")

    # ── Layer 4: 硬拒答兜底 ──
    logger.critical("❌ 四层容灾全部耗尽")
    return {
        "final_answer": _HARD_REFUSAL,
        "sources": [],
        "model": "never-empty-guarantee",
        "route_status": "complete",
    }


def _build_final_answer(answer: str, context_docs: list, model: str) -> dict:
    """组装最终回答字典。"""
    sources = list(set(
        doc.metadata.get("source", "未知")
        for doc in context_docs
    )) if context_docs else []
    return {
        "final_answer": answer,
        "sources": sources,
        "model": model,
        "route_status": "complete",
    }


# ============================================================
# Graph 构建与编译
# ============================================================

# 模块级单例：编译好的 LangGraph 实例
_compiled_graph = None

# 模块级 vector_store 引用（由 app.py 在启动时注入）
_graph_vector_store = None


def _get_graph_vector_store():
    """获取当前注入的 vector_store 实例。"""
    global _graph_vector_store
    return _graph_vector_store


def set_graph_vector_store(vs):
    """
    向 Graph 引擎注入 ChromaDB 向量库实例。

    应在 app.py 的 startup 事件中调用。
    """
    global _graph_vector_store
    _graph_vector_store = vs
    logger.info("✅ Graph 引擎已注入 vector_store 实例")


def _build_graph() -> StateGraph:
    """
    构建并编译 LangGraph StateGraph。

    返回编译后的图实例（带状态校验）。
    """
    graph = StateGraph(RAGState)

    # ── 注册节点 ──
    graph.add_node("query_fusion", query_fusion_node)
    graph.add_node("product_routing", product_routing_node)
    graph.add_node("build_direct_response", build_direct_response_node)
    graph.add_node("hybrid_retrieval", hybrid_retrieval_node)
    graph.add_node("llm_generation", llm_generation_node)

    # ── 注册边 ──
    graph.set_entry_point("query_fusion")
    graph.add_edge("query_fusion", "product_routing")

    # 条件边：根据 route_status 决定是否跳过检索
    graph.add_conditional_edges(
        "product_routing",
        _route_after_product_routing,
        {
            "build_direct_response": "build_direct_response",
            "hybrid_retrieval": "hybrid_retrieval",
        },
    )

    graph.add_edge("build_direct_response", END)
    graph.add_edge("hybrid_retrieval", "llm_generation")
    graph.add_edge("llm_generation", END)

    return graph.compile()


def get_graph():
    """
    获取编译后的 LangGraph 实例（懒加载单例）。
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph()
        logger.info("✅ LangGraph RAG 状态图已编译")
    return _compiled_graph


# ============================================================
# 公开 API — 与 rag_chat / rag_chat_stream 兼容
# ============================================================

def run_graph(
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    product_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    使用 LangGraph 引擎执行一次 RAG 对话（非流式）。

    与 rag_chat() 接口完全兼容：
      - 相同输入参数
      - 相同输出格式：{"answer", "sources", "model", "needs_clarification"}

    Args:
        query: 用户问题
        chat_history: 多轮对话历史
        product_id: 产品标识（可选）

    Returns:
        {"answer": str, "sources": [str], "model": str}
    """
    graph = get_graph()

    initial_state: RAGState = {
        "query": query,
        "fused_query": "",
        "product_id": product_id,
        "chat_history": chat_history or [],
        "retrieved_docs": [],
        "final_answer": "",
        "sources": [],
        "model": "",
        "route_status": "",
    }

    # invoke() 同步执行完整图，返回最终状态
    final_state = graph.invoke(initial_state)

    needs_clarification = final_state.get("route_status") == "clarify"

    return {
        "answer": final_state.get("final_answer", ""),
        "sources": final_state.get("sources", []),
        "model": final_state.get("model", "langgraph"),
        "needs_clarification": needs_clarification,
    }


def run_graph_stream(
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    product_id: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    使用 LangGraph 引擎执行一次 RAG 对话（流式）。

    与 rag_chat_stream() 接口完全兼容。

    流式策略：
      1. 使用 graph.astream() 异步追踪节点执行
      2. 在 llm_generation 节点内部使用 _stream_llm（真正的 token 级流式）
      3. 对于非 LLM 节点，yield 预生成的回答文本（模拟打字机）
    """
    # 对于流式场景，我们直接复用 rag_chat_stream() 的内部逻辑，
    # 以保持 token 级流式输出。LangGraph 的 astream() 是节点级事件，
    # 而非 token 级，因此流式生成更适合直接在 LLMGenerationNode 内部处理。
    #
    # 这里提供一个基于 Graph 状态流的模拟实现，将图节点作为编排层。

    # 运行前置节点（query_fusion → product_routing）获取路由决策
    initial_state: RAGState = {
        "query": query,
        "fused_query": "",
        "product_id": product_id,
        "chat_history": chat_history or [],
        "retrieved_docs": [],
        "final_answer": "",
        "sources": [],
        "model": "",
        "route_status": "",
    }

    # Step 1 & 2: query_fusion → product_routing
    # 🔴 关键：使用 {**base, **overrides} 模式确保节点输出覆盖初始默认值，
    # 而非被空默认值反向覆盖（这是上一版的致命状态污染 Bug）。
    s1 = query_fusion_node(initial_state)
    state = {**initial_state, **s1}           # s1 的输出优先
    s2 = product_routing_node(state)
    state.update(s2)                          # 合并路由决策
    route_status = state.get("route_status", "generate")

    if route_status in ("clarify", "chitchat", "refuse"):
        # 直接回复，模拟打字机效果
        answer = state.get("final_answer", "")
        chunk_size = 15
        for i in range(0, len(answer), chunk_size):
            yield answer[i:i + chunk_size]
        return

    # Step 3: hybrid_retrieval
    s3 = hybrid_retrieval_node(state)
    state.update(s3)                          # 合并检索结果
    context_docs = state.get("retrieved_docs", [])
    fused_query = state.get("fused_query") or state.get("query", "")

    # 🔴 父子切片扩展
    if context_docs:
        context_docs = _expand_parent_sections(
            context_docs, _get_graph_vector_store(),
            product_id=state.get("product_id"), max_siblings=2,
        )

    # Step 4: LLM generation with real token streaming
    # 构建消息
    from .rag_chain import _build_messages, _check_vllm_health, _acquire_vllm_lock, _release_vllm_lock
    from .rag_chain import _get_client, _get_deepseek_client, _resolve_vllm_model, _stream_llm
    from .rag_chain import _FALLBACK_EXCEPTIONS, _FALLBACK_ENABLED
    from .rag_chain import _direct_retrieval_response_stream, _hard_refusal_stream

    try:
        messages = _build_messages(fused_query, context_docs, chat_history)
    except Exception:
        yield from _hard_refusal_stream()
        return

    # 🔴 数字请求无上下文硬防护
    if _last_numeric_context_missing:
        logger.info("🚫 [Graph Stream] 数字请求无上下文 → 硬拒答")
        yield from _hard_refusal_stream()
        return

    _yielded = [False]

    def _track(gen):
        for chunk in gen:
            _yielded[0] = True
            yield chunk

    # Layer 1
    vllm_healthy = _check_vllm_health()
    if vllm_healthy:
        lock_acquired = _acquire_vllm_lock()
        try:
            if lock_acquired:
                yield from _track(_stream_llm(_get_client(), _resolve_vllm_model(), messages))
                if _yielded[0]:
                    return
        except _FALLBACK_EXCEPTIONS:
            pass
        except Exception:
            pass
        finally:
            if lock_acquired:
                _release_vllm_lock()

    # Layer 2
    if _FALLBACK_ENABLED:
        try:
            from .config import DEEPSEEK_MODEL
            yield from _track(_stream_llm(_get_deepseek_client(), DEEPSEEK_MODEL, messages))
            if _yielded[0]:
                return
        except _FALLBACK_EXCEPTIONS:
            pass
        except Exception:
            pass

    # Layer 3
    try:
        yield from _track(_direct_retrieval_response_stream(context_docs, fused_query))
        if _yielded[0]:
            return
    except Exception:
        pass

    # Layer 4
    yield from _hard_refusal_stream()


# ============================================================
# 模块初始化
# ============================================================

logger.info("📐 LangGraph RAG 引擎模块已加载（图实例将在首次调用时编译）")
