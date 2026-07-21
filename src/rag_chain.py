"""
=============================================================================
RAG 对话管线 — 检索增强生成的核心编排层
=============================================================================

【经典 RAG 四步法】

  用户提问: "张三在2024年获得了什么奖？"
        │
        ▼
  ┌─────────────────────────────────────────┐
  │ ① 检索 (Retrieve)                       │
  │   将问题转为向量，在知识库中搜索 Top-K   │
  │   结果: ["张三，2024年，诺贝尔物理学奖"] │
  ├─────────────────────────────────────────┤
  │ ② 增强 (Augment)                        │
  │   将检索结果拼入 Prompt 模板作为上下文   │
  │   Prompt = 系统指令 + 参考资料 + 问题    │
  ├─────────────────────────────────────────┤
  │ ③ 生成 (Generate)                       │
  │   调用 LLM 基于增强 Prompt 生成回答      │
  │   LLM 输出: "张三获得了诺贝尔物理学奖"   │
  ├─────────────────────────────────────────┤
  │ ④ 返回 (Response)                       │
  │   将回答返回给用户                       │
  └─────────────────────────────────────────┘

【多轮对话支持】

  本模块维护一个对话历史列表，格式为：
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
  每次发起新对话时，将历史一并传给 LLM，使其能够理解上下文。
  例如：
    用户: "张三得了什么奖？"     ← 第一次提问
    AI: "诺贝尔物理学奖。"
    用户: "哪一年得的？"         ← 第二次提问（省略了主语"张三"和"奖"）
                                          但 AI 从历史中能推断出指代对象

【四层金字塔容灾架构 — ADR-5】

  本模块实现了"本地 vLLM → DeepSeek API → 纯检索直出 → 优雅错误"的四层降级链路：

    ┌─────────────────────────────────────────────────────────────────┐
    │  第 1 层 │ 本地 vLLM 推理服务 (GPU 物理推理，零 API 费用)      │
    │         │ 超时: connect=3s / read=15s，假死快速切断             │
    ├─────────────────────────────────────────────────────────────────┤
    │  第 2 层 │ 云端 DeepSeek API (云端算力备份，自动无缝切换)       │
    ├─────────────────────────────────────────────────────────────────┤
    │  第 3 层 │ 纯向量检索直出模式 (纯 CPU 运行，零显存 / 零 API)    │
    │         │ 从 ChromaDB 检索原文片段，模板化组装后直接返回        │
    ├─────────────────────────────────────────────────────────────────┤
    │  第 4 层 │ 优雅中文错误提示 (仅在向量库损坏等极端情况触发)      │
    │         │ "大模型服务暂时不可用，请稍后重试"                    │
    └─────────────────────────────────────────────────────────────────┘

  详见 dev_log.md「ADR-5：四层金字塔容灾架构」。

=============================================================================
"""

import logging
from typing import List, Dict, Optional, Generator

import httpx
from openai import OpenAI, APITimeoutError, APIConnectionError

from .config import (
    BASE_URL, API_KEY, MODEL_NAME, RETRIEVAL_K,
    DEEPSEEK_BASE_URL, DEEPSEEK_API_KEY, DEEPSEEK_MODEL,
)
from .vector_store import search_similar

logger = logging.getLogger(__name__)

# ============================================================
# 超时配置 — 解决 vLLM 假死时前端无限等待问题
# ============================================================

# 显式配置 httpx 超时参数，防止 vLLM 进程假死（GPU 卡死但 TCP 端口仍监听）时
# 系统陷入无限等待。默认 openai 库的 read timeout 为 600s，对用户不可接受。
#
# - connect=3.0s : TCP 连接建立超时（vLLM 未启动 → 3 秒内快速失败）
# - read=15.0s   : 读取超时（vLLM 假死/GPU 卡死 → 15 秒内切断并降级）
# - write=15.0s  : 写入超时
# - pool=3.0s    : 连接池获取超时
LLM_TIMEOUT = httpx.Timeout(connect=3.0, read=15.0, write=15.0, pool=3.0)

# ============================================================
# 用户友好错误提示
# ============================================================

FRIENDLY_ERROR_MSG = "大模型服务暂时不可用，请稍后重试"


class LLMServiceError(Exception):
    """
    LLM 服务不可用异常（第 4 层兜底）。

    当本地 vLLM、云端 DeepSeek API 和纯检索直出模式全部失败时抛出。
    上层（app.py）应捕获此异常并向前端返回用户可读的中文提示，
    而非原始 Python 堆栈或 HTTP 500。
    """
    pass


# ============================================================
# 降级触发条件
# ============================================================

# 以下异常类型表示"网络层/传输层故障"，触发降级切换；
# 使用 httpx 原生异常 + OpenAI SDK 封装异常的双重覆盖，
# 确保不同版本的 SDK 行为差异不会导致降级失效。
_FALLBACK_EXCEPTIONS = (
    httpx.TimeoutException,       # 父类：涵盖 ConnectTimeout / ReadTimeout / WriteTimeout / PoolTimeout
    httpx.NetworkError,            # 父类：涵盖 ConnectError / ReadError / WriteError / RemoteProtocolError
    APITimeoutError,               # OpenAI SDK 封装的超时异常
    APIConnectionError,            # OpenAI SDK 封装的连接异常
)

# 当主 BASE_URL 已经是 DeepSeek 时，跳过同源降级（避免无意义的重复请求）
_FALLBACK_ENABLED = BASE_URL != DEEPSEEK_BASE_URL

# ============================================================
# LLM 客户端（OpenAI 兼容接口，单例模式）
# ============================================================

_client: Optional[OpenAI] = None
_deepseek_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """
    获取主 LLM 客户端实例（懒加载，单例）。

    【兼容性说明】
    此客户端同时兼容：
    - DeepSeek API（Anthropic 兼容端点）
    - 本地 vLLM 服务（OpenAI 兼容端点）
    - 任何 OpenAI 兼容的 API 服务

    切换方式：修改 src/config.py 中的 BASE_URL / API_KEY / MODEL_NAME 即可，
    无需改动此文件的任何代码。这是"依赖倒置原则"的体现。
    """
    global _client
    if _client is None:
        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=LLM_TIMEOUT)
        logger.info(
            f"LLM 客户端已初始化: base_url={BASE_URL}, model={MODEL_NAME}, "
            f"timeout=connect:{LLM_TIMEOUT.connect}s/read:{LLM_TIMEOUT.read}s"
        )
    return _client


def _get_deepseek_client() -> OpenAI:
    """
    获取 DeepSeek 降级客户端实例（懒加载，单例）。

    仅当主 LLM 不可用触发降级时才首次初始化，避免不必要的资源占用。
    使用与主客户端相同的超时配置。
    """
    global _deepseek_client
    if _deepseek_client is None:
        _deepseek_client = OpenAI(
            base_url=DEEPSEEK_BASE_URL,
            api_key=DEEPSEEK_API_KEY,
            timeout=LLM_TIMEOUT,
        )
        logger.info(
            f"DeepSeek 降级客户端已初始化: base_url={DEEPSEEK_BASE_URL}, "
            f"timeout=connect:{LLM_TIMEOUT.connect}s/read:{LLM_TIMEOUT.read}s"
        )
    return _deepseek_client


# ============================================================
# LLM 调用辅助函数（DRY 原则 — 双通道复用同一调用逻辑）
# ============================================================

def _call_llm(client: OpenAI, model: str, messages: List[Dict[str, str]]) -> str:
    """
    调用 LLM 完成非流式推理，返回完整回答文本。

    将 client.chat.completions.create() 封装为独立函数，
    便于 rag_chat() 中 Layer 1 / Layer 2 复用同一调用逻辑。
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,        # 低温度 → 输出更确定性，减少幻觉
        max_tokens=2048,
    )
    return response.choices[0].message.content


def _stream_llm(
    client: OpenAI, model: str, messages: List[Dict[str, str]]
) -> Generator[str, None, None]:
    """
    调用 LLM 完成流式推理，逐 token 产出文本增量。

    将流式调用封装为独立生成器函数，
    便于 rag_chat_stream() 中 Layer 1 / Layer 2 复用同一调用逻辑。
    """
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
        stream=True,  # ← 关键：开启流式模式
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


# ============================================================
# 第 3 层降级：纯向量检索直出模式 (Direct Retrieval Fallback)
# ============================================================
#
# 当 Layer 1（本地 vLLM）和 Layer 2（DeepSeek API）全部失败或超时后触发。
#
# 特点：
#   - 纯 CPU 运行：仅使用 ChromaDB 向量检索 + 模板组装，不调用任何 LLM
#   - 零显存消耗：不经过 vLLM / PyTorch GPU 推理
#   - 零 API 费用：不产生任何云端 API 调用
#   - 秒级响应：省略 LLM 推理延迟，直接返回检索结果
#   - 支持流式：分段 yield 文本，前端打字机效果正常运作
#
# 局限性：
#   - 不做内容理解与总结，仅提供原文片段
#   - 多轮对话上下文不会影响检索结果（仅基于当前 query 检索）
# ============================================================

# 纯检索模式使用的 Top-K 值（可独立于常规 RAG 的 RETRIEVAL_K 调整）
# 纯检索模式使用的 Top-K 值（与 RETRIEVAL_K 保持一致，确保 Layer 3 输出完整召回）
DIRECT_RETRIEVAL_K = 5

# 纯检索模式的提示文本模板
DIRECT_RETRIEVAL_HEADER = (
    "【提示：当前大模型生成服务未就绪，已为您开启纯文档检索直出模式】\n\n"
    "根据比邻星技术文档，找到以下相关内容：\n\n"
)

DIRECT_RETRIEVAL_FOOTER = (
    "\n---\n"
    "💡 以上为文档原文检索结果。如需更深入的分析与总结，请等待大模型服务恢复后重试。"
)


def _format_direct_retrieval_answer(context_docs: List) -> str:
    """
    将检索到的文档片段格式化为用户可读的纯文本回答。

    Args:
        context_docs: 检索到的 LangChain Document 列表

    Returns:
        格式化的纯文本回答字符串
    """
    parts = []
    for i, doc in enumerate(context_docs, start=1):
        source = doc.metadata.get("source", "未知来源")
        content = doc.page_content.strip()
        parts.append(f"{i}. [来源: {source}]\n{content}")

    return DIRECT_RETRIEVAL_HEADER + "\n".join(parts) + DIRECT_RETRIEVAL_FOOTER


def _direct_retrieval_response(
    context_docs: List,
    query: str,
) -> Dict[str, any]:
    """
    第 3 层降级 — 纯向量检索直出模式（非流式）。

    仅依赖 ChromaDB 检索结果 + 模板组装，不调用任何 LLM。
    整个过程纯 CPU 运行，零显存消耗，零 API 费用，秒级响应。

    Args:
        context_docs: 已检索到的文档片段列表
        query: 用户原始问题（保留以备未来扩展，如日志记录）

    Returns:
        {"answer": ..., "sources": [...], "model": "direct-retrieval (CPU-only)"}
    """
    logger.info(
        f"🔷 进入纯检索直出模式（第 3 层降级），"
        f"返回 Top-{len(context_docs)} 文档原文片段"
    )

    direct_answer = _format_direct_retrieval_answer(context_docs)

    sources = list(set(
        doc.metadata.get("source", "未知")
        for doc in context_docs
    ))

    return {
        "answer": direct_answer,
        "sources": sources,
        "model": "direct-retrieval (CPU-only, zero-GPU/API)",
    }


def _direct_retrieval_response_stream(
    context_docs: List,
    query: str,
) -> Generator[str, None, None]:
    """
    第 3 层降级 — 纯向量检索直出模式（流式）。

    将组装好的文本以 ~15 字符/块的速率分段 yield，
    模拟打字机效果，确保前端 SSE 流式渲染正常工作。

    Args:
        context_docs: 已检索到的文档片段列表
        query: 用户原始问题（保留以备未来扩展）

    Yields:
        文本增量（每次约 15 个字符）
    """
    logger.info(
        f"🔷 进入纯检索直出模式-流式（第 3 层降级），"
        f"返回 Top-{len(context_docs)} 文档原文片段"
    )

    direct_answer = _format_direct_retrieval_answer(context_docs)

    # 分段 yield 模拟流式打字机效果
    # 块大小 ~15 字符：平衡前端渲染频率与网络开销
    chunk_size = 15
    for i in range(0, len(direct_answer), chunk_size):
        yield direct_answer[i:i + chunk_size]


# ============================================================
# Prompt 模板 — RAG 的核心"咒语"
# ============================================================

RAG_SYSTEM_PROMPT = """你是由湖南比邻星科技有限公司开发的官方开发与使用文档智能助手。
你的任务是基于提供的公司内部文档资料，准确、专业地回答用户关于公司产品、
API 接口、开发指南和使用手册的问题。

请严格遵守以下规则：
1. 回答必须严格基于【参考资料】中的内容，不得编造或臆测信息
2. 如果参考资料不足以回答问题，请明确告知用户"根据现有文档，无法找到相关信息，建议联系技术支持或查阅最新文档"
3. 回答应条理清晰、专业规范，尽量使用简洁的语言
4. 可以适当引用参考资料中的原文（使用引号标注），便于用户对照查阅
5. 如果用户的问题涉及代码实现，请同时注明参考的文档来源
"""


def _build_messages(
    query: str,
    context_docs: List,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """
    构建发送给 LLM 的完整消息列表。

    【消息结构】
    [
      {"role": "system",    "content": "系统指令"},
      {"role": "user",      "content": "历史对话 + 参考资料 + 当前问题"},
      {"role": "assistant", "content": "..."},   ← 历史回答
      {"role": "user",      "content": "..."},   ← 上一轮问题
      ...
      {"role": "user",      "content": "（最终增强后的 Prompt）"}
    ]

    Args:
        query: 当前用户问题
        context_docs: 检索到的相关文档片段列表
        chat_history: 历史对话 [{"role": "...", "content": "..."}, ...]

    Returns:
        messages 列表，可直接传给 OpenAI API
    """

    # ---- 拼接参考资料 ----
    context_parts = []
    for i, doc in enumerate(context_docs, start=1):
        source = doc.metadata.get("source", "未知来源")
        content = doc.page_content.strip()
        context_parts.append(f"[参考资料 {i}]（来源：{source}）\n{content}")

    context_text = "\n\n---\n\n".join(context_parts)

    # ---- 构建当前轮次的用户消息 ----
    current_user_message = f"""【参考资料】
{context_text}

---
【用户问题】
{query}

请基于以上参考资料回答问题。如果参考资料不足以回答，请明确说明。"""

    # ---- 组装完整消息列表 ----
    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
    ]

    if chat_history:
        messages.extend(chat_history)

    messages.append({"role": "user", "content": current_user_message})

    return messages


# ============================================================
# 核心 API：RAG 对话（非流式）— 四层金字塔容灾
# ============================================================

def rag_chat(
    vector_store,
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    k: int = RETRIEVAL_K,
) -> Dict[str, any]:
    """
    执行一次完整的 RAG 对话（非流式，一次性返回完整结果）。

    【完整调用链 — 四层容灾】
    query → [向量检索] → context_docs → [构建 Prompt] → messages
         │
         ├── 第 1 层：本地 vLLM 推理 (GPU)
         │     └── 成功 → 返回 LLM 生成的回答
         │     └── 失败/超时 → 进入第 2 层
         │
         ├── 第 2 层：云端 DeepSeek API 降级 (Cloud)
         │     └── 成功 → 返回 LLM 生成的回答（日志标注"降级成功"）
         │     └── 失败/超时 → 进入第 3 层
         │
         ├── 第 3 层：纯向量检索直出模式 (CPU-only, 零显存/零API)
         │     └── 成功 → 返回检索原文片段（模板组装）
         │     └── 失败 → 进入第 4 层
         │
         └── 第 4 层：优雅中文错误提示
               └── 抛出 LLMServiceError("大模型服务暂时不可用，请稍后重试")

    Args:
        vector_store: ChromaDB 向量库实例
        query: 用户问题
        chat_history: 历史对话列表
        k: 检索文档数量

    Returns:
        {
            "answer": "LLM 的回答文本 或 纯检索直出结果",
            "sources": ["来源文件名1", "来源文件名2", ...],
            "model": "使用的模型名称 或 direct-retrieval"
        }

    Raises:
        LLMServiceError: 四层全部失败时抛出（第 4 层兜底）
    """
    # ---- ① 检索 (Retrieve) ----
    context_docs = search_similar(vector_store, query, k=k)

    # ---- ② 增强 (Augment) ----
    messages = _build_messages(query, context_docs, chat_history)

    # ================================================================
    # 第 1 层：本地 vLLM 推理服务
    # ================================================================
    try:
        answer = _call_llm(_get_client(), MODEL_NAME, messages)
        logger.info(f"✅ 第 1 层（本地 vLLM）调用成功")

        sources = list(set(
            doc.metadata.get("source", "未知")
            for doc in context_docs
        ))
        return {"answer": answer, "sources": sources, "model": MODEL_NAME}

    except _FALLBACK_EXCEPTIONS as e:
        logger.warning(f"⚠️  第 1 层（本地 vLLM）不可用（网络/超时）: {e}")
    except Exception as e:
        logger.warning(f"⚠️  第 1 层（本地 vLLM）调用异常: {type(e).__name__}: {e}")

    # ================================================================
    # 第 2 层：云端 DeepSeek API 降级
    # ================================================================
    if _FALLBACK_ENABLED:
        logger.info("🔄 正在切换到第 2 层（DeepSeek API）...")
        try:
            answer = _call_llm(_get_deepseek_client(), DEEPSEEK_MODEL, messages)
            logger.info("✅ 第 2 层（DeepSeek API）降级成功")

            sources = list(set(
                doc.metadata.get("source", "未知")
                for doc in context_docs
            ))
            return {"answer": answer, "sources": sources, "model": DEEPSEEK_MODEL}

        except _FALLBACK_EXCEPTIONS as e:
            logger.warning(f"⚠️  第 2 层（DeepSeek API）不可用（网络/超时）: {e}")
        except Exception as e:
            logger.warning(f"⚠️  第 2 层（DeepSeek API）调用异常: {type(e).__name__}: {e}")
    else:
        logger.info("主 BASE_URL 已是 DeepSeek API，跳过第 2 层同源降级")

    # ================================================================
    # 第 3 层：纯向量检索直出模式 (CPU-only, 零显存 / 零 API)
    # ================================================================
    logger.info("🔄 正在切换到第 3 层（纯向量检索直出模式）...")
    try:
        # 使用已检索到的 context_docs，取前 DIRECT_RETRIEVAL_K 条
        top_docs = context_docs[:DIRECT_RETRIEVAL_K]
        return _direct_retrieval_response(top_docs, query)
    except Exception as e:
        logger.error(f"❌ 第 3 层（纯检索直出模式）失败: {type(e).__name__}: {e}")

    # ================================================================
    # 第 4 层：优雅中文错误提示（最终兜底）
    # ================================================================
    logger.critical("❌ 四层容灾全部耗尽！所有 LLM 通道和检索通道均不可用")
    raise LLMServiceError(FRIENDLY_ERROR_MSG)


# ============================================================
# 核心 API：RAG 对话（流式）— 四层金字塔容灾
# ============================================================

def rag_chat_stream(
    vector_store,
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    k: int = RETRIEVAL_K,
) -> Generator[str, None, None]:
    """
    执行一次完整的 RAG 对话（流式，逐 token 返回）。

    【流式输出 vs 非流式输出】

    非流式：LLM 生成完整个回答后一次性返回 → 用户需要等待
    流式（Streaming）：LLM 每生成一个 token 就立即返回 → 打字机效果
      优势：
        - 用户体验更好（不用盯着空白等待）
        - 首字延迟（TTFT）更低
        - 更接近 ChatGPT 等产品的交互体验

    【容灾降级 — 四层金字塔】
    同 rag_chat() 的四层策略。流式场景下：
    - 第 1/2 层：真正的 token 级流式输出（LLM 逐 token 生成）
    - 第 3 层：模拟流式效果（将组装文本分块 yield，前端打字机正常运作）
    - 第 4 层：抛出 LLMServiceError（前端收到结构化错误）

    Args:
        vector_store: ChromaDB 向量库实例
        query: 用户问题
        chat_history: 历史对话列表
        k: 检索文档数量

    Yields:
        文本增量（每个 chunk 是几个 token 的字符串）

    Raises:
        LLMServiceError: 四层全部失败时抛出（第 4 层兜底）
    """
    # ---- ① 检索 ----
    context_docs = search_similar(vector_store, query, k=k)

    # ---- ② 增强 ----
    messages = _build_messages(query, context_docs, chat_history)

    # ================================================================
    # 第 1 层：本地 vLLM 推理服务（流式）
    # ================================================================
    try:
        yield from _stream_llm(_get_client(), MODEL_NAME, messages)
        logger.info(f"✅ 第 1 层（本地 vLLM 流式）调用成功")
        return  # ← 成功，生成器结束

    except _FALLBACK_EXCEPTIONS as e:
        logger.warning(f"⚠️  第 1 层（本地 vLLM 流式）不可用（网络/超时）: {e}")
    except Exception as e:
        logger.warning(f"⚠️  第 1 层（本地 vLLM 流式）调用异常: {type(e).__name__}: {e}")

    # ================================================================
    # 第 2 层：云端 DeepSeek API 降级（流式）
    # ================================================================
    if _FALLBACK_ENABLED:
        logger.info("🔄 正在切换到第 2 层（DeepSeek API 流式）...")
        try:
            yield from _stream_llm(_get_deepseek_client(), DEEPSEEK_MODEL, messages)
            logger.info("✅ 第 2 层（DeepSeek API 流式）降级成功")
            return  # ← 成功，生成器结束

        except _FALLBACK_EXCEPTIONS as e:
            logger.warning(f"⚠️  第 2 层（DeepSeek API 流式）不可用（网络/超时）: {e}")
        except Exception as e:
            logger.warning(f"⚠️  第 2 层（DeepSeek API 流式）调用异常: {type(e).__name__}: {e}")
    else:
        logger.info("主 BASE_URL 已是 DeepSeek API，跳过第 2 层同源降级")

    # ================================================================
    # 第 3 层：纯向量检索直出模式（模拟流式）
    # ================================================================
    logger.info("🔄 正在切换到第 3 层（纯向量检索直出模式-流式）...")
    try:
        top_docs = context_docs[:DIRECT_RETRIEVAL_K]
        yield from _direct_retrieval_response_stream(top_docs, query)
        logger.info("✅ 第 3 层（纯检索直出模式-流式）成功")
        return  # ← 成功，生成器结束

    except Exception as e:
        logger.error(f"❌ 第 3 层（纯检索直出模式-流式）失败: {type(e).__name__}: {e}")

    # ================================================================
    # 第 4 层：优雅中文错误提示（最终兜底）
    # ================================================================
    logger.critical("❌ 四层容灾全部耗尽！所有 LLM 通道和检索通道均不可用")
    raise LLMServiceError(FRIENDLY_ERROR_MSG)


# ============================================================
# 命令行测试入口
# ============================================================
if __name__ == "__main__":
    import sys
    from .vector_store import load_vector_store

    print("=== RAG 对话测试 ===\n")

    vs = load_vector_store()
    if vs is None:
        print("❌ 向量库为空，请先上传 PDF 并创建知识库。")
        sys.exit(1)

    while True:
        try:
            query = input("\n🧑 你: ")
            if query.lower() in ("exit", "quit", "q"):
                break
            if not query.strip():
                continue

            print("🤖 AI: ", end="", flush=True)
            for token in rag_chat_stream(vs, query):
                print(token, end="", flush=True)
            print()

        except KeyboardInterrupt:
            break
        except LLMServiceError as e:
            print(f"\n⚠️  {e}")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
