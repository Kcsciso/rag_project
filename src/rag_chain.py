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

=============================================================================
"""

import logging
from typing import List, Dict, Optional, Generator

from openai import OpenAI

from .config import BASE_URL, API_KEY, MODEL_NAME, RETRIEVAL_K
from .vector_store import search_similar

logger = logging.getLogger(__name__)

# ============================================================
# LLM 客户端（OpenAI 兼容接口）
# ============================================================

# 单例模式：整个应用共用一个客户端实例
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """
    获取 OpenAI 兼容客户端实例（懒加载）。

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
        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        logger.info(f"LLM 客户端已初始化: base_url={BASE_URL}, model={MODEL_NAME}")
    return _client


# ============================================================
# Prompt 模板 — RAG 的核心"咒语"
# ============================================================

RAG_SYSTEM_PROMPT = """你是一个专业的知识助手。你的任务是基于提供的参考资料，准确、简洁地回答用户的问题。

请严格遵守以下规则：
1. 回答必须基于【参考资料】中的内容，不得编造信息
2. 如果参考资料不足以回答问题，请明确告知用户"参考资料中未包含相关信息"
3. 如果用户的问题与参考资料无关，你可以用自己的知识进行回答，但应注明
4. 回答应条理清晰，尽量使用简洁的语言
5. 可以适当引用参考资料中的原文（使用引号标注）
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

    【为什么要用 Chat 格式而非纯文本？】
    LLM 的 Chat API 区分三种角色：
    - system: 设定 AI 的行为准则（"你是一个...请遵守..."）
    - user: 用户的输入
    - assistant: AI 之前的回答
    这种结构化格式让模型能清晰区分"指令"和"对话内容"，
    比把所有内容塞进一段纯文本效果好得多。

    Args:
        query: 当前用户问题
        context_docs: 检索到的相关文档片段列表
        chat_history: 历史对话 [{"role": "...", "content": "..."}, ...]

    Returns:
        messages 列表，可直接传给 OpenAI API
    """

    # ---- 拼接参考资料 ----
    # 将检索到的文档片段编号后拼接，方便 LLM 引用
    context_parts = []
    for i, doc in enumerate(context_docs, start=1):
        source = doc.metadata.get("source", "未知来源")
        content = doc.page_content.strip()
        context_parts.append(f"[参考资料 {i}]（来源：{source}）\n{content}")

    context_text = "\n\n---\n\n".join(context_parts)

    # ---- 构建当前轮次的用户消息 ----
    # 把参考资料和用户问题放在同一条 user 消息中
    # 这样 LLM 能在一屏内看到所有相关信息
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

    # 如果有历史对话，插入历史
    if chat_history:
        messages.extend(chat_history)

    # 最后追加当前轮次的增强 Prompt
    messages.append({"role": "user", "content": current_user_message})

    return messages


# ============================================================
# 核心 API：RAG 对话（非流式）
# ============================================================

def rag_chat(
    vector_store,
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    k: int = RETRIEVAL_K,
) -> Dict[str, any]:
    """
    执行一次完整的 RAG 对话（非流式，一次性返回完整结果）。

    【完整调用链】
    query → [向量检索] → context_docs → [构建 Prompt] → messages → [LLM 推理] → answer

    Args:
        vector_store: ChromaDB 向量库实例
        query: 用户问题
        chat_history: 历史对话列表
        k: 检索文档数量

    Returns:
        {
            "answer": "LLM 的回答文本",
            "sources": ["来源文件名1", "来源文件名2", ...],
            "model": "使用的模型名称"
        }
    """
    # ---- ① 检索 (Retrieve) ----
    # 从向量库中搜索与问题最相关的 k 个文档片段
    context_docs = search_similar(vector_store, query, k=k)

    # ---- ② 增强 (Augment) ----
    # 将检索结果和用户问题组装为结构化的 Prompt
    messages = _build_messages(query, context_docs, chat_history)

    # ---- ③ 生成 (Generate) ----
    # 调用 LLM 进行推理
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3,        # 低温度 → 输出更确定性，减少幻觉
            max_tokens=2048,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        raise

    # ---- ④ 返回 (Response) ----
    # 提取来源文件名（去重）
    sources = list(set(
        doc.metadata.get("source", "未知")
        for doc in context_docs
    ))

    return {
        "answer": answer,
        "sources": sources,
        "model": MODEL_NAME,
    }


# ============================================================
# 核心 API：RAG 对话（流式）
# ============================================================

def rag_chat_stream(
    vector_store,
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    k: int = RETRIEVAL_K,
) -> Generator[str, None, Dict[str, any]]:
    """
    执行一次完整的 RAG 对话（流式，逐 token 返回）。

    【流式输出 vs 非流式输出】

    非流式：LLM 生成完整个回答后一次性返回 → 用户需要等待
    流式（Streaming）：LLM 每生成一个 token 就立即返回 → 打字机效果
      优势：
        - 用户体验更好（不用盯着空白等待）
        - 首字延迟（TTFT）更低
        - 更接近 ChatGPT 等产品的交互体验

    技术实现：
      设置 stream=True，API 返回一个迭代器，
      每次迭代产出一个小增量（通常是几个 token），
      我们通过 yield 将其逐段传给上层。

    Args:
        vector_store: ChromaDB 向量库实例
        query: 用户问题
        chat_history: 历史对话列表
        k: 检索文档数量

    Yields:
        文本增量（每个 chunk 是几个 token 的字符串）

    Returns:
        （生成器结束后）返回 {"sources": [...], "model": "..."}
        通过 generator.return_value 或特殊约定传递
    """
    # ---- ① 检索 ----
    context_docs = search_similar(vector_store, query, k=k)

    # ---- ② 增强 ----
    messages = _build_messages(query, context_docs, chat_history)

    # ---- ③ 流式生成 ----
    client = _get_client()
    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
            stream=True,  # ← 关键：开启流式模式
        )

        # 逐 chunk 产出文本
        for chunk in stream:
            # delta.content 是本次增量文本（可能为 None 或空字符串）
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    except Exception as e:
        logger.error(f"LLM 流式调用失败: {e}")
        raise

    # ---- ④ 后处理 ----
    # 流式输出结束后，通过附加属性传递元信息
    # 注意：Python 生成器无法在 yield 后 return 值给调用方，
    # 所以来源信息需要另一个渠道传递给上层。
    # 实际使用中通过 app.py 的 SSE 事件来传递。


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
        except Exception as e:
            print(f"\n❌ 错误: {e}")
