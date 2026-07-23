"""
=============================================================================
RAG Agent State — LangGraph 状态图状态定义
=============================================================================

定义了 RAG 管线各节点之间流转的 TypedDict 状态结构。

每个图节点读取 state，返回部分更新（dict），LangGraph 自动合并。

=============================================================================
"""
from typing import List, Dict, Optional, Any, TypedDict


class RAGState(TypedDict, total=False):
    """
    RAG 管线完整状态。

    【字段说明】
      query:              当前用户原始提问（经 sanitate 清洗后的字符串）
      fused_query:        QueryFusionNode 融合历史后的完整检索词
      product_id:         产品标识（"OpenR6" / "OpenC3" / "JAKA" / None）
      chat_history:       多轮对话历史 [{"role":"user","content":...}, ...]
      retrieved_docs:     HybridRetrievalNode 召回并排序后的 LangChain Document 列表
      final_answer:       LLMGenerationNode 生成的最终回答文本
      sources:            回答中引用的文档来源列表
      model:              实际使用的 LLM 模型标识
      route_status:       路由状态标记：
                            - "clarify"    → 需要反问产品，直接输出澄清文本
                            - "chitchat"   → 闲聊/身份询问，直接回复
                            - "refuse"     → 不可能组合，硬拒答
                            - "generate"   → 正常检索+生成
                            - "fallback"   → 检索为空但仍尝试 LLM
                            - "complete"   → 生成完成
                            - "error"      → 发生错误
    """
    query: str
    fused_query: str
    product_id: Optional[str]
    chat_history: Optional[List[Dict[str, str]]]
    retrieved_docs: List[Any]
    final_answer: str
    sources: List[str]
    model: str
    route_status: str
