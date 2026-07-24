"""
=============================================================================
RAG Agent State — LangGraph 状态图状态定义（v2 — Post-Generation Control）
=============================================================================

定义了 RAG 管线各节点之间流转的 TypedDict 状态结构。

每个图节点读取 state，返回部分更新（dict），LangGraph 自动合并。

【v2 新增字段 — 后处理控制层】（ADR-11, 2026-07-24）
  - extracted_entities: 从 Context 中提取的通用 KV 属性映射
  - feedback:           SDK 代码自纠错反馈信息
  - retry_count:        自纠错重试计数器（上限 2 次）
  - context_text:       原始 Context 拼接文本（供后处理节点对齐用）
  - raw_llm_answer:     未修改的 LLM 原始输出（供 extract_align 节点对比）

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
      final_answer:       LLMGenerationNode 生成的最终回答文本（经后处理修正）
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

      ── v2 后处理控制字段 ──
      extracted_entities: 从 Context 中扫描提取的通用 KV 属性映射，
                          格式 {"端口": "6502", "波特率": "9600"}。
                          由 hybrid_retrieval_node 填充，extract_align_node 消费。
      feedback:           SDK 代码自纠错反馈文本。
                          若 SDK 代码缺失关键前缀/调用链，写入具体提示；
                          为空字符串表示代码无需修正。
                          由 sdk_verify_node 填充，llm_generation_node 消费。
      retry_count:        自纠错重试计数器（整数 0-2）。
                          每次 SDK 校验失败 +1，上限 2 次防止死循环。
      context_text:       原始 Context 拼接文本（所有检索切片拼接为纯文本）。
                          供 extract_align_node 与 raw_llm_answer 做逐数字对齐。
      raw_llm_answer:     未经后处理的 LLM 原始输出文本。
                          extract_align_node 对此文本做属性词硬改写后，
                          将修正结果写入 final_answer。
    """
    # ── 现有基础字段 ──
    query: str
    fused_query: str
    product_id: Optional[str]
    chat_history: Optional[List[Dict[str, str]]]
    retrieved_docs: List[Any]
    final_answer: str
    sources: List[str]
    model: str
    route_status: str

    # ── v2 后处理控制字段 ──
    extracted_entities: Dict[str, str]
    feedback: str
    retry_count: int
    context_text: str
    raw_llm_answer: str
