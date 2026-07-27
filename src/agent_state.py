"""
=============================================================================
RAG Agent State — LangGraph 状态图状态定义（v3 — Plan-Execute-Synthesize）
=============================================================================

定义了 RAG 管线各节点之间流转的 TypedDict 状态结构。

每个图节点读取 state，返回部分更新（dict），LangGraph 自动合并。

【v3 新增字段 — Plan-Execute-Synthesize 架构】（ADR-14, 2026-07-25）
  - sub_goals:              SubGoalPlanner 拆分的子目标列表
  - sub_results:            各子目标并行执行的结果
  - cross_product_candidates: 跨产品检索候选
  - attribute_intent:       动态属性意图提取结果（替代静态 KV 表）
  - code_entities:          从 query 中提取的代码实体名列表
  - plan_mode:              执行模式 "single"|"multi"|"cross_product"|"attribute"
  - skip_planner:           快速路径标志 — 明确 product_id 时绕过 Planner

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

      ── v3 Plan-Execute-Synthesize 字段 ──
      sub_goals:          子目标列表，每项为 dict:
                            {"type":"product_qa"|"attribute_lookup"|"code_search"|"cross_product",
                             "product_id":"JAKA"|None, "query":"...", "priority":1-3}
      sub_results:        各子目标执行结果列表，每项为 dict:
                            {"goal_index":0, "type":"...", "answer":"...", "sources":[...], "model":"..."}
      cross_product_candidates: 跨产品检索结果，每项为 dict:
                            {"product_id":"JAKA", "snippet":"...", "relevance":0.85}
      attribute_intent:   动态属性意图 dict:
                            {"query_keyword":"波特率", "extracted_value":"9600",
                             "normalized_key":"波特率", "bm25_hits":2, "resolved":True}
      code_entities:      从 query 提取的代码实体名列表 ["robot_movl","set_robot_power_on"]
      plan_mode:          当前执行模式: "single" | "multi" | "cross_product" | "attribute"
      skip_planner:       True = 快速路径，绕过 SubGoalPlanner

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

    # ── v3 Plan-Execute-Synthesize 字段 ──
    sub_goals: List[Dict[str, Any]]
    sub_results: List[Dict[str, Any]]
    cross_product_candidates: List[Dict[str, Any]]
    attribute_intent: Dict[str, Any]
    code_entities: List[str]
    plan_mode: str
    skip_planner: bool

    # ── v2 后处理控制字段 ──
    extracted_entities: Dict[str, str]
    feedback: str
    retry_count: int
    context_text: str
    raw_llm_answer: str
    # ── v2.1 Agentic RAG 控制字段 ──
    retrieval_retry: int
    doc_sufficient: bool
