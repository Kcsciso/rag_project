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
import re
import threading
from typing import List, Dict, Optional, Generator, Any, Tuple

from langgraph.graph import StateGraph, END

from .agent_state import RAGState
from . import config as _cfg

logger = logging.getLogger(__name__)

# ============================================================
# 通用属性词库 — 用于 Context KV 实体提取（零特定数字硬编码）
# ============================================================

# 物理属性词 / 参数词列表（按类别组织，便于维护和扩展）
_GENERIC_PHYSICAL_ATTRS: List[str] = [
    # ── 网络 / 通信 ──
    "端口", "端口号", "port", "IP", "ip", "IP地址", "地址", "从站地址", "主站地址",
    "MAC", "mac", "子网掩码", "网关", "DNS", "域名",
    # ── 串口 / Modbus ──
    "波特率", "baud", "数据位", "停止位", "校验位", "校验方式",
    "奇偶校验", "从站ID", "站号", "设备地址", "寄存器地址",
    # ── 电气参数 ──
    "电压", "电流", "功率", "电阻", "频率", "输入电压", "输出电压",
    "额定电压", "额定电流", "额定功率", "功耗",
    # ── 设备标识 ──
    "设备标识", "设备ID", "序列号", "型号", "版本号", "固件版本",
    # ── 时序参数 ──
    "超时", "间隔", "周期", "采样率", "刷新率",
    # ── 机械参数 ──
    "力矩", "扭矩", "速度", "加速度", "减速度", "角度", "位置",
    "关节角度", "末端速度", "最大速度",
    # ── 密码 / 凭据 ──
    "密码", "口令", "默认密码", "用户名", "登录密码",
]

# 编译正则：匹配 "属性词: 数值" / "属性词=数值" / "属性词 数值" 等通用 KV 模式
# 属性词从词库动态拼接，数值匹配 ≥2 位数字（可带小数、单位后缀如 V/A/Ω/ms）
# 连接词支持：冒号、等号、中文"为/是"、空格
_ATTR_VALUE_RE = re.compile(
    r'(?P<attr>' + '|'.join(re.escape(a) for a in _GENERIC_PHYSICAL_ATTRS) + r')'
    r'\s*'
    r'(?:[：:=\s]|[为是]\s*)?'     # 连接词：冒号/等号/空格/中文"为/是"
    r'\s*'
    r'(?P<value>\d{1,6}(?:\.\d+)?(?:\s*[A-Za-zμΩΩ%℃VAmswHz]+)?)',
    re.IGNORECASE,
)

# IP 地址专用正则（处理 "192.168.11.214" 这类点分四段格式）
_IP_VALUE_RE = re.compile(
    r'(?P<attr>(?:IP|ip)\s*地址?|IP地址|ip地址)'
    r'\s*'
    r'(?:[：:=\s]|[为是]\s*)?'
    r'\s*'
    r'(?P<value>\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})',
    re.IGNORECASE,
)

# SDK 相关查询特征词（触发 SDK 代码自纠错的条件）
_SDK_QUERY_PATTERNS = re.compile(
    r'(?:SDK|sdk|函数|代码|示例|example|code|接口|API|api|调用|怎么写|如何写)',
    re.IGNORECASE,
)

# SDK 代码常见缺失项检测规则
# 每条规则: (pattern, feedback_message)
_SDK_MISSING_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # OpenR6: 缺少 set_ 前缀（使用负向后顾排除已有 set_ 的正确写法）
    (re.compile(r'(?<!set_)(?:robot_arm_init|robot_power|robot_motor|robot_socket|robot_move)'
                r'(?!\s*=\s*set_)'),
     "OpenR6 SDK 函数调用缺少 'set_' 前缀，正确写法例如 set_robot_arm_init()"),
    # 缺少 CDLL 加载
    (re.compile(r'(?:set_robot_|get_robot_|robot_)'),
     "SDK 代码中使用了 robot_ 函数但未找到 ctypes.CDLL 动态库加载语句"),
    # 缺少 argtypes 声明
    (re.compile(r'(?:set_robot_|robot_Power_on|robot_enable)\s*\('),
     "SDK 函数调用前缺少 argtypes 参数类型声明"),
]


def _extract_generic_kv_entities(context_text: str) -> Dict[str, str]:
    """
    从检索 Context 中扫描提取通用物理属性 KV 映射。

    使用通用属性词库 + 正则扫描，自动识别如：
      - "端口号为 6502"  → {"端口号": "6502"}
      - "波特率 9600"    → {"波特率": "9600"}
      - "IP 地址 192.168.11.214" → {"IP地址": "192.168.11.214"}
      - "输入电压 24V"   → {"输入电压": "24V"}

    【设计原则】
    - 零特定数字硬编码：不写死 6502、9600 等具体数值
    - 属性词库可维护：新增领域词只需在 _GENERIC_PHYSICAL_ATTRS 中追加
    - 同名去重：同一属性词出现多次时，保留第一次匹配的值
    - IP 地址专用匹配：独立正则以支持点分四段格式

    Args:
        context_text: 所有检索切片的纯文本拼接

    Returns:
        {"属性词": "数值", ...} 字典
    """
    entities: Dict[str, str] = {}
    seen_attrs: set = set()

    # 第 1 步：匹配 IP 地址（点分四段格式 — 独立正则）
    for match in _IP_VALUE_RE.finditer(context_text):
        attr = match.group("attr").strip()
        value = match.group("value").strip()
        attr_lower = attr.lower()
        if attr_lower not in seen_attrs:
            entities[attr] = value
            seen_attrs.add(attr_lower)
            logger.debug(f"  🔑 KV提取(IP): '{attr}' → '{value}'")

    # 第 2 步：匹配通用属性-数值 KV
    for match in _ATTR_VALUE_RE.finditer(context_text):
        attr = match.group("attr").strip()
        value = match.group("value").strip()
        attr_lower = attr.lower()
        # 去重：同一属性词保留首次匹配
        if attr_lower not in seen_attrs:
            # 二次校验：排除 Context 中的错误关联（如"端口"出现在"波特率"后面等）
            entities[attr] = value
            seen_attrs.add(attr_lower)
            logger.debug(f"  🔑 KV提取: '{attr}' → '{value}'")

    return entities


def _is_sdk_query(query: str) -> bool:
    """检测用户查询是否涉及 SDK 函数/代码编写。"""
    return bool(_SDK_QUERY_PATTERNS.search(query))


def _check_sdk_code_issues(code_text: str) -> List[str]:
    """
    扫描生成的代码文本，检测常见的 SDK 代码缺失项。

    每条规则独立检测，支持 CDLL/argtypes 的"存在则豁免"逻辑：
      - set_ 前缀缺失：检测不带 set_ 前缀的 robot_* 函数调用
      - CDLL 缺失：检查代码中是否包含 ctypes.CDLL 加载语句
      - argtypes 缺失：检查代码中是否包含 .argtypes 声明

    Returns:
        发现的问题描述列表（空列表 = 代码无问题）
    """
    issues: List[str] = []
    for pattern, feedback in _SDK_MISSING_PATTERNS:
        if pattern.search(code_text):
            # ── 智能豁免：CDLL 已存在时不报 CDLL 缺失 ──
            if 'CDLL' in feedback and 'ctypes.CDLL' in code_text:
                continue
            # ── 智能豁免：argtypes 已存在时不报 argtypes 缺失 ──
            if 'argtypes' in feedback and '.argtypes' in code_text:
                continue
            issues.append(feedback)
    return issues

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
    此节点保留原始 route_status（如 "clarify"）以便 run_graph() 正确返回
    needs_clarification 标志。
    """
    current_status = state.get("route_status", "")
    logger.info(f"🟠 [Node 2b] BuildDirectResponse: route_status='{current_status}'")
    # 保留原始路由状态（clarify/chitchat/refuse），不覆盖为 complete
    return {}


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

    # ── v2: 提取 Context KV 实体 + 拼接原始文本 ──
    context_text = ""
    kv_entities: Dict[str, str] = {}
    if context_docs:
        context_parts = []
        for doc in context_docs:
            context_parts.append(doc.page_content)
        context_text = "\n".join(context_parts)
        kv_entities = _extract_generic_kv_entities(context_text)
        if kv_entities:
            logger.info(f"  ↳ 提取 {len(kv_entities)} 个 KV 实体: {list(kv_entities.keys())[:6]}")

    return {
        "retrieved_docs": context_docs,
        "route_status": route_status,
        "context_text": context_text,
        "extracted_entities": kv_entities,
    }


# ============================================================
# Node 4: LLMGenerationNode — 四层容灾 + 章节溯源约束
# ============================================================

def llm_generation_node(state: RAGState) -> dict:
    """
    使用检索到的文档片段，调用四层金字塔容灾 LLM 生成最终回答。

    容灾链路：本地 vLLM → 云端智谱 API → 纯检索直出 → 硬拒答兜底

    【v2 增强】
    - 若 state["feedback"] 非空（SDK 自纠错重试），将反馈作为额外 user 消息追加
    - 成功生成后始终保存 raw_llm_answer 供 post-processing 节点使用
    - 重试成功时清除 feedback 并重置 retry_count
    """
    query = state.get("fused_query") or state.get("query", "")
    context_docs = state.get("retrieved_docs", [])
    chat_history = state.get("chat_history")
    feedback = state.get("feedback", "")
    retry_count = state.get("retry_count", 0)

    if feedback:
        logger.info(f"🟣 [Node 4] LLMGeneration (retry #{retry_count}): "
                     f"feedback='{feedback[:80]}'")

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
                "raw_llm_answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "model": result.get("model", "direct-retrieval-fallback"),
                "route_status": "complete",
                "feedback": "",
                "retry_count": 0,
            }
        except Exception:
            return {
                "final_answer": _HARD_REFUSAL,
                "raw_llm_answer": _HARD_REFUSAL,
                "sources": [],
                "model": "fatal-fallback",
                "route_status": "complete",
                "feedback": "",
                "retry_count": 0,
            }

    # ── v2: SDK 自纠错反馈注入 ──
    if feedback:
        correction_msg = (
            f"【自纠错提示 — 上一轮代码问题】\n{feedback}\n\n"
            f"请修正上述问题，重新生成完整、正确的回答（包含修正后的代码）。"
        )
        messages.append({"role": "user", "content": correction_msg})
        logger.info(f"  ↳ 已注入自纠错反馈到消息列表")

    # 🔴 数字请求无上下文硬防护
    if _last_numeric_context_missing:
        logger.info("🚫 [Graph] 数字请求无上下文 → 直接返回硬拒答")
        return {
            "final_answer": _HARD_REFUSAL,
            "raw_llm_answer": _HARD_REFUSAL,
            "sources": [],
            "model": "numeric-guard",
            "route_status": "complete",
            "feedback": "",
            "retry_count": 0,
        }

    # ── 生成结果容器 ──
    generated_answer: Optional[str] = None
    used_model: str = ""

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
                    generated_answer = answer
                    used_model = model
        except _FALLBACK_EXCEPTIONS as e:
            logger.warning(f"⚠️  Layer 1 不可用: {e}")
        except Exception as e:
            logger.warning(f"⚠️  Layer 1 异常: {type(e).__name__}: {e}")
        finally:
            if lock_acquired:
                _release_vllm_lock()

    # ── Layer 2: 云端智谱 API ──
    if generated_answer is None and _FALLBACK_ENABLED:
        logger.info("🔄 降级 Layer 2 (智谱 API)...")
        try:
            from .config import DEEPSEEK_MODEL
            answer = _call_llm(_get_deepseek_client(), DEEPSEEK_MODEL, messages)
            if answer and answer.strip():
                logger.info("✅ Layer 2 (智谱 API) 成功")
                generated_answer = answer
                used_model = DEEPSEEK_MODEL
        except _FALLBACK_EXCEPTIONS as e:
            logger.warning(f"⚠️  Layer 2 不可用: {e}")
        except Exception as e:
            logger.warning(f"⚠️  Layer 2 异常: {type(e).__name__}: {e}")

    # ── Layer 3: 纯检索直出 ──
    if generated_answer is None:
        logger.info("🔄 降级 Layer 3 (纯检索直出)...")
        try:
            result = _direct_retrieval_response(context_docs, query)
            answer = result.get("answer", "")
            if answer.strip():
                generated_answer = answer
                used_model = result.get("model", "direct-retrieval")
        except Exception as e:
            logger.error(f"❌ Layer 3 失败: {e}")

    # ── Layer 4: 硬拒答兜底 ──
    if generated_answer is None or not generated_answer.strip():
        logger.critical("❌ 四层容灾全部耗尽")
        return {
            "final_answer": _HARD_REFUSAL,
            "raw_llm_answer": _HARD_REFUSAL,
            "sources": [],
            "model": "never-empty-guarantee",
            "route_status": "complete",
            "feedback": "",
            "retry_count": 0,
        }

    # ── 成功：组装结果（保留 raw_llm_answer 供后续 extract_align 使用）──
    sources = list(set(
        doc.metadata.get("source", "未知")
        for doc in context_docs
    )) if context_docs else []

    return {
        "final_answer": generated_answer,
        "raw_llm_answer": generated_answer,
        "sources": sources,
        "model": used_model,
        "route_status": "complete",
        "feedback": "",       # 成功后清除反馈
        "retry_count": 0,     # 成功后重置计数器
    }


# ============================================================
# Node 5: SDK_VerifyNode — SDK 代码自纠错校验（v2）
# ============================================================

def sdk_verify_node(state: RAGState) -> dict:
    """
    SDK 代码自纠错校验节点。

    检测规则（按优先级）：
      1. 用户 query 是否涉及 SDK/函数/代码？ → 否 → 跳过，feedback=""
      2. 模型输出中是否包含代码块？ → 否 → 跳过
      3. 代码是否缺失 set_ 前缀？ → 是 → 写入 feedback
      4. 代码是否缺失 ctypes.CDLL？ → 是 → 写入 feedback
      5. 代码是否缺失 argtypes 声明？ → 是 → 写入 feedback

    若发现问题 → feedback 非空 → 递增 retry_count → 条件边路由回 llm_generation
    若无问题   → feedback="" → 条件边路由到 extract_align
    """
    query = state.get("fused_query") or state.get("query", "")
    raw_answer = state.get("raw_llm_answer") or state.get("final_answer", "")
    retry_count = state.get("retry_count", 0)

    logger.info(f"🔴 [Node 5] SDK_Verify: retry_count={retry_count}")

    # ── 规则 1: 非 SDK 查询 → 跳过 ──
    if not _is_sdk_query(query):
        logger.info("  ↳ 非 SDK 查询，跳过 SDK 校验")
        return {"feedback": "", "retry_count": retry_count}

    # ── 规则 2: 无代码块 → 跳过 ──
    if "```" not in raw_answer and "robot_" not in raw_answer.lower():
        logger.info("  ↳ 回答中无代码内容，跳过 SDK 校验")
        return {"feedback": "", "retry_count": retry_count}

    # ── 规则 3-5: 扫描代码问题 ──
    issues = _check_sdk_code_issues(raw_answer)

    if issues:
        feedback = "；".join(issues)
        new_retry_count = retry_count + 1
        logger.warning(
            f"  ↳ SDK 代码问题 ({len(issues)} 项): {feedback[:120]}"
        )
        logger.info(f"  ↳ retry_count: {retry_count} → {new_retry_count}")
        return {
            "feedback": feedback,
            "retry_count": new_retry_count,
        }

    logger.info("  ↳ SDK 代码校验通过 ✓")
    return {"feedback": "", "retry_count": retry_count}


# ============================================================
# Node 6: ExtractAlignNode — 通用属性对齐校验（v2）
# ============================================================

def extract_align_node(state: RAGState) -> dict:
    """
    通用属性对齐校验节点 — 后处理阶段的最后防线。

    【设计原则】
    - 零特定数字补丁：不硬编码 6502、9600 等具体数值
    - 通用属性词库 + 正则扫描：自动识别 Context 中的 KV 映射
    - 硬改写对齐：若模型将属性词颠倒/篡改，用 Context 原文强制覆盖

    【算法】
    1. 从 state["extracted_entities"] 读取 Context 中的真实 KV 映射
    2. 对 state["raw_llm_answer"] 中的每个数值，扫描其紧邻的属性词
    3. 若属性词与 Context 中的属性词不匹配 → 用 Context 原词硬改写
    4. 输出修正后的 final_answer

    【示例】
    Context:  "端口号 6502"
    LLM 输出: "Modbus 从站地址为 6502"  ← "从站地址" ≠ "端口号"
    → 硬改写为: "Modbus 端口号为 6502"

    Args:
        state: 当前图状态（含 extracted_entities, raw_llm_answer）

    Returns:
        {"final_answer": 修正后的回答, "route_status": "complete"}
    """
    raw_answer = state.get("raw_llm_answer") or state.get("final_answer", "")
    kv_entities = state.get("extracted_entities", {})

    logger.info(f"🟢 [Node 6] ExtractAlign: {len(kv_entities)} KV entities, "
                 f"answer_len={len(raw_answer)}")

    # 若没有可对齐的实体，直接透传
    if not kv_entities or not raw_answer:
        logger.info("  ↳ 无 KV 实体或回答为空，透传原始回答")
        return {
            "final_answer": raw_answer,
            "route_status": "complete",
        }

    corrected = raw_answer

    # ── 对每个 Context 中的 KV 实体，检查模型输出是否正确 ──
    fixes_applied = 0
    for context_attr, context_value in kv_entities.items():
        # 仅处理 ≥2 位数值（过滤单数字噪声）
        value_digits = re.sub(r'[^\d]', '', context_value)
        if len(value_digits) < 2:
            continue

        # 在模型输出中查找该数值
        value_pattern = re.compile(
            r'(?P<prefix>.{0,12})'           # 数值前最多 12 字符（属性词窗口）
            + re.escape(context_value) +      # 数值本身
            r'(?P<suffix>.{0,8})',            # 数值后最多 8 字符
        )

        for vm in value_pattern.finditer(corrected):
            prefix = vm.group("prefix")
            suffix = vm.group("suffix")
            nearby_text = prefix + context_value + suffix

            # 检查是否存在与 Context 属性词冲突的"错误属性词"
            for candidate_attr, candidate_val in kv_entities.items():
                if candidate_val == context_value:
                    continue  # 跳过自己
                # 若错误属性词出现在数值附近，且正确属性词不在附近 → 需要修正
                attr_lower = candidate_attr.lower().strip()
                ctx_attr_lower = context_attr.lower().strip()
                if (attr_lower in nearby_text.lower()
                        and ctx_attr_lower not in nearby_text.lower()):
                    # ── 硬改写：用 Context 中的正确属性词替换错误属性词 ──
                    escaped_bad = re.escape(candidate_attr)
                    new_nearby = re.sub(
                        escaped_bad, context_attr,
                        nearby_text, count=1, flags=re.IGNORECASE,
                    )
                    if new_nearby != nearby_text:
                        corrected = corrected.replace(nearby_text, new_nearby, 1)
                        fixes_applied += 1
                        logger.info(
                            f"  🔧 属性对齐: '{candidate_attr}' → '{context_attr}' "
                            f"(数值 {context_value})"
                        )

    if fixes_applied > 0:
        logger.info(f"  ↳ 共修正 {fixes_applied} 处属性词颠倒/篡改")

    return {
        "final_answer": corrected,
        "route_status": "complete",
    }


# ============================================================
# 条件路由函数（v2 扩展：后处理路由）
# ============================================================

def _route_after_llm(state: RAGState) -> str:
    """
    LLM 生成后的条件路由。

    Return:
      - "sdk_verify"     → 回答中含代码且查询涉及 SDK → SDK 校验
      - "extract_align"  → 直接进入属性对齐（默认路径）
    """
    query = state.get("fused_query") or state.get("query", "")
    raw_answer = state.get("raw_llm_answer") or state.get("final_answer", "")

    # 若回答为空/拒答 → 直接对齐（跳过 SDK 校验）
    if not raw_answer or raw_answer == _HARD_REFUSAL:
        return "extract_align"

    # 若查询涉及 SDK 且回答含代码 → SDK 校验
    if _is_sdk_query(query) and ("```" in raw_answer or "robot_" in raw_answer.lower()):
        logger.info("  ↳ 路由: SDK 查询 + 代码输出 → sdk_verify")
        return "sdk_verify"

    return "extract_align"


def _route_after_sdk_verify(state: RAGState) -> str:
    """
    SDK 校验后的条件路由。

    Return:
      - "llm_generation"  → 代码有问题且重试次数未达上限 → 回环重试
      - "extract_align"   → 代码无问题或重试次数耗尽 → 进入属性对齐
    """
    feedback = state.get("feedback", "")
    retry_count = state.get("retry_count", 0)
    max_retries = 2

    if feedback and retry_count <= max_retries:
        logger.info(
            f"  ↳ 路由: SDK 问题未修复 → 回环重试 (retry #{retry_count})"
        )
        return "llm_generation"

    if feedback and retry_count > max_retries:
        logger.warning(
            f"  ↳ 路由: SDK 重试次数耗尽 ({retry_count} > {max_retries}) → 放弃修复"
        )

    return "extract_align"


# ============================================================
# Graph 构建与编译（v2 — 后处理节点 + 自纠错环路）
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
    构建并编译 LangGraph StateGraph（v2 — 含后处理节点与自纠错环路）。

    图结构:

      START
        │
        ▼
      query_fusion ──→ product_routing
                          │
              ┌───────────┼───────────┐
              │           │           │
          clarify/    chitchat/    generate/
          refuse      (→ END)     fallback
              │                       │
              ▼                       ▼
      build_direct_response    hybrid_retrieval
              │                       │
              ▼                       ▼
             END               llm_generation
                                   │
                    ┌──────────────┼──────────────┐
                    │                             │
               sdk_verify                   extract_align
                    │                             │
            ┌───────┼───────┐                     ▼
            │               │                    END
      llm_generation   extract_align
      (retry loop)        │
                          ▼
                         END

    返回编译后的图实例（带状态校验）。
    """
    graph = StateGraph(RAGState)

    # ── 注册节点 ──
    graph.add_node("query_fusion", query_fusion_node)
    graph.add_node("product_routing", product_routing_node)
    graph.add_node("build_direct_response", build_direct_response_node)
    graph.add_node("hybrid_retrieval", hybrid_retrieval_node)
    graph.add_node("llm_generation", llm_generation_node)
    # ── v2 新增节点 ──
    graph.add_node("sdk_verify", sdk_verify_node)
    graph.add_node("extract_align", extract_align_node)

    # ── 注册边（前置管线不变）──
    graph.set_entry_point("query_fusion")
    graph.add_edge("query_fusion", "product_routing")

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

    # ── v2 新增条件边：LLM 生成后 → SDK 校验 或 属性对齐 ──
    graph.add_conditional_edges(
        "llm_generation",
        _route_after_llm,
        {
            "sdk_verify": "sdk_verify",
            "extract_align": "extract_align",
        },
    )

    # ── v2 新增条件边：SDK 校验后 → 回环重试 或 属性对齐 ──
    graph.add_conditional_edges(
        "sdk_verify",
        _route_after_sdk_verify,
        {
            "llm_generation": "llm_generation",   # 🔄 自纠错回环
            "extract_align": "extract_align",      # → 最终后处理
        },
    )

    # ── v2: 属性对齐后结束 ──
    graph.add_edge("extract_align", END)

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

    【v2 图执行流程】
      query_fusion → product_routing → hybrid_retrieval → llm_generation
          → [sdk_verify ⇄ llm_generation 回环重试]
          → extract_align → END

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
        # ── v2 后处理控制字段 ──
        "extracted_entities": {},
        "feedback": "",
        "retry_count": 0,
        "context_text": "",
        "raw_llm_answer": "",
    }

    # invoke() 同步执行完整图，返回最终状态
    final_state = graph.invoke(initial_state)

    needs_clarification = final_state.get("route_status") == "clarify"

    # extract_align_node 会将修正后的 answer 写回 final_answer
    answer = final_state.get("final_answer", "")
    sources = final_state.get("sources", [])
    model = final_state.get("model", "langgraph")

    # ── v2: 记录后处理信息 ──
    retry_used = final_state.get("retry_count", 0)
    if retry_used > 0:
        logger.info(f"📊 SDK 自纠错: 共重试 {retry_used} 次")

    kv_count = len(final_state.get("extracted_entities", {}))
    if kv_count > 0:
        raw = final_state.get("raw_llm_answer", "")
        if raw and raw != answer:
            logger.info(f"📊 属性对齐: 对 {kv_count} 个 KV 实体进行了校验")

    return {
        "answer": answer,
        "sources": sources,
        "model": model,
        "needs_clarification": needs_clarification,
    }


def run_graph_stream(
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    product_id: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    使用 LangGraph 引擎执行一次 RAG 对话（流式 v2）。

    与 rag_chat_stream() 接口完全兼容。

    流式策略：
      1. 前置节点手动执行（query_fusion → product_routing → hybrid_retrieval）
      2. llm_generation 节点使用 _stream_llm 实现真正的 token 级流式
      3. v2 新增：流式输出后，执行 SDK 校验 + 属性对齐（非流式后处理）
         — SDK 校验发现问题时，回环重试并重新流式输出修正结果
    """
    # ── 初始状态（含 v2 字段）──
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
        "extracted_entities": {},
        "feedback": "",
        "retry_count": 0,
        "context_text": "",
        "raw_llm_answer": "",
    }

    # Step 1 & 2: query_fusion → product_routing
    s1 = query_fusion_node(initial_state)
    state = {**initial_state, **s1}
    s2 = product_routing_node(state)
    state.update(s2)
    route_status = state.get("route_status", "generate")

    if route_status in ("clarify", "chitchat", "refuse"):
        answer = state.get("final_answer", "")
        chunk_size = 15
        for i in range(0, len(answer), chunk_size):
            yield answer[i:i + chunk_size]
        return

    # Step 3: hybrid_retrieval
    s3 = hybrid_retrieval_node(state)
    state.update(s3)
    context_docs = state.get("retrieved_docs", [])
    fused_query = state.get("fused_query") or state.get("query", "")

    if context_docs:
        context_docs = _expand_parent_sections(
            context_docs, _get_graph_vector_store(),
            product_id=state.get("product_id"), max_siblings=2,
        )

    # ── 导入流式所需函数 ──
    from .rag_chain import _build_messages, _check_vllm_health, _acquire_vllm_lock, _release_vllm_lock
    from .rag_chain import _get_client, _get_deepseek_client, _resolve_vllm_model, _stream_llm
    from .rag_chain import _FALLBACK_EXCEPTIONS, _FALLBACK_ENABLED
    from .rag_chain import _direct_retrieval_response_stream, _hard_refusal_stream

    # ── v2 SDK 自纠错流式回路 ──
    max_retries = 2
    retry_count = 0
    feedback = ""

    while True:
        # 构建消息（含自纠错反馈）
        try:
            messages = _build_messages(fused_query, context_docs, chat_history)
        except Exception:
            yield from _hard_refusal_stream()
            return

        if _last_numeric_context_missing:
            logger.info("🚫 [Graph Stream] 数字请求无上下文 → 硬拒答")
            yield from _hard_refusal_stream()
            return

        # 注入自纠错反馈
        if feedback:
            correction_msg = (
                f"【自纠错提示 — 上一轮代码问题】\n{feedback}\n\n"
                f"请修正上述问题，重新生成完整、正确的回答（包含修正后的代码）。"
            )
            messages.append({"role": "user", "content": correction_msg})
            logger.info(f"  ↳ [Stream] 已注入自纠错反馈 (retry #{retry_count})")

        # ── 四层容灾流式生成 ──
        _yielded = [False]
        streaming_buffer: List[str] = []  # 🔴 收集完整流式输出用于后处理

        def _track_and_collect(gen):
            for chunk in gen:
                _yielded[0] = True
                streaming_buffer.append(chunk)
                yield chunk

        # Layer 1: 本地 vLLM
        vllm_healthy = _check_vllm_health()
        if vllm_healthy:
            lock_acquired = _acquire_vllm_lock()
            try:
                if lock_acquired:
                    yield from _track_and_collect(
                        _stream_llm(_get_client(), _resolve_vllm_model(), messages)
                    )
            except _FALLBACK_EXCEPTIONS:
                pass
            except Exception:
                pass
            finally:
                if lock_acquired:
                    _release_vllm_lock()

        # Layer 2: 云端智谱
        if not _yielded[0] and _FALLBACK_ENABLED:
            streaming_buffer.clear()
            try:
                from .config import DEEPSEEK_MODEL
                yield from _track_and_collect(
                    _stream_llm(_get_deepseek_client(), DEEPSEEK_MODEL, messages)
                )
            except _FALLBACK_EXCEPTIONS:
                pass
            except Exception:
                pass

        # Layer 3: 纯检索直出
        if not _yielded[0]:
            streaming_buffer.clear()
            try:
                yield from _track_and_collect(
                    _direct_retrieval_response_stream(context_docs, fused_query)
                )
            except Exception:
                pass

        # Layer 4: 全部失败
        if not _yielded[0]:
            yield from _hard_refusal_stream()
            return

        # ── 生成成功：更新状态 ──
        full_answer = "".join(streaming_buffer)
        state["raw_llm_answer"] = full_answer
        state["final_answer"] = full_answer
        state["retry_count"] = retry_count
        state["feedback"] = feedback

        # ── v2 SDK 自纠错检测（循环内） ──
        if _is_sdk_query(fused_query) and retry_count < max_retries:
            sdk_result = sdk_verify_node(state)
            new_feedback = sdk_result.get("feedback", "")
            new_retry = sdk_result.get("retry_count", retry_count)

            if new_feedback and new_retry <= max_retries:
                logger.info(
                    f"🔄 [Stream] SDK 自纠错: retry_count={new_retry}, "
                    f"feedback='{new_feedback[:80]}' → 回环重试"
                )
                retry_count = new_retry
                feedback = new_feedback
                streaming_buffer.clear()
                continue  # ← 回到 while 顶部，重新生成

        # ── 无 SDK 问题或重试耗尽 → 跳出循环，进入后处理 ──
        break

    # ── v2: 属性对齐后处理（循环外） ──
    align_result = extract_align_node(state)
    corrected_answer = align_result.get("final_answer", "")

    if corrected_answer != state.get("final_answer", ""):
        state["final_answer"] = corrected_answer
        # 流式场景下原始 token 已发送，对齐差异仅记录日志
        logger.info("📊 [Stream] 属性对齐完成（差异已记录，流式输出不做回溯修改）")

    # ── 确保生成器正常结束 ──
    return


# ============================================================
# 模块初始化
# ============================================================

logger.info("📐 LangGraph RAG 引擎模块已加载（图实例将在首次调用时编译）")
