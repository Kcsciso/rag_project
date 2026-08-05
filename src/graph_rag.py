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
from . import rag_chain as _rag_chain_mod  # 模块引用 — 访问可变全局变量

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


# ── v3 CodeEntityAnchor: 代码实体提取与 BM25 增强 ──

# 函数名模式: robot_*, set_*, get_*, movl/movc/movj/movp
_CODE_ENTITY_PATTERNS = [
    re.compile(r'\b(?:robot_|set_|get_)\w+\b', re.IGNORECASE),
    re.compile(r'\b(?:movl|movc|movj|movp|movb)\b', re.IGNORECASE),
    re.compile(r'\b(?:py_dll|collrob_sdk)\b', re.IGNORECASE),
    re.compile(r'\b(?:ctypes\.CDLL|ctypes\.c_\w+|POINTER|byref)\b', re.IGNORECASE),
    re.compile(r'\b(?:power_on|enable|brkopen|home|joint_angle|io_output)\b', re.IGNORECASE),
    re.compile(r'\b(?:POSE|RobJoint|RobPos|JNT)\b', re.IGNORECASE),
]

# 运动类型归一化 — 防止 movl/movc 混淆
_MOTION_ALIASES = {
    "movl": "直线运动 (movl)",
    "movc": "圆弧运动 (movc)",
    "movj": "关节运动 (movj)",
    "movp": "点位运动 (movp)",
    "movb": "样条运动 (movb)",
}


def _extract_code_entities(query: str) -> List[str]:
    """
    从 query 中提取代码实体名（函数名/DLL/结构体），用于增强 BM25 检索精度。

    比如 query "怎么用 movl 走直线" → ["movl", "robot_movl"]
    检索时 "[CODE:movl]" 标签会被 BM25 tokenizer 保护，强制精确匹配。
    """
    entities = []
    seen = set()
    for pat in _CODE_ENTITY_PATTERNS:
        for m in pat.finditer(query):
            entity = m.group(0).lower()
            if entity not in seen:
                seen.add(entity)
                entities.append(entity)
    # 如果是运动类型缩写，追加完整形式到检索词
    for e in list(entities):
        if e in _MOTION_ALIASES:
            alias = _MOTION_ALIASES[e]
            # 不修改 query，仅记录以供 BM25 boost
    return entities[:8]  # 最多 8 个实体，防止 query 过长


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
    # 🔴 删除了 _fuse_short_query, _resolve_clarification_followup 等旧轮子
    _resolve_product_from_query,
    _is_chitchat,
    _is_impossible_query,
    # 🔴 删除了 _has_business_intent, _is_product_name_only
    _build_messages,
    _expand_parent_sections,
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
    _stream_guardrail,          # 🔴 v25: 流式代码围栏闭合兜底
    _fix_and_close_sdk_code,    # 🔴 v25: 非流式代码闭合兜底
    _build_clarification_response,
    _build_clarification_response_stream,
    _rewrite_query_with_llm,  # 🟢 引入大模型重写引擎
)
from .vector_store import get_registered_products, search_similar_with_threshold
# ── 短词查询最大长度阈值（低于此值触发短词融合）──
_SHORT_QUERY_MAX_LEN = 15


# ============================================================
# 🔴 v2.2: 全图节点全局异常捕获装饰器 (Fail-Safe)
# ============================================================

def _node_safe(fallback: dict):
    """
    节点安全装饰器 — 捕获节点内所有未处理异常，返回平滑兜底 State，
    绝对禁止向外抛出 Unhandled Exception 导致 HTTP 500。

    用法: @_node_safe(fallback_dict)  或  手动包装: node = _node_safe({...})(node)
    """
    def decorator(fn):
        def wrapper(state):
            try:
                return fn(state)
            except Exception as e:
                logger.error(
                    f"❌ [{fn.__name__}] 运行时异常: {e}", exc_info=True
                )
                return fallback
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper
    return decorator


# ============================================================
# Node 1: QueryFusionNode — 多轮对话融合 (接入大模型意图重写)
# ============================================================

def query_fusion_node(state: RAGState) -> dict:
    """
    接入大模型重写引擎：对原始 query 进行意图补全与代词消解 → 口语噪音剥离。

    输入：state["query"], state["chat_history"]
    输出：fused_query (清洗后供向量检索), query (重写后的完整意图，供全局使用)
    """
    query = state.get("query", "")
    chat_history = state.get("chat_history")

    # 历史沉渣净化 — 传入融合节点前剥离拒答/免责套话
    if chat_history:
        from .rag_chain import sanitize_chat_history as _sanitize_hist
        chat_history = _sanitize_hist(chat_history)

    logger.info(f"🟢 [Node 1] QueryFusion: raw='{query[:60]}'")

    # 🔴 核心：让大模型接管历史阅读、代词消解与意图重写
    rewritten_query = _rewrite_query_with_llm(query, chat_history)

    # 剥离口语噪音（交给底层 ChromaDB 检索的形态）
    cleaned = _preprocess_query(rewritten_query)

    return {
        "fused_query": cleaned,
        "query": rewritten_query,  # 🔴 状态图中的 query 更新为补全后的完美意图
        "raw_query": query,        # 🔴 v27: 保留原始输入（重写前）—— 产品路由责任切分需要
    }

# ============================================================
# Node 2: ProductRoutingNode — 产品识别 + 意图分类
# ============================================================

# ── v3.0 多产品检测: 返回 query 中匹配的所有产品 ID ──
def _detect_all_products(query: str) -> List[str]:
    """扫描 PRODUCT_ROUTER_RULES，返回 query 中命中的所有产品 ID（去重）。"""
    q = query.lower()
    products = []
    for rule in _cfg.PRODUCT_ROUTER_RULES:
        for kw in rule["keywords"]:
            if kw.lower() in q:
                if rule["product_id"] not in products:
                    products.append(rule["product_id"])
                break
    return products


# ── 🔴 v17: Search-First 预检索软路由 ──
def _search_first_soft_route(query: str) -> Optional[str]:
    """
    后台跨产品全库预检索：若某产品得分断层领先，自动锁定 product_id。

    算法:
      1. 调用 search_similar_with_threshold 做全库检索 (k=5, threshold=None)
      2. 按 metadata["product_id"] 分组，取各组最高相似度得分
      3. 若最高分产品 > 0.5 且比第二高产品领先 ≥ 0.15 → 返回该产品的 product_id
      4. 否则返回 None
    """
    try:
        from .vector_store import search_similar_with_threshold as _vsearch
        docs = _vsearch(query, k=5, threshold=None, product_id=None)
        if not docs:
            return None
        # 按产品分组取最高得分
        product_best: Dict[str, float] = {}
        for doc in docs:
            pid = doc.metadata.get("product_id", "") if hasattr(doc, "metadata") else ""
            if not pid or pid == "General":
                continue
            # ChromaDB 相似度: 距离越小越相似，需有 score
            score = getattr(doc, '_score', None) or doc.metadata.get('_score', 0.5)
            if pid not in product_best or score > product_best[pid]:
                product_best[pid] = score
        if len(product_best) < 1:
            return None
        # 找最高分产品
        sorted_products = sorted(product_best.items(), key=lambda x: x[1], reverse=True)
        top_pid, top_score = sorted_products[0]
        second_score = sorted_products[1][1] if len(sorted_products) > 1 else 0.0
        # 断层领先判定
        if top_score > 0.5 and (top_score - second_score) >= 0.15:
            logger.info(
                f"🔍 Search-First: 锁定 '{top_pid}' (score={top_score:.3f}, "
                f"lead={top_score - second_score:.3f})"
            )
            return top_pid
    except Exception as e:
        logger.debug(f"Search-First 预检索异常: {e}")
    return None


# ── 🔴 v17: 确定性产品反问生成器 ──
def build_product_clarification_response() -> dict:
    """
    纯 Python 确定性反问 — 彻底废除 LLM 生成的占位符澄清模板。

    直接调用 get_registered_products() 读取 ChromaDB 中已入库产品列表，
    组装为确定性的反问文案。
    """
    try:
        from .vector_store import get_registered_products as _get_products
        products = _get_products() or []
    except Exception:
        products = []
    # 🔴 过滤无效值，若为空则硬编码物理产品列表兜底 — 绝不用占位符
    valid_prods = [p for p in products if p and "具体产品型号" not in str(p)]
    if not valid_prods:
        valid_prods = ["JAKA", "OpenC3", "OpenR6"]
    products_str = "、".join(valid_prods)
    msg = (
        f"请问您询问的是哪一款产品呢？（当前已支持：{products_str}）\n"
        "不同产品的 SDK 接口与操作逻辑有所不同，请告知具体型号以便为您准确解答。"
    )
    return {"answer": msg, "sources": [], "model": "product-clarification",
            "needs_clarification": True}


def product_routing_node(state: RAGState) -> dict:
    """
    确定 route_status (v3.0 — 多产品对比支持):
      - "chitchat"       → 闲聊/身份询问
      - "refuse"         → 不可能组合（如 JAKA+NumPy）
      - "clarify"        → 产品未识别，需反问
      - "multi_product"  → 2+ 产品同时命中 → 拆分检索
      - "generate"       → 正常单产品检索+生成
      - "fallback"       → 产品已识别但检索词较弱
    """
    query = state.get("fused_query") or state.get("query", "")
    product_id = state.get("product_id")
    # 🔴 v27: 原始输入（重写前）与历史 —— 产品路由责任切分的数据源
    _orig_raw = state.get("raw_query") or state.get("query", "") or query
    _history = state.get("chat_history") or []

    logger.info(f"🟡 [Node 2] ProductRouting: query='{query[:60]}', product_id='{product_id}'")

    # ── 意图 1: 闲聊/身份询问 ──
    if _is_chitchat(query):
        logger.info("  ↳ route_status='chitchat'")
        resp = _chitchat_response()
        return {"route_status": "chitchat", "final_answer": resp["answer"],
                "sources": resp.get("sources", []), "model": resp.get("model", "identity-router")}

    # ── 意图 2: 不可能组合 ──
    if _is_impossible_query(query):
        logger.info("  ↳ route_status='refuse'")
        resp = _hard_refusal_response()
        return {"route_status": "refuse", "final_answer": resp["answer"],
                "sources": [], "model": "hard-refusal"}

    # ── v3.0 意图 2.5: 多产品对比检测（🔴 v27: 用原始 query 判定，防重写脑补产品名）──
    if not product_id:
        all_products = _detect_all_products(_orig_raw)
        if len(all_products) >= 2:
            logger.info(f"  ↳ route_status='multi_product': {all_products}")
            return {
                "route_status": "multi_product",
                "product_id": None,  # 不绑定单一产品 → 检索节点拆分
            }
        elif len(all_products) == 1:
            product_id = all_products[0]

    # ── 意图 3: 产品未识别 → 责任切分 → Search-First 预检索 → 反问澄清 ──
    if not product_id:
        from .rag_chain import _resolve_product_from_query as _resolve
        from .rag_chain import _COVERAGE_QUERY_RE as _coverage_re
        from .rag_chain import _resolve_product_from_history as _resolve_hist

        # 🔴 v27: 单轮 + 原始 query 无产品名 + 非覆盖性提问 → 直接澄清
        # （重写器不得越权补产品；同时跳过 Search-First 软路由）
        if (not _history and not _resolve(_orig_raw)
                and not _coverage_re.search(_orig_raw)):
            registered = get_registered_products()
            resp = build_product_clarification_response()
            logger.info("  ↳ route_status='clarify' (v27: 单轮无产品名直接澄清)")
            return {"route_status": "clarify", "final_answer": resp["answer"],
                    "sources": [], "model": "product-clarification", "product_id": None}

        product_id = _resolve(_orig_raw) or _resolve(query)
        # 多轮第三兜底：历史文本扫描（重写器不服从时仍能锁定产品）
        if not product_id and _history:
            product_id = _resolve_hist(_history)

        if not product_id:
            # 🔴 v27: coverage 例外 —— "有没有提到 X" 不得澄清，进跨产品检索由 L3 模板守卫拒答
            if _coverage_re.search(_orig_raw):
                logger.info("  ↳ coverage 提问例外：跳过澄清，进入跨产品检索（L3 拒答）")
                return {"route_status": "generate", "product_id": None,
                        "sources": [], "model": "coverage-router"}
            _orig_query = state.get("query", "") or query
            # 🔴 v17: Search-First 软路由 — 后台跨产品预检索，断层领先则自动锁定
            _auto_locked_pid = None
            # 🟢 移除了 _has_business_intent 校验，因为重写后的 query 已完全自洽
            if len(_orig_query.strip()) >= 4:
                try:
                    _auto_locked_pid = _search_first_soft_route(query)
                except Exception:
                    pass
            if _auto_locked_pid:
                logger.info(f"  ↳ Search-First 自动锁定: product_id='{_auto_locked_pid}'")
                return {"route_status": "generate", "product_id": _auto_locked_pid}
            # 短词 + 无显式产品 → 澄清反问
            registered = get_registered_products()
            resp = build_product_clarification_response()
            logger.info("  ↳ route_status='clarify'")
            return {"route_status": "clarify", "final_answer": resp["answer"],
                    "sources": [], "model": "product-clarification", "product_id": None}

    # ── 意图 4: 正常生成 ──
    logger.info(f"  ↳ route_status='generate', product_id='{product_id}'")
    return {"route_status": "generate", "product_id": product_id}


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
    if status in ("chitchat", "refuse", "clarify"):
        # 🔴 clarify 直接返回反问文本，不进入 Planner（避免 Planner 降级为跨产品搜索后产生幻觉代码）
        return "build_direct_response"
    return "subgoal_planner"


def _route_after_planner(state: RAGState) -> str:
    """
    SubGoalPlannerNode 之后的分发逻辑:

      - plan_mode == "cross_product" 且 product_id 为 None → cross_product_retrieval
      - 否则 → hybrid_retrieval（Fast Path, 保持零额外延迟）
    """
    plan_mode = state.get("plan_mode", "single")
    product_id = state.get("product_id")

    if plan_mode == "cross_product" and not product_id:
        return "cross_product_retrieval"
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
      0. v3 CodeEntityAnchor: 从 query 提取代码实体，注入 BM25 boost
      1. 主检索：阈值过滤 + 产品隔离
      2. 若主检索为空 → 第二机会：无阈值原始向量 Top-3 兜底
      3. 若仍为空 → route_status="fallback"（仍尝试 LLM 生成）

    需要外部注入 vector_store 实例。通过模块级 _graph_vector_store 变量传递。
    """
    query = state.get("fused_query") or state.get("query", "")
    product_id = state.get("product_id")

    logger.info(f"🔵 [Node 3] HybridRetrieval: query='{query[:60]}', product_id='{product_id}'")

    # ── v3 CodeEntityAnchor: 代码实体提取与 BM25 增强 ──
    code_entities = _extract_code_entities(query)
    if code_entities:
        logger.info(f"  ↳ [CodeEntityAnchor] 检测到 {len(code_entities)} 个代码实体: {code_entities}")
        # 将代码实体追加到检索 query 中，提升 BM25 权重
        # 用特殊标签标记，在 BM25 tokenizer 中会被保护
        _entity_suffix = " ".join(f"[CODE:{e}]" for e in code_entities)
        query = f"{query} {_entity_suffix}"
    else:
        code_entities = []

    vector_store = _get_graph_vector_store()
    if vector_store is None:
        logger.error("❌ vector_store 未注入到 Graph 引擎")
        return {
            "retrieved_docs": [],
            "route_status": "fallback",
        }

    # ── v3.0: 多产品拆分检索 ──
    route_status_in = state.get("route_status", "generate")
    if route_status_in == "multi_product":
        all_products = _detect_all_products(query)
        logger.info(f"  ↳ 多产品拆分检索: {all_products}")
        merged: List[Any] = []
        seen_fps: set = set()
        for pid in all_products:
            sub_docs = _hybrid_retrieve(
                vector_store, query,
                k=2, threshold=_cfg.SIMILARITY_THRESHOLD,
                fetch_factor=3, product_id=pid,
            )
            for doc in sub_docs:
                fp = doc.page_content[:120]
                if fp not in seen_fps:
                    seen_fps.add(fp)
                    merged.append(doc)
            logger.info(f"    [{pid}]: {len(sub_docs)} chunks → merged {len(merged)}")
        context_docs = merged
    else:
        # 主检索（单产品）
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

    # ── 🔴 v3.0 ABSTAIN 网关: 查询含实体/数字但 Context 中不存在 → 硬弃答 ──
    _query_entities = set(
        re.findall(r'\b(\d{2,})\b', query) +
        re.findall(r'(?:Modbus|Profinet|EtherCAT|TCP|RTU|RS485|RS232|'
                   r'波特率|端口号|IP地址|寄存器|从站|主站|末端传感器|'
                   r'电控柜|制动电阻|MiniCab|VBrake)', query, re.IGNORECASE)
    )
    if _query_entities and context_docs:
        _context_combined = " ".join(
            d.page_content if hasattr(d, 'page_content') else str(d)
            for d in context_docs
        )
        _missing = [e for e in _query_entities
                    if e.lower() not in _context_combined.lower()]
        if _missing:
            _doc_name = (context_docs[0].metadata.get("source", "相关文档")
                         if hasattr(context_docs[0], 'metadata') else "相关文档")
            _abstain_msg = (
                f"根据《{_doc_name}》，未找到关于 '{_missing[0]}' 的明确记载。"
                f"请确认参数名称或联系技术支持。"
            )
            logger.info(f"🚫 ABSTAIN: 实体 {_missing} 不在 Context 中 → 硬弃答")
            return {
                "final_answer": _abstain_msg,
                "raw_llm_answer": _abstain_msg,
                "sources": [],
                "model": "abstain-gateway",
                "route_status": "complete",
                "feedback": "",
                "retry_count": state.get("retry_count", 0),
            }

    # ── 父子切片扩展 + 构建消息 ──
    if context_docs:
        context_docs = _expand_parent_sections(
            context_docs, _get_graph_vector_store(),
            product_id=state.get("product_id"), max_siblings=2,
        )
    try:
        # 🔴 v29: 返回侧信道 (messages, refusal_flag) —— Fast-Path 确定性拒答
        messages, _refusal_flag = _build_messages(query, context_docs, chat_history)
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

    # 🔴 v29: Fast-Path 确定性拒答 —— 模板守卫命中 → 跳过 LLM 直接返回固定话术
    # 检查点在生成金字塔之前（含 Layer 3 —— 否则守卫命中 + LLM 全挂时
    # _direct_retrieval_response 会直出非拒答内容绕过拒答模板）
    if _refusal_flag:
        logger.info("🚫 [Fast-Path] 模板守卫命中 → 确定性拒答（跳过 LLM 生成）")
        return {
            "final_answer": _HARD_REFUSAL,
            "raw_llm_answer": _HARD_REFUSAL,
            "sources": [],
            "model": "refusal-fast-path",
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

    # 🔴 数字请求无上下文硬防护 + KV 属性检索 (ADR-13)
    # 🔴 v25: 数字意图查询即尝试 KV 属性注入（不依赖 Context 缺失守卫），
    # 使 E05(端口6502)/GT-6(6502含义)/E07(波特率9600) 的正确答案确定性出现在 Prompt 中
    _numeric_guard = _rag_chain_mod._last_numeric_context_missing
    if _numeric_guard or _rag_chain_mod._NUMERIC_QUERY_RE.search(query):
        # ── 第零机会: KV 属性存储检索 ──
        try:
            from .kv_extractor import lookup_attribute as _kv_lookup
            _kv_result = _kv_lookup(query, product_id=state.get("product_id"))
            if _kv_result:
                logger.info(f"✅ [Graph] KV 属性检索命中: {_kv_result}")
                # 🔴 将 KV 结果注入用户消息头部作为最高优先级事实参考
                _kv_prefix = (
                    f"【⚠️ 以下为系统属性库中的已知事实，必须在回答中直接引用：{_kv_result}。"
                    f"请以系统属性库为准，不要引用PDF中不完整的信息。】\n\n"
                )
                for _i, _m in enumerate(messages):
                    if _m["role"] == "user":
                        messages[_i]["content"] = _kv_prefix + messages[_i]["content"]
                        break
                _numeric_guard = False
        except Exception:
            pass

    if _numeric_guard:
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
        "feedback": "",                              # 成功后清除反馈
        "retry_count": state.get("retry_count", 0),  # 🔴 保留当前重试计数，不重置（防死循环）
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
    _MAX_RETRIES = 2

    logger.info(f"🔴 [Node 5] SDK_Verify: retry_count={retry_count}")

    # ── 🔴 硬熔断: 重试次数已达上限 → 无条件放弃修复，透传当前回答 ──
    if retry_count >= _MAX_RETRIES:
        logger.warning(
            f"  ↳ 🔴 SDK 重试硬熔断: retry_count={retry_count} >= {_MAX_RETRIES}，"
            f"放弃修复，透传当前回答"
        )
        return {"feedback": "", "retry_count": retry_count}

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
# Node 5b: RenderNode — 提取模式 JSON 渲染器 (v4.0 Extract-Render)
# ============================================================

def render_node(state: RAGState) -> dict:
    """
    退化版 RenderNode: 放弃 JSON 解析，直接透传大模型的 Markdown 输出。
    后处理交由 extract_align_node 和 rag_chain 中的正则防线兜底。
    """
    raw_answer = state.get("raw_llm_answer") or state.get("final_answer", "")
    
    return {
        "final_answer": raw_answer.strip(),
        "route_status": state.get("route_status", "complete"),
    }


# ============================================================
# Node 6: ExtractAlignNode — 通用属性对齐校验（v2）
# ============================================================

# ── 🔴 v16: 免责套话后处理剥离 ──
_HEDGING_TAIL_RE = re.compile(
    r'(?:'
    r'参考文档(?:中)?未(?:包含|记载|找到|涵盖|提供)(?:详细)?[^。]{0,30}[。]?'
    r'|上述代码(?:假设|仅为示例|假设存在)[^。]*[。]?'
    r'|具体操作步骤未在文档中[^。]*[。]?'
    r'|建议(?:您)?(?:联系技术支持|查阅最新文档|参照)[^。]*[。]?'
    r'|请注意[，,]\s*以上[^。]*[。]?'
    r'|由于参考资料[^。]*[。]?'
    r')[\s\n]*$'
)


def _strip_hedging_tail(text: str) -> str:
    """
    v16: 后处理剥离 — 擦除回答末尾自相矛盾的免责/假设套话。

    场景: LLM 已正确给出 API 代码，但末尾又追加了
    "参考文档未包含详细步骤" 或 "上述代码假设存在"，形成前后矛盾。
    """
    if not text:
        return text
    stripped = _HEDGING_TAIL_RE.sub('', text).strip()
    if stripped != text:
        logger.info("  ✂️  HedgingTail: 剥离末尾免责套话")
    return stripped


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
    # 🔴 v25: SDK 代码兜底（代码块闭合）——Graph 路径此前从未应用过，
    # 非流式 ⑧ 代码截断的直接原因
    raw_answer = _fix_and_close_sdk_code(raw_answer)

    logger.info(f"🟢 [Node 6] ExtractAlign: {len(kv_entities)} KV entities, "
                 f"answer_len={len(raw_answer)}")

    # 回答为空 → 直接透传（去重/闭合需要非空文本）
    if not raw_answer:
        logger.info("  ↳ 回答为空，透传原始回答")
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

    # ── 🔴 v25: 精确段落去重（无条件执行）—— 复读机整段复读兜底 ──
    # 🔴 v27: 比较前规范化（忽略空白与尾标点差异）—— 换行/标点差异的重复段也能捕获
    # 仅移除"连续两段基本相同"的段落（≥80 字符，与 eval ② 定义一致），对代码/步骤零误伤
    def _norm_para(_p: str) -> str:
        return re.sub(r'\s+', '', _p).rstrip('。！？；;，,：:.、…')
    _paras = corrected.split("\n\n")
    _paras_deduped = []
    for _p in _paras:
        _p = _p.strip()
        if (_paras_deduped and _p and len(_p) >= 80
                and _norm_para(_p) == _norm_para(_paras_deduped[-1])):
            logger.info(f"  ✂️  段落去重: 移除连续重复段落 ({len(_p)} 字符)")
            continue
        _paras_deduped.append(_p)
    corrected = "\n\n".join(_paras_deduped)

    # ── 🔴 v3.0 SemanticDedup: 后处理语义去重，消除1.5B小模型段落级重复生成 ──
    # 取消 JAKA 的豁免权，全量开启去重防御复读机
    # 🔴 v25: 含代码块的回答跳过模糊去重（避免 trigram 误伤代码行），仅做精确段落去重
    if "```" not in corrected:
        import re as _re_dedup
        _sentences = _re_dedup.split(r'(?<=[。！？\n])\s*', corrected)
        _sentences = [s.strip() for s in _sentences if len(s.strip()) >= 8]

        if len(_sentences) >= 4:
            _deduped = [_sentences[0]]  # 保留首句
            _cut_at = len(_sentences)
            for _i in range(1, len(_sentences)):
                # 滑动窗口: 检查当前句是否与前面任一已保留句高度重复
                _is_dup = False
                _cur = _sentences[_i]
                for _j in range(max(0, _i - 3), _i):
                    _prev = _sentences[_j]
                    # Jaccard-like trigram overlap
                    _cur_grams = set(_cur[i:i+3] for i in range(len(_cur) - 2))
                    _prev_grams = set(_prev[i:i+3] for i in range(len(_prev) - 2))
                    if _cur_grams and _prev_grams:
                        _overlap = len(_cur_grams & _prev_grams) / min(len(_cur_grams), len(_prev_grams))
                        if _overlap > 0.55:
                            _is_dup = True
                            break
                if _is_dup:
                    _cut_at = _i
                    break
                _deduped.append(_sentences[_i])

            if _cut_at < len(_sentences):
                _trimmed_count = len(_sentences) - _cut_at
                logger.info(
                    f"  ✂️  SemanticDedup: 截断 {_trimmed_count} 个重复句 "
                    f"(trigram_overlap > 0.55 @ pos {_cut_at})"
                )
                corrected = "\n".join(_deduped)

    # 🔴 v16: 剥离末尾自相矛盾的免责套话
    corrected = _strip_hedging_tail(corrected)
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


# ============================================================
# v3 Node: SubGoalPlannerNode — 任务分解（ADR-14 Plan-Execute-Synthesize）
# ============================================================

# 快速判断：是否需要 Planner（单产品简单查询跳过）
def _needs_planner(state: RAGState) -> bool:
    """有明确 product_id 且 query 无跨产品/属性意图时跳过 Planner"""
    query = state.get("query", "")
    product_id = state.get("product_id", "")
    # Fast Path: 有明确产品 ID 的单产品查询
    if product_id and product_id != "multi":
        return False
    # 无产品 ID 或 query 含跨产品意图 → 需要 Planner
    return True


def _parse_subgoals_markdown(text: str) -> List[Dict]:
    """从 LLM 输出的 Markdown 中解析子目标列表。含严密异常捕获。"""
    goals = []
    try:
        in_list = False
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # 匹配 "- 类型: product_qa" 等
            if line.startswith("-") and "类型:" in line:
                in_list = True
                goal = {"type": "product_qa", "product_id": None, "query": "", "priority": 1}
                # 提取字段
                type_match = re.search(r'类型\s*[:：]\s*(\w+)', line)
                if type_match:
                    goal["type"] = type_match.group(1).strip()
                pid_match = re.search(r'产品\s*[:：]\s*(\S+)', line)
                if pid_match and pid_match.group(1).lower() != "无":
                    goal["product_id"] = pid_match.group(1).strip()
                q_match = re.search(r'查询\s*[:：]\s*["“]?(.+?)["”]?\s*(?:,|，|$)', line)
                if q_match:
                    goal["query"] = q_match.group(1).strip()
                elif goal["type"] == "cross_product":
                    goal["query"] = text[:200]  # fallback
                goals.append(goal)
            elif line.startswith("-") and in_list:
                # 续行
                if goals:
                    q_match = re.search(r'查询\s*[:：]\s*["“]?(.+?)["”]?\s*$', line)
                    if q_match and not goals[-1].get("query"):
                        goals[-1]["query"] = q_match.group(1).strip()
        if not goals:
            logger.warning("SubGoalPlanner: Markdown 解析为空，降级为单路检索")
    except Exception as e:
        logger.warning(f"SubGoalPlanner: 解析异常 {e}，降级为单路检索")
        goals = []
    return goals


def subgoal_planner_node(state: RAGState) -> dict:
    """
    SubGoalPlannerNode — 用 LLM 轻量调用拆分子问题。

    防崩兜底：Markdown 解析失败 / LLM 调用失败 / 返回空结果 →
    全部降级为标准单路检索（plan_mode="single"）。
    """
    query = state.get("query", "")
    product_id = state.get("product_id") or ""
    skip = state.get("skip_planner", False) or not _needs_planner(state)

    if skip:
        logger.info(f"🟢 [SubGoalPlanner] Fast Path: product_id='{product_id}' → 跳过规划")
        return {
            "plan_mode": "single",
            "sub_goals": [{"type": "product_qa", "product_id": product_id or None,
                           "query": query, "priority": 1}],
            "skip_planner": True,
        }

    logger.info(f"🟡 [SubGoalPlanner] 启动任务规划: query='{query[:60]}'")

    # ── 构建规划 Prompt ──
    product_list = "JAKA, OpenC3, OpenR6"
    if product_id:
        product_list = product_id

    planner_prompt = f"""分析用户问题，判断是否需要拆分为多个子问题。

已知产品: {product_list}
用户问题: {query}

输出规则（严格按 Markdown 列表）:
- 如果是单产品简单查询 → 只输出 1 行:
- 类型: product_qa, 产品: {product_id or '自动检测'}, 查询: "{query}"

- 如果涉及跨产品对比 → 每个产品输出 1 行（最多 3 个）:
- 类型: product_qa, 产品: JAKA, 查询: "JAKA 相关子问题"
- 类型: product_qa, 产品: OpenR6, 查询: "OpenR6 相关子问题"

- 如果涉及端口号/密码/波特率等参数 → 增加 1 行:
- 类型: attribute_lookup, 产品: 产品名或自动, 查询: "属性关键词"

- 如果涉及 SDK 函数名（如 movl/power_on/joint）→ 增加 1 行:
- 类型: code_search, 产品: 产品名或自动, 查询: "函数名"

- 如果产品不明确 → 输出:
- 类型: cross_product, 产品: 无, 查询: "{query}"

只输出列表，不要解释。"""

    try:
        from .rag_chain import _call_llm, _get_client, _resolve_vllm_model, _build_messages
        # 构建最简 messages（不需要 context retrieval）
        messages = [
            {"role": "system", "content": "你是任务规划助手。根据用户问题判断是否需要拆分为多个子任务。"},
            {"role": "user", "content": planner_prompt},
        ]
        response = _call_llm(_get_client(), _resolve_vllm_model(), messages, max_tokens=256)
        sub_goals = _parse_subgoals_markdown(response)
    except Exception as e:
        logger.warning(f"SubGoalPlanner: LLM 调用失败 ({e})，降级为单路检索")
        sub_goals = []

    # ── 兜底：空结果 → 根据是否有 product_id 选择策略 ──
    if not sub_goals:
        if product_id:
            sub_goals = [{"type": "product_qa", "product_id": product_id,
                          "query": query, "priority": 1}]
            plan_mode = "single"
        else:
            sub_goals = [{"type": "cross_product", "product_id": None,
                          "query": query, "priority": 1}]
            plan_mode = "cross_product"
    else:
        plan_mode = "multi" if len(sub_goals) > 1 else "single"

    logger.info(f"🟢 [SubGoalPlanner] 规划完成: mode={plan_mode}, {len(sub_goals)} 个子目标")
    for g in sub_goals:
        logger.info(f"  ↳ type={g['type']}, pid={g.get('product_id','?')}, q={g.get('query','')[:60]}")

    return {
        "sub_goals": sub_goals,
        "plan_mode": plan_mode,
        "skip_planner": plan_mode == "single",
    }


# ============================================================
# v3 Node: CrossProductRetrievalNode — 全库检索（product_id=None）
# ============================================================

def cross_product_retrieval_node(state: RAGState) -> dict:
    """
    当 product_id 为 None 时，执行全库混合检索，而非仅输出反问。

    行为:
      1. 对所有已注册产品执行 Top-3 检索
      2. 返回综合候选 + 反问（两者兼有，不 Only 反问）
    """
    query = state.get("fused_query") or state.get("query", "")
    logger.info(f"🔵 [CrossProductRetrieval] 全库检索: '{query[:60]}'")

    try:
        from .vector_store import get_registered_products, search_similar_with_threshold
        products = get_registered_products() or ["JAKA", "OpenC3", "OpenR6"]
    except Exception:
        products = ["JAKA", "OpenC3", "OpenR6"]

    candidates = []
    all_docs = []

    for pid in products:
        try:
            docs = search_similar_with_threshold(
                _get_graph_vector_store(), query, k=3, threshold=0.55, product_id=pid
            )
            for doc in docs:
                relevance = getattr(doc, "relevance", 0.7) if hasattr(doc, "relevance") else 0.7
                snippet = doc.page_content[:150] if hasattr(doc, "page_content") else str(doc)[:150]
                candidates.append({
                    "product_id": pid,
                    "snippet": snippet,
                    "relevance": relevance,
                })
                all_docs.append(doc)
        except Exception as e:
            logger.debug(f"CrossProductRetrieval: {pid} 检索失败: {e}")

    # 按相关性排序
    candidates.sort(key=lambda x: x["relevance"], reverse=True)

    logger.info(f"🔵 [CrossProductRetrieval] 找到 {len(candidates)} 个候选（{len(products)} 产品）")
    for c in candidates[:5]:
        logger.info(f"  ↳ {c['product_id']} r={c['relevance']:.2f}: {c['snippet'][:80]}")

    return {
        "cross_product_candidates": candidates[:9],
        "retrieved_docs": all_docs,  # 填充 retrieved_docs 以便后续节点使用
    }


# ============================================================
# v3 Node: SynthesizeNode — 多路结果融合
# ============================================================

def synthesize_node(state: RAGState) -> dict:
    """
    融合多路子目标结果，生成最终回答。

    策略:
      - 单子目标（plan_mode="single"）→ 透传原始回答，零额外延迟
      - 多子目标 → LLM 融合：对比、并列、反问
      - 全库检索 → 先反问确认产品，同时附上各产品候选信息
    """
    plan_mode = state.get("plan_mode", "single")
    sub_results = state.get("sub_results") or []
    candidates = state.get("cross_product_candidates") or []

    # ── Fast Path: 单子目标透传 ──
    if plan_mode == "single" or (not sub_results and not candidates):
        logger.info(f"🟢 [Synthesize] Fast Path: 单路透传 (mode={plan_mode})")
        return {}

    logger.info(f"🟡 [Synthesize] 融合 {len(sub_results)} 路结果 + {len(candidates)} 候选")

    # ── 全库检索模式：反问 + 候选信息 ──
    if candidates and not sub_results:
        product_names = list(dict.fromkeys(c["product_id"] for c in candidates[:6]))
        product_list = "、".join(product_names) if product_names else "多个产品"
        snippets = "\n".join(
            f"- {c['product_id']}: {c['snippet'][:100]}" for c in candidates[:5]
        )
        answer = (
            f"您的问题可能涉及{product_list}。以下是各产品的相关信息：\n\n"
            f"{snippets}\n\n"
            f"请问您具体询问的是哪一款产品？我将为您提供更精准的回答。"
        )
        return {"final_answer": answer, "sources": [], "model": "cross-product-synthesis"}

    # ── 多子目标融合 ──
    parts = []
    for i, sr in enumerate(sub_results):
        if sr and sr.get("answer"):
            goal_type = sr.get("type", "?")
            answer = sr["answer"]
            if goal_type == "attribute_lookup":
                parts.append(f"【参数查询结果】\n{answer}")
            elif goal_type == "code_search":
                parts.append(f"【代码查询结果】\n{answer}")
            else:
                pid = sr.get("product_id", "?")
                parts.append(f"【{pid} 产品】\n{answer}")

    if parts:
        final_answer = "\n\n---\n\n".join(parts)
        return {"final_answer": final_answer, "sources": [], "model": "multi-path-synthesis"}

    return {}


def _build_graph() -> StateGraph:
    """
    构建并编译 LangGraph StateGraph（v3 — Plan-Execute-Synthesize 架构）。

    图结构:

      START
        │
        ▼
      query_fusion ──→ product_routing
                          │
              ┌───────────┼───────────┐
              │           │           │
          clarify/    chitchat/    generate/
          refuse      (→ END)     (→ subgoal_planner)
              │                       │
              ▼               ┌───────┴───────┐
      build_direct_response   │               │
              │           plan_mode=      plan_mode=
              ▼            single       multi/cross_product
             END               │               │
                               ▼               ▼
                        hybrid_retrieval  cross_product_retrieval
                               │               │
                               ▼               ▼
                        llm_generation   llm_generation
                               │               │
                               ▼               ▼
                          synthesize  ←────────┘
                               │
                    ┌──────────┼──────────┐
                    │                     │
               sdk_verify           extract_align
                    │                     │
            ┌───────┴───────┐             ▼
            │               │            END
      llm_generation   extract_align
      (retry ≤ 2)          │
                           ▼
                          END

    返回编译后的图实例（带状态校验）。
    """
    graph = StateGraph(RAGState)

    # ── 注册现有节点 ──
    graph.add_node("query_fusion", query_fusion_node)
    graph.add_node("product_routing", product_routing_node)
    graph.add_node("build_direct_response", build_direct_response_node)
    graph.add_node("hybrid_retrieval", hybrid_retrieval_node)
    graph.add_node("llm_generation", llm_generation_node)
    graph.add_node("render", render_node)
    graph.add_node("sdk_verify", sdk_verify_node)
    graph.add_node("extract_align", extract_align_node)

    # ── v3 新增节点 ──
    graph.add_node("subgoal_planner", subgoal_planner_node)
    graph.add_node("cross_product_retrieval", cross_product_retrieval_node)
    graph.add_node("synthesize", synthesize_node)

    # ── 前置管线不变 ──
    graph.set_entry_point("query_fusion")
    graph.add_edge("query_fusion", "product_routing")

    # ── v3 条件路由: product_routing → 3 路分发 ──
    graph.add_conditional_edges(
        "product_routing",
        _route_after_product_routing,
        {
            "build_direct_response": "build_direct_response",
            "subgoal_planner": "subgoal_planner",
        },
    )

    graph.add_edge("build_direct_response", END)

    # ── v3 条件路由: subgoal_planner → Fast Path 或 Multi Path ──
    graph.add_conditional_edges(
        "subgoal_planner",
        _route_after_planner,
        {
            "hybrid_retrieval": "hybrid_retrieval",           # Fast Path: single product
            "cross_product_retrieval": "cross_product_retrieval",  # product_id=None
        },
    )

    # ── 检索 → 生成（两路汇聚到 llm_generation）──
    graph.add_edge("hybrid_retrieval", "llm_generation")
    graph.add_edge("cross_product_retrieval", "llm_generation")

    # ── llm_generation → render → synthesize ──
    graph.add_edge("llm_generation", "render")
    graph.add_edge("render", "synthesize")

    # ── synthesize → 后处理路由（复用 v2 结构）──
    graph.add_conditional_edges(
        "synthesize",
        _route_after_llm,  # 复用现有路由（SDK校验/属性对齐）
        {
            "sdk_verify": "sdk_verify",
            "extract_align": "extract_align",
        },
    )

    # ── v2 自纠错回环 + 属性对齐（与现有逻辑一致）──
    graph.add_conditional_edges(
        "sdk_verify",
        _route_after_sdk_verify,
        {
            "llm_generation": "llm_generation",
            "extract_align": "extract_align",
        },
    )

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
        # ── 🔴 硬熔断: 重试次数已达上限 → 跳出循环，透传当前回答 ──
        if retry_count > max_retries:
            logger.warning(
                f"🔴 [Stream] SDK 重试硬熔断: retry_count={retry_count} > {max_retries}，"
                f"跳出循环，透传当前回答"
            )
            break

        # 构建消息（含自纠错反馈）
        try:
            # 🔴 v29: 返回侧信道 (messages, refusal_flag) —— Fast-Path 确定性拒答
            messages, _refusal_flag_s = _build_messages(fused_query, context_docs, chat_history)
        except Exception:
            yield from _hard_refusal_stream()
            return

        # 🔴 v29: Fast-Path 确定性拒答（流式）—— 模板守卫命中 → 跳过 LLM 直接输出固定话术
        if _refusal_flag_s:
            logger.info("🚫 [Fast-Path] 模板守卫命中 → 确定性拒答（跳过 LLM 生成）")
            yield from _hard_refusal_stream()
            return

        _numeric_guard_s = _rag_chain_mod._last_numeric_context_missing
        # 🔴 v25: 数字意图查询即尝试 KV 属性注入（不依赖 Context 缺失守卫）
        if _numeric_guard_s or _rag_chain_mod._NUMERIC_QUERY_RE.search(fused_query):
            # ── 第零机会: KV 属性存储检索 ──
            try:
                from .kv_extractor import lookup_attribute as _kv_lookup_s
                _kv_result = _kv_lookup_s(query, product_id=state.get("product_id"))
                if _kv_result:
                    logger.info(f"✅ [Graph Stream] KV 属性检索命中: {_kv_result}")
                    _kv_fact = (
                        f"\n\n【⚠️ 系统属性库 — 高优先级已知事实，优先于检索结果】\n"
                        f"{_kv_result}\n"
                    )
                    for _i, _m in enumerate(messages):
                        if _m["role"] == "system":
                            messages[_i]["content"] = _kv_fact + messages[_i]["content"]
                            break
                    _numeric_guard_s = False
            except Exception:
                pass

        if _numeric_guard_s:
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
                yield chunk  # 🔴 核心：把它放回来！恢复流式输出极速体验！

        # Layer 1: 本地 vLLM
        vllm_healthy = _check_vllm_health()
        if vllm_healthy:
            lock_acquired = _acquire_vllm_lock()
            try:
                if lock_acquired:
                    yield from _track_and_collect(
                        _stream_guardrail(_stream_llm(_get_client(), _resolve_vllm_model(), messages))
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
                    _stream_guardrail(_stream_llm(_get_deepseek_client(), DEEPSEEK_MODEL, messages))
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

    # ========================================================
    # 🔴 核心修改：只更新内部状态存入历史，绝不再次 yield 给前端！
    # ========================================================
    
    # 1. 渲染退化（直接透传）
    render_result = render_node(state)
    state.update(render_result)

    # 2. 属性对齐后处理（更新的是 final_answer 供历史记录使用，不再流式输出）
    align_result = extract_align_node(state)
    state.update(align_result)

    # 🔴 彻底删除这三行导致“双重输出”的代码：
    # chunk_size = 15
    # for i in range(0, len(final_rendered_answer), chunk_size):
    #     yield final_rendered_answer[i:i + chunk_size]

    return


# ============================================================
# 模块初始化
# ============================================================

# ============================================================
# 🔴 v2.2: 全图节点全局异常捕获 (Fail-Safe Wrapping)
# 所有节点函数在模块加载时自动包装 try/except，兜底返回安全 State，
# 绝对禁止 Unhandled Exception 向外传播导致 HTTP 500
# ============================================================

_NODE_FALLBACKS = {
    "query_fusion_node": {
        "fused_query": "", "query": "", "product_id": None,
    },
    "product_routing_node": {
        "route_status": "generate", "product_id": None,
    },
    "build_direct_response_node": {},
    "hybrid_retrieval_node": {
        "retrieved_docs": [], "route_status": "fallback",
        "context_text": "", "extracted_entities": {},
    },
    "llm_generation_node": {
        "final_answer": "服务暂时不可用，请稍后重试",
        "raw_llm_answer": "", "sources": [], "model": "node-fallback",
        "route_status": "complete", "feedback": "", "retry_count": 0,
    },
    "sdk_verify_node": {
        "feedback": "", "retry_count": 0,
    },
    "extract_align_node": {
        "final_answer": "", "route_status": "complete",
    },
}

# 对每个节点应用安全包装器
for _name, _fallback in _NODE_FALLBACKS.items():
    _fn = globals().get(_name)
    if _fn is not None and callable(_fn):
        globals()[_name] = _node_safe(_fallback)(_fn)

logger.info("📐 LangGraph RAG 引擎模块已加载（图实例将在首次调用时编译）")
