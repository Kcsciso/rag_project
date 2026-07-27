"""
=============================================================================
Attribute Intent Tool — 动态属性意图提取与检索（ADR-14, v3）
=============================================================================

替代静态 KV 正则的架构升级。核心思路：
  1. 用 LLM 轻量调用（max_tokens=128）从 query 中提取"属性意图"
  2. 用提取的属性意图做 BM25 精准关键词搜索
  3. 在返回切片中用正则提取具体数值
  4. 返回结构化属性结果，注入 RAG context

与静态 KV 表的本质区别：
  - 不需要人工维护别名映射表（"初始化波特率"→"RS485默认波特率"）
  - LLM 天然理解语义等价（"9600 用于什么通信"="波特率 9600 = Modbus RTU"）
  - 自动适配新产品、新参数类型，零配置

调用点：
  - SubGoalPlannerNode 检测到属性意图时创建 attribute_lookup 子目标
  - 各子目标的执行器调用 resolve_attribute()
  - 也可由 _build_messages 中的 KV 通道直接调用
=============================================================================
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ============================================================
# 属性意图提取 Prompt（轻量，max_tokens=128）
# ============================================================
_ATTRIBUTE_INTENT_PROMPT = """从用户问题中提取属性查询意图。

用户问题：{query}

请输出：
属性关键词: <用户想查询的属性名称，如"端口号"、"密码"、"波特率">
查询数值: <如果问题中提到了具体数值则输出，否则输出"无"，如"6502"、"9600">
产品: <如果问题中提到了产品名则输出，否则输出"无">

只输出以上三行，不要解释。"""


# ============================================================
# 属性值正则 — 从检索切片中提取具体数值
# ============================================================
_ATTR_VALUE_PATTERNS = [
    # "端口号: 6502" 或 "端口号 = 6502"
    re.compile(r'(?:端口号?|port)\s*[：:=]\s*(\d{2,6})', re.IGNORECASE),
    # "波特率...9600" 或 "baud rate...9600"
    re.compile(r'(?:波特率|baud\s*rate)\s*[：:=]\s*(\d{2,8})', re.IGNORECASE),
    # "默认密码：xxx" 或 "密码 = xxx"
    re.compile(r'(?:默认\s*)?密码\s*[：:=]\s*(\S{1,30})', re.IGNORECASE),
    # "默认密码：管理员：xxx"（嵌套格式）
    re.compile(r'管理员\s*[：:]\s*(\S{1,30})', re.IGNORECASE),
    # "IP 地址：xxx"
    re.compile(r'IP\s*(?:地址|address)?\s*[：:=]\s*([\d.]{7,15})', re.IGNORECASE),
    # 通用 "key = value" 数字格式
    re.compile(r'(?:^|\n)\s*([^：:\n]{1,20})\s*[：:]\s*(\d{2,8})\s*(?:$|\n)'),
]


# ============================================================
# Public API
# ============================================================

def extract_attribute_intent(
    query: str,
    llm_call_fn=None,
) -> Dict:
    """
    从 query 中提取属性查询意图。

    如果提供了 llm_call_fn，使用 LLM 提取（更准确）；
    否则回退到纯正则提取（零延迟但覆盖率低）。

    Returns:
        {
            "query_keyword": "波特率",
            "extracted_value": "9600",
            "normalized_key": "波特率",
            "resolved": False,  # 尚未做 BM25 搜索
        }
    """
    intent = {
        "query_keyword": "",
        "extracted_value": "",
        "normalized_key": "",
        "resolved": False,
        "bm25_hits": 0,
    }

    # ── Step 1: 从 query 中提取候选数字 ──
    query_numbers = re.findall(r'\b(\d{2,})\b', query)
    if query_numbers:
        intent["extracted_value"] = query_numbers[0]

    # ── Step 2: 用 LLM 提取属性关键词（若可用）──
    if llm_call_fn:
        try:
            prompt = _ATTRIBUTE_INTENT_PROMPT.format(query=query)
            response = llm_call_fn(prompt, max_tokens=128, temperature=0.0)
            for line in response.strip().split("\n"):
                line = line.strip()
                if line.startswith("属性关键词:") or line.startswith("属性关键词："):
                    intent["query_keyword"] = line.split(":", 1)[-1].split("：", 1)[-1].strip()
                elif line.startswith("查询数值:") or line.startswith("查询数值："):
                    val = line.split(":", 1)[-1].split("：", 1)[-1].strip()
                    if val and val != "无":
                        intent["extracted_value"] = val
        except Exception as e:
            logger.debug(f"LLM 属性意图提取失败，回退正则: {e}")

    # ── Step 3: 正则回退（LLM 不可用或失败时）──
    if not intent["query_keyword"]:
        query_lower = query.lower()
        for pattern_kw, normalized in [
            ("端口", "端口号"), ("port", "端口号"),
            ("波特率", "波特率"), ("baud", "波特率"),
            ("密码", "密码"), ("password", "密码"),
            ("ip", "IP地址"), ("地址", "IP地址"),
            ("速率", "速率"), ("频率", "频率"),
            ("超时", "超时"), ("周期", "周期"),
        ]:
            if pattern_kw in query_lower:
                intent["query_keyword"] = pattern_kw
                intent["normalized_key"] = normalized
                break
        if not intent["query_keyword"] and intent["extracted_value"]:
            intent["query_keyword"] = "参数"
            intent["normalized_key"] = "参数"

    if not intent["normalized_key"]:
        intent["normalized_key"] = intent["query_keyword"]

    return intent


def resolve_attribute_value(
    intent: Dict,
    bm25_search_fn=None,
    vector_store=None,
    product_id: Optional[str] = None,
) -> Optional[str]:
    """
    基于属性意图，用 BM25 精准搜索 ChromaDB，提取实际属性值。

    Args:
        intent: extract_attribute_intent 的输出
        bm25_search_fn: BM25 搜索函数 (query, product_id, k) → [(Document, score), ...]
        vector_store: ChromaDB 实例（回退用）
        product_id: 产品 ID 限制搜索范围

    Returns:
        属性描述字符串 "JAKA: Modbus TCP 端口号 = 6502"，或 None
    """
    keyword = intent.get("query_keyword", "")
    expected_value = intent.get("extracted_value", "")

    if not keyword and not expected_value:
        return None

    # ── BM25 精准搜索 ──
    bm25_docs = []
    search_query = keyword if keyword else expected_value

    if bm25_search_fn:
        try:
            bm25_docs = bm25_search_fn(search_query, product_id=product_id, k=5)
        except Exception as e:
            logger.debug(f"BM25 属性搜索失败: {e}")

    # ── 在返回切片中提取属性值 ──
    best_match = None
    best_score = 0

    for doc, bm25_score in bm25_docs:
        text = doc.page_content if hasattr(doc, "page_content") else str(doc)

        # 尝试所有正则模式
        for pat in _ATTR_VALUE_PATTERNS:
            for m in pat.finditer(text):
                groups = m.groups()
                if len(groups) >= 2:
                    found_key, found_value = groups[0], groups[1]
                elif len(groups) == 1:
                    found_key, found_value = keyword, groups[0]
                else:
                    continue

                # 评分：关键词匹配 + 数字匹配
                score = bm25_score * 10
                if keyword.lower() in found_key.lower():
                    score += 5
                if expected_value and expected_value in found_value:
                    score += 10

                if score > best_score:
                    best_score = score
                    pid = (
                        doc.metadata.get("product_id", product_id or "?")
                        if hasattr(doc, "metadata") else (product_id or "?")
                    )
                    section = (
                        doc.metadata.get("section_id", "")
                        if hasattr(doc, "metadata") else ""
                    )
                    best_match = (pid, found_key.strip(), found_value.strip(), section)

    if best_match:
        pid, key, value, section = best_match
        intent["resolved"] = True
        intent["bm25_hits"] = len(bm25_docs)
        if section:
            return f"[属性检索] {pid}: {key} = {value}（出处: {section}）"
        return f"[属性检索] {pid}: {key} = {value}"
    elif expected_value:
        # BM25 未找到但 query 中有数值 → 返回弱信号
        intent["resolved"] = True
        pid = product_id or "?"
        return f"[属性检索·弱信号] {pid}: {keyword} = {expected_value}（来源: 用户查询中直接提及）"

    return None


def attribute_lookup(
    query: str,
    product_id: Optional[str] = None,
    *,
    llm_call_fn=None,
    bm25_search_fn=None,
    vector_store=None,
) -> Optional[str]:
    """
    一站式属性查询：提取意图 → BM25 搜索 → 提取值。

    这是替代静态 KV 表的主入口。SubGoalPlanner 和 _build_messages KV 通道
    均通过此函数进行属性解析。
    """
    try:
        intent = extract_attribute_intent(query, llm_call_fn=llm_call_fn)
        if not intent["query_keyword"] and not intent["extracted_value"]:
            return None
        return resolve_attribute_value(
            intent,
            bm25_search_fn=bm25_search_fn,
            vector_store=vector_store,
            product_id=product_id,
        )
    except Exception as e:
        logger.warning(f"attribute_lookup 失败: {e}")
        return None


# ============================================================
# 兼容层 — 保留旧 kv_extractor.lookup_attribute 接口
# ============================================================

def lookup_attribute_fallback(query: str, product_id: Optional[str] = None) -> Optional[str]:
    """
    回退到旧静态 KV 表（兼容层）。
    当 Dynamic AttributeIntent 不可用时使用。
    """
    try:
        from .kv_extractor import lookup_attribute as _old_lookup
        return _old_lookup(query, product_id=product_id)
    except Exception:
        return None
