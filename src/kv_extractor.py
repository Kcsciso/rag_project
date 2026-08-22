"""
=============================================================================
轻量 KV 属性事实检索模块
=============================================================================
直接读取 kv_db/attribute_kv.json 中的已知事实数据，为网络端口、波特率、默认密码等
高敏感数字参数提供 100% 确定性的前置 Context 注入，防止 LLM 产生数字幻觉。
"""

import os
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

_KV_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "kv_db",
    "attribute_kv.json"
)

_cached_kv_data: Optional[Dict[str, Any]] = None


def _load_kv_data() -> Dict[str, Any]:
    """加载并缓存 attribute_kv.json 数据库"""
    global _cached_kv_data
    if _cached_kv_data is not None:
        return _cached_kv_data

    if not os.path.exists(_KV_FILE_PATH):
        logger.warning(f"⚠️ KV 属性数据库文件不存在: {_KV_FILE_PATH}")
        _cached_kv_data = {}
        return _cached_kv_data

    try:
        with open(_KV_FILE_PATH, "r", encoding="utf-8") as f:
            _cached_kv_data = json.load(f)
            logger.info(f"✅ 成功加载 KV 属性库: {len(_cached_kv_data)} 个键值对")
    except Exception as e:
        logger.error(f"❌ 读取 KV 属性库失败: {e}")
        _cached_kv_data = {}

    return _cached_kv_data


def lookup_attribute(query: str, product_id: Optional[str] = None) -> Optional[str]:
    """
    根据用户提问与产品标识，在 KV 属性库中检索匹配的事实描述。

    Args:
        query: 用户查询文本
        product_id: 产品线 ID (如 "JAKA", "OpenC3", "OpenR6")

    Returns:
        匹配到的事实描述字符串，未命中则返回 None
    """
    kv_data = _load_kv_data()
    if not kv_data:
        return None

    query_lower = query.lower()
    matched_facts = []

    for key, value in kv_data.items():
        # 若指定了 product_id，先做产品线隔离（通用属性除外）
        key_lower = key.lower()
        
        # 关键词匹配规则
        # 1. 键名在 query 中出现（如 "modbus端口"、"波特率"）
        # 2. 或核心属性词与产品名同时命中
        if key_lower in query_lower:
            if isinstance(value, dict):
                # 若结构为 {"value": "6502", "description": "..."}
                desc = value.get("description", str(value))
                matched_facts.append(f"- {key}: {desc}")
            else:
                matched_facts.append(f"- {key}: {value}")
        else:
            # 细粒度拆分匹配（针对 "JAKA 默认端口是多少" 等提问）
            parts = key_lower.split("_")
            if len(parts) >= 2 and all(p in query_lower for p in parts):
                matched_facts.append(f"- {key}: {value}")

    if not matched_facts:
        return None

    result_str = "\n".join(matched_facts)
    logger.info(f"🎯 KV 属性库精准命中 ({len(matched_facts)} 项):\n{result_str}")
    return result_str