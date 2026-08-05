#!/usr/bin/env python3
"""
=============================================================================
PDF 结构化属性提取器 — KV Attribute Extractor (ADR-13)
=============================================================================

背景：
  JAKA 等产品 PDF 中大量关键数字参数（端口号 6502、波特率 9600、默认密码等）
  存在于截图/图片中，pypdf/pdfplumber 无法提取。纯文本层仅包含描述性文字。
  这导致数字参数查询被 ADR-9/10 的硬拒答逻辑拦截（Context 中无 ≥2 位数字）。

设计目标：
  1. 从已索引的 ChromaDB 文本切片中，用正则提取 KV 属性对（键=值）。
  2. 支持手动补充截图中的已知数字参数（人工校准）。
  3. 输出为独立轻量 JSON KV 存储，不依赖向量检索。
  4. 集成到 RAG Pipeline 的预检索通道 — 硬拒答之前先查 KV 存储。

架构位置：
  rag_chat() / run_graph()
    → _kv_attribute_lookup(query, product_id)   ← 新增：查 KV 存储
    → 命中 → 注入 Context
    → 未命中 → 继续原有硬拒答逻辑

运行方式：
  # 构建/重建 KV 存储
  python src/kv_extractor.py --build

  # 查询
  python src/kv_extractor.py --query "JAKA Modbus 端口号"
=============================================================================
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("kv_extractor")

# ============================================================
# KV 存储路径
# ============================================================
_KV_STORE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "kv_db"
_KV_STORE_FILE = _KV_STORE_DIR / "attribute_kv.json"

# ============================================================
# 通用属性提取正则 — 零特定数值硬编码
# ============================================================

# 模式 1: "默认XX：YYY" 格式（密码、参数等）
_RE_DEFAULT_VALUE = re.compile(
    r'(?:默认|初始|预设)\s*'
    r'('
    r'(?:密码|端口号?|波特率|IP地址|地址|参数|值|用户名|账号|速率|频率)'
    r')[\s：:]*'
    r'([^\s，。,.\n]{1,50})',
    re.IGNORECASE,
)

# 模式 2: "XXX：YYY" 格式（数字参数表）
_RE_KV_PAIR = re.compile(
    r'('
    r'(?:端口号?|波特率|IP地址|速率|频率|超时|周期|间隔|'
    r'默认密码|管理员密码|操作员|技术员|密码|用户名|账号|'
    r'数据位|停止位|校验位|从站地址|从站节点号|节点号|站号|通道)'
    r')[\s：:]*'
    r'([^\s，。,.\n]{1,80})',
    re.IGNORECASE,
)

# 模式 3: "支持最大 XXXXX" 格式
_RE_MAX_VALUE = re.compile(
    r'(支持\s*(?:最大|最高|最多|最低|最小)?\s*)'
    r'('
    r'(?:波特率|速率|频率|超时|周期|间隔)'
    r')[\s：:]*'
    r'([\d,，]+)',
    re.IGNORECASE,
)

# 模式 4: 括号内数字标注 — "端口号 (6502)" 等
_RE_PAREN_VALUE = re.compile(
    r'('
    r'(?:端口号?|波特率|速率|频率|密码)'
    r')\s*[\(（]\s*'
    r'([\d]+)'
    r'\s*[\)）]',
    re.IGNORECASE,
)

# ============================================================
# 手动校准数据 — 截图/图片中的已知数字
# ============================================================
# 这些值从 PDF 截图中人工确认，文本提取无法获得。
# 格式: {product_id: {key: value}}
_MANUAL_CALIBRATION: Dict[str, Dict[str, str]] = {
    "JAKA": {
        # Modbus TCP 端口号 — 截图第 3.1.5.1 节
        "Modbus TCP 端口号": "6502",
        "Modbus TCP Server 端口": "6502",
        # RS485 / Modbus RTU 默认波特率
        "Modbus RTU 默认波特率": "9600",
        "RS485 默认波特率": "9600",
        # 管理员默认密码（文本中已有，二次确认）
        "管理员默认密码": "jakazuadmin",
        "技术员默认密码": "0000",
        "操作员默认密码": "0",
        # 从站地址
        "Modbus 默认从站地址": "1",
    },
}

# ============================================================
# 归一化映射 — 统一不同表述
# ============================================================
_KEY_ALIASES = {
    "端口": "端口号",
    "port": "端口号",
    "端口号": "端口号",
    "modbus端口": "Modbus TCP 端口号",
    "modbus tcp端口": "Modbus TCP 端口号",
    "modbus rtu波特率": "Modbus RTU 默认波特率",
    "默认波特率": "RS485 默认波特率",
    "波特率": "RS485 默认波特率",
    "baud": "RS485 默认波特率",
    "9600": "RS485 默认波特率",
    "管理员密码": "管理员默认密码",
    "admin密码": "管理员默认密码",
    "登录密码": "管理员默认密码",
    "密码": "管理员默认密码",
}


def _normalize_key(key: str) -> str:
    """将各种表述归一化为标准属性名"""
    key_lower = key.lower().strip().rstrip("：:")
    return _KEY_ALIASES.get(key_lower, key_lower)


# ============================================================
# 提取引擎
# ============================================================

def extract_kv_from_text(text: str, source: str = "", section: str = "") -> List[Dict]:
    """
    从文本中提取所有 KV 属性对。

    Returns:
        [{"product_id": "JAKA", "key": "端口号", "value": "6502",
          "source": "xxx.pdf", "section": "3.1.5.1", "method": "regex"}, ...]
    """
    entries = []

    # 推断 product_id
    product_id = _infer_product(source)

    def _add(key: str, value: str, method: str):
        key = key.strip().rstrip("：:")
        value = value.strip().rstrip("：:。，,;；")
        if not key or not value:
            return
        if len(value) > 60:  # 太长的不是参数值
            return
        normalized = _normalize_key(key)
        entries.append({
            "product_id": product_id,
            "key": normalized,
            "value": value,
            "raw_key": key,
            "source": source,
            "section": section,
            "method": method,
        })

    # 模式 1: 默认XX：YYY
    for m in _RE_DEFAULT_VALUE.finditer(text):
        _add(m.group(1), m.group(2), "default_pattern")

    # 模式 2: XX：YYY
    for m in _RE_KV_PAIR.finditer(text):
        _add(m.group(1), m.group(2), "kv_pattern")

    # 模式 3: 支持最大 XXXXX
    for m in _RE_MAX_VALUE.finditer(text):
        _add(f"{m.group(2)}（最大）", m.group(3), "max_pattern")

    # 模式 4: 括号数字
    for m in _RE_PAREN_VALUE.finditer(text):
        _add(m.group(1), m.group(2), "paren_pattern")

    # 去重：同一 key 保留最长 value
    deduped = {}
    for e in entries:
        k = (e["product_id"], e["key"])
        if k not in deduped or len(e["value"]) > len(deduped[k]["value"]):
            deduped[k] = e

    return list(deduped.values())


def _infer_product(source: str) -> str:
    """从文件名推断 product_id"""
    source_lower = source.lower()
    if "jaka" in source_lower or "zu" in source_lower:
        return "JAKA"
    if "openc3" in source_lower or "六轴" in source_lower:
        return "OpenC3"
    if "openr6" in source_lower or "r6" in source_lower:
        return "OpenR6"
    return "unknown"


# ============================================================
# KV 存储管理
# ============================================================

def build_kv_store(force: bool = False) -> Dict[str, Dict[str, str]]:
    """
    从 ChromaDB + 手动校准数据构建完整 KV 存储。

    Returns:
        {product_id: {normalized_key: value}}
    """
    _KV_STORE_DIR.mkdir(parents=True, exist_ok=True)

    store: Dict[str, Dict[str, str]] = {}

    # ── Layer 1: 从 ChromaDB 文本切片提取 ──
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.vector_store import load_vector_store
        from src.config import CHROMA_PERSIST_DIR

        vs = load_vector_store(CHROMA_PERSIST_DIR)
        if vs:
            collection = vs._collection
            all_data = collection.get(include=["documents", "metadatas"])
            logger.info(f"从 ChromaDB 加载 {len(all_data['ids'])} 个切片，开始提取 KV...")

            for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
                source = meta.get("source", "")
                section = meta.get("section_id", "")
                entries = extract_kv_from_text(doc, source=source, section=section)
                for entry in entries:
                    pid = entry["product_id"]
                    if pid not in store:
                        store[pid] = {}
                    store[pid][entry["key"]] = entry["value"]

            logger.info(f"文本提取完成: {sum(len(v) for v in store.values())} 条属性")
    except Exception as e:
        logger.warning(f"ChromaDB 提取失败（将仅使用手动校准数据）: {e}")

    # ── Layer 2: 手动校准数据覆盖（强制覆盖文本提取值）──
    for pid, attrs in _MANUAL_CALIBRATION.items():
        if pid not in store:
            store[pid] = {}
        for key, value in attrs.items():
            store[pid][key] = value  # 覆盖 Layer 1 同名 key
    # 清理明显的噪声条目（value 长度 > 20 或含特殊字符）
    for pid in list(store.keys()):
        cleaned = {}
        for k, v in store[pid].items():
            if len(v) <= 30 and not any(c in v for c in "（）【】《》()[]"):
                cleaned[k] = v
        store[pid] = cleaned
    logger.info(f"手动校准合并完成: 总计 {sum(len(v) for v in store.values())} 条属性")

    # ── 持久化 ──
    with open(_KV_STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)
    logger.info(f"KV 存储已写入: {_KV_STORE_FILE} ({_KV_STORE_FILE.stat().st_size} bytes)")

    return store


def load_kv_store() -> Dict[str, Dict[str, str]]:
    """加载 KV 存储（若不存在则自动构建）"""
    if not _KV_STORE_FILE.exists():
        logger.info("KV 存储不存在，自动构建...")
        return build_kv_store()

    with open(_KV_STORE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def lookup_attribute(
    query: str,
    product_id: Optional[str] = None,
) -> Optional[str]:
    """
    从 KV 存储中检索与 query 匹配的属性值。

    匹配策略（按优先级）：
      1. query 中包含 KV key 的完整字符串 → 得分 15
      2. query 中包含 KV value 的数字 → 得分 12
      3. query 和 KV key 的关键词有交集 → 得分 6
      4. query 中包含纯数字（如 6502）且 KV value 匹配 → 得分 8
    """
    store = load_kv_store()

    if product_id and product_id in store:
        candidates = [(product_id, store[product_id])]
    else:
        candidates = list(store.items())

    query_lower = query.lower()
    matches = []

    # 从 query 中提取数字 token
    query_numbers = re.findall(r'\d{3,}', query)

    for pid, attrs in candidates:
        for key, value in attrs.items():
            key_lower = key.lower()
            value_lower = value.lower()
            score = 0

            # 规则 1: query 包含完整 key 关键词
            key_tokens = re.split(r'[\s：:]+', key_lower)
            key_tokens = [t for t in key_tokens if len(t) >= 2]
            matched_tokens = sum(1 for t in key_tokens if t in query_lower)
            if matched_tokens >= 1:
                score = max(score, 6 + matched_tokens * 2)

            # 规则 2: query 数字匹配 value 数字
            for qnum in query_numbers:
                if qnum in value:
                    score = max(score, 12)
                    break

            # 规则 3: query 中包含 value 本身
            if len(value) >= 3 and value_lower in query_lower:
                score = max(score, 8)

            # 规则 4: 别名映射匹配
            for alias, target in _KEY_ALIASES.items():
                if alias in query_lower and target == key:
                    score = max(score, 10)
                    break

            if score > 0:
                matches.append((pid, key, value, score, matched_tokens))

    if not matches:
        return None

    # 🔴 v25: 同分时按 query 关键词命中数决胜
    #（"波特率 9600" 类查询优先 "Modbus RTU 默认波特率" 而非 "RS485 默认波特率"）
    matches.sort(key=lambda x: (x[3], x[4]), reverse=True)
    best_pid, best_key, best_value, _, _ = matches[0]

    # 构建返回结果 — 包含 product context
    return f"[KV属性] {best_pid}: {best_key} = {best_value}"


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if "--build" in sys.argv:
        store = build_kv_store(force=True)
        for pid, attrs in store.items():
            print(f"\n{'='*40}\n  {pid}\n{'='*40}")
            for k, v in sorted(attrs.items()):
                print(f"  {k}: {v}")

    elif "--query" in sys.argv:
        idx = sys.argv.index("--query")
        q = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "端口号"
        pid = None
        if "--product" in sys.argv:
            pid_idx = sys.argv.index("--product")
            pid = sys.argv[pid_idx + 1] if pid_idx + 1 < len(sys.argv) else None

        result = lookup_attribute(q, product_id=pid)
        if result:
            print(f"✅ 命中: {result}")
        else:
            print(f"❌ 未找到匹配属性")

    else:
        print("用法: python kv_extractor.py --build | --query <关键词> [--product <产品ID>]")
        # 默认：构建并显示
        store = build_kv_store(force="--force" in sys.argv)
        for pid, attrs in store.items():
            print(f"\n{pid}: {len(attrs)} 条属性")
            for k, v in sorted(attrs.items()):
                print(f"  {k}: {v}")
