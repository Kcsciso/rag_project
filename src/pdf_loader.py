"""
=============================================================================
PDF 加载与文本分块模块（v4 — ADR-15 切片机制升级）
=============================================================================

v4 新增策略:
  - API-Level Atomic Chunking: SDK 函数定义+代码示例保持在同一原子块
  - Header-Aware Chunking: 标题层级树感知切分 + 面包屑注入
  - Parent-Child Dual Indexing: H2 父层(粗召回) + H3/H4 子层(精匹配)

v3 兼容: load_pdfs_from_directory() 保持不变，供回退使用。
v4 入口: load_pdfs_v4_dual() → 返回 (parent_docs, child_docs) 元组。
=============================================================================
"""

import bisect
import logging
import os
import re
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import PRODUCT_MAPPING_RULES

logger = logging.getLogger(__name__)


def _resolve_product_id_from_filename(filename: str) -> str:
    """
    根据文件名解析对应的产品标识 — 强归一化映射。

    优先级:
      1. PRODUCT_MAPPING_RULES 精确匹配（保留向后兼容）
      2. 内置归一化回退: case-insensitive 匹配已知产品名
      3. 全部未命中 → 返回 "General" + WARNING

    严禁返回 null / "unknown" / 小写杂乱的 product_id。

    Args:
        filename: PDF 文件名

    Returns:
        product_id 字符串（OpenC3 / JAKA / OpenR6 / General）
    """
    filename_lower = filename.lower()

    # 第 1 层: PRODUCT_MAPPING_RULES 精确匹配
    for rule in PRODUCT_MAPPING_RULES:
        for pattern in rule["filename_patterns"]:
            if pattern.lower() in filename_lower:
                logger.info(
                    f"🏷️  产品识别: '{filename}' → product_id='{rule['product_id']}' "
                    f"(命中模式: '{pattern}')"
                )
                return rule["product_id"]

    # 第 2 层: 内置归一化回退（case-insensitive 直接子串匹配）
    _NORMALIZED_PRODUCT_IDS = {
        "openc3": "OpenC3",
        "openr6": "OpenR6",
        "jaka": "JAKA",
        "zu": "JAKA",
        "minicab": "JAKA",
        "collrob": "OpenC3",
        "六轴": "OpenC3",
        "windows系统": "OpenR6",
        "py_dll": "OpenR6",
    }
    for key, pid in _NORMALIZED_PRODUCT_IDS.items():
        if key in filename_lower:
            logger.info(
                f"🏷️  产品识别(回退): '{filename}' → product_id='{pid}' "
                f"(命中关键词: '{key}')"
            )
            return pid

    logger.warning(f"⚠️  无法识别产品: '{filename}'，标记为 'General'")
    return "General"


def _resolve_doc_type(product_id: str) -> str:
    """
    根据 product_id 解析文档类型（双轨制核心）。

    - JAKA → "gui_app" (纯界面操作手册，严禁生成 SDK 代码)
    - OpenC3 / OpenR6 → "c_sdk" (C 语言动态库 SDK，API 即答案)
    """
    _DOC_TYPE_MAP = {
        "JAKA": "gui_app",
        "OpenC3": "c_sdk",
        "OpenR6": "c_sdk",
    }
    return _DOC_TYPE_MAP.get(product_id, "general")


def extract_text_from_pdf(file_path: str) -> str:
    """
    从单个 PDF 文件中提取全部文本内容。

    【技术细节】
    - 使用 pypdf（纯 Python 实现，零 C 扩展依赖）逐页读取
    - 跳过无法提取文本的页面（如图片型 PDF 页面）
    - 在页面之间插入换行符以保持段落结构

    【局限说明】
    - pypdf 只能提取"文字型 PDF"中的文本
    - 对于"扫描型 PDF"（图片中的文字），需要 OCR 引擎（如 Tesseract），
      但 OCR 会引入额外依赖，违反本项目"不升级核心依赖"的原则
    - 在实际应用中，建议优先使用文字型 PDF

    Args:
        file_path: PDF 文件的绝对路径

    Returns:
        整个 PDF 的纯文本字符串，页面之间用 "\n\n" 分隔
    """
    reader = PdfReader(file_path)
    full_text_parts = []

    for page_idx, page in enumerate(reader.pages):
        # 逐页提取文本
        page_text = page.extract_text()
        if page_text:
            full_text_parts.append(page_text.strip())

    # 拼接所有页面，页面间用双换行分隔
    return "\n\n".join(full_text_parts)

def debug_print_chunks(chunks: List[Document], max_show: int = 3):
    """
    【DEBUG 辅助函数】详细打印切片内容，帮助观察：
    1. 每个 Chunk 的字符数是否超出限制
    2. Chunk 之间的 Overlap（重叠区域）是否正常工作
    3. Metadata 是否被完好保留
    """
    print("\n" + "=" * 25 + " [DEBUG: 切片效果观察] " + "=" * 25)
    total_chunks = len(chunks)
    show_count = min(max_show, total_chunks)

    for i in range(show_count):
        chunk = chunks[i]
        content = chunk.page_content.strip()
        print(f"\n🧩 [Chunk {i + 1}/{total_chunks}] (字符数: {len(content)})")
        print(f"📌 元数据 (Metadata): {chunk.metadata}")
        print("📝 切片内容:")
        print("┌" + "─" * 60)
        # 逐行打印并加缩进，避免格式混乱
        for line in content.split("\n"):
            print(f"│ {line}")
        print("└" + "─" * 60)

        # 打印与下一个 Chunk 的重叠交叉区（验证 chunk_overlap）
        if i < show_count - 1 and len(chunks) > 1:
            next_content = chunks[i + 1].page_content.strip()
            # 简单寻找末尾与下一个开头重合的部分
            overlap_preview = content[-60:]  # 取当前 chunk 结尾 60 字
            print(f"🔍 [与 Chunk {i + 2} 的交界预览 (尾部)]: ...{overlap_preview}")


def load_pdfs_from_directory(
    data_dir: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    批量加载指定目录下的所有 PDF 文件，并进行文本分块。

    【完整流程】
    1. 扫描目录 → 找到所有 .pdf 文件
    2. 逐个调用 extract_text_from_pdf() 提取全文
    3. 给每个文档附加元数据（来源文件名）
    4. 使用 RecursiveCharacterTextSplitter 将所有文档统一分块
    5. 返回分块后的 Document 列表

    Args:
        data_dir: 存放 PDF 文件的目录路径（通常是 data/）
        chunk_size: 每个文本块的最大字符数（默认 500）
        chunk_overlap: 相邻块之间的重叠字符数（默认 50）

    Returns:
        List[Document]: 分块后的文档列表，每个 Document 的 metadata
                        中包含 "source"（来源文件名）字段
    """
    # ---- 第 1 步：扫描 PDF 文件 ----
    pdf_files = [
        f for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print(f"[pdf_loader] ⚠️  目录 '{data_dir}' 中未找到任何 PDF 文件。")
        return []

    print(f"[pdf_loader] 📄 发现 {len(pdf_files)} 个 PDF 文件，开始加载...")

    # ---- 第 2-3 步：提取文本 + 附加元数据 ----
    all_documents = []
    for pdf_file in pdf_files:
        file_path = os.path.join(data_dir, pdf_file)
        try:
            text = extract_text_from_pdf(file_path)
            if text.strip():
                # 创建一个 Document 对象，metadata 记录来源文件名和产品标识
                # 🏷️ 产品打标：根据文件名自动识别产品线（OpenR6 / OpenC3 / ...）
                product_id = _resolve_product_id_from_filename(pdf_file)
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": pdf_file,
                        "product_id": product_id,
                    }
                )
                all_documents.append(doc)
                logger.info(f"   ✅ {pdf_file}: {len(text)} 字符 (product_id={product_id})")
            else:
                print(f"[pdf_loader]   ⚠️  {pdf_file}: 未提取到有效文本（可能是扫描件）")
        except Exception as e:
            print(f"[pdf_loader]   ❌ {pdf_file}: 解析失败 — {e}")

    if not all_documents:
        print("[pdf_loader] ❌ 所有 PDF 均无法提取有效文本。")
        return []

    # ---- 第 4 步：递归文本分块 ----
    # 这里的分隔符列表体现了"由粗到细"的递归策略
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",    # 段落分隔（最高优先级）
            "\n",      # 换行
            "。",      # 中文句号
            ".",       # 英文句号
            "！",      # 中文感叹号
            "!",       # 英文感叹号
            "？",      # 中文问号
            "?",       # 英文问号
            "；",      # 中文分号
            ";",       # 英文分号
            "，",      # 中文逗号
            ",",       # 英文逗号
            " ",       # 空格（最低优先级）
        ],
        length_function=len,        # 使用字符数（而非 token 数）衡量长度
        is_separator_regex=False,   # 分隔符按字面匹配
    )

    # split_documents 会将每个长 Document 拆分为多个短 Document
    chunks = text_splitter.split_documents(all_documents)

    # ================================================================
    # 🔴 Section Injection: 自动提取章节标题并注入切片头部
    # ================================================================
    import bisect as _bisect
    import re as _re
    _HEADING_PATTERNS = [
        _re.compile(r'^(\d+(?:\.\d+)+)\s+(.+?)(?:\r?\n|$)', _re.MULTILINE),
        _re.compile(r'^(第[一二三四五六七八九十\d]+[章节])\s*(.+?)(?:\r?\n|$)', _re.MULTILINE),
        _re.compile(r'^([（(]?[一二三四五六七八九十]+[）)]?[\s、,，])\s*(.+?)(?:\r?\n|$)', _re.MULTILINE),
        _re.compile(r'^(#{1,4})\s+(.+?)(?:\r?\n|$)', _re.MULTILINE),
        _re.compile(r'^(?:[■□◆◇●○]|##?)\s*(?:(\d+(?:\.\d+)*)\s+)?(.+?)(?:\r?\n|$)', _re.MULTILINE),
    ]

    def _extract_headings_pdf(text: str) -> list:
        seen_positions = set()
        headings = []
        for pattern in _HEADING_PATTERNS:
            for m in pattern.finditer(text):
                pos = m.start()
                if pos in seen_positions:
                    continue
                seen_positions.add(pos)
                full = m.group(0).strip()
                if 3 <= len(full) <= 80:
                    headings.append((pos, full))
        headings.sort(key=lambda x: x[0])
        deduped = []
        for pos, title in headings:
            if deduped and pos - deduped[-1][0] < len(deduped[-1][1]) + 5:
                if len(title) > len(deduped[-1][1]):
                    deduped[-1] = (pos, title)
            else:
                deduped.append((pos, title))
        return deduped

    def _resolve_section_pdf(chunk_text: str, full_text: str, headings: list) -> str:
        if not headings:
            return ""
        fingerprint = chunk_text.strip()[:80]
        pos = full_text.find(fingerprint)
        if pos < 0:
            fingerprint = chunk_text.strip()[:40]
            pos = full_text.find(fingerprint)
        if pos < 0:
            return ""
        idx = _bisect.bisect_right([h[0] for h in headings], pos) - 1
        if idx >= 0:
            return headings[idx][1]
        return ""

    # ── 🔴 v3.0: 先做 Section Injection（此时 chunk 内容尚未被前缀污染）──
    section_injected_count = 0
    for original_doc in all_documents:
        full_text = original_doc.page_content
        headings = _extract_headings_pdf(full_text)
        if not headings:
            continue
        source = original_doc.metadata.get("source", "?")
        for chunk in chunks:
            if chunk.metadata.get("source") != source:
                continue
            section = _resolve_section_pdf(chunk.page_content, full_text, headings)
            if section:
                chunk.page_content = f"[章节: {section}]\n{chunk.page_content}"
                section_injected_count += 1

    # ── 🔴 v3.0: 再做 Document Prefixing — 统一 [文档: X | 章节: Y] 前缀 ──
    for chunk in chunks:
        source = chunk.metadata.get("source", "未知文档")
        has_section = chunk.page_content.startswith("[章节:")
        if has_section:
            # 提取已有章节前缀，合并为统一格式
            chunk.page_content = re.sub(
                r'^\[章节:\s*([^\]]+)\]\n',
                f'[文档: {source} | 章节: \\1]\n',
                chunk.page_content,
                count=1,
            )
        else:
            chunk.page_content = f"[文档: {source}]\n{chunk.page_content}"

    # 🔴 Header Injection: 为每个 chunk 提取 C 函数名并注入文本头部
    # 极大增强 Dense Vector 和 Sparse BM25 对特定函数名的敏感度
    _FUNC_RE = _re.compile(
        r'\b([a-zA-Z_][a-zA-Z0-9_]*_[a-zA-Z0-9_]+)\s*\(',  # snake_case 函数名( → "set_move_line("
        # 🔴 保留原始大小写，检索匹配端统一 .lower()
    )
    func_injected_count = 0
    for chunk in chunks:
        funcs = set()
        for m in _FUNC_RE.finditer(chunk.page_content):
            fname = m.group(1).strip('_')
            # 过滤掉明显不是 SDK 函数的短名
            if len(fname) >= 6 and ('_' in fname):
                funcs.add(fname)  # 🔴 保留原始大小写，不做 .lower()
        if funcs:
            funcs_sorted = sorted(funcs, key=lambda x: x.lower())[:10]  # 最多 10 个，避免过长
            header = f"[Functions: {', '.join(f.lower() for f in funcs_sorted)}]\n"  # header tag 用小写保持一致性
            chunk.page_content = header + chunk.page_content
            func_injected_count += 1

    print(f"[pdf_loader] ✅ 加载完成：{len(all_documents)} 个原始文档 → "
          f"{len(chunks)} 个文本块（chunk_size={chunk_size}, overlap={chunk_overlap}）"
          f" [Section Injected: {section_injected_count}, Header Injected: {func_injected_count}]")

    return chunks


# ============================================================
# v4 切片策略: API 原子化 + 标题感知 + 父子双层索引
# ============================================================

# ── 标题识别模式（兼容 Markdown 和非 Markdown 格式）──
_V4_HEADING_PATTERNS = [
    # 数字编号: 3.1.5 / 3.1.5.1 标题
    (re.compile(r'^(\d+(?:\.\d+){1,3})\s+(.{3,80}?)(?:\r?\n|$)', re.MULTILINE), 1),
    # 中文编号: 第一章 / 第一节
    (re.compile(r'^(第[一二三四五六七八九十\d]+[章节])\s*(.{3,80}?)(?:\r?\n|$)', re.MULTILINE), 1),
    # 中文序号: 一、/ （一）/ (一)
    (re.compile(r'^[（(]?[一二三四五六七八九十]+[）)]?\s*[、,，\s]\s*(.{3,80}?)(?:\r?\n|$)', re.MULTILINE), 1),
    # Markdown H: ## / ### / ####
    (re.compile(r'^(#{1,4})\s+(.{3,80}?)(?:\r?\n|$)', re.MULTILINE), 1),
    # 纯数字+点号: 1. / 2) 标题
    (re.compile(r'^(\d{1,2})[\.\)）]\s+(.{3,80}?)(?:\r?\n|$)', re.MULTILINE), 1),
    # 无编号中文标题: 概述 / 功能说明 / 安装步骤
    (re.compile(r'^([一-鿿]{2,20})\s*$', re.MULTILINE), 3),
]

# ── SDK 函数原子块识别 ──
_API_BLOCK_PATTERNS = [
    # Python ctypes 函数签名: lib.robot_movl.restype = c_int
    re.compile(
        r'(?:^|\n)((?:(?:lib|robot|py_dll|cdll)\.\w+\.(?:restype|argtypes)\s*=.+)|'
        r'(?:def\s+\w+\([^)]*\)\s*:)|'
        r'(?:```[\s\S]*?```))',
        re.MULTILINE,
    ),
    # C 函数声明: int robot_Power_on(void);
    re.compile(r'(?:^|\n)(\w+\s+\w+\([^)]*\)\s*;)', re.MULTILINE),
    # SDK 函数调用块: robot = CDLL(...) ... robot.xxx()
    re.compile(
        r'(?:robot\s*=\s*CDLL\([^)]+\)[\s\S]{0,500}?(?=\n\n|\n\s*\n|\Z))',
        re.MULTILINE,
    ),
    # ── 🔴 中文 SDK 手册格式: 函数名称 / 函数说明 / 功能描述 段落边界 ──
    # 匹配以 "函数名称 xxx" 开头直到下一个函数边界或空行的连续段落，
    # 防止 robot_movj 和 robot_movl 两个紧邻函数混入同一切片
    re.compile(
        r'(?:^|\n)(?:函数名称|函数名)\s+(\w+)\s*\(?[^)]*\)?\s*'
        r'(?:(?:功能描述|函数说明|功能说明|参数说明|返回值|返回值说明|'
        r'注意事项|调用示例|示例代码)[\s\S]{0,800}?)'
        r'(?=\n(?:函数名称|函数名)\s+\w+|'
        r'\n\s*\n(?:函数名称|函数名)|\n\n\n|\Z)',
        re.MULTILINE,
    ),
    # ── 🔴 中文 SDK 手册: 独立的 "函数名称" 行作为原子块起点 ──
    # 匹配单行 "函数名称 robot_xxx" 格式（简化版），确保至少该行不被切碎
    re.compile(
        r'^函数名称\s+(\w+(?:\([^)]*\))?)',
        re.MULTILINE,
    ),
]

# ── KV 参数行识别 ──
_KV_LINE_RE = re.compile(
    r'(?:'
    r'(?:默认|初始|预设)?\s*(?:端口号?|波特率|密码|用户名|IP\s*地址|从站地址|速率|频率|超时|周期)\s*[：:=]\s*\S+'
    r'|'
    r'\b(?:\d{1,3}\.){3}\d{1,3}\b'  # IP 地址
    r')',
    re.IGNORECASE,
)


# ── PDF 文本清洗：连字替换 + 函数括号空格规范化 ──
# PDF 内部编码中常见的连字（ligature）字符，在文本提取时不会被自动分解，
# 会导致 SDK 函数名匹配失败（如 "oﬀ" ≠ "off"）

_LIGATURE_MAP = {
    # Latin 连字 → 分解形式
    'ﬀ': 'ff',    # ﬀ (U+FB00) → ff
    'ﬁ': 'fi',    # ﬁ (U+FB01) → fi
    'ﬂ': 'fl',    # ﬂ (U+FB02) → fl
    'ﬃ': 'ffi',   # ﬃ (U+FB03) → ffi
    'ﬄ': 'ffl',   # ﬄ (U+FB04) → ffl
    'ﬅ': 'ft',    # ﬅ (U+FB05) → ft  (少见)
    'ﬆ': 'st',    # ﬆ (U+FB06) → st  (少见)
    # 常见 Unicode 等价替换
    '–': '-',     # – (EN DASH) → -
    '—': '--',    # — (EM DASH) → --
    '‘': "'",     # ' (LEFT SINGLE QUOTE) → '
    '’': "'",     # ' (RIGHT SINGLE QUOTE) → '
    '“': '"',     # " (LEFT DOUBLE QUOTE) → "
    '”': '"',     # " (RIGHT DOUBLE QUOTE) → "
    ' ': ' ',     # NBSP → 普通空格
}
_LIGATURE_TRANS_TABLE = str.maketrans(_LIGATURE_MAP)

# 函数括号内外多余空格清洗正则（3 种模式）
# ① "函数名 ( args )" → "函数名(args)"
_FUNC_PAREN_SPACE_RE = re.compile(
    r'(?:'
    r'(\w+)\s+\(\s+'          # 函数名后跟空格+左括号+空格 → 函数名(
    r'|'
    r'\s+\)'                   # 空格+右括号 → )
    r'|'
    r'\(\s+([^)\n]{1,200}?)\s+\)'  # ( 空格 args 空格 ) → (args)
    r')',
)

def _clean_pdf_text(text: str) -> str:
    """
    对 PDF 提取文本做通用规范化清洗（Universal Sanitizer）。

    处理步骤:
      1. PDF 常见乱码与控制字符剔除:
         - \\x00 (null), \\x0c (form feed), \\x0b (vertical tab)
         - \\x01-\\x08, \\x0e-\\x1f（除 \\t\\n 外的 ASCII 控制字符）
         - \\x0c 替换为 \\n\\n 以保留分页语义
      2. PDF 连字替换: oﬀ→off, ﬁ→fi, ﬂ→fl 等（防止函数名匹配失败）
      3. Unicode 符号规范化: EN/EM DASH, 智能引号, NBSP
      4. 函数括号内外空格清理（仅限括号内部，不破坏正文）:
         - "robot_Power_on ( )" → "robot_Power_on()"
         - 括号内部逗号前后多余空格归一化
      5. 收尾: 压缩 ≥3 个连续空行 → 2 个空行，保护 Markdown 缩进与代码换行

    Args:
        text: PDF 原始提取文本

    Returns:
        清洗后的规范化文本
    """
    if not text:
        return text

    # ── Step 1: 控制字符剔除 ──
    cleaned = text.replace('\x0c', '\n\n').replace('\x0b', '\n')
    cleaned = re.sub(r'[\x00-\x08\x0e-\x1f]', '', cleaned)

    # ── Step 2+3: 连字 + Unicode 符号替换（单次 str.translate）──
    cleaned = cleaned.translate(_LIGATURE_TRANS_TABLE)

    # ── Step 4: 函数括号内外空格清理（限定在括号内部，不触碰正文换行/缩进）──
    # 策略：在每个 (...) 括号对上做局部清洗，不影响括号外部的 Markdown 结构
    def _clean_parens(match: re.Match) -> str:
        """对单个 (args) 括号内容做局部清洗。兼容 1 或 2 个捕获组。"""
        groups = match.groups()
        if len(groups) >= 2 and groups[0] is not None:
            # 模式 "函数名 ( args )": group(1)=函数名, group(2)=参数
            prefix = groups[0]
            inner = groups[1] or ""
        elif len(groups) >= 1 and groups[0] is not None:
            # 模式 "( args )": group(1)=参数
            prefix = ""
            inner = groups[0]
        else:
            return match.group(0)  # 不应该出现，安全返回原文本
        inner = re.sub(r'\s*,\s*', ', ', inner)
        inner = re.sub(r'[ \t]+', ' ', inner).strip()
        if prefix:
            return f"{prefix}({inner})"
        else:
            return f"({inner})"

    # 4a: "函数名 ( args )" → "函数名(args)"（2 个捕获组）
    cleaned = re.sub(r'(\w+)\s+\(\s*([^)]*?)\s*\)', _clean_parens, cleaned)
    # 4b: 处理剩余的 "( args )"（1 个捕获组）
    cleaned = re.sub(r'\(\s*([^)]*?)\s*\)', _clean_parens, cleaned)

    # ── Step 5: 收尾 — 仅压缩 ≥3 个连续空行，保护缩进与正文换行 ──
    cleaned = re.sub(r'\n{4,}', '\n\n\n', cleaned)  # 4+ → 3
    cleaned = re.sub(r'\n{3}', '\n\n', cleaned)       # 3 → 2

    logger.debug(f"🧹 PDF 文本清洗: {len(text)} → {len(cleaned)} 字符"
                 f" (连字={text != cleaned})")
    return cleaned


def _v4_extract_headings(text: str) -> List[Tuple[int, str, int]]:
    """
    从文本中提取所有标题及其层级。

    兼容数字编号（3.1.5）、Markdown（##）、中文序号（一、）等多种格式。

    Returns:
        [(position, title_text, level), ...]
        level: 1=H1, 2=H2, 3=H3, 4=H4
    """
    headings = []
    seen_positions = set()

    for pattern, base_level in _V4_HEADING_PATTERNS:
        for m in pattern.finditer(text):
            pos = m.start()
            if pos in seen_positions:
                continue
            seen_positions.add(pos)
            full = m.group(0).strip()
            if not full or len(full) < 3 or len(full) > 85:
                continue

            # 推断层级
            groups = m.groups()
            if len(groups) >= 2 and groups[0] and groups[1]:
                title_num = groups[0]
                # "3.1.5" → 层级 = 点号数量 + 1
                dots = title_num.count('.')
                if dots >= 1:
                    level = min(dots + 1, 4)  # 最多 4 级
                elif title_num.startswith('#'):
                    level = min(len(title_num), 4)
                elif '章' in title_num or '节' in title_num:
                    level = 1 if '章' in title_num else 2
                elif base_level >= 3:
                    level = base_level
                else:
                    level = base_level
            else:
                level = base_level

            headings.append((pos, full, level))

    # 按位置排序 + 去重
    headings.sort(key=lambda x: x[0])
    deduped = []
    for pos, title, level in headings:
        if deduped and pos - deduped[-1][0] < len(deduped[-1][1]) + 5:
            # 重叠 → 保留更长的
            if len(title) > len(deduped[-1][1]):
                deduped[-1] = (pos, title, level)
        else:
            deduped.append((pos, title, level))
    return deduped


def _v4_build_breadcrumb(headings: List[Tuple[int, str, int]], chunk_pos: int, chunk_end: int) -> str:
    """
    为给定位置范围构建层级面包屑。

    例: [路径: JAKA Zu APP > 硬件与通讯 > Modbus 通讯设置]

    确保即使数字编号格式（3.1.5）也能正确生成面包屑。
    """
    # 找到该位置之前最近的各级标题
    path_parts = []
    current_level = 0
    for pos, title, level in headings:
        if pos > chunk_pos:
            break
        if level > current_level:
            path_parts.append(title)
            current_level = level
        elif level <= current_level and path_parts:
            # 同级或上级标题 → 弹出并替换
            while len(path_parts) >= max(level, 1):
                path_parts.pop()
            path_parts.append(title)
            current_level = level

    if not path_parts:
        return ""

    return " > ".join(p for p in path_parts if p)


def _v4_extract_api_blocks(text: str) -> List[Tuple[int, int, str]]:
    """
    预提取 SDK API 原子块，标记为不可分割区域。

    支持多种 SDK 文档格式:
      - Python ctypes 代码块
      - C 函数声明
      - CDLL 加载块
      - 中文 SDK 手册格式（函数名称 + 功能描述段落）

    Returns:
        [(start, end, block_label), ...]
        例如: [(1450, 1680, "API: robot_movl")]
    """
    blocks = []
    for pat in _API_BLOCK_PATTERNS:
        for m in pat.finditer(text):
            start, end = m.start(), m.end()
            block_text = m.group(0)
            # 提取函数名作为 label — 支持多种命名风格
            func_match = re.search(
                r'(?:robot_|set_|get_)\w+|'          # snake_case SDK 函数
                r'(?<=函数名称\s)\w+|'                 # 中文 "函数名称 xxx"
                r'(?<=函数名\s)\w+',                   # 中文 "函数名 xxx"
                block_text, re.IGNORECASE,
            )
            if func_match:
                label = f"API: {func_match.group(0)}"
            elif block_text.strip().startswith("函数名称"):
                # fallback: 从中文标签行提取
                name_m = re.search(r'函数名称\s+(\w+)', block_text)
                label = f"API: {name_m.group(1)}" if name_m else "API: block"
            else:
                label = "API: block"
            blocks.append((start, end, label))
    blocks.sort(key=lambda x: x[0])
    # 合并重叠块
    merged = []
    for b in blocks:
        if merged and b[0] < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b[1]), merged[-1][2])
        else:
            merged.append(b)
    return merged


# ── v4 OCR: 多模态图片文本抽取 + 通用页面级提取器 (ADR-16 增强) ──

_ocr_engine_cache = None
_ocr_available_cache = None


def _v4_get_ocr_engine():
    """懒加载 RapidOCR ONNX 引擎（纯 CPU，约 15MB 模型）。"""
    global _ocr_engine_cache, _ocr_available_cache
    if _ocr_available_cache is not None:
        return _ocr_engine_cache
    try:
        from rapidocr_onnxruntime import RapidOCR
        import numpy as np
        _ocr_engine_cache = RapidOCR()
        _ocr_engine_cache(np.zeros((100, 100, 3), dtype=np.uint8))
        _ocr_available_cache = True
        logger.info("✅ RapidOCR ONNX 引擎就绪（多模态图片文本抽取）")
    except Exception as e:
        logger.warning(f"RapidOCR 不可用，跳过图片文本抽取: {e}")
        _ocr_engine_cache = None
        _ocr_available_cache = False
    return _ocr_engine_cache


# ── 页面级字符密度阈值：低于此值判定为扫描件/纯图页，强制 OCR ──
_PAGE_DENSITY_THRESHOLD = 30


def _v4_extract_text_universal(pdf_path: str) -> Tuple[str, int, int]:
    """
    通用 PDF 文本提取器 — 逐页字符密度检测 + OCR 归位融合 + 章节上下文继承。

    算法:
      ① 使用 PyMuPDF 逐页提取文本
      ② 维护全局 Last Known Header 追踪器：
         - 遍历过程中记录最近遇到的章节标题（匹配 _V4_HEADING_PATTERNS）
         - 当后续低密度页触发 OCR 时，OCR 文字自动继承前置章节的 [路径] 和 [章节]
      ③ 若某页有效字符 < _PAGE_DENSITY_THRESHOLD (30 字) → 判定为扫描件/纯图页
         → 自动对该页所有 >100×100 px 的图片执行 RapidOCR 补漏识别
      ④ OCR 文字直接融入当前页面内容区域，并注入 Last Known Header 上下文，
         而非追加到文档末尾 — 确保 OCR 文本继承所在页面的章节层级
      ⑤ 各页之间保留双换行分隔

    Args:
        pdf_path: PDF 文件路径

    Returns:
        (full_text, total_pages, ocr_pages):
          - full_text: 融合 OCR 后的完整文本（OCR 已归位）
          - total_pages: 总页数
          - ocr_pages: 触发 OCR 补漏的页数
    """
    ocr = _v4_get_ocr_engine()

    try:
        import fitz
        import numpy as np
        from PIL import Image
        import io
    except ImportError:
        logger.warning("PyMuPDF 不可用，回退 pypdf 基础提取")
        text = extract_text_from_pdf(pdf_path)
        return text, 1, 0

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"fitz 打开失败: {pdf_path}: {e}")
        return "", 0, 0

    total_pages = len(doc)
    page_texts: List[str] = []
    ocr_pages = 0
    total_ocr_chars = 0

    # ── Last Known Header 追踪器 ──
    # 格式: {"number": "3.1.5", "title": "3.1.5 Modbus 通讯设置",
    #        "path": "JAKA Zu APP > 硬件与通讯 > Modbus 通讯设置", "level": 3}
    last_header: dict = {"number": "", "title": "", "path": "", "level": 0}

    # 预编译标题正则（与 _V4_HEADING_PATTERNS 兼容）
    _HEADER_TRACK_RE = re.compile(
        r'(?:^|\n)\s*('
        r'\d+(?:\.\d+){1,3}\s+.+?'          # 数字编号: 3.1.5 标题
        r'|第[一二三四五六七八九十\d]+[章节]\s*.+?'  # 中文编号: 第一章
        r'|[（(]?[一二三四五六七八九十]+[）)]?\s*[、,，\s].+?'  # 中文序号
        r'|#{1,4}\s+.+?'                     # Markdown H
        r'|\d{1,2}[\.\)）]\s+.+?'             # 纯数字+点号
        r')(?:\r?\n|$)',
        re.MULTILINE,
    )

    def _try_update_header(page_text: str, page_idx: int):
        """从页面文本中提取标题，更新 Last Known Header。"""
        matches = list(_HEADER_TRACK_RE.finditer(page_text))
        if matches:
            # 取本页最后一个标题作为当前章节锚点
            last_match = matches[-1]
            raw_title = last_match.group(1).strip()
            if 4 <= len(raw_title) <= 85:
                # 构建面包屑路径
                path_parts = [raw_title]
                if last_header["number"]:
                    # 简单层级推断：数字编号深度 + 中文章/节判定
                    num_match = re.match(r'^(\d+(?:\.\d+)*)', raw_title)
                    if num_match:
                        num_dots = num_match.group(1).count('.')
                        # 继承上级路径中层级较低的项
                        if last_header.get("_prev_path"):
                            deeper = last_header["_prev_path"][:num_dots]
                            path_parts = deeper + [raw_title]
                last_header["_prev_path"] = path_parts.copy()
                last_header["title"] = raw_title
                last_header["path"] = " > ".join(path_parts)
                last_header["level"] = 1 + (raw_title.count('.') if '.' in raw_title else 0)

    for page_idx in range(total_pages):
        page = doc[page_idx]
        page_text = page.get_text("text") or ""

        # ── Step 1: 字符密度检测 ──
        effective_chars = len(re.sub(r'\s', '', page_text))
        needs_ocr = effective_chars < _PAGE_DENSITY_THRESHOLD

        # ── Step 2: 更新标题追踪器（正常密度页）──
        if not needs_ocr and effective_chars >= _PAGE_DENSITY_THRESHOLD:
            _try_update_header(page_text, page_idx)

        # ── Step 3: OCR 补漏（仅低密度页触发）──
        ocr_lines = []
        if needs_ocr and ocr is not None:
            image_list = page.get_images(full=True)
            for img_info in (image_list or []):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    if base_image is None:
                        continue
                    image_bytes = base_image.get("image")
                    if not image_bytes:
                        continue

                    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    h, w = pil_img.size[1], pil_img.size[0]
                    if h < 100 or w < 100:
                        continue

                    np_img = np.array(pil_img)
                    result = ocr(np_img)
                    if result is None:
                        continue

                    lines = []
                    for item in result:
                        text = str(item[1]).strip()
                        if text and len(text) >= 2:
                            lines.append(text)

                    if lines:
                        ocr_lines.append(" | ".join(lines))
                        total_ocr_chars += sum(len(l) for l in lines)
                except Exception:
                    continue

            if ocr_lines:
                ocr_pages += 1

        # ── Step 4: 页面内容组装 — OCR 归位 + 章节上下文继承 ──
        page_parts = []
        if page_text.strip():
            page_parts.append(page_text.strip())

        if ocr_lines:
            # 🔴 章节上下文继承：OCR 文字继承 Last Known Header
            section_header = ""
            if last_header.get("title"):
                section_header = (
                    f"\n[路径: {last_header['path']}]"
                    f"\n[章节: {last_header['title']}]"
                )

            ocr_header = (
                f"[OCR补漏: page={page_idx + 1}, chars={effective_chars}<{_PAGE_DENSITY_THRESHOLD}]"
                f"{section_header}"
            )
            page_parts.append(ocr_header + "\n" + "\n".join(ocr_lines))

        if page_parts:
            page_texts.append("\n".join(page_parts))
        elif page_text.strip():
            page_texts.append(page_text.strip())

    doc.close()

    full_text = "\n\n".join(page_texts)

    if ocr_pages > 0:
        logger.info(
            f"  🖼️  通用提取: {total_pages} 页, 低密度触发 OCR: {ocr_pages} 页 "
            f"({total_ocr_chars} OCR 字符), 已归位融合 → ({pdf_path})"
        )
    else:
        logger.info(f"  📄 通用提取: {total_pages} 页, 无需 OCR → ({pdf_path})")

    return full_text, total_pages, ocr_pages


# ── 受保护区域正则: 代码块 ```...``` 与 Markdown 表格 |...| ──
_PROTECTED_BLOCK_RE = re.compile(
    r'(```[\s\S]*?```)'                              # 代码块
    r'|'                                              # 或
    r'((?:^\|.+\|[\s\S]*?)(?=\n\n|\n(?:[^|]|\Z)|\Z))',  # Markdown 表格
    re.MULTILINE,
)


def _v4_find_protected_ranges(text: str) -> List[Tuple[int, int, str]]:
    """
    扫描全文，标记不可切分的受保护区域（代码块 + 表格）。
    Returns: [(start, end, type), ...]  例如 [(100, 300, "code"), (500, 700, "table")]
    """
    ranges = []
    for m in _PROTECTED_BLOCK_RE.finditer(text):
        rtype = "code" if m.group(1) else "table"
        ranges.append((m.start(), m.end(), rtype))
    return ranges


def _safe_boundary(pos: int, protected: List[Tuple[int, int, str]]) -> int:
    """若 pos 落在受保护区域内部，将其推移到区域之前（安全切分点）。"""
    for p_start, p_end, _ in protected:
        if p_start < pos < p_end:
            return p_start
    return pos


def _build_child_prefix(
    source: str,
    breadcrumb: str,
    section_title: str,
    func_names: List[str],
) -> str:
    """构建 Child Doc 正文前缀（面包屑 + 章节 + 函数名标注）。"""
    parts = [f"[文档: {source}]"]
    if breadcrumb:
        parts.append(f"[路径: {breadcrumb}]")
    if section_title:
        parts.append(f"[章节: {section_title}]")
    if func_names:
        parts.append(f"[Functions: {', '.join(f.lower() for f in func_names)}]")
    return "\n".join(parts) + "\n"


# ── SDK 全局代码头提取（CDLL 加载 + POSE/Joint 结构体）──
_SDK_HEADER_RE = re.compile(
    r'(?:'
    r'(?:import\s+ctypes|from\s+ctypes\s+import).*?'                                     # import ctypes
    r'(?:robot\s*=\s*ctypes\.CDLL\s*\(["\'](?:collrob_sdk|py_dll)\.dll["\']\)|'         # CDLL load
    r'ctypes\.CDLL\s*\(["\'](?:collrob_sdk|py_dll)\.dll["\']\))'
    r'[\s\S]{0,3000}?'                                                                    # struct definitions follow
    r')',
    re.IGNORECASE,
)

_STRUCT_DEF_RE = re.compile(
    r'class\s+(?:POSE|Joint|RobJoint|RobPos|JNT)\s*\(.*?\)\s*:[\s\S]{0,800}?(?=\n(?:def |class |\Z))',
    re.IGNORECASE,
)


def _extract_sdk_header(full_text: str) -> str:
    """
    从 SDK 文档全文提取全局代码头（CDLL 加载 + POSE/Joint 结构体定义）。

    用于在前置依赖自动挂载：每个 API Child Chunk 顶部注入此头，
    确保任意 API 切片被检索时都包含可直接运行的完整代码上下文。
    """
    header_parts = []

    # 1. 提取 CDLL 加载行
    cdll_match = re.search(
        r'(?:robot\s*=\s*)?ctypes\.CDLL\s*\(\s*["\']([^"\']+)["\']\s*\)',
        full_text, re.IGNORECASE,
    )
    if cdll_match:
        dll_name = cdll_match.group(1)
        header_parts.append(f"import ctypes")
        header_parts.append(f"robot = ctypes.CDLL(\"{dll_name}\")")
    else:
        # Fallback: 搜索任何 ctypes import + CDLL 的模式
        cdll_block = re.search(
            r'(?:import\s+ctypes|from\s+ctypes\s+import\s+\*).*?'
            r'ctypes\.CDLL\s*\(["\'][^"\']+["\']\)',
            full_text, re.IGNORECASE | re.DOTALL,
        )
        if cdll_block:
            header_parts.append(cdll_block.group(0).strip())

    # 2. 提取 POSE / Joint 结构体定义
    for m in _STRUCT_DEF_RE.finditer(full_text):
        struct_def = m.group(0).strip()
        if struct_def and len(struct_def) >= 20:
            header_parts.append(struct_def)

    if not header_parts:
        return ""

    header = "\n".join(header_parts)
    return f"```python\n{header}\n```\n"


def _v4_build_parent_child_docs(
    full_text: str,
    source: str,
    product_id: str,
    child_chunk_size: int = 400,
    parent_chunk_size: int = 1000,
) -> Tuple[List[Document], List[Document]]:
    """
    为单个 PDF 文档构建 Parent-Child 双层 Document（纯标题树驱动，零厂商硬正则）。

    切分策略:
      - 提取标题树 (Markdown / 数字编号 / 中文序号 通用兼容)
      - Parent 层: 按 H2 章节边界切分，~1000-1500 字符，提供宏观背景
      - Child 层: 按 H3/H4 子标题边界切分，~200-400 字符，用于精准向量检索
      - 受保护区域 (```代码块``` + |Markdown 表格|): 绝不拦腰切断
      - 每个 Child Doc 强制注入 [路径: H1 > H2 > H3] 面包屑
      - metadata.parent_id 关联到所属 Parent 切片

    Returns:
        (parent_docs, child_docs)
    """
    headings = _v4_extract_headings(full_text)
    protected = _v4_find_protected_ranges(full_text)

    # ── 🔴 SDK 全局代码头提取（c_sdk 文档专用）──
    doc_type = _resolve_doc_type(product_id)
    sdk_header = _extract_sdk_header(full_text) if doc_type == "c_sdk" else ""
    if sdk_header:
        logger.info(f"  📦 SDK 代码头已提取: {len(sdk_header)} 字符 → 自动挂载至所有 API Child")

    # ── Step 1: 确定 Parent 层级 (H2 优先，不存在则自适应提升) ──
    all_levels = set(lv for _, _, lv in headings)
    parent_level = 2
    if not all_levels:
        parent_level = 1
    elif 2 not in all_levels:
        parent_level = min(all_levels)

    h_parent = [(pos, title, lv) for pos, title, lv in headings if lv == parent_level]

    # Fallback: 无任何标题 → 全文作为 1 个 Parent
    if not h_parent:
        parent_id = f"parent_{product_id}_0"
        parent_doc = Document(
            page_content=f"[文档: {source}]\n\n{full_text[:parent_chunk_size]}",
            metadata={
                "source": source, "product_id": product_id,
                "doc_type": _resolve_doc_type(product_id),
                "chunk_type": "parent", "parent_id": None,
                "section_title": "全文", "section_level": parent_level,
            },
        )
        children = _v4_build_child_docs_v2(
            full_text, source, product_id, parent_id,
            0, len(full_text), headings, protected, child_chunk_size,
            sdk_header=sdk_header,
        )
        return [parent_doc], children

    # ── Step 2: 构建 Parent Docs + 对应 Child Docs ──
    parent_docs = []
    all_children = []

    parent_boundaries = [p_start for p_start, _, _ in h_parent] + [len(full_text)]
    for i, (p_start, p_title, p_lv) in enumerate(h_parent):
        p_end = parent_boundaries[i + 1]
        p_start_safe = _safe_boundary(p_start, protected)
        p_end_safe = _safe_boundary(p_end, protected)

        parent_text = full_text[p_start_safe:p_end_safe].strip()
        if len(parent_text) < 30:
            continue

        if len(parent_text) > parent_chunk_size:
            cutoff = max(
                parent_text.rfind('\n\n', parent_chunk_size - 200, parent_chunk_size + 200),
                parent_text.rfind('\n', parent_chunk_size - 100, parent_chunk_size + 100),
                parent_chunk_size,
            )
            parent_text = parent_text[:cutoff].strip()

        # 子章节 TOC
        child_titles = [
            title for pos, title, lv in headings
            if p_start <= pos < p_end and lv > parent_level
        ]
        toc = "\n".join(f"- {t}" for t in child_titles[:15])
        if toc:
            parent_text = f"{parent_text}\n\n【子章节】\n{toc}"

        parent_breadcrumb = _v4_build_breadcrumb(headings, p_start, p_end) or p_title
        parent_id = f"parent_{product_id}_{i}"

        parent_doc = Document(
            page_content=f"[文档: {source}]\n[路径: {parent_breadcrumb}]\n\n{parent_text}",
            metadata={
                "source": source, "product_id": product_id,
                "doc_type": _resolve_doc_type(product_id),
                "chunk_type": "parent", "parent_id": None,
                "section_title": p_title, "section_level": parent_level,
            },
        )
        parent_docs.append(parent_doc)

        # Child Docs
        children = _v4_build_child_docs_v2(
            full_text, source, product_id, parent_id,
            p_start_safe, p_end_safe, headings, protected, child_chunk_size,
            sdk_header=sdk_header,
        )
        all_children.extend(children)

    return parent_docs, all_children


def _v4_build_child_docs_v2(
    full_text: str,
    source: str,
    product_id: str,
    parent_id: str,
    section_start: int,
    section_end: int,
    headings: List[Tuple[int, str, int]],
    protected: List[Tuple[int, int, str]],
    child_chunk_size: int = 400,
    sdk_header: str = "",
) -> List[Document]:
    """
    在指定 Parent 范围内，按子标题驱动构建 Child Docs（v2 纯标题版）。

    切分逻辑:
      1. 找出范围内的 H3+ 子标题作为第一级边界
      2. 边界推移至受保护区域外部（不切代码块/表格）
      3. 若子段仍超过 child_chunk_size → 按段落 (\\n\\n) 软断块
      4. 每个 Child 强制注入 [路径: ...] 面包屑 + [章节: ...]
      5. metadata 写入 parent_id / section_title / function_names(原始大小写)
    """
    children = []

    sub_headings = [
        (pos, title, lv) for pos, title, lv in headings
        if section_start <= pos < section_end and lv > 2
    ]

    if not sub_headings:
        text = full_text[section_start:section_end].strip()
        if text:
            breadcrumb = _v4_build_breadcrumb(headings, section_start, section_end)
            children = _split_text_into_children(
                text, source, product_id, parent_id,
                child_chunk_size, breadcrumb, "",
                sdk_header=sdk_header,
            )
        return children

    boundaries = [section_start]
    for pos, _, _ in sub_headings:
        boundaries.append(_safe_boundary(pos, protected))
    boundaries.append(section_end)

    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if s >= e:
            continue
        text = full_text[s:e].strip()
        if not text or len(text) < 15:
            continue

        current_title = ""
        current_level = 3
        for pos, title, lv in sub_headings:
            if _safe_boundary(pos, protected) == s:
                current_title = title
                current_level = lv
                break

        breadcrumb = _v4_build_breadcrumb(headings, s, e)
        sub_children = _split_text_into_children(
            text, source, product_id, parent_id,
            child_chunk_size, breadcrumb, current_title, current_level,
            sdk_header=sdk_header,
        )
        children.extend(sub_children)

    return children


def _split_text_into_children(
    text: str,
    source: str,
    product_id: str,
    parent_id: str,
    chunk_size: int = 400,
    breadcrumb: str = "",
    section_title: str = "",
    section_level: int = 3,
    sdk_header: str = "",
) -> List[Document]:
    """
    将文本段落按「先提取受保护块，再按段落切分」的 Tokenize 模式切分为 Child Docs。

    保护机制:
      - ```代码块``` 先整体提取，无论多大绝不拦腰切断，防止 Markdown AST 崩溃
      - |表格行| 作为受保护整体保留
      - 普通文本按 \\n\\n 段落边界累积合并，超出 chunk_size 时 flush
    """
    children = []
    child_idx = 0

    def _emit_child(content: str):
        nonlocal child_idx
        content = content.strip()
        if len(content) < 10:
            return
        func_names = _v4_extract_function_names(content)
        is_api = len(func_names) > 0
        prefix = _build_child_prefix(source, breadcrumb, section_title, func_names)
        # 🔴 SDK 全局代码头自动挂载: c_sdk 文档的每个 API Child 注入 CDLL+结构体
        header_block = ""
        if sdk_header and is_api:
            header_block = f"\n【前置依赖 — 可直接运行】\n{sdk_header}\n"
        children.append(Document(
            page_content=f"{prefix}{header_block}{content}",
            metadata={
                "source": source, "product_id": product_id,
                "doc_type": _resolve_doc_type(product_id),
                "chunk_type": "child", "parent_id": parent_id,
                "section_title": section_title,
                "section_level": section_level,
                "function_names": ",".join(func_names) if func_names else "",
                "api_atomic": is_api,
                "is_api": is_api,
            },
        ))
        child_idx += 1

    # ── 1. 将文本解析为 [受保护段] 与 [普通段] ──
    segments = []
    last_end = 0
    for m in _PROTECTED_BLOCK_RE.finditer(text):
        normal_text = text[last_end:m.start()].strip()
        if normal_text:
            segments.append({"type": "normal", "text": normal_text})
        segments.append({"type": "protected", "text": m.group(0)})
        last_end = m.end()

    remaining = text[last_end:].strip()
    if remaining:
        segments.append({"type": "normal", "text": remaining})

    # ── 2. 遍历合并切片 ──
    current_chunk = ""
    for seg in segments:
        if seg["type"] == "protected":
            if current_chunk.strip():
                _emit_child(current_chunk)
                current_chunk = ""
            # 保护块整体作为一个切片，无论多大绝不拦腰切断
            _emit_child(seg["text"])
        else:
            paras = re.split(r'\n\n+', seg["text"])
            for para in paras:
                para = para.strip()
                if not para:
                    continue
                if current_chunk and len(current_chunk) + len(para) + 2 > chunk_size:
                    _emit_child(current_chunk)
                    current_chunk = para
                else:
                    current_chunk = (current_chunk + "\n\n" + para) if current_chunk else para

    if current_chunk.strip():
        _emit_child(current_chunk)

    return children


def _v4_extract_function_names(text: str) -> List[str]:
    """
    从文本中提取 SDK 函数名/API 标识符列表（用于 metadata 和 header tag）。

    多层匹配策略（按优先级）：
      ① snake_case SDK 函数: robot_Power_on(...), set_move_line(...)
      ② C 函数声明: int robot_movl(...), void set_robot_power_on(...)
      ③ 中文 SDK 格式: 函数名称 Robot_socket_start(...)
      ④ Python dotted 方法: robot.movl(...), lib.set_robot_power_on(...)
      ⑤ API 指示器模式: POSE, JointValue, argtypes, restype, ctypes.CDLL 等
      ⑥ 代码块内无括号的函数名: robot_xxx, set_xxx, get_xxx 等前缀标识符

    保留文档原始大小写，检索匹配端通过 .lower() 兼容。
    """
    funcs_raw = set()

    # ── ① snake_case 函数调用: robot_Power_on(arg1, arg2) ──
    for m in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*_[a-zA-Z0-9_]+)\s*\(', text):
        fname = m.group(1).strip('_')
        if len(fname) >= 6 and '_' in fname:
            funcs_raw.add(fname)

    # ── ② C 函数声明: int robot_movl(...), void set_robot_power_on(...) ──
    for m in re.finditer(
        r'(?:^|\n)\s*(?:int|void|bool|char|float|double|long|unsigned|short|'
        r'ctypes\.c_\w+|POINTER|HDC|HWND|HANDLE)\s+'
        r'([a-zA-Z_]\w{4,})\s*\(', text, re.MULTILINE,
    ):
        fname = m.group(1).strip('_')
        if len(fname) >= 6 and re.search(r'[a-z]', fname):
            funcs_raw.add(fname)

    # ── ③ 中文 SDK 格式: 函数名称 xxx( ──
    for m in re.finditer(r'(?:函数名称|函数名|方法名)\s+([a-zA-Z_]\w{3,})\s*\(?', text):
        fname = m.group(1).strip('_')
        if len(fname) >= 4:
            funcs_raw.add(fname)

    # ── ④ Python dotted 方法: robot.xxx(...), lib.xxx(...) ──
    for m in re.finditer(r'\b(?:robot|lib|sdk|cdll|py_dll|collrob)\s*\.\s*'
                         r'([a-zA-Z_]\w{3,})\s*\(', text, re.IGNORECASE):
        fname = m.group(1).strip('_')
        if len(fname) >= 4:
            funcs_raw.add(fname)

    # ── ⑤ API 指示器模式: 含典型 SDK 参数关键词的代码段中的标识符 ──
    _api_indicator = re.compile(
        r'\b(POSE|JointValue|RobJoint|RobPos|JNT|argtypes|restype|'
        r'ctypes\.CDLL|ctypes\.c_\w+|Structure|_fields_|POINTER|byref)',
        re.IGNORECASE,
    )
    if _api_indicator.search(text):
        # 此段文本包含 SDK 代码特征 → 额外提取所有前缀标识符
        for m in re.finditer(
            r'\b(?:robot_|set_|get_|arm_|jaka_|collrob_|mov[lcpjb])'
            r'([a-zA-Z0-9_]*)\b',
            text, re.IGNORECASE,
        ):
            fname = m.group(0).strip('_')
            if len(fname) >= 4:
                funcs_raw.add(fname)

    # ── ⑥ 无括号前缀标识符: robot_xxx, set_xxx, get_xxx 在代码块中 ──
    if '```' in text or 'robot_' in text.lower() or 'ctypes' in text.lower():
        for m in re.finditer(
            r'\b((?:robot|set|get|arm|py|collrob)_[a-zA-Z]\w{2,})\b',
            text, re.IGNORECASE,
        ):
            fname = m.group(1).strip('_')
            if len(fname) >= 5:
                funcs_raw.add(fname)

    # 按小写去重 + 排序（保留第一个出现的原始大小写）
    seen_lower = set()
    funcs = []
    for f in sorted(funcs_raw, key=lambda x: (x.lower(), len(x))):
        if f.lower() not in seen_lower:
            seen_lower.add(f.lower())
            funcs.append(f)
    return funcs[:15]


# ============================================================
# v4 主入口: load_pdfs_v4_dual
# ============================================================

def load_pdfs_v4_dual(
    data_dir: str,
    child_chunk_size: int = 400,
    parent_chunk_size: int = 1000,
) -> Tuple[List[Document], List[Document]]:
    """
    v4 双层索引加载器 — Parent-Child Dual Indexing。

    返回 (parent_docs, child_docs) 两个独立的 Document 列表，
    分别写入 ChromaDB 的两个 Collection。

    Args:
        data_dir: PDF 文件目录
        child_chunk_size: Child 层最大字符数（默认 400）
        parent_chunk_size: Parent 层最大字符数（默认 1000）

    Returns:
        (parents, children): 两个 Document 列表
    """
    pdf_files = [
        f for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        logger.warning(f"目录 '{data_dir}' 中未找到 PDF 文件")
        return [], []

    logger.info(f"📄 v4 Dual: 发现 {len(pdf_files)} 个 PDF，child={child_chunk_size}, parent={parent_chunk_size}")

    all_parents = []
    all_children = []

    for pdf_file in pdf_files:
        file_path = os.path.join(data_dir, pdf_file)
        try:
            # ── v4 通用提取: 字符密度检测 + OCR 归位融合 ──
            text, total_pages, ocr_pages = _v4_extract_text_universal(file_path)
            if not text.strip():
                logger.warning(f"  ⚠️  {pdf_file}: 无有效文本")
                continue

            # ── 🔴 PDF 文本清洗: 控制字符 + 连字替换 + 括号空格规范化 ──
            text = _clean_pdf_text(text)

            product_id = _resolve_product_id_from_filename(pdf_file)
            parents, children = _v4_build_parent_child_docs(
                text, pdf_file, product_id,
                child_chunk_size=child_chunk_size,
                parent_chunk_size=parent_chunk_size,
            )
            all_parents.extend(parents)
            all_children.extend(children)
            logger.info(
                f"  ✅ {pdf_file}: {len(parents)} parents + {len(children)} children "
                f"(product={product_id})"
            )
        except Exception as e:
            logger.error(f"  ❌ {pdf_file}: {e}")

    # 统计
    api_atomic = sum(1 for d in all_children if d.metadata.get("api_atomic"))
    with_funcs = sum(1 for d in all_children if d.metadata.get("function_names"))
    logger.info(
        f"✅ v4 Dual 加载完成: {len(all_parents)} parents + {len(all_children)} children "
        f"(api_atomic={api_atomic}, with_funcs={with_funcs})"
    )
    return all_parents, all_children


# ============================================================
# 命令行测试入口
# ============================================================
if __name__ == "__main__":
    from .config import PDF_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

    docs = load_pdfs_from_directory(
        data_dir=PDF_DATA_DIR,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    if docs:
        debug_print_chunks(docs, max_show=3)
        print(f"\n--- 示例：第一个文本块 ---")
        print(f"来源: {docs[0].metadata['source']}")
        print(f"内容预览: {docs[0].page_content[:200]}...")
