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
    根据文件名解析对应的产品标识。

    使用 PRODUCT_MAPPING_RULES 中的 filename_patterns 进行匹配，
    不区分大小写，任一模式命中即返回对应 product_id。

    Args:
        filename: PDF 文件名

    Returns:
        product_id 字符串，若无法识别则返回 "unknown"
    """
    filename_lower = filename.lower()
    for rule in PRODUCT_MAPPING_RULES:
        for pattern in rule["filename_patterns"]:
            if pattern.lower() in filename_lower:
                logger.info(
                    f"🏷️  产品识别: '{filename}' → product_id='{rule['product_id']}' "
                    f"(命中模式: '{pattern}')"
                )
                return rule["product_id"]

    logger.warning(f"⚠️  无法识别产品: '{filename}'，标记为 'unknown'")
    return "unknown"


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
    对 PDF 提取文本做规范化清洗。

    处理步骤:
      1. PDF 连字替换: oﬀ→off, ﬁ→fi, ﬂ→fl 等（防止函数名匹配失败）
      2. Unicode 符号规范化: EN/EM DASH, 智能引号, NBSP
      3. 函数括号内外多余空格清洗:
         - "robot_Power_on ( )" → "robot_Power_on()"
         - "robot_movl ( POSE pose , float vel )" → "robot_movl(POSE pose, float vel)"
      4. 参数列表内多余空格压缩: ",  " → ", "

    Args:
        text: PDF 原始提取文本

    Returns:
        清洗后的规范化文本
    """
    if not text:
        return text

    # Step 1+2: 连字 + Unicode 符号替换（单次 str.translate）
    cleaned = text.translate(_LIGATURE_TRANS_TABLE)

    # Step 3: 函数括号空格清洗 — 多轮迭代至收敛
    _max_iter = 3
    for _ in range(_max_iter):
        prev = cleaned
        # 3a: "函数名 ( args )" → "函数名(args)"
        cleaned = re.sub(r'(\w+)\s+\(\s+', r'\1(', cleaned)
        # 3b: " )" → ")"（孤立右括号前的空格）
        cleaned = re.sub(r'\s+\)', ')', cleaned)
        # 3c: "( args , args )" → "(args, args)"
        cleaned = re.sub(r'\(\s+', '(', cleaned)
        # 3d: 压缩参数列表内多余空格: ",  " → ", "
        cleaned = re.sub(r'\s*,\s*', ', ', cleaned)
        # 3e: 压缩连续空格
        cleaned = re.sub(r'  +', ' ', cleaned)
        if cleaned == prev:
            break

    # Step 4: 移除 null 字节
    cleaned = cleaned.replace('\x00', '')

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


# ── v4 OCR: 多模态图片文本抽取 (ADR-16) ──

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


def _v4_inject_ocr_text(pdf_path: str, full_text: str) -> str:
    """
    对 PDF 内嵌图片做 OCR，将识别文本注入页面内容中。

    只处理 >100×100 px 的图片（跳过图标/装饰元素）。
    OCR 结果以 [OCR识别] 标签注入，含数字参数的行双写为 KV 标签。
    同时注入页面所在章节的 [路径: ...] 和 [章节: ...] 面包屑前缀。

    Returns:
        增强后的 full_text（在原文本中追加 OCR 段落）
    """
    ocr = _v4_get_ocr_engine()
    if ocr is None:
        return full_text

    try:
        import fitz
        import numpy as np
        from PIL import Image
        import io
    except ImportError:
        return full_text

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return full_text

    # ── 预计算章节标题位置，用于 OCR 面包屑注入 ──
    headings = _v4_extract_headings(full_text)
    # 预计算每页在 full_text 中的位置（用于面包屑定位）
    _page_positions: List[int] = []
    _page_fingerprints: List[str] = []
    _cursor = 0
    for page_idx in range(len(doc)):
        try:
            page_text = doc[page_idx].get_text("text")
            fp = page_text.strip()[:80] if page_text else ""
            pos = full_text.find(fp, _cursor) if fp else -1
            if pos >= 0:
                _page_positions.append(pos)
                _cursor = pos + max(len(fp), 1)
            else:
                _page_positions.append(-1)
            _page_fingerprints.append(fp)
        except Exception:
            _page_positions.append(-1)
            _page_fingerprints.append("")

    ocr_parts = []
    total_ocr_chars = 0
    processed_images = 0

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        image_list = page.get_images(full=True)
        if not image_list:
            continue

        page_ocr_lines = []
        for img_info in image_list:
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
                    continue  # 跳过小图标

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
                    processed_images += 1
                    ocr_text = " | ".join(lines)
                    page_ocr_lines.append(ocr_text)
                    total_ocr_chars += len(ocr_text)
            except Exception:
                continue

        if page_ocr_lines:
            # ── 🔴 章节面包屑注入: 为 OCR 文字添加上下文 ──
            page_pos = _page_positions[page_idx] if page_idx < len(_page_positions) else -1
            breadcrumb = ""
            section_title = ""
            if page_pos >= 0 and headings:
                breadcrumb = _v4_build_breadcrumb(headings, page_pos, page_pos + 500)
                # 提取最近的章节标题
                _idx = bisect.bisect_right([h[0] for h in headings], page_pos) - 1
                if _idx >= 0:
                    section_title = headings[_idx][1]

            _header = f"[OCR识别: page={page_idx + 1}, images={len(page_ocr_lines)}]"
            if breadcrumb:
                _header += f"\n[路径: {breadcrumb}]"
            if section_title:
                _header += f"\n[章节: {section_title}]"
            ocr_parts.append(_header + "\n" + "\n".join(page_ocr_lines))

    doc.close()

    if ocr_parts:
        ocr_block = "\n\n" + "\n\n".join(ocr_parts)
        logger.info(
            f"  🖼️  OCR: {processed_images} 张图片 → {total_ocr_chars} 字符 "
            f"({pdf_path})"
        )
        return full_text + ocr_block

    return full_text


def _v4_build_parent_child_docs(
    full_text: str,
    source: str,
    product_id: str,
    child_chunk_size: int = 400,
    parent_chunk_size: int = 1000,
) -> Tuple[List[Document], List[Document]]:
    """
    为单个 PDF 文档构建 Parent-Child 双层 Document。

    Parent 层（H2 级别，~800-1500 字符）:
      - 按 H2 标题切分，包含概述段落 + 子章节标题列表
      - metadata.type = "parent"
      - metadata.parent_id = None

    Child 层（H3/H4 级别，~100-500 字符，API 原子）:
      - 按 H3/H4 切分，API 块保持完整
      - metadata.type = "child"
      - metadata.parent_id → 对应父层 chunk 的 ID

    Returns:
        (parent_docs, child_docs)
    """
    headings = _v4_extract_headings(full_text)
    api_blocks = _v4_extract_api_blocks(full_text)

    # ── Step 1: 自适应层级提升 ──
    # 如果没有 H2 标题，将最低层级提升为 Parent 边界
    all_levels = set(lv for _, _, lv in headings)
    if not all_levels:
        # 完全无标题 → 1 Parent, API 块作为 Child
        h2_positions = []
        h3_positions = []
    else:
        min_lv = min(all_levels)
        max_lv = max(all_levels)
        if 2 not in all_levels:
            # 不存在 H2 → 将当前最低层级提升为 H2（Parent）
            h2_positions = [(pos, title) for pos, title, lv in headings if lv == min_lv]
            h3_positions = [(pos, title, lv) for pos, title, lv in headings if lv > min_lv]
        else:
            h2_positions = [(pos, title) for pos, title, lv in headings if lv == 2]
            h3_positions = [(pos, title, lv) for pos, title, lv in headings if lv >= 3]

    # ── Step 2: 构建 Parent Docs ──
    parent_docs = []
    parent_boundaries = [p[0] for p in h2_positions] + [len(full_text)]
    for i, (p_start, p_title) in enumerate(h2_positions):
        p_end = parent_boundaries[i + 1]
        parent_text = full_text[p_start:p_end].strip()
        if len(parent_text) < 50:
            continue

        # 截取到 parent_chunk_size，保留完整段落结尾
        if len(parent_text) > parent_chunk_size:
            cutoff = parent_text.rfind('\n\n', parent_chunk_size - 200, parent_chunk_size + 200)
            if cutoff < 0:
                cutoff = parent_text.rfind('\n', parent_chunk_size - 100, parent_chunk_size + 100)
            if cutoff < 0:
                cutoff = parent_chunk_size
            parent_text = parent_text[:cutoff].strip()

        # 收集子章节标题列表
        child_titles_in_range = [
            title for pos, title, lv in h3_positions
            if p_start <= pos < p_end
        ]
        child_toc = "\n".join(f"- {t}" for t in child_titles_in_range[:15])
        if child_toc:
            parent_text = f"{parent_text}\n\n【子章节】\n{child_toc}"

        parent_id = f"parent_{product_id}_{i}"
        parent_doc = Document(
            page_content=f"[文档: {source}]\n[路径: {p_title}]\n\n{parent_text}",
            metadata={
                "source": source,
                "product_id": product_id,
                "chunk_type": "parent",
                "parent_id": None,
                "section_title": p_title,
                "section_level": 2,
            },
        )
        parent_docs.append(parent_doc)

        # ── Step 3: 在此 Parent 范围内构建 Child Docs ──
        child_docs_in_parent = _v4_build_child_docs(
            full_text, source, product_id, parent_id,
            p_start, p_end, headings, api_blocks, child_chunk_size,
        )
        parent_docs.extend(child_docs_in_parent)  # flattened — 实际使用会分 Collection

    # Fallback: 如果没有 H2 标题 → 全文作为 1 个 Parent
    if not parent_docs:
        parent_id = f"parent_{product_id}_0"
        parent_doc = Document(
            page_content=f"[文档: {source}]\n\n{full_text[:parent_chunk_size]}",
            metadata={
                "source": source, "product_id": product_id,
                "chunk_type": "parent", "parent_id": None,
                "section_title": "全文", "section_level": 1,
            },
        )
        parent_docs.append(parent_doc)

    # ── Step 4: 分离 Parent 和 Child（目前混在一起）──
    pure_parents = [d for d in parent_docs if d.metadata.get("chunk_type") == "parent"]
    pure_children = [d for d in parent_docs if d.metadata.get("chunk_type") == "child"]
    parent_docs.clear()
    parent_docs.extend(pure_parents)
    # 将 children 从 parent_docs 移出
    # （已经在上面分离）

    return pure_parents, pure_children


def _v4_build_child_docs(
    full_text: str,
    source: str,
    product_id: str,
    parent_id: str,
    section_start: int,
    section_end: int,
    headings: List[Tuple[int, str, int]],
    api_blocks: List[Tuple[int, int, str]],
    child_chunk_size: int = 400,
) -> List[Document]:
    """
    在指定 Parent 范围内构建 Child Docs。

    Child 切分策略:
      1. API 原子块保持完整（绝不切分）
      2. H3/H4 标题处软断块（内容 < 400 字则合并）
      3. KV 参数行周围的上下文保留（最小 80 字符 padding）
    """
    children = []

    # 找出此范围内的 H3+ 标题
    sub_headings = [
        (pos, title, lv) for pos, title, lv in headings
        if section_start <= pos < section_end and lv >= 3
    ]

    if not sub_headings:
        # 无子标题 → 按 api_blocks 切分
        text = full_text[section_start:section_end].strip()
        children = _v4_split_by_api_blocks(
            text, source, product_id, parent_id, child_chunk_size,
        )
        return children

    # 按 H3 边界切分
    boundaries = [section_start] + [p[0] for p in sub_headings] + [section_end]
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        text = full_text[s:e].strip()
        if not text or len(text) < 20:
            continue

        # 找到当前 H3 标题
        current_title = ""
        current_level = 3
        for pos, title, lv in sub_headings:
            if pos == s:
                current_title = title
                current_level = lv
                break

        # Breadcrumb
        breadcrumb = _v4_build_breadcrumb(headings, s, e)

        # API 块内的进一步原子切分
        sub_children = _v4_split_by_api_blocks(
            text, source, product_id, parent_id, child_chunk_size,
            breadcrumb=breadcrumb, section_title=current_title,
        )
        for child in sub_children:
            child.metadata["section_title"] = current_title or child.metadata.get("section_title", "")
            child.metadata["section_level"] = current_level
        children.extend(sub_children)

    return children


def _v4_split_by_api_blocks(
    text: str,
    source: str,
    product_id: str,
    parent_id: str,
    chunk_size: int = 400,
    breadcrumb: str = "",
    section_title: str = "",
) -> List[Document]:
    """
    对文本段做 API 感知的原子切分。

    规则:
      - API 块标记为 [api_atomic=True]，永不切分
      - 非 API 文本用 RecursiveCharacterTextSplitter 正常切分
      - KV 参数行周围的 padding 最小化为 80 字符
      - 每个 Child 带 [函数名: xxx] header tag（用于 CodeEntityAnchor 匹配）
    """
    children = []
    api_blocks = _v4_extract_api_blocks(text)

    if not api_blocks:
        # 纯文本 → 标准切分
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=30,
            separators=["\n\n", "\n", "。", ".", " ", ""],
            length_function=len, is_separator_regex=False,
        )
        temp_doc = Document(page_content=text, metadata={})
        chunks = splitter.split_documents([temp_doc])
        for i, c in enumerate(chunks):
            child_id = f"{parent_id}_c{i}"
            func_names = _v4_extract_function_names(c.page_content)
            func_header = f"[Functions: {', '.join(func_names)}]\n" if func_names else ""
            prefix = f"[文档: {source}]\n"
            if breadcrumb:
                prefix += f"[路径: {breadcrumb}]\n"
            if section_title:
                prefix += f"[章节: {section_title}]\n"
            children.append(Document(
                page_content=f"{prefix}{func_header}{c.page_content}",
                metadata={
                    "source": source, "product_id": product_id,
                    "chunk_type": "child", "parent_id": parent_id,
                    "api_atomic": False,
                    "function_names": ",".join(func_names) if func_names else "",
                },
            ))
        return children

    # 有 API 块 → 保护它们不被打散
    last_end = 0
    child_idx = 0
    for api_start, api_end, api_label in api_blocks:
        # API 块之前的文本
        pre_text = text[last_end:api_start].strip()
        if pre_text and len(pre_text) >= 30:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=20,
                separators=["\n\n", "\n", "。", ".", " "],
                length_function=len, is_separator_regex=False,
            )
            temp_doc = Document(page_content=pre_text, metadata={})
            for c in splitter.split_documents([temp_doc]):
                child_id = f"{parent_id}_c{child_idx}"
                child_idx += 1
                func_names = _v4_extract_function_names(c.page_content)
                func_header = f"[Functions: {', '.join(func_names)}]\n" if func_names else ""
                prefix = f"[文档: {source}]\n"
                if breadcrumb:
                    prefix += f"[路径: {breadcrumb}]\n"
                if section_title:
                    prefix += f"[章节: {section_title}]\n"
                children.append(Document(
                    page_content=f"{prefix}{func_header}{c.page_content}",
                    metadata={
                        "source": source, "product_id": product_id,
                        "chunk_type": "child", "parent_id": parent_id,
                        "api_atomic": False,
                        "function_names": ",".join(func_names) if func_names else "",
                    },
                ))

        # API 原子块本身 — 永不切分
        api_text = text[api_start:api_end].strip()
        if api_text:
            child_id = f"{parent_id}_c{child_idx}"
            child_idx += 1
            func_names = _v4_extract_function_names(api_text)
            func_header = f"[Functions: {', '.join(func_names)}]\n" if func_names else ""
            prefix = f"[文档: {source}]\n"
            if breadcrumb:
                prefix += f"[路径: {breadcrumb}]\n"
            if section_title:
                prefix += f"[章节: {section_title}]\n"
            # 对于 API 块，额外标记函数签名行
            api_funcs = re.findall(r'(?:robot_|set_|get_)\w+', api_text.lower())
            if api_funcs:
                prefix += f"[API原子块: {', '.join(api_funcs[:5])}]\n"
            children.append(Document(
                page_content=f"{prefix}{func_header}{api_text}",
                metadata={
                    "source": source, "product_id": product_id,
                    "chunk_type": "child", "parent_id": parent_id,
                    "api_atomic": True,
                    "function_names": ",".join(sorted(set(func_names + api_funcs))),
                },
            ))

        last_end = api_end

    # API 块之后的文本
    remaining = text[last_end:].strip()
    if remaining and len(remaining) >= 30:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=20,
            separators=["\n\n", "\n", "。", ".", " "],
            length_function=len, is_separator_regex=False,
        )
        temp_doc = Document(page_content=remaining, metadata={})
        for c in splitter.split_documents([temp_doc]):
            child_id = f"{parent_id}_c{child_idx}"
            child_idx += 1
            func_names = _v4_extract_function_names(c.page_content)
            func_header = f"[Functions: {', '.join(func_names)}]\n" if func_names else ""
            prefix = f"[文档: {source}]\n"
            if breadcrumb:
                prefix += f"[路径: {breadcrumb}]\n"
            if section_title:
                prefix += f"[章节: {section_title}]\n"
            children.append(Document(
                page_content=f"{prefix}{func_header}{c.page_content}",
                metadata={
                    "source": source, "product_id": product_id,
                    "chunk_type": "child", "parent_id": parent_id,
                    "api_atomic": False,
                    "function_names": ",".join(func_names) if func_names else "",
                },
            ))

    return children


def _v4_extract_function_names(text: str) -> List[str]:
    """
    从文本中提取 SDK 函数名列表（用于 metadata 和 header tag）。

    保留文档原始大小写（如 robot_Power_on），检索匹配端通过 .lower()
    实现忽略大小写比对。header tag 中输出为小写以保持一致性。
    """
    funcs_raw = set()
    for m in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*_[a-zA-Z0-9_]+)\s*\(', text):
        fname = m.group(1).strip('_')  # 🔴 保留原始大小写，不做 .lower()
        if len(fname) >= 6 and '_' in fname:
            funcs_raw.add(fname)
    # 按小写去重 + 排序（原始大小写中的第一个出现被保留）
    seen_lower = set()
    funcs = []
    for f in sorted(funcs_raw, key=lambda x: x.lower()):
        if f.lower() not in seen_lower:
            seen_lower.add(f.lower())
            funcs.append(f)
    return funcs[:10]


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
            text = extract_text_from_pdf(file_path)
            if not text.strip():
                logger.warning(f"  ⚠️  {pdf_file}: 无有效文本")
                continue

            # ── v4 OCR: 多模态图片文本注入 ──
            text = _v4_inject_ocr_text(file_path, text)

            # ── 🔴 PDF 文本清洗: 连字替换 + 括号空格规范化 ──
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
