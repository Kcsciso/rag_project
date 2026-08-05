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
        # 多级数字: 3.1.5 通讯设置 / 3.1设置
        _re.compile(r'^(\d+(?:\.\d+)+)\s*(.+?)(?:\r?\n|$)', _re.MULTILINE),
        # 🔴 新增：单级数字标题兼容: 1. / 1、 / 1.安装需要的模块 (兼容无空格)
        _re.compile(r'^(\d+)\.\s*([^\d\s].+?)(?:\r?\n|$)', _re.MULTILINE),
        # 中文章节: 第1章 / 第一章前言
        _re.compile(r'^(第[一二三四五六七八九十\d]+[章节])\s*(.+?)(?:\r?\n|$)', _re.MULTILINE),
        # 中文序号: 一、/ （一）
        _re.compile(r'^([（(]?[一二三四五六七八九十]+[）)]?[\s、,，])\s*(.+?)(?:\r?\n|$)', _re.MULTILINE),
        # Markdown H1~H4
        _re.compile(r'^(#{1,4})\s+(.+?)(?:\r?\n|$)', _re.MULTILINE),
        # 特殊符号标题
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

# ── SDK 表头黑名单：绝不允许被识别为 Heading 切分点的表头关键词 ──
# 这些词常见于 SDK 手册的表格/段落头部，若被误识别为 Heading 会把 API 块切碎
_SDK_TABLE_HEADER_BLACKLIST = frozenset({
    "函数名称", "函数名", "方法名", "方法名称",
    "功能描述", "功能说明", "函数说明", "函数功能",
    "参数说明", "参数列表", "参数", "输入参数", "输出参数",
    "返回值", "返回值说明", "返回参数", "返回值类型",
    "示例代码", "调用示例", "代码示例", "示例",
    "注意事项", "备注", "说明", "注",
    "头文件", "库文件", "依赖",
})

# ── 标题识别模式（兼容 Markdown 和非 Markdown 格式）──
# ── 🔴 标题识别模式（兼容无空格、极短标题及明确层级）──
_V4_HEADING_PATTERNS = [
    # 层级 1: 中文章节 (H1) -> 第1章 前言 / 第一章前言 / 第1 章 前言 (v28 容忍章号与"章"间空格)
    (re.compile(r'^(第[一二三四五六七八九十\d]+\s*[章节])\s*(.{1,80}?)(?:\r?\n|$)', re.MULTILINE), 1),
    
    # 层级 2: 多级数字编号 (H2/H3/H4) -> 3.1.5 通讯设置 / 3.1设置 (兼容无空格，点号收尾)
    # 🔴 [终极修复 2] 多级数字编号：将 {1,3} 放宽至 {1,5}，支持高达 6 级深度的标题 (如 3.1.5.2.1)
    # \.?\s* 确保完美兼容末尾带点和数字汉字粘连（无空格）的极端排版
    (re.compile(r'^(\d+(?:\.\d+){1,5})\.?\s*(.{1,80}?)(?:\r?\n|$)', re.MULTILINE), 2),
    
    # 🔴 [终极修复 1] 层级 2: 纯数字+点 (H2) -> 1. 标题 (移除 、 和 ） 防止将 1、列表项误判为大纲导致父块碎裂)
    (re.compile(r'^(\d{1,2})\.\s*(.{1,80}?)(?:\r?\n|$)', re.MULTILINE), 2),
    
    # 层级 2: 中文序号 -> 一、/ （一）/ (一)
    (re.compile(r'^[（(]?[一二三四五六七八九十]+[）)]?\s*[、,，\s]\s*(.{1,80}?)(?:\r?\n|$)', re.MULTILINE), 2),
    
    # 层级 1: Markdown H -> ## / ### / ####
    (re.compile(r'^(#{1,4})\s+(.{1,80}?)(?:\r?\n|$)', re.MULTILINE), 1),
    
    # 层级 3: 无编号中文标题 (兜底 H3) -> 概述 / 功能说明
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
    # ── 🔴 中文 SDK 手册: "数字序号 + 函数功能标题" 作为 API 原子块起点 ──
    # 匹配 "3. 连接机械臂" / "4. 机械臂上电" 等序号标题，
    # 将其后的函数名称+功能描述+参数说明+返回值+示例代码 完整封装在 1 个原子块中
    re.compile(
        r'(?:^|\n)(\d{1,2})[\.\)、）]\s*'
        r'((?:连接|断开|上电|下电|使能|回零|复位|急停|抱闸|松闸|'
        r'运动|移动|直线|圆弧|关节|设置|获取|读取|写入|初始化|配置|'
        r'启动|停止|控制|加载|卸载|注册|注销|登录|登出|同步|异步|'
        r'IO\s*操作|信号|输出|输入|状态|错误|异常)[^\n]{0,60})'
        r'(?:[\s\S]{0,2000}?)'
        r'(?=\n\d{1,2}[\.\)、）]\s*(?:连接|断开|上电|下电|使能|回零|复位|急停|抱闸|'
        r'运动|移动|直线|圆弧|关节|设置|获取|读取|写入|初始化|配置|'
        r'启动|停止|控制|加载|卸载|注册|注销)|\n\n\d{1,2}[\.\)、）]|\Z)',
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

def _clean_pdf_text(text: str, doc_type: str = "general", product_id: str = "") -> str:
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

    # ── Step 4.3: 下划线空格断裂归一化（PDF OCR artifact）──
    # "rob _ ip" → "rob_ip", "_ _" → "__", "collrob _ sdk" → "collrob_sdk"
    cleaned = re.sub(r'(\w+)\s+_\s+(\w+)', r'\1_\2', cleaned)
    cleaned = re.sub(r'_\s+_', '__', cleaned)
    cleaned = re.sub(r'(\w)\s+_\s+', r'\1_', cleaned)
    cleaned = re.sub(r'\s+_\s+(\w)', r'_\1', cleaned)

    # ── Step 4.5: I/O 乱码归一化（PDF 字体映射问题：I→1, O→0）──
    # 通用字符级替换，不依赖特定工业名词列表：
    #   "配置1/0信息" → "配置I/O信息", "1/0 信号" → "I/O 信号"
    #   "I/0"、"1/O" 等 PDF 字体错位变体 → "I/O"
    cleaned = re.sub(r'\b1/[0O]\b|\bI/0\b|\b1/O\b', 'I/O', cleaned)
    # 处理中文紧邻情况: "配置1/0信息" / "电控柜 1/0" / "IO 配置"
    cleaned = re.sub(r'([一-鿿])\s*1/0\s*([一-鿿])', r'\1I/O\2', cleaned)
    cleaned = re.sub(r'([一-鿿])\s+I0\b', r'\1 I/O', cleaned)
    cleaned = re.sub(r'\bIO\s*([一-鿿])', r'I/O \1', cleaned)

    # ── Step 4.6: 源头清洗 — 修复 PDF 提取导致的边界错位与标题漏抓 ──
    # 1. 修复无空格标题 (如 "23.设置" -> "23. 设置")
    cleaned = re.sub(r'^(\d{1,2})[\.\)）]([一-龥a-zA-Z])', r'\1. \2', cleaned, flags=re.MULTILINE)
    # 2. 修复 OCR IO 乱码 (如 "25.10状态" -> "25. IO状态")
    cleaned = re.sub(r'(\d{1,2})\.10\s*([一-鿿])', r'\1. IO\2', cleaned)
    cleaned = re.sub(r'\b10\s*([口状态判断输出输入接口]|编号)', r'IO\1', cleaned)
    # 3. 剥离表格竖线，防止阻断后续提取 (如 "| Robot_socket_start" -> "Robot_socket_start")
    cleaned = re.sub(r'^\s*\|\s*', '', cleaned, flags=re.MULTILINE)

    # ── Step 5: 收尾 — 仅压缩 ≥3 个连续空行，保护缩进与正文换行 ──
    cleaned = re.sub(r'\n{4,}', '\n\n\n', cleaned)  # 4+ → 3
    cleaned = re.sub(r'\n{3}', '\n\n', cleaned)       # 3 → 2

    logger.debug(f"🧹 PDF 文本清洗: {len(text)} → {len(cleaned)} 字符"
                 f" (连字={text != cleaned})")
    # ------------------ 🔴 以下为直接新增的代码 ------------------

    # ── Step 6: 修复 PDF OCR 代码断线与下划线换行错乱 ──
    # ① 修复像 robot\n_\n_power\n_\non 这种下划线被多行拆碎的代码
    cleaned = re.sub(r'([a-zA-Z0-9_]+)\s*\n\s*_\s*([a-zA-Z0-9_]+)', r'\1_\2', cleaned)
    cleaned = re.sub(r'([a-zA-Z0-9]+)_\s*\n\s*([a-zA-Z0-9]+)', r'\1_\2', cleaned)
    
    # ② 修复像 set\nrobot 或 robot\npower 这种 SDK 关键字被折行切断的代码
    cleaned = re.sub(r'\b(set|get|robot|arm)\s*\n\s*(robot|power|arm|cmd|time|mode|on|off|send)\b', r'\1_\2', cleaned, flags=re.IGNORECASE)
    
    # ③ 修复对象点号跨行（如 robot.\nset）
    cleaned = re.sub(r'([a-zA-Z0-9_]+)\.\s*\n\s*([a-zA-Z0-9_]+)', r'\1.\2', cleaned)

    # ----- JAKA 手册专属版式清洗 -----
    # 1. 剔除全屏乱飞的页眉、页脚、版本号
    cleaned = re.sub(r'(?:JAKA\s*节卡|节卡机器人|APP\s*使用手册|JAKA\s*ZU(?:®)?\s*APP|Zu\s*APP|APP版本号.*|www\.jaka\.com)', '', cleaned, flags=re.IGNORECASE)
    # 2. 剔除独立存在的页码数字 (例如孤立的一行 '10' 或 '12')
    # 🔴 v25: JAKA/gui_app 专属强化 —— 仅删除 1-2 位孤立数字（页码），保护 ≥3 位数字
    #（端口/波特率/节点地址等表格参数值，如 6502/9600），防止参数单元格被误删
    # 其余轨（C-SDK 等）保持原逻辑 ^\s*\d+\s*$ 完全不变
    if doc_type == "gui_app" or product_id == "JAKA":
        cleaned = re.sub(r'^\s*\d{1,2}\s*$', '', cleaned, flags=re.MULTILINE)
    else:
        cleaned = re.sub(r'^\s*\d+\s*$', '', cleaned, flags=re.MULTILINE)

    # ------------------ ---------------------------------------
    
    return cleaned

# ============================================================
# 🔴 [SDK 轨道专有] Golden TOC 目录树预解析引擎
# ============================================================
def _v4_extract_sdk_toc(full_text: str) -> dict:
    """
    从 SDK 文档前 2000 字符中提取官方 1~30 级标准目录树 (Golden TOC)。
    返回: {28: "28. 机械臂电源上电", 3: "3. 连接机械臂", ...}
    """
    toc_map = {}
    # 截取文档前 2500 字符（涵盖 TOC 目录页）
    front_matter = full_text[:2500]
    
    # 匹配 "28. 机械臂电源上电" 或 "28.机械臂电源上电" 格式
    toc_pattern = re.compile(
        r'^\s*(\d{1,2})\s*[\.、\s]\s*([\u4e00-\u9fa5a-zA-Z0-9_\(\)（）]+)',
        re.MULTILINE
    )
    for m in toc_pattern.finditer(front_matter):
        sec_num = int(m.group(1))
        sec_title = f"{sec_num}. {m.group(2).strip()}"
        # 排除非标题杂音（如 1.安装 属于 valid, 100. 排除）
        if 1 <= sec_num <= 60 and len(sec_title) <= 50:
            toc_map[sec_num] = sec_title

    return toc_map


# 🔴 [新增 1/2] 函数签名增加 doc_type 参数，默认为 gui_app
def _v4_extract_headings(text: str, doc_type: str = "gui_app") -> List[Tuple[int, str, int]]:
    """
    从文本中提取所有标题及其层级。

    兼容数字编号（3.1.5）、Markdown（##）、中文序号（一、）等多种格式。
    支持 doc_type 动态双轨策略，防止 GUI 步骤列表被误判为大纲。

    Returns:
        [(position, title_text, level), ...]
        level: 1=H1, 2=H2, 3=H3, 4=H4
    """
    headings = []
    seen_positions = set()

    # 🔴 v28: 区域状态机 —— 标题提取感知受保护区域（代码块/表格/OCR 补充块）。
    # 此前标题提取与区域保护完全解耦：假标题（OCR 坐标行/表格单元格）照样进标题树、
    # 决定 parent/child 边界与面包屑（实测 309 个污染路径）
    _protected_ranges = _v4_find_protected_ranges(text)

    for pattern, base_level in _V4_HEADING_PATTERNS:
        for m in pattern.finditer(text):
            # 👇 -------------------- 🔴 [新增 2/2] 动态双轨标题拦截 -------------------- 👇
            # 检查当前正则是否为单数字编号模式（例如匹配了 "4. 机械臂上电" 或 "1. 将升级包..."）
            _is_single_digit_pattern = pattern.pattern.startswith(r'^(\d{1,2})\.')
            if _is_single_digit_pattern and doc_type == "gui_app":
                # GUI 轨道：绝对禁止将单数字识别为标题，保护操作步骤 (1. 2. 3.) 不被切碎！
                continue
            # 👆 ------------------------------------------------------------------------- 👆

            pos = m.start()
            if pos in seen_positions:
                continue
            # 🔴 v28: 保护区内的匹配直接跳过（与 _safe_boundary 开区间语义一致）
            if any(p_start < pos < p_end for p_start, p_end, _ in _protected_ranges):
                continue
            seen_positions.add(pos)
            full = m.group(0).strip()
            if not full or len(full) < 3 or len(full) > 85:
                continue
            # 🔴 v28: 数字编号标题形态校验（负向判定）—— 标题文字首字符为 '|' 或
            # 整段仅数字/点/竖线/空白 → 拒绝（OCR 坐标行 "0.000 | 0.000 | 0.000" 等假标题）
            if re.match(r'^[\d.\s|]+$', full) or full.lstrip().startswith('|'):
                continue

            # -------------------- 🔴 [修改 1/3] 拦截 Python 代码注释行 --------------------
            # 排除 SDK 示例代码中的 Python 单行注释（如 `# 机械臂初始化运动` / `#时间等待3秒`）
            if full.startswith('#'):
                # 获取该匹配点前后 120 字符的上下文
                context_start = max(0, pos - 120)
                context_end = min(len(text), pos + 120)
                line_context = text[context_start:context_end]
                
                # 若上下文包含代码特征词，说明这是代码块内部的注释行，坚决拒绝提权为 Heading！
                _CODE_KEYWORDS = ['restype', 'argtypes', 'CDLL', 'ctypes', 'robot.', 'c_int', 'c_float', 'import ']
                if any(kw in line_context for kw in _CODE_KEYWORDS):
                    continue
            # -----------------------------------------------------------------------------

            # 🔴 SDK 表头黑名单过滤：禁止将 SDK 手册表格表头识别为 Heading
            # 剥离数字编号前缀后检查（如 "3. 函数名称" → "函数名称"）
            _stripped_title = re.sub(r'^[\d.]+\s*', '', full).strip()
            if _stripped_title in _SDK_TABLE_HEADER_BLACKLIST:
                continue

            # 推断层级
            groups = m.groups()
            if len(groups) >= 2 and groups[0] and groups[1]:
                title_num = groups[0]
                # "3.1.5" → 层级 = 点号数量 + 1
                dots = title_num.count('.')
                if dots >= 1:
                    level = min(dots + 1, 4)  # 最多 4 级
                    # 🔴 仅 1 个点号（如 "4.1" / "3.2"）→ 强制 H2，
                    # 确保 "X.Y" 格式不被错误提升为更深层级
                    if dots == 1:
                        level = 2
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
    为给定位置范围构建层级面包屑（固定 4 槽位数组）。

    使用固定槽位 slots = ["", "", "", ""] 对应 H1~H4。
    更新第 k 层标题时，强制清空 index > k 的所有旧槽位，
    杜绝平级标题串联嵌套和跨章继承错误。

    大章跳变检测：从标题中提取数字编号的整数部分（兼容 "3.1.5"、"第4章" 等格式），
    跨章时强制重置所有槽位。

    例: [路径: JAKA Zu APP > 硬件与通讯 > Modbus 通讯设置]
    """
    slots = ["", "", "", ""]  # [H1, H2, H3, H4]
    last_root_number = 0

    for pos, title, level in headings:
        if pos > chunk_pos:
            break

        # 提取标题的数字编号整数部分（兼容 "3.1.5"、"第4章"、"4.1" 等格式）
        root_match = re.search(r'(?:第|\b)(\d+)', title)
        current_root = int(root_match.group(1)) if root_match else 0

        # 🔴 顶级大章重置：整数部分跳变时（如 3.x → 4.x），强制清空所有槽位
        if current_root > 0 and last_root_number > 0 and current_root != last_root_number:
            slots = ["", "", "", ""]

        last_root_number = current_root if current_root > 0 else last_root_number

        # 槽位索引（level 1-based → 0-based）
        slot_idx = max(0, min(level - 1, 3))

        # 🔴 关键：写入当前槽位，并强制清空所有更深层槽位
        slots[slot_idx] = title
        for clear_idx in range(slot_idx + 1, 4):
            slots[clear_idx] = ""

    # 组装非空槽位
    path_parts = [s for s in slots if s]
    if not path_parts:
        return ""

    return " > ".join(path_parts)


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


def _ocr_kv_normalize_row(cells: List[str]) -> str:
    """
    🔴 v29: OCR 行键值归一化 —— "端口： | 6502" → "端口：6502"。
    将 `|` 离散分隔转为 Dense 友好的键值语义（标签与数值处于同一局部窗口）。
    标签形态：≤12 字且以 ：/: 结尾；其后紧随的非标签项合并为值。
    """
    out = []
    _i = 0
    while _i < len(cells):
        _c = cells[_i].strip()
        if re.match(r'^.{1,12}[：:]$', _c) and _i + 1 < len(cells):
            _v = cells[_i + 1].strip()
            if not re.match(r'^.{1,12}[：:]$', _v):
                out.append(f"{_c}{_v}")
                _i += 2
                continue
        out.append(_c)
        _i += 1
    return "，".join(out)


def _ocr_merge_cross_line(lines: List[str], page_num: int) -> List[str]:
    """
    🔴 v29: OCR 跨行键值合并 —— "从站节点号：" + 下一行纯数值 → "从站节点号：1"。
    防误伤：排除与当前页码 ±1 相等的纯数字（页脚页码）；下一行仍是标签（以 ：结尾）则不合并。
    """
    merged = []
    for _ln in lines:
        _ln = _ln.strip()
        if not _ln:
            continue
        _prev = merged[-1] if merged else ""
        _is_label = bool(re.match(r'^.{1,12}[：:]$', _prev))
        _is_value = bool(re.match(r'^\d{1,6}$', _ln) or re.match(r'^[\d.~/]{1,12}$', _ln))
        _is_page = _ln in (str(page_num), str(page_num - 1), str(page_num + 1))
        if _is_label and _is_value and not _is_page:
            merged[-1] = merged[-1] + _ln
        else:
            merged.append(_ln)
    return merged


def _v4_extract_text_universal(pdf_path: str, doc_type: str = "general") -> Tuple[str, int, int]:
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

    # 🔴 v29: xref 放置次数预扫描 + 全局 OCR 去重
    # 页眉 logo 类（xref 253 实测放置 137 页）在任何尺寸阈值下都漏网——放置 >20 页的 xref 跳过
    _xref_page_count = {}
    try:
        for _pi in range(total_pages):
            for _img in (doc[_pi].get_images(full=True) or []):
                _xref_page_count[_img[0]] = _xref_page_count.get(_img[0], 0) + 1
    except Exception:
        pass
    _ocr_done_xrefs = set()

    # ── Last Known Header 追踪器 ──
    # 格式: {"number": "3.1.5", "title": "3.1.5 Modbus 通讯设置",
    #        "path": "JAKA Zu APP > 硬件与通讯 > Modbus 通讯设置", "level": 3}
    last_header: dict = {"number": "", "title": "", "path": "", "level": 0}

    # 预编译标题正则（与 _V4_HEADING_PATTERNS 兼容）
    _HEADER_TRACK_RE = re.compile(
        r'(?:^|\n)\s*('
        r'\d+(?:\.\d+){1,3}\s+.+?'          # 数字编号: 3.1.5 标题
        r'|第[一二三四五六七八九十\d]+\s*[章节]\s*.+?'  # 中文编号: 第一章 / 第1 章 (v28 容忍空格)
        r'|[（(]?[一二三四五六七八九十]+[）)]?\s*[、,，\s].+?'  # 中文序号
        r'|#{1,4}\s+.+?'                     # Markdown H
        r'|\d{1,2}[\.\)）]\s+.+?'             # 纯数字+点号
        r')(?:\r?\n|$)',
        re.MULTILINE,
    )

    def _try_update_header(page_text: str, page_idx: int):
        """
        🔴 v28: 数字编号标题层级栈 —— 仅接受数字编号/章节编号标题；
        弹栈规则 = 栈顶 level >= 新 level 或 新编号不以栈顶编号为前缀。
        修复跨章叠加（旧实现 `_prev_path[:num_dots]` 用新标题点号数截旧路径头部，
        产生 "关闭恒力柔顺控制 > 3.1.5.5" 式污染）与 OCR 行污染（配合保护区跳过 + 形态校验）。
        """
        # 保护区跳过：OCR 补充块内的行不参与标题追踪
        _page_protected = _v4_find_protected_ranges(page_text)
        _candidates = []
        for _m in _HEADER_TRACK_RE.finditer(page_text):
            if any(p_start < _m.start() < p_end for p_start, p_end, _ in _page_protected):
                continue
            _raw = _m.group(1).strip()
            if 4 <= len(_raw) <= 85:
                _candidates.append(_raw)
        if not _candidates:
            return
        raw_title = _candidates[-1]  # 取本页最后一个有效标题作为章节锚点

        # 🔴 v28: 数字编号标题形态校验（负向判定）—— OCR 坐标行 "0.000 | 0.000" 拒绝
        if re.match(r'^[\d.\s|]+$', raw_title) or raw_title.lstrip().startswith('|'):
            return

        _num_match = re.match(r'^(\d+(?:\.\d+)*)', raw_title)
        if _num_match:
            number = _num_match.group(1)
            level = min(number.count('.') + 1, 4)   # 与标题树 L713 统一封顶
        elif re.match(r'^第[一二三四五六七八九十\d]+\s*[章节]', raw_title):
            number = ""
            level = 1 if '章' in raw_title[:3] else 2
        else:
            # 裸字/中文序号标题无可靠层级语义 → 仅更新锚点 title，不入栈
            last_header["title"] = raw_title
            return

        stack = last_header.setdefault("stack", [])
        # 弹栈：层级不降 或 编号前缀不匹配（3.1.5.5 的祖先只可能是 3/3.1/3.1.5）
        while stack and (
                stack[-1][0] >= level
                or (stack[-1][1] and number and not number.startswith(stack[-1][1]))):
            stack.pop()
        stack.append((level, number, raw_title))
        last_header["number"] = number
        last_header["title"] = raw_title
        last_header["path"] = " > ".join(t for _, _, t in stack)
        last_header["level"] = level

    for page_idx in range(total_pages):
        page = doc[page_idx]
        
        # -------------------- 🔴 [终极修复 1] Y 坐标物理排序 --------------------
        # 🔴 v28: gui_app 轨升级为 line 级几何重建 —— PyMuPDF block 会把表格整行
        # 单元格合并为单 block（内部 \n 分隔），block 级排序无法重建表格行。
        # line 级按 y 中心聚类 + x 排序；仅 ≥2 项且单元格短的带包装为 Markdown
        # 表格行（自动受 _PROTECTED_BLOCK_RE 保护 + 单元格不再裸行被 H3 兜底提权）
        if doc_type == "gui_app":
            try:
                _dict_data = page.get_text("dict")
                _lines_all = []
                for _blk in _dict_data.get("blocks", []):
                    if _blk.get("type") != 0:
                        continue
                    for _ln in _blk.get("lines", []):
                        _txt = "".join(_sp.get("text", "") for _sp in _ln.get("spans", [])).strip()
                        if not _txt:
                            continue
                        _bbox = _ln.get("bbox") or (0, 0, 0, 0)
                        _lines_all.append(
                            (round((_bbox[1] + _bbox[3]) / 2 / 12) * 12, _bbox[0], _txt))
                _lines_all.sort(key=lambda t: (t[0], t[1]))
                _merged = []
                _cur_row = None
                _cur_items = []
                for _yb, _xb, _txt in _lines_all:
                    if _cur_row is None:
                        _cur_row = _yb
                    if _yb == _cur_row:
                        _cur_items.append((_xb, _txt))
                    else:
                        _merged.append((_cur_row, _cur_items))
                        _cur_row = _yb
                        _cur_items = [(_xb, _txt)]
                if _cur_items:
                    _merged.append((_cur_row, _cur_items))
                _row_texts = []
                for _yb, _items in _merged:
                    _items.sort(key=lambda t: t[0])
                    _cells = [t for _, t in _items]
                    if len(_cells) >= 2 and all(len(c) <= 40 for c in _cells):
                        # 表格行：包装为 Markdown 形态 → 受保护 + 不被标题提取
                        _row_texts.append("| " + " | ".join(_cells) + " |")
                    else:
                        # 单 item 带 / 长文本行：原样输出（保住标题/图注/散文）
                        _row_texts.append(_cells[0] if _cells else "")
                page_text = "\n".join(_row_texts)
            except Exception:
                page_text = ""
        else:
            # C-SDK 轨保持原 block 级逻辑（严禁触碰 OpenC3/OpenR6）
            blocks = page.get_text("blocks")
            if blocks:
                text_blocks = [b for b in blocks if b[6] == 0]
                text_blocks.sort(key=lambda b: (round(b[1] / 10) * 10, b[0]))
                page_text = "\n\n".join([b[4].strip() for b in text_blocks if b[4].strip()])
            else:
                page_text = ""
        # ---------------------------------------------------------------------------

        # ── Step 1: 字符密度检测 ──
        effective_chars = len(re.sub(r'\s', '', page_text))
        needs_ocr = effective_chars < _PAGE_DENSITY_THRESHOLD

        # 👇 修改点 2：定义强制 OCR 标识，赋予 JAKA (gui_app) 专属特权
        force_image_ocr = (doc_type == "gui_app")

        # ── Step 2: 更新标题追踪器（正常密度页）──
        if not needs_ocr and effective_chars >= _PAGE_DENSITY_THRESHOLD:
            _try_update_header(page_text, page_idx)

        # ── Step 3: OCR 补漏（低密度页 或 🔴 强制图片OCR）──
        ocr_lines = []

        # 👇 修改点 3：条件放宽，即使文本很密，只要是 GUI 手册也强制扫图
        if (needs_ocr or force_image_ocr) and ocr is not None:
            image_list = page.get_images(full=True)
            _img_seq = 0   # 🔴 v29: 本页图片序号（按图子块化）
            for img_info in (image_list or []):
                xref = img_info[0]
                # 🔴 v26 保留: 该图在页面上的放置矩形（供 gui_app 面积比过滤）
                _img_rect = None
                try:
                    _rects = page.get_image_rects(xref)
                    if _rects:
                        _img_rect = max(_rects, key=lambda r: r.width * r.height)
                except Exception:
                    pass
                try:
                    base_image = doc.extract_image(xref)
                    if base_image is None:
                        continue
                    image_bytes = base_image.get("image")
                    if not image_bytes:
                        continue

                    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    h, w = pil_img.size[1], pil_img.size[0]
                    if force_image_ocr:
                        # 🔴 v29: 过滤重构 —— 废除 1.5% 面积比（实测拦截了 230 个
                        # 0.5%~1.5% 带的目标小参数图）；保留数据支撑的绝对下限
                        #（面积 ≥0.5% 且边长 ≥40px，仅拦图标级）
                        if _img_rect is None:
                            continue
                        _page_area = page.rect.width * page.rect.height
                        _area_ratio = (_img_rect.width * _img_rect.height) / max(1.0, _page_area)
                        if _area_ratio < 0.005 or _img_rect.width < 40 or _img_rect.height < 40:
                            continue
                        # 页眉/页脚 logo 类：放置 >20 页的 xref 跳过（任何尺寸阈值都拦不住）
                        if _xref_page_count.get(xref, 0) > 20:
                            continue
                        # 同一 xref 全局只 OCR 一次（页眉 logo 曾被 OCR 137 次纯浪费）
                        if xref in _ocr_done_xrefs:
                            continue
                        _ocr_done_xrefs.add(xref)
                    elif h < 100 or w < 100:
                        continue

                    np_img = np.array(pil_img)
                    # 🔴 核心修复：正确解包 RapidOCR 的返回值 (result_list, elapse_time)
                    ocr_res = ocr(np_img)
                    if not ocr_res or ocr_res[0] is None:
                        continue

                    if doc_type == "gui_app":
                        # 🔴 v25: JAKA/gui_app 专属 OCR 表格行对齐 —— 按 Y 坐标聚类成行、
                        # X 坐标排序，保证"标签 | 值"同行输出（6502/9600 与属性词不分离）
                        # 🔴 v27: 回退 CTM Y 归位（v26 因 PDF 坐标系不一致插错位置污染切片）
                        # 🔴 v29: 键值法语义 —— 行内"标签：| 值"归一为"标签：值"、
                        # 图内跨行"标签：/数值"配对、按图子块化（[图表内容包含：] 前缀，
                        # 治 Dense 块级稀释 —— 多图拼一页尾块会稀释参数上下文）
                        _img_ocr_lines = []   # 每图独立收集（按图子块化）
                        _ocr_items = []
                        for _item in ocr_res[0]:
                            _box = _item[0] if _item[0] else None
                            _text = str(_item[1]).strip()
                            if len(_text) < 2:
                                continue
                            if _box:
                                _y_c = sum(_p[1] for _p in _box) / len(_box)
                                _x_l = min(_p[0] for _p in _box)
                            else:
                                _y_c, _x_l = 0.0, 0.0
                            _ocr_items.append((_y_c, _x_l, _text))
                        _ocr_items.sort(key=lambda t: (round(t[0] / 12), t[1]))
                        _row_lines = []
                        _cur_row = None
                        for _y_c, _x_l, _text in _ocr_items:
                            _row_key = round(_y_c / 12)
                            if _cur_row is None:
                                _cur_row = _row_key
                            if _row_key == _cur_row:
                                _row_lines.append(_text)
                            else:
                                _img_ocr_lines.append(_ocr_kv_normalize_row(_row_lines))
                                total_ocr_chars += sum(len(l) for l in _row_lines)
                                _row_lines = [_text]
                                _cur_row = _row_key
                        if _row_lines:
                            _img_ocr_lines.append(_ocr_kv_normalize_row(_row_lines))
                            total_ocr_chars += sum(len(l) for l in _row_lines)
                        # 图内跨行键值合并 + 按图子块追加
                        if _img_ocr_lines:
                            _img_ocr_lines = _ocr_merge_cross_line(_img_ocr_lines, page_idx + 1)
                            _img_seq += 1
                            ocr_lines.append(f"[图表内容包含：本页第{_img_seq}张截图]")
                            ocr_lines.extend(_img_ocr_lines)
                    else:
                        # C-SDK 轨保持原逻辑（严禁触碰 OpenC3/OpenR6 切片）
                        lines = []
                        for item in ocr_res[0]:
                            # item 的结构是: [box坐标, 识别文本, 置信度得分]
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

        # ── Step 4: 页面内容组装 — OCR 页尾追加 + 章节上下文继承 ──
        page_parts = []
        if page_text.strip():
            page_parts.append(page_text.strip())

        if ocr_lines:
            # 🔴 v27: 低密度页（OCR 触发页）把 OCR 文本也喂给标题追踪器，
            # 避免 OCR 内容继承上一节的章节归属（低成本改进）
            if needs_ocr:
                try:
                    _try_update_header("\n".join(ocr_lines), page_idx)
                except Exception:
                    pass
            # 章节上下文继承：OCR 文字继承 Last Known Header
            section_header = ""
            if last_header.get("title"):
                section_header = (
                    f"\n[路径: {last_header['path']}]"
                    f"\n[章节: {last_header['title']}]"
                )
            # 🔴 v27: 安全追加在该页最后一个 Header 层级之下（回退 v26 CTM Y 归位）
            # 🔴 v29: 块级前缀简化（子块级 [图表内容包含：] 已带语义）
            ocr_header = (
                f"[本页图片解析参数: page={page_idx + 1}]"
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


# ── 受保护区域正则: 代码块 ```...```、Markdown 表格 |...|、OCR 补充块 ──
# 🔴 v28: 第三分支 OCR 补充块 —— 标记行起，锚定 \n\n 页分隔止。
# 必须锚定页分隔而非"下一个标记"（否则会吞掉下一页全部正文，标题树塌陷）
# 🔴 v29: 标记同步 —— 块级 [本页图片解析参数: page=N] 与子块级 [图表内容包含：] 均入保护区
_PROTECTED_BLOCK_RE = re.compile(
    r'(```[\s\S]*?```)'                              # 代码块
    r'|'                                              # 或
    r'((?:^\|.+\|[\s\S]*?)(?=\n\n|\n(?:[^|]|\Z)|\Z))'  # Markdown 表格
    r'|'                                              # 或
    r'(^\[(?:本页图片解析参数|图表内容包含)：[^\]]*\][\s\S]*?(?=\n\n|\Z))',  # OCR 补充块
    re.MULTILINE,
)


def _v4_find_protected_ranges(text: str) -> List[Tuple[int, int, str]]:
    """
    扫描全文，标记不可切分的受保护区域（代码块 + 表格 + OCR 补充块）。
    Returns: [(start, end, type), ...]  例如 [(100, 300, "code"), (500, 700, "table")]
    """
    ranges = []
    for m in _PROTECTED_BLOCK_RE.finditer(text):
        if m.group(1):
            rtype = "code"
        elif m.group(2):
            rtype = "table"
        else:
            rtype = "ocr"   # 🔴 v28: OCR 补充块（页尾图片解析文本，不得参与标题提取）
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
    """构建 Child Doc 正文前缀（面包屑 + 章节 + 函数名标注）。

    调用方已通过 _sanitize_section_title / 4 级 Fallback 链完成清洗，
    此处不再重复调用 _sanitize_section_title，避免双重清洗。
    """
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
    # 🔴 v27: 兼容文档原文 `robot = CDLL(...)`（`from ctypes import *` 后省略 `ctypes.` 前缀）
    # 实测两份 SDK 文档均为此写法——此前正则只匹配 ctypes.CDLL，导致 sdk_header 丢失 CDLL 行
    cdll_match = re.search(
        r'(?:robot\s*=\s*)?(?:ctypes\.)?CDLL\s*\(\s*(?:[rR])?["\']([^"\']+)["\']\s*\)',
        full_text, re.IGNORECASE,
    )
    if cdll_match:
        dll_name = cdll_match.group(1)
        header_parts.append(f"import ctypes")
        header_parts.append(f"robot = ctypes.CDLL(\"{dll_name}\")")
    else:
        # Fallback: 搜索任何 ctypes import + CDLL 的模式（含 from ctypes import * 与裸 CDLL）
        cdll_block = re.search(
            r'(?:import\s+ctypes|from\s+ctypes\s+import\s+\*).*?'
            r'(?:ctypes\.)?CDLL\s*\(\s*(?:[rR])?["\'][^"\']+["\']\)',
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
    # 👇 ================= 🔴 新增：调整顺序，先获取 doc_type ================= 👇
    # ── 1. 提前解析 doc_type ──
    doc_type = _resolve_doc_type(product_id)
    
    # ── 2. 将 doc_type 喂给提取引擎，实现动态拦截 ──
    headings = _v4_extract_headings(full_text, doc_type=doc_type)
    protected = _v4_find_protected_ranges(full_text)
    # 👆 ========================================================================= 👆

    # ── 🔴 SDK 全局代码头提取与动态切片大小分配 ──
    
    # 👇 ================= v22: 动态切片策略 ================= 👇
    if doc_type == "gui_app" or product_id == "JAKA":
        # GUI 手册 (JAKA) 包含大量长连续步骤，切片扩容防断裂
        child_chunk_size = 1500  
        parent_chunk_size = max(parent_chunk_size, 2000) # 父块同步扩容
        logger.info(f"  📐 GUI 轨触发: 切片容量扩容至 Child={child_chunk_size}, Parent={parent_chunk_size}")
    else:
        # SDK 文档保持原有精细切片，防止多个 API 混叠
        # 使用传入的 default 值 (通常为 400 左右)
        pass 
    # 👆 ======================================================= 👆

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

    # 🔴 核心修复：必须使用 <=，否则“第1章”(lv=1) 会被直接跳过丢弃！
    h_parent = [(pos, title, lv) for pos, title, lv in headings if lv <= parent_level]

    # 🔴 架构级补全：如果第一章标题上方还有前导文字（扉页、目录），将其保护起来
    parent_boundaries = [p_start for p_start, _, _ in h_parent] + [len(full_text)]
    if parent_boundaries and parent_boundaries[0] > 0:
        h_parent.insert(0, (0, "文档说明与前言", parent_level))

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

        # 👇 ================= 🔴 核心修复：跨级大纲扫描 ================= 👇
        # 寻找下一个同级或更高级别的标题位置，作为大纲的扫描终点
        # 这样 H1(第1章) 的大纲就能一直扫描到下一个 H1(第2章) 之前，把所有 1.x 都囊括进来！
        toc_end = len(full_text)
        for j in range(i + 1, len(h_parent)):
            if h_parent[j][2] <= p_lv:
                toc_end = h_parent[j][0]
                break

        # 子章节 TOC
        child_titles = [
            title for pos, title, lv in headings
            if p_start < pos < toc_end and lv > p_lv
        ]
        toc = "\n".join(f"- {t}" for t in child_titles[:25])  # 扩容大纲条数到 25
        
        # 🔴 注意这里改成了 [章节大纲参考]: ，为了配合我们在 rag_chain.py 里的提权触发词！
        if toc:
            parent_text = f"{parent_text}\n\n[章节大纲参考]:\n{toc}"
        # 👆 ============================================================= 👆

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


# ── 🔴 v13: SDK 轨状态机 API 块解析器 ──

# -------------------- 🔴 [终极修复 2] 纯净版状态机 --------------------
_SDK_BLOCK_BOUNDARY_RE = re.compile(
    r'(?:^|\n)'
    r'(?='
    r'[ \t]*\d{1,2}\.\s*[^\n]+'      # ① 数字小节标题 (严格限定为 点号. 排除 1、 列表项)
    r'|'
    r'[ \t]*(?:函数名称|函数说明|函数名)\s*'  # ② 中文表头 (兼容前导空格)
    r')',
    re.MULTILINE,
)


# -------------------- 🔴 [修改 3/3] 标题清洗与伪标题过滤 --------------------
# 绝不允许被识别为 section_title 的伪标题关键词
_PSEUDO_SECTION_BLACKLIST = frozenset({
    "时间等待", "命令发送", "示例代码", "代码示例", "调用示例",
    "参数说明", "返回值", "功能描述", "函数说明", "注意事项", "备注",
})

def _sanitize_section_title(title: str) -> str:
    """
    标题元数据清洗器 — 剥离 PDF 提取引入的脏字符并过滤伪标题。
    """
    if not title:
        return ""
    # Step 1: 换行 → 空格，去掉 # 和多余空白
    cleaned = re.sub(r'[\n\r]+', ' ', title).strip()
    cleaned = re.sub(r'^#+\s*', '', cleaned)
    # Step 2: 剥离已知前缀
    cleaned = re.sub(r'^(?:函数说明|函数名称|函数名|方法名|API)[\s\n:：]*', '', cleaned, flags=re.IGNORECASE)
    # Step 3: 压缩连续空白
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # Step 4: 🔴 伪标题黑名单校验 (如 "时间等待3秒", "示例代码" 等直接剔除)
    for bad_kw in _PSEUDO_SECTION_BLACKLIST:
        if bad_kw in cleaned and len(cleaned) < 15:
            return ""  # 返回空字符串，促使调用方继承父级 H2 标题

    return cleaned
# -----------------------------------------------------------------------------


def _is_skeleton_chunk(content: str) -> bool:
    """
    检测并过滤骨架/目录占位块 — 在建库阶段直接丢弃。

    判定规则（全部满足 → True → 丢弃）:
      1. 文本长度 < 150 字符
      2. 不含实际代码特征（def、argtypes、restype、ctypes.CDLL、```python）
      3. 不含实质参数/功能描述（参数说明、返回值、功能描述）
      4. 含占位符文本（示例代码、代码示例、纯数字标题序列）
    """
    if not content or len(content.strip()) >= 150:
        return False
    content = content.strip()

    # 🔴 注入免死金牌：只要这段话里有底层 SDK 初始化的核心词，绝对不准丢弃！
    if any(kw in content for kw in ["import ctypes", "CDLL", "collrob_sdk", "py_dll", "动态链接库"]):
        return False

    # 有实际代码 → 保留
    _has_code = bool(re.search(
        r'(?:def\s+\w+\s*\(|\.restype|\.argtypes|\bctypes\.|```python|=.*ctypes\.CDLL)',
        content,
    ))
    # 🔴 只要包含实质 SDK 代码调用特征，绝对保留（防止误杀含代码的碎片）
    if not _has_code and bool(re.search(
        r'(?:robot\s*\.\s*\w+|def\s+\w+|\.restype|\.argtypes|CDLL|ctypes|'
        r'\b(?:set|get|end|close|start)_robot_\w+)',
        content, re.IGNORECASE,
    )):
        return False
    if _has_code:
        return False

    # 有实质参数/功能描述 → 保留
    _has_substance = bool(re.search(
        r'(?:参数说明|返回值|功能描述|函数说明|功能说明|注意事项)',
        content,
    ))
    if _has_substance:
        return False

    # 纯占位符文本
    _skeleton_markers = ['示例代码', '代码示例', '调用示例', '见示例', '详见示例']
    if any(m in content for m in _skeleton_markers):
        return True

    # 纯数字标题序列（目录骨架）
    lines = [l.strip() for l in content.split('\n') if l.strip()]
    if len(lines) <= 2 and all(re.match(r'^\d{1,2}[\.\、\s]', l) for l in lines):
        return True

    # 🔴 v28: TOC 目录行特征 —— 点线目录格式（"3.2.5 Ethernet/IP IO ...... 31"）
    # 目录页 chunk 无正文参数，被召回会触发误拒答（E29 根因之一）；通用形态判定非业务词
    _toc_lines = 0
    for l in lines:
        if re.match(r'^\d+(?:\.\d+){1,3}\s+.*?\.{2,}\s*\d+\s*$', l):
            _toc_lines += 1
    if _toc_lines >= max(2, len(lines) // 2):
        return True

    return False


def _v4_parse_sdk_state_machine(text: str) -> List[Tuple[int, int, str]]:
    if not text:
        return []

    # ================= 修改开始 =================
    # 不预设第一个边界为 0，让正则自己找边界。
    # 如果开头有非标题内容，自然会形成一个独立的块，我们可以后续过滤。
    matches = list(_SDK_BLOCK_BOUNDARY_RE.finditer(text))
    
    if not matches:
        return [(0, len(text), "")]

    boundaries = []
    titles = []
    
    # 补齐开头的前言/目录块 (如果第一个匹配不是从 0 开始)
    if matches[0].start() > 0:
        boundaries.append(0)
        titles.append("")  # 前言部分没有明确提取的 SDK 标题

    for m in matches:
        boundaries.append(m.start())
        titles.append(m.group(0).strip())

    boundaries.append(len(text))
    titles.append("") # 最后一个结尾补充
    # ================= 修改结束 =================

    _MIN_BLOCK_GAP = 20
    merged_boundaries = [boundaries[0]]
    merged_titles = [titles[0]]
    for i in range(1, len(boundaries)-1):
        gap = boundaries[i] - merged_boundaries[-1]
        if gap < _MIN_BLOCK_GAP and i < len(boundaries) - 1:
            if len(titles[i]) > len(merged_titles[-1]):
                merged_titles[-1] = titles[i]
        else:
            merged_boundaries.append(boundaries[i])
            merged_titles.append(titles[i])
    # 收尾最后一个边界
    if len(text) - merged_boundaries[-1] > 0:
        merged_boundaries.append(len(text))

    blocks = []
    for i in range(len(merged_boundaries) - 1):
        start = merged_boundaries[i]
        end = merged_boundaries[i + 1]
        if start >= end:
            continue

        raw_title = merged_titles[i]
        block_text = text[start:end]
        # 提取真正的标题（兼容前导空格）
        _title_match = re.search(
            r'(?:^[ \t]*\d{1,2}[\.\、\s]\s*[^\n]+|^[ \t]*函数名称\s*\n?\s*(\w+)|^[ \t]*函数说明\s*\n?\s*(\w+))',
            block_text, re.MULTILINE,
        )
        if _title_match:
            title = _title_match.group(0).strip()
        else:
            title = raw_title
        blocks.append((start, end, title))

    return blocks


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

    doc_type = _resolve_doc_type(product_id)  # 🔴 v5: 双轨制切片策略

    # ── 🔴 v11: SDK 轨状态机快速路径 — 替代标题树切分 ──
    # ── 🔴 SDK 轨状态机快速路径：整块解包，禁止二次拆切 ──
    if doc_type == "c_sdk":
        section_text = full_text[section_start:section_end]
        sdk_blocks = _v4_parse_sdk_state_machine(section_text)
        if sdk_blocks:
            # 1. 提取所有原始候选块
            raw_blocks = []
            for block_start_rel, block_end_rel, block_title in sdk_blocks:
                block_abs_start = section_start + block_start_rel
                block_abs_end = section_start + block_end_rel
                block_text = full_text[block_abs_start:block_abs_end].strip()
                if not block_text or _is_skeleton_chunk(block_text):
                    continue
                raw_blocks.append((block_abs_start, block_abs_end, block_text, block_title))

            # 2. 🔴 微碎片向下自动缝合 (Micro-Chunk Auto-Merge + API 排他锁)
            # 解决 "28. 机械臂电源上电" 等短标题被切成孤儿碎片的问题
            # 同时确保不同 API 绝不缝合：提取双方主 API 函数名，不同则强制提交
            merged_blocks = []
            buf_start, buf_end = None, None
            buf_text, buf_title = "", ""

            for b_start, b_end, b_text, b_title in raw_blocks:
                if buf_text:
                    buf_text = buf_text + "\n\n" + b_text
                    buf_end = b_end
                    if not buf_title:
                        buf_title = b_title
                else:
                    buf_start = b_start
                    buf_end = b_end
                    buf_text = b_text
                    buf_title = b_title

                # 判定条件：合并后文本长度 >= 60 字符，或包含明显的 SDK 代码/方法调用 → 提交为一个独立 API 块
                has_code = any(kw in buf_text for kw in ["robot.", "dll.", "ctypes", "("])
                if len(buf_text) >= 60 or has_code:
                    merged_blocks.append((buf_start, buf_end, buf_text, buf_title))
                    buf_start, buf_end, buf_text, buf_title = None, None, "", ""

            # 处理末尾残留的微碎片
            if buf_text:
                if merged_blocks:
                    prev_start, prev_end, prev_text, prev_title = merged_blocks[-1]
                    merged_blocks[-1] = (prev_start, buf_end, prev_text + "\n\n" + buf_text, prev_title)
                else:
                    merged_blocks.append((buf_start, buf_end, buf_text, buf_title))

            # 3. 将缝合后的整块生成 Document (含 4 级 Title Fallback)
            for block_abs_start, block_abs_end, block_text, block_title in merged_blocks:
                if len(block_text) < 15 or _is_skeleton_chunk(block_text):
                    continue

                breadcrumb = _v4_build_breadcrumb(headings, block_abs_start, block_abs_end)
                func_names = _v4_extract_function_names(block_text)
                is_api = len(func_names) > 0

                # 🔴 4 级 Title Fallback 链 — section_title 永不落空
                _clean_sec = _sanitize_section_title(block_title)          # L1: 状态机标题
                if not _clean_sec:
                    _clean_sec = _sanitize_section_title(breadcrumb)       # L2: 面包屑路径
                if not _clean_sec:
                    # L3: 回溯当前 Parent 切片的 section_title (H2 父标题)
                    for pos, title, lv in headings:
                        if pos <= block_abs_start and lv == 2:
                            _candidate = _sanitize_section_title(title)
                            if _candidate:
                                _clean_sec = _candidate
                        elif pos > block_abs_start:
                            break
                if not _clean_sec:
                    _clean_sec = "SDK 接口说明"                            # L4: 硬兜底

                prefix = _build_child_prefix(source, breadcrumb, _clean_sec, func_names)

                children.append(Document(
                    page_content=f"{prefix}{block_text}",
                    metadata={
                        "source": source, "product_id": product_id,
                        "doc_type": doc_type, "chunk_type": "child", "parent_id": parent_id,
                        "section_title": _clean_sec,
                        "section_level": 3,
                        "function_names": ",".join(func_names) if func_names else "",
                        "api_atomic": is_api, "is_api": is_api,
                        "sdk_header": sdk_header if (sdk_header and is_api) else "",
                    },
                ))
        return children

    # =========================================================
    # 🔴 以下为 GUI 轨 (及默认轨) 的核心切分与微缩大纲注入逻辑
    # =========================================================

    # 🟢 注入点 1：跨边界搜集当前章节下的所有子标题，生成 "微缩大纲 (Mini-TOC)"
    toc_text = ""
    if doc_type == "gui_app":
        # 获取当前章节的层级
        current_lv = 2  # 默认兜底
        for pos, title, lv in headings:
            if pos == section_start:
                current_lv = lv
                break
        
        _local_subs = []
        # 突破 section_end 限制，向后搜寻隶属于本章的所有子标题
        for pos, title, lv in headings:
            if pos <= section_start:
                continue
            if lv <= current_lv:
                break  # 遇到同级或更高级别（如第2章），停止大纲搜集
            if lv <= current_lv + 2: # 收集往下两级的标题
                _local_subs.append(title.strip())

        # 👇 ================= v22 修复 HALL-1.1 ================= 👇
        if _local_subs:
            # 限制微缩大纲上限严格为 5 条，避免"标题噪声"引发大模型幻觉
            _MAX_TOC_ITEMS = 5
            toc_lines = _local_subs[:_MAX_TOC_ITEMS]
            if len(_local_subs) > _MAX_TOC_ITEMS:
                toc_lines.append("... (更多章节略)")
            
            toc_text = "\n\n[章节大纲参考]:\n- " + "\n- ".join(toc_lines)
        # 👆 ===================================================== 👆

    sub_headings = [
        (pos, title, lv) for pos, title, lv in headings
        if section_start <= pos < section_end and lv > 2
    ]

    if not sub_headings:
        # 🔴 H2 导言区：无 H3 子标题 → text 是 Parent 标题后的导言段落
        text = full_text[section_start:section_end].strip()
        
        # 🟢 注入点 2：如果该块只有大标题和几句话，强制把大纲拼在后面（作为纯背景提供）
        if toc_text:
            text += toc_text
            
        if text:
            breadcrumb = _v4_build_breadcrumb(headings, section_start, section_end)
            # 从 headings 中回溯当前 Parent 的 H2/H1 标题
            _parent_title = ""
            for pos, title, lv in headings:
                if pos <= section_start and lv <= 2: # 兼容 H1 和 H2
                    _parent_title = title
                elif pos > section_start:
                    break
            children = _split_text_into_children(
                text, source, product_id, parent_id,
                child_chunk_size, breadcrumb, _parent_title or breadcrumb,
                sdk_header=sdk_header, doc_type=doc_type,
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

        # 🟢 注入点 3：仅在该大章节切分出的第一块 (即章节导言部分) 末尾注入大纲
        if i == 0 and toc_text:
            text += toc_text

        if not text or len(text) < 15:
            continue

        current_title = ""
        current_level = 3
        for pos, title, lv in sub_headings:
            if _safe_boundary(pos, protected) == s:
                current_title = title
                current_level = lv
                break

        # 🔴 架构级补全：若当前为章节总览/导言段落（未匹配到 H3 子标题），自动继承父级标题
        if not current_title:
            for pos, title, lv in headings:
                if pos <= s and lv <= 2:
                    current_title = title
                    current_level = lv
                elif pos > s:
                    break

        breadcrumb = _v4_build_breadcrumb(headings, s, e)
        sub_children = _split_text_into_children(
            text, source, product_id, parent_id,
            child_chunk_size, breadcrumb, current_title, current_level,
            sdk_header=sdk_header, doc_type=doc_type,
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
    doc_type: str = "general",  # 🔴 v5: 双轨制切片策略
) -> List[Document]:
    """
    将文本段落按「先提取受保护块，再按段落切分」的 Tokenize 模式切分为 Child Docs。

    保护机制:
      - ```代码块``` 先整体提取，无论多大绝不拦腰切断，防止 Markdown AST 崩溃
      - |表格行| 作为受保护整体保留
      - 普通文本按 \\n\\n 段落边界累积合并，超出 chunk_size 时 flush

    🔴 v5 双轨制:
      - gui_app: 废除字符硬切，按 Heading-to-Heading 完整保留 UI 步骤
      - c_sdk / general: 保留段落累积逻辑，放宽代码块保护
    """
    children = []
    child_idx = 0

    def _emit_child(content: str):
        nonlocal child_idx
        content = content.strip()
        if len(content) < 10:
            return
        # 🔴 v13: 离线骨架块过滤 — 建库阶段直接丢弃占位/目录块
        if _is_skeleton_chunk(content):
            return
        func_names = _v4_extract_function_names(content)
        is_api = len(func_names) > 0
        # 🔴 3 级 Title Fallback 链 (非 c_sdk 路径 — 无 headings 可用，L3 跳过)
        _clean_sec = _sanitize_section_title(section_title)          # L1: 传入标题
        if not _clean_sec:
            _clean_sec = _sanitize_section_title(breadcrumb)         # L2: 面包屑路径
        if not _clean_sec:
            _clean_sec = "技术文档"                                    # L4: 硬兜底
        prefix = _build_child_prefix(source, breadcrumb, _clean_sec, func_names)
        children.append(Document(
            page_content=f"{prefix}{content}",
            metadata={
                "source": source, "product_id": product_id,
                "doc_type": _resolve_doc_type(product_id),
                "chunk_type": "child", "parent_id": parent_id,
                "section_title": _clean_sec,
                "section_level": section_level,
                "function_names": ",".join(func_names) if func_names else "",
                "api_atomic": is_api,
                "is_api": is_api,
                # 🔴 v5: sdk_header 存入 metadata，线上按需单次挂载
                "sdk_header": sdk_header if (sdk_header and is_api) else "",
            },
        ))
        child_idx += 1

    # ── 🔴 v5: GUI 轨 — 废除字符硬切，按 Heading-to-Heading 完整保留 ──
    if doc_type == "gui_app":
        content = text.strip()
        if len(content) >= 10:
            _emit_child(content)
        return children

    # ── 🔴 v12: SDK 轨 — 状态机已定义完整 API 边界，绝对禁止二次拆切 ──
    if doc_type == "c_sdk":
        content = text.strip()
        if len(content) >= 10:
            _emit_child(content)
        return children

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
            r'\b((?:robot|set|get|arm|py|collrob|end|close|start|Reset)_[a-zA-Z0-9_]{2,})\b',
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

    # ── 🔴 Step1: 主 API 优先级排序 — 将 chunk 内核心 API 函数名排在首位 ──
    # 策略:
    #   1. 提取 "函数说明/函数名称 xxx" 直接声明的函数名 → 最高优先级
    #   2. 提取独立行上的 snake_case 函数定义 → 次优先级
    #   3. 其余按原有顺序排在后面
    _primary_funcs = []
    for m in re.finditer(
        r'(?:函数说明|函数名称|函数名)\s*\n?\s*([a-zA-Z_]\w{3,})\s*\(?',
        text,
    ):
        fname = m.group(1).strip('_')
        if len(fname) >= 4:
            _primary_funcs.append(fname)
    # 独立行函数定义
    for m in re.finditer(
        r'^(?:[a-zA-Z_][a-zA-Z0-9_]*_[a-zA-Z0-9_]{2,})\s*\([^)]*\)\s*$',
        text, re.MULTILINE,
    ):
        fname = m.group(0).split('(')[0].strip().strip('_')
        if fname not in _primary_funcs and len(fname) >= 6:
            _primary_funcs.append(fname)

    # 将主 API 排在首位，其余按原顺序跟随（去重）
    _primary_deduped = []
    _seen_p = set()
    for f in _primary_funcs:
        key = f.lower()
        if key not in _seen_p:
            _seen_p.add(key)
            _primary_deduped.append(f)

    _rest = [f for f in funcs if f.lower() not in _seen_p]
    return (_primary_deduped + _rest)[:15]


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
            # 👇 修改点 5：把产品 ID 和文档类型的解析提前
            product_id = _resolve_product_id_from_filename(pdf_file)
            doc_type = _resolve_doc_type(product_id)
            
            # ── 🔴 v4 增强提取: 将 doc_type 传给底层，JAKA 开启全量图像 OCR ──
            text, total_pages, ocr_pages = _v4_extract_text_universal(file_path, doc_type=doc_type)
            
            if not text.strip():
                logger.warning(f"  ⚠️  {pdf_file}: 无有效文本")
                continue

            # ── 🔴 PDF 文本清洗: 控制字符 + 连字替换 + 括号空格规范化 ──
            # 🔴 v25: 传入 doc_type/product_id，JAKA/gui_app 启用数字保护特判
            text = _clean_pdf_text(text, doc_type=doc_type, product_id=product_id)

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
