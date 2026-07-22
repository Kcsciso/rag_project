"""
=============================================================================
多模态文档解析模块 — 表格/图片/文本一体化提取
=============================================================================

支持格式: PDF (pypdf + pdfplumber + PyMuPDF)

核心能力:
  1. 表格 → Markdown 表格（保持行列映射关系）
  2. 图片 → [Image: caption/OCR_text] 元数据注入
  3. 文本 → 保留原有段落结构
  4. 兼容原有 Header Injection + BM25 正则 Token 保护

依赖: PyMuPDF, pdfplumber, pypdf (已有)

=============================================================================
"""
import logging
import os
import re
from typing import List, Optional, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ============================================================
# 表格提取 — pdfplumber
# ============================================================

def _extract_tables_from_page(pdf_path: str, page_num: int) -> List[str]:
    """
    使用 pdfplumber 提取单页中的表格，返回 Markdown 表格字符串列表。
    """
    import pdfplumber

    md_tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num >= len(pdf.pages):
                return md_tables
            page = pdf.pages[page_num]
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue
                # 清洗：去除 None 和空白单元格
                cleaned = []
                for row in table:
                    cleaned_row = [(cell or "").strip().replace("\n", " ") for cell in row]
                    if any(cleaned_row):  # 至少有一个非空单元格
                        cleaned.append(cleaned_row)
                if len(cleaned) < 2:
                    continue
                # 转换为 Markdown 表格
                md = _table_to_markdown(cleaned)
                md_tables.append(md)
    except Exception as e:
        logger.debug(f"pdfplumber 表格提取失败 (page {page_num}): {e}")

    return md_tables


def _table_to_markdown(rows: List[List[str]]) -> str:
    """将二维列表转为 Markdown 表格字符串。"""
    if not rows:
        return ""
    lines = []
    # 表头
    header = rows[0]
    lines.append("| " + " | ".join(header) + " |")
    # 分隔线
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    # 数据行
    for row in rows[1:]:
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded[:len(header)]) + " |")
    return "\n".join(lines)


# ============================================================
# 图片检测与 Caption — PyMuPDF
# ============================================================

def _extract_images_info(pdf_path: str) -> List[dict]:
    """
    使用 PyMuPDF 提取 PDF 中所有嵌入图片的位置信息。

    Returns:
        [{"page": int, "bbox": (x0,y0,x1,y1), "size": (w,h)}, ...]
    """
    import fitz  # PyMuPDF

    images = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            # 获取页面中的所有图片
            img_list = page.get_images(full=True)
            for img_info in img_list:
                xref = img_info[0]
                try:
                    # 获取图片在页面上的位置
                    img_rects = page.get_image_rects(xref)
                    for rect in img_rects:
                        bbox = rect.irect if hasattr(rect, 'irect') else rect
                        images.append({
                            "page": page_num,
                            "bbox": (
                                getattr(bbox, 'x0', 0), getattr(bbox, 'y0', 0),
                                getattr(bbox, 'x1', 0), getattr(bbox, 'y1', 0),
                            ),
                            "size": (img_info[2], img_info[3]) if len(img_info) > 3 else (0, 0),
                        })
                except Exception:
                    # 图片可能无法定位，记录存在即可
                    images.append({
                        "page": page_num,
                        "bbox": (0, 0, 0, 0),
                        "size": (img_info[2], img_info[3]) if len(img_info) > 3 else (0, 0),
                    })
        doc.close()
    except Exception as e:
        logger.warning(f"PyMuPDF 图片提取失败: {e}")

    return images


def _generate_image_caption(
    pdf_path: str, page_num: int, img_info: dict, page_text: str
) -> str:
    """
    为图片生成描述标签。

    策略（由简到繁）:
      1. 从图片周围的文本中提取可能的 Caption（图注、Fig.、表注等）
      2. 从图片所在页面的文本中提取关键词作为语义标签
      3. 使用图片尺寸和位置作为元数据

    Returns:
        "[Image: 描述文字]" 格式的标签字符串
    """
    captions = []

    # 策略1: 在页面文本中查找 Caption 模式
    caption_patterns = [
        r'(?:图|Fig\.?|Figure|图表|示意图|流程图)[\s.]*[\d]+[:：\s]*(.+?)(?:\n|$)',
        r'(?:表|Table)[\s.]*[\d]+[:：\s]*(.+?)(?:\n|$)',
        r'(?:如上图|下图|如图所示|见下图)[\s]*(.+?)(?:\n|$)',
    ]
    for pat in caption_patterns:
        matches = re.findall(pat, page_text, re.IGNORECASE)
        for m in matches:
            cap = m.strip()[:80]
            if cap and cap not in captions:
                captions.append(cap)

    # 策略2: 提取页面中的关键中文/英文短语（可能是 UI 标签/按钮名）
    if not captions:
        # 提取页面中出现的中文短语 (2-6字) 和英文标识符
        zh_phrases = re.findall(r'[一-鿿]{2,6}', page_text)
        en_terms = re.findall(r'\b[A-Z][a-zA-Z]{2,15}\b', page_text)
        key_terms = (zh_phrases[:5] + en_terms[:3])
        if key_terms:
            captions.append("页面关键词: " + ", ".join(key_terms[:8]))

    # 策略3: 元数据兜底
    if not captions:
        captions.append(f"第{page_num+1}页嵌入图片 ({img_info['size'][0]}x{img_info['size'][1]}px)")

    caption_text = "; ".join(captions[:3])
    return f"[Image: {caption_text}]"


# ============================================================
# 增强文本提取 — pdfplumber (表格→MD) + pypdf (文本)
# ============================================================

def extract_enhanced_text(pdf_path: str) -> str:
    """
    增强版 PDF 文本提取。

    流程:
      1. pdfplumber 逐页提取文本 + 表格
      2. 表格转为 Markdown 表嵌入文本流
      3. pypdf 作为纯文本兜底
      4. PyMuPDF 检测图片并注入 [Image: ...] 标签

    Returns:
        增强后的 Markdown 格式文本
    """
    import pdfplumber
    from pypdf import PdfReader

    # 1. 提取图片信息
    images_info = _extract_images_info(pdf_path)

    # 2. pdfplumber 逐页提取
    pages_output = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_parts = []

                # 提取文本
                text = page.extract_text()
                if text:
                    page_parts.append(text.strip())

                # 提取表格并转为 Markdown
                tables = page.tables if hasattr(page, 'tables') else []
                for table in tables:
                    try:
                        extracted = table.extract()
                        if extracted and len(extracted) >= 2:
                            cleaned = []
                            for row in extracted:
                                cleaned_row = [(c or "").strip()[:100] for c in row]
                                if any(cleaned_row):
                                    cleaned.append(cleaned_row)
                            if len(cleaned) >= 2:
                                md_table = _table_to_markdown(cleaned)
                                page_parts.append(f"\n[Table: 第{page_num+1}页数据表]\n{md_table}\n")
                    except Exception:
                        pass

                # 图片 Caption 注入
                page_imgs = [img for img in images_info if img["page"] == page_num]
                page_text_for_caption = text or ""
                for img in page_imgs:
                    caption = _generate_image_caption(pdf_path, page_num, img, page_text_for_caption)
                    page_parts.append(caption)

                if page_parts:
                    pages_output.append("\n\n".join(page_parts))
    except Exception as e:
        logger.warning(f"pdfplumber 提取失败: {e}，回退到 pypdf")

    # 3. 如果 pdfplumber 完全失败，回退 pypdf
    if not pages_output or not any(p.strip() for p in pages_output):
        logger.info(f"pdfplumber 提取为空，使用 pypdf 兜底")
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_output.append(text.strip())

    result = "\n\n---\n\n".join(pages_output)
    logger.info(
        f"📄 {os.path.basename(pdf_path)}: "
        f"pdfplumber={len(pages_output)} pages, "
        f"images={len(images_info)}, "
        f"total_chars={len(result)}"
    )
    return result


# ============================================================
# 独立使用的文档解析入口
# ============================================================

def load_enhanced_documents(
    data_dir: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    增强版文档加载与分块。

    相比 `pdf_loader.load_pdfs_from_directory()`，额外提供:
      - Markdown 表格结构保留
      - 图片 Caption 注入
      - 多列排版容错

    Args:
        data_dir: PDF 文件目录
        chunk_size: 文本块大小
        chunk_overlap: 重叠字符数

    Returns:
        LangChain Document 列表（含 product_id + Header Injection）
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from .config import PRODUCT_MAPPING_RULES

    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logger.warning(f"目录 '{data_dir}' 中未找到 PDF 文件")
        return []

    logger.info(f"📄 发现 {len(pdf_files)} 个 PDF 文件，开始增强解析...")

    # 产品识别
    def _resolve_product(filename: str) -> str:
        fn = filename.lower()
        for rule in PRODUCT_MAPPING_RULES:
            for pat in rule["filename_patterns"]:
                if pat.lower() in fn:
                    return rule["product_id"]
        return "unknown"

    # 提取文本
    all_docs = []
    for pdf_file in pdf_files:
        file_path = os.path.join(data_dir, pdf_file)
        try:
            text = extract_enhanced_text(file_path)
            if text.strip():
                pid = _resolve_product(pdf_file)
                doc = Document(
                    page_content=text,
                    metadata={"source": pdf_file, "product_id": pid},
                )
                all_docs.append(doc)
                logger.info(f"  ✅ {pdf_file}: {len(text)} chars (pid={pid})")
            else:
                logger.warning(f"  ⚠️  {pdf_file}: 无有效文本")
        except Exception as e:
            logger.error(f"  ❌ {pdf_file}: {e}")

    if not all_docs:
        logger.error("所有 PDF 均无法解析")
        return []

    # 分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", "！", "!", "？", "?", "；", ";", "，", ",", " "],
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(all_docs)

    # 🔴 Header Injection（保持与阶段一兼容）
    _FUNC_RE = re.compile(
        r'\b([a-z_][a-z0-9_]*_[a-z0-9_]+)\s*\(', re.IGNORECASE
    )
    for chunk in chunks:
        funcs = set()
        for m in _FUNC_RE.finditer(chunk.page_content):
            fname = m.group(1).lower().strip('_')
            if len(fname) >= 6 and '_' in fname:
                funcs.add(fname)
        if funcs:
            header = f"[Functions: {', '.join(sorted(funcs)[:10])}]\n"
            chunk.page_content = header + chunk.page_content

    injected = sum(1 for c in chunks if c.page_content.startswith("[Functions:"))
    logger.info(
        f"✅ 增强加载完成: {len(all_docs)} 文档 → {len(chunks)} chunks "
        f"(Header Injected: {injected})"
    )
    return chunks
