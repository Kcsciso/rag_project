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
# 图片检测与 Caption — PyMuPDF + OCR 双轨
# ============================================================

# ── 模块级 OCR 引擎（懒加载单例）──
_ocr_engine = None
_ocr_available = None  # None=未检测, True=可用, False=不可用


def _get_ocr_engine():
    """
    获取 RapidOCR 引擎实例（懒加载 + 优雅回退）。

    RapidOCR 基于 ONNX Runtime，无需 PyTorch/CUDA，模型约 15MB，
    支持中英文混合识别，是嵌入式场景的最优选择。
    """
    global _ocr_engine, _ocr_available
    if _ocr_available is not None:
        return _ocr_engine  # 已检测过，直接返回（可能为 None）

    try:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()
        # 冒烟测试：空白图片不报错即可
        import numpy as np
        test_img = np.zeros((50, 100, 3), dtype=np.uint8) + 255
        _ocr_engine(test_img)
        _ocr_available = True
        logger.info("✅ RapidOCR (ONNX) OCR 引擎已就绪")
    except ImportError:
        logger.warning("⚠️  rapidocr_onnxruntime 未安装，OCR 功能不可用。安装: pip install rapidocr-onnxruntime")
        _ocr_available = False
    except Exception as e:
        logger.warning(f"⚠️  OCR 引擎初始化失败: {e}")
        _ocr_available = False

    return _ocr_engine


def _ocr_image_from_page(pdf_path: str, page_num: int, xref: int) -> str:
    """
    从 PDF 页面中提取指定 xref 的图片字节流 → OCR 识别。

    流程:
      1. PyMuPDF 提取图片原始字节（PNG/JPEG）
      2. 解码为 numpy 数组
      3. RapidOCR 做文字识别
      4. 返回拼接后的 OCR 文本

    Args:
        pdf_path: PDF 文件路径
        page_num: 页面编号 (0-based)
        xref: PyMuPDF 图片交叉引用号

    Returns:
        OCR 识别出的文本（失败时返回空字符串）
    """
    ocr = _get_ocr_engine()
    if ocr is None:
        return ""

    import fitz
    import numpy as np
    from PIL import Image
    import io

    try:
        doc = fitz.open(pdf_path)
        # 提取图片字节
        base_image = doc.extract_image(xref)
        if base_image is None:
            doc.close()
            return ""

        image_bytes = base_image.get("image")
        if not image_bytes:
            doc.close()
            return ""

        # PIL 解码 → numpy (RGB)
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_img = np.array(pil_img)

        # 跳过极小图片（< 50×50 px，通常是图标/装饰）
        h, w = np_img.shape[:2]
        if h < 50 or w < 50:
            doc.close()
            return ""

        doc.close()

        # OCR 识别
        result, _ = ocr(np_img)
        if not result:
            return ""

        # 拼接识别文本（按置信度过滤）
        lines = []
        for block in result:
            text = str(block[1]).strip()
            confidence = float(block[2]) if len(block) > 2 else 1.0
            if text and confidence > 0.5:
                lines.append(text)

        ocr_text = " ".join(lines)
        return ocr_text

    except Exception as e:
        logger.debug(f"OCR 提取失败 (page={page_num}, xref={xref}): {e}")
        return ""

def _extract_images_info(pdf_path: str) -> List[dict]:
    """
    使用 PyMuPDF 提取 PDF 中所有嵌入图片的位置信息 + xref 引用。

    Returns:
        [{"page": int, "xref": int, "bbox": (x0,y0,x1,y1), "size": (w,h)}, ...]
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
                            "xref": xref,
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
                        "xref": xref,
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
    为图片生成描述标签（含 OCR 双轨提取）。

    策略（由简到繁）:
      1. 从图片周围的文本中提取可能的 Caption（图注、Fig.、表注等）
      2. 🔴 OCR: 对图片像素运行 RapidOCR，提取截图内的参数文字（如端口号 6502）
      3. 从图片所在页面的文本中提取关键词作为语义标签
      4. 使用图片尺寸和位置作为元数据

    Returns:
        "[Image: 描述文字 | OCR内容: xxx]" 格式的标签字符串
    """
    captions = []
    ocr_text = ""

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

    # 🔴 策略2: OCR 图片像素提取（双轨核心 — 截图文字参数提取）
    xref = img_info.get("xref", 0)
    if xref > 0:
        ocr_text = _ocr_image_from_page(pdf_path, page_num, xref)
        if ocr_text:
            # 截断过长的 OCR 文本（最多 150 字符）
            ocr_text = ocr_text[:150]

    # 策略3: 提取页面中的关键中文/英文短语（无 Caption 时补充）
    if not captions:
        zh_phrases = re.findall(r'[一-鿿]{2,6}', page_text)
        en_terms = re.findall(r'\b[A-Z][a-zA-Z]{2,15}\b', page_text)
        key_terms = (zh_phrases[:5] + en_terms[:3])
        if key_terms:
            captions.append("页面关键词: " + ", ".join(key_terms[:8]))

    # 策略4: 元数据兜底
    if not captions and not ocr_text:
        captions.append(f"第{page_num+1}页嵌入图片 ({img_info['size'][0]}x{img_info['size'][1]}px)")

    # ── 组装输出：Caption 标签 + OCR 文本作为独立可检索行 ──
    # 🔴 双轨策略：
    #    1. [Image: ...] 保留元数据（图片位置、周围文本 Caption）
    #    2. OCR文本作为独立行注入，参与向量嵌入 + BM25 索引
    #    格式: [Image: -34 Modbus参数界面]\n[图片文字: 端口：6502 IP 192.168.1.100]
    parts = []
    if captions:
        parts.append("; ".join(captions[:3]))
    if ocr_text:
        parts.append(f"OCR内容: {ocr_text}")
    tag = f"[Image: {' | '.join(parts)}]" if parts else ""
    if ocr_text:
        # 过滤明显的 UI 噪声词
        cleaned = re.sub(r'菌\s*|日惠\s*|信号强度\s*\S{0,10}\s*', '', ocr_text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if len(cleaned) >= 10:
            return f"{tag}\n[图片文字: {cleaned}]"
    return tag


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

                # 图片 Caption + OCR 文本（由 _generate_image_caption 内部完成，避免重复 OCR）
                page_imgs = [img for img in images_info if img["page"] == page_num]
                page_text_for_caption = text or ""
                for img in page_imgs:
                    caption = _generate_image_caption(pdf_path, page_num, img, page_text_for_caption)
                    page_parts.append(caption)

                if page_parts:
                    # 🔴 通用溯源注入：每页开头追加页码标记，分块后自然携带到切片中
                    page_content = "\n\n".join(page_parts)
                    pages_output.append(f"[Page: {page_num + 1}]\n{page_content}")
    except Exception as e:
        logger.warning(f"pdfplumber 提取失败: {e}，回退到 pypdf")

    # 3. 如果 pdfplumber 完全失败，回退 pypdf（同样注入页码）
    if not pages_output or not any(p.strip() for p in pages_output):
        logger.info(f"pdfplumber 提取为空，使用 pypdf 兜底")
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages_output.append(f"[Page: {page_num + 1}]\n{text.strip()}")

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

    # ================================================================
    # 🔴 章节标题自动提取（用于后续注入到切片头部）
    # ================================================================
    # 在每个文档的原始文本中提取所有标题行及其字符位置，
    # 分块后根据 Chunk 在原文中的位置自动注入最近的父级标题。
    _HEADING_PATTERNS = [
        # 编号型：2.2.4.3 版本升级、1.1 概述
        re.compile(r'^(\d+(?:\.\d+)+)\s+(.+?)(?:\r?\n|$)', re.MULTILINE),
        # 章/节型：第5章 通信协议、第3.2节 电气参数
        re.compile(r'^(第[一二三四五六七八九十\d]+[章节])\s*(.+?)(?:\r?\n|$)', re.MULTILINE),
        # 中文序号型：一、系统概述、（一）功能说明
        re.compile(r'^([（(]?[一二三四五六七八九十]+[）)]?[\s、,，])\s*(.+?)(?:\r?\n|$)', re.MULTILINE),
        # Markdown 标题型：# 标题、## 子标题
        re.compile(r'^(#{1,4})\s+(.+?)(?:\r?\n|$)', re.MULTILINE),
        # 功能/模块标题型（如 "■ 功能说明"、"## 2. 通信协议"）
        re.compile(r'^(?:[■□◆◇●○]|##?)\s*(?:(\d+(?:\.\d+)*)\s+)?(.+?)(?:\r?\n|$)', re.MULTILINE),
    ]

    def _extract_headings(text: str) -> list:
        """扫描全文提取所有标题行，返回 [(字符位置, 标题文本), ...] 升序。"""
        headings = []
        seen_positions = set()
        for pattern in _HEADING_PATTERNS:
            for m in pattern.finditer(text):
                pos = m.start()
                if pos in seen_positions:
                    continue
                seen_positions.add(pos)
                full = m.group(0).strip()
                # 过滤过短（< 3 字）或过长（> 80 字）的标题
                if 3 <= len(full) <= 80:
                    headings.append((pos, full))
        headings.sort(key=lambda x: x[0])
        # 合并相邻重复标题（同一标题可能被多个 pattern 匹配到）
        deduped = []
        for pos, title in headings:
            if deduped and pos - deduped[-1][0] < len(deduped[-1][1]) + 5:
                # 位置太近，优先保留更长的标题
                if len(title) > len(deduped[-1][1]):
                    deduped[-1] = (pos, title)
            else:
                deduped.append((pos, title))
        return deduped

    def _resolve_section(chunk_text: str, full_text: str, headings: list) -> str:
        """根据 chunk 在原文中的位置，找到最近的父级标题。"""
        if not headings:
            return ""
        # 用前 80 字符作为指纹在原文中定位（重叠区域的重复风险很低）
        fingerprint = chunk_text.strip()[:80]
        pos = full_text.find(fingerprint)
        if pos < 0:
            # 回退：用前 40 字符
            fingerprint = chunk_text.strip()[:40]
            pos = full_text.find(fingerprint)
        if pos < 0:
            return ""
        # 二分查找最近的前置标题
        import bisect
        idx = bisect.bisect_right([h[0] for h in headings], pos) - 1
        if idx >= 0:
            return headings[idx][1]
        return ""

    # 分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", "！", "!", "？", "?", "；", ";", "，", ",", " "],
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(all_docs)

    # 🔴 Section Injection: 为每个 chunk 注入章节标题上下文
    section_injected = 0
    for doc_idx, original_doc in enumerate(all_docs):
        full_text = original_doc.page_content
        headings = _extract_headings(full_text)
        if not headings:
            continue
        logger.info(
            f"📑 章节扫描: {original_doc.metadata.get('source', '?')} "
            f"→ {len(headings)} 个标题"
        )
        # 为属于该文档的所有 chunk 注入标题
        for chunk in chunks:
            if chunk.metadata.get("source") != original_doc.metadata.get("source"):
                continue
            section = _resolve_section(chunk.page_content, full_text, headings)
            if section:
                chunk.page_content = f"[章节: {section}]\n{chunk.page_content}"
                section_injected += 1

    # 🔴 Function Header Injection（保持与阶段一兼容）
    _FUNC_RE = re.compile(
        r'\b([a-z_][a-z0-9_]*_[a-z0-9_]+)\s*\(', re.IGNORECASE
    )
    func_injected = 0
    for chunk in chunks:
        funcs = set()
        for m in _FUNC_RE.finditer(chunk.page_content):
            fname = m.group(1).lower().strip('_')
            if len(fname) >= 6 and '_' in fname:
                funcs.add(fname)
        if funcs:
            # 追加到已有 section header 之后（而非覆盖）
            chunk.page_content = f"[Functions: {', '.join(sorted(funcs)[:10])}]\n{chunk.page_content}"
            func_injected += 1

    logger.info(
        f"✅ 增强加载完成: {len(all_docs)} 文档 → {len(chunks)} chunks "
        f"(Section Injected: {section_injected}, Func Injected: {func_injected})"
    )
    return chunks
