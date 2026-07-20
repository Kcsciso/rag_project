"""
=============================================================================
PDF 加载与文本分块模块
=============================================================================

本模块负责将 data/ 目录下的 PDF 文件转化为 LangChain Document 列表，
供后续向量化使用。核心流程如下：

  PDF 文件 → [pypdf 提取文本] → 纯文本字符串
           → [RecursiveCharacterTextSplitter 递归字符分割]
           → List[Document]（每个 Document 对应一个文本块）

【算法说明 — RecursiveCharacterTextSplitter】
  这是一个"由粗到细"的递归分割器：
  1. 首先尝试用段落分隔符（\n\n）分割文本
  2. 如果分出来的块仍然超过 chunk_size，则尝试用换行符（\n）再分割
  3. 如果还超长，则用句号（。）分割
  4. 最后手段：按字符硬切
  这种层级式分割能最大程度保留文本的语义完整性，
  避免在句子/词语中间"拦腰截断"。

【chunk_overlap 的作用】
  假设原文："...张三在2024年获得了诺贝尔物理学奖..."
  如果不设 overlap（重叠），这些信息可能被切成两块：
    块A: "...张三在2024年..."（断在"诺贝尔"前面）
    块B: "...获得了诺贝尔物理学奖..."
  检索"张三 诺贝尔奖"时，块A和块B各自只包含部分信息。
  设置 overlap 后，相邻块之间会有重叠区间，保证关键信息
  至少在一整个块内是完整的。

=============================================================================
"""

import os
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


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
                # 创建一个 Document 对象，metadata 记录来源文件名
                doc = Document(
                    page_content=text,
                    metadata={"source": pdf_file}
                )
                all_documents.append(doc)
                print(f"[pdf_loader]   ✅ {pdf_file}: {len(text)} 字符")
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

    print(f"[pdf_loader] ✅ 加载完成：{len(all_documents)} 个原始文档 → "
          f"{len(chunks)} 个文本块（chunk_size={chunk_size}, overlap={chunk_overlap}）")

    return chunks


# ============================================================
# 命令行测试入口
# ============================================================
if __name__ == "__main__":
    from .config import PDF_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

    docs = load_pdfs_from_directory(PDF_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP)
    if docs:
        print(f"\n--- 示例：第一个文本块 ---")
        print(f"来源: {docs[0].metadata['source']}")
        print(f"内容预览: {docs[0].page_content[:200]}...")
