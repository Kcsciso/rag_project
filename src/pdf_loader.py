"""
=============================================================================
统一数据摄入与切片模块（v32 — 数据摄入双轨制 + 多模态 VLM 提纯 + Cache-First）
=============================================================================
"""

import os
import re
import json
import base64
import logging
import requests
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from pypdf import PdfReader
from tqdm import tqdm
from langchain_core.documents import Document

try:
    from .config import PRODUCT_MAPPING_RULES, VLM_BASE_URL, VLM_MODEL_NAME
except ImportError:
    from config import PRODUCT_MAPPING_RULES, VLM_BASE_URL, VLM_MODEL_NAME

logger = logging.getLogger(__name__)
session = requests.Session()

# ============================================================
# 通用路由与元数据解析
# ============================================================

def _resolve_product_id_from_filename(filename: str) -> str:
    filename_lower = filename.lower()
    for rule in PRODUCT_MAPPING_RULES:
        for pattern in rule["filename_patterns"]:
            if pattern.lower() in filename_lower:
                return rule["product_id"]

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
            return pid
    return "General"

def _resolve_doc_type(product_id: str) -> str:
    _DOC_TYPE_MAP = {
        "JAKA": "gui_app",
        "OpenC3": "c_sdk",
        "OpenR6": "c_sdk",
    }
    return _DOC_TYPE_MAP.get(product_id, "general")

# ============================================================
# JAKA 专轨：HTML 表格规整、VLM 并发提纯与 Markdown 切片
# ============================================================

def clean_html_tables(content: str) -> str:
    """彻底剥离 HTML 标签并将 <table> 转换为标准 GitHub Markdown 表格"""
    def replace_table(match):
        html = match.group(0)
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.I | re.DOTALL)
        if not rows:
            return ""
        
        md_lines = []
        col_count = 0
        for i, row in enumerate(rows):
            cells = [
                re.sub(r'<[^>]+>', '', c).strip().replace('\n', ' ') 
                for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, re.I | re.DOTALL)
            ]
            if not any(cells):
                continue
            if i == 0 or col_count == 0:
                col_count = len(cells)
                md_lines.append("| " + " | ".join(cells) + " |")
                md_lines.append("| " + " | ".join(["---"] * col_count) + " |")
            else:
                if len(cells) < col_count:
                    cells.extend([""] * (col_count - len(cells)))
                md_lines.append("| " + " | ".join(cells[:col_count]) + " |")
                
        return "\n\n" + "\n".join(md_lines) + "\n\n"

    # 1. 转换 table 为 Markdown 格式
    content = re.sub(r'<table[^>]*>.*?</table>', replace_table, content, flags=re.I | re.DOTALL)
    # 2. 全局清除残留的 html/body/p/div/span/thead/tbody 等标签
    content = re.sub(r'</?(?:html|body|p|div|span|thead|tbody|tr|td|th)[^>]*>', '', content, flags=re.I)
    # 3. 规范化空行
    content = re.sub(r'\n{3,}', '\n\n', content)
    return content

def _encode_image_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def _call_vlm_worker(task: Tuple[str, str]) -> Tuple[str, str]:
    abs_path, context = task
    try:
        base64_image = _encode_image_b64(abs_path)
    except Exception as e:
        return abs_path, ""

    prompt = (
        f"当前图片说明：'{context}'。\n"
        f"请提取图中所有可见的界面配置项名称、输入框数值、IP地址、端口号、坐标系参数、错误码或表格数据。\n"
        f"输出要求：以简洁的 Markdown 列表或键值对输出，不要带有'无'或'未提供'的项目。"
    )

    payload = {
        "model": VLM_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 256,
        "temperature": 0.1
    }

    try:
        resp = session.post(f"{VLM_BASE_URL}/chat/completions", json=payload, timeout=60)
        resp.raise_for_status()
        res = resp.json()['choices'][0]['message']['content'].strip()
        res = re.sub(r'^```[a-zA-Z]*\n?', '', res)
        res = re.sub(r'\n?```$', '', res).strip()
        
        lines = [l.strip() for l in res.splitlines() if l.strip()]
        valid_lines = [l for l in lines if not re.search(r'[:：]\s*(?:无|未提供|None|暂无)$', l)]
        
        if not valid_lines or len("".join(valid_lines)) < 8:
            return abs_path, "仅为UI示意图"
            
        return abs_path, "\n".join(valid_lines)
    except Exception:
        return abs_path, ""

def _preprocess_all_images(content: str, base_dir: str, max_workers: int = 6) -> Dict[str, str]:
    img_pattern = re.compile(r'(!\[(.*?)\]\((images/[^)]+)\))')
    tasks = {}
    
    for match in img_pattern.finditer(content):
        full_tag = match.group(1)
        rel_path = match.group(3)
        abs_path = os.path.join(base_dir, rel_path)
        
        if not os.path.exists(abs_path) or abs_path in tasks:
            continue
            
        try:
            with Image.open(abs_path) as img:
                w, h = img.size
                if w < 80 or h < 80 or (w / max(h, 1) > 8) or (h / max(w, 1) > 8):
                    continue
        except Exception:
            continue
            
        start_idx = max(0, match.start() - 100)
        end_idx = min(len(content), match.end() + 100)
        context_window = content[start_idx:end_idx].replace(full_tag, "").strip()
        
        has_caption = bool(re.search(r'(图\s*\d+|表\s*\d+|Figure|Table|界面|设置|配置|网络|参数|如下|说明)', context_window, re.I))
        if has_caption:
            tasks[abs_path] = context_window

    if not tasks:
        return {}

    logger.info(f"🎯 [JAKA多模态] 筛选出 {len(tasks)} 张待提纯截图，启动 {max_workers} 线程并发提取...")
    results_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_call_vlm_worker, (path, ctx)): path for path, ctx in tasks.items()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="🚀 Qwen2-VL 提纯进度"):
            path, extracted_text = future.result()
            if extracted_text and extracted_text != "图片信息提取服务暂不可用。":
                results_map[path] = extracted_text
            
    return results_map

def load_jaka_mineru_dual(
    md_file_path: str,
    source_name: str = "JAKA_Manual.pdf",
    child_chunk_size: int = 1500,
    parent_chunk_size: int = 2000,
    cached_json_path: str = "/home/kasm-user/rag_project/data/jaka_manual_chunks.json",
) -> Tuple[List[Document], List[Document]]:
    """JAKA 专轨：支持复用已提纯的 JSON 缓存，避免重复耗费 GPU 推理"""
    parents: List[Document] = []
    children: List[Document] = []

    # 1. 优先从已完成提纯的 JSON 文件构建（极速秒级恢复）
    if os.path.exists(cached_json_path):
        try:
            with open(cached_json_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            
            # 校验缓存是否为包含真实提纯的有效数据
            valid_vlm_count = sum(1 for c in cached_data if "[图表参数智能提纯]" in c.get("content", "") and "图片信息提取服务暂不可用" not in c.get("content", "") and "仅为UI示意图" not in c.get("content", ""))
            if valid_vlm_count > 50:
                logger.info(f"⚡ [JAKA专轨] 命中高质量持久化切片缓存: {cached_json_path} (含 {valid_vlm_count} 个多模态实体切片)")
                parent_map = {}
                for idx, item in enumerate(cached_data):
                    path = item.get("path", "JAKA Zu APP 使用手册")
                    content = item.get("content", "")
                    h1_title = path.split(" > ")[0] if " > " in path else path
                    parent_id = f"parent_JAKA_{abs(hash(h1_title)) % 10000}"
                    
                    if parent_id not in parent_map:
                        p_doc = Document(
                            page_content=f"[文档: {source_name}]\n[章节: {h1_title}]\n\n{h1_title}",
                            metadata={
                                "source": source_name,
                                "product_id": "JAKA",
                                "doc_type": "gui_app",
                                "chunk_type": "parent",
                                "parent_id": None,
                                "section_title": h1_title,
                                "section_level": 1,
                            }
                        )
                        parents.append(p_doc)
                        parent_map[parent_id] = True

                    has_vlm = "[图表参数智能提纯]" in content and "仅为UI示意图" not in content
                    c_doc = Document(
                        page_content=f"[文档: {source_name}]\n[路径: {path}]\n\n{content}",
                        metadata={
                            "source": source_name,
                            "product_id": "JAKA",
                            "doc_type": "gui_app",
                            "chunk_type": "child",
                            "parent_id": parent_id,
                            "section_title": path.split(" > ")[-1] if " > " in path else path,
                            "section_level": 2,
                            "function_names": "",
                            "api_atomic": False,
                            "is_api": False,
                            "has_multimodal_data": has_vlm,
                        }
                    )
                    children.append(c_doc)
                
                logger.info(f"✅ [JAKA专轨] 成功从缓存加载: {len(parents)} parents + {len(children)} children")
                return parents, children
        except Exception as e:
            logger.warning(f"⚠️ 读取 JAKA 缓存失败，将重新全量解析: {e}")

    # 2. 无缓存时走全量解析流程
    if not os.path.exists(md_file_path):
        logger.error(f"❌ 找不到 JAKA Markdown 文件: {md_file_path}")
        return [], []

    base_dir = os.path.dirname(md_file_path)
    with open(md_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    vlm_cache = _preprocess_all_images(content, base_dir, max_workers=6)
    content = clean_html_tables(content)
    
    img_pattern = re.compile(r'(!\[(.*?)\]\((images/[^)]+)\))')
    def replace_img(match):
        full_tag = match.group(1)
        abs_path = os.path.join(base_dir, match.group(3))
        if abs_path in vlm_cache and vlm_cache[abs_path]:
            extracted_info = vlm_cache[abs_path]
            return f"\n{full_tag}\n> **[图表参数智能提纯]**:\n> {extracted_info.replace(chr(10), chr(10)+'> ')}\n"
        return full_tag
        
    content = img_pattern.sub(replace_img, content)

    paragraphs = re.split(r'\n{2,}', content)
    slots = ["", "", "", ""]
    current_path_str = "JAKA Zu APP 使用手册"
    current_parent_title = "前言与说明"
    
    current_buffer = []
    current_length = 0
    parent_idx = 0
    child_idx = 0
    
    def is_toc(text: str) -> bool:
        """精准识别并过滤目录页 (TOC)"""
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if not lines:
            return False
        # 统计带有页码引导点（如 ".. 12"、".... 23"、"… 8"）的行数
        toc_lines = sum(
            1 for line in lines 
            if re.search(r'(?:(?:\.|\s|·|…){2,}\s*\d+|\b\d+\.\d+\s+.*?\s+\d+$)', line)
        )
        # 若超过 30% 的行是目录特征行，判定为目录直接丢弃
        return toc_lines >= 3 or (len(lines) > 0 and toc_lines / len(lines) >= 0.3)

    def flush_buffer():
        nonlocal current_buffer, current_length, parent_idx, child_idx
        if current_buffer:
            text = "\n\n".join(current_buffer)
            if len(text.strip()) >= 20 and not is_toc(text):
                parent_id = f"parent_JAKA_{parent_idx}"
                has_vlm = "[图表参数智能提纯]" in text and "仅为UI示意图" not in text
                
                doc_child = Document(
                    page_content=f"[文档: {source_name}]\n[路径: {current_path_str}]\n\n{text}",
                    metadata={
                        "source": source_name,
                        "product_id": "JAKA",
                        "doc_type": "gui_app",
                        "chunk_type": "child",
                        "parent_id": parent_id,
                        "section_title": current_parent_title,
                        "section_level": 2,
                        "function_names": "",
                        "api_atomic": False,
                        "is_api": False,
                        "has_multimodal_data": has_vlm,
                    }
                )
                children.append(doc_child)
                child_idx += 1
                
            current_buffer = []
            current_length = 0

    chapter_pattern = re.compile(
        r'^#+\s*(第\s*[\d一二三四五六七八九十]+\s*章|附录\s*[一二三四五六七八九十\d]+|\d+(?:\.\d+)+)\s*(.*)',
        re.I
    )

    for para in paragraphs:
        para = para.strip()
        if not para: continue
            
        chap_match = chapter_pattern.match(para)
        if chap_match:
            flush_buffer()
            sec_num = chap_match.group(1).strip()
            sec_title = chap_match.group(2).strip()
            
            if '章' in sec_num or '附录' in sec_num:
                level = 0
                slots[0] = f"{sec_num} {sec_title}".strip()
                slots[1] = ""
                slots[2] = ""
                slots[3] = ""
                current_parent_title = f"{sec_num} {sec_title}".strip()
                parent_idx += 1
                
                parents.append(Document(
                    page_content=f"[文档: {source_name}]\n[章节: {current_parent_title}]\n\n{para}",
                    metadata={
                        "source": source_name,
                        "product_id": "JAKA",
                        "doc_type": "gui_app",
                        "chunk_type": "parent",
                        "parent_id": None,
                        "section_title": current_parent_title,
                        "section_level": 1,
                    }
                ))
            else:
                level = min(sec_num.count('.'), 3)
                slots[level] = f"{sec_num} {sec_title}".strip()
                for i in range(level + 1, 4): slots[i] = ""
                
            current_path_str = " > ".join([s for s in slots if s])
            current_buffer.append(para)
            current_length += len(para)
            continue
            
        if current_length + len(para) > child_chunk_size and current_length > 0:
            flush_buffer()
            
        current_buffer.append(para)
        current_length += len(para)
        
    flush_buffer()
    logger.info(f"✅ [JAKA专轨] 加载完成: {len(parents)} parents + {len(children)} children")
    return parents, children

# ============================================================
# SDK 专轨：基于 PyMuPDF 布局排序的 API 原子切片
# ============================================================

import fitz  # PyMuPDF（必须使用 fitz 保障表格与代码的阅读流顺序）

# 严格按编号章节切分：匹配行首的 "1. ", "16. ", "1、" 等
_SDK_CHAPTER_BOUNDARY_RE = re.compile(
    r'(?:^|\n)(?=[ \t]*\d{1,2}\s*\.\s*[\u4e00-\u9fa5a-zA-Z])',
    re.MULTILINE,
)

_CTYPES_BLACKLIST = frozenset({
    "c_float", "c_int", "c_char_p", "c_char", "c_void_p", "c_double",
    "c_short", "c_long", "c_uint", "c_ubyte", "c_bool", "c_byte",
    "restype", "argtypes", "structure", "pointer", "byref", "cast",
    "time_sleep", "os_path", "print", "decode", "encode",
})

def _extract_text_with_fitz(file_path: str) -> str:
    """使用 PyMuPDF 按视觉坐标从上到下精确排序抽取文本，彻底解决 pypdf 表格乱序"""
    doc = fitz.open(file_path)
    pages_text = []
    for page in doc:
        # sort=True 保证表格内外内容按真实垂直版面顺序排列
        text = page.get_text("text", sort=True)
        if text:
            pages_text.append(text)
    doc.close()
    return "\n\n".join(pages_text)

def _clean_sdk_pdf_text(text: str) -> str:
    """清洗 SDK 文本：消除换行断字、拼合函数定义"""
    cleaned = text.replace('\x0c', '\n\n').replace('\x0b', '\n')
    cleaned = re.sub(r'[\x00-\x08\x0e-\x1f]', '', cleaned)
    
    # 修复表格纵向断字（"函数名\n称" -> "函数名称"）
    cleaned = re.sub(r'函\s*数\s*名\s*[\n\r]*\s*称?', '函数名称', cleaned)
    cleaned = re.sub(r'功\s*能\s*描\s*[\n\r]*\s*述?', '功能描述', cleaned)
    cleaned = re.sub(r'参\s*数\s*说\s*[\n\r]*\s*明?', '参数说明', cleaned)
    cleaned = re.sub(r'返\s*回\s*[\n\r]*\s*值?', '返回值', cleaned)
    cleaned = re.sub(r'示\s*例\s*代\s*[\n\r]*\s*码?', '示例代码', cleaned)
    
    # 修复下划线断行与 API 名字断裂
    cleaned = re.sub(r'([a-zA-Z0-9_]+)\s+_\s+([a-zA-Z0-9_]+)', r'\1_\2', cleaned)
    cleaned = re.sub(r'([a-zA-Z0-9_]+)\s*\n\s*_\s*([a-zA-Z0-9_]+)', r'\1_\2', cleaned)
    cleaned = re.sub(r'([a-zA-Z0-9_]+)\.\s*\n\s*([a-zA-Z0-9_]+)', r'\1.\2', cleaned)
    cleaned = re.sub(r'\b(set|get|robot|arm)\s*\n\s*(robot|power|arm|cmd|time|mode|on|off|send|mov|socket)\b', r'\1_\2', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b1/[0O]\b|\bI/0\b|\b1/O\b', 'I/O', cleaned)
    cleaned = re.sub(r'^\s*\|\s*', '', cleaned, flags=re.MULTILINE)
    
    return cleaned

def _extract_sdk_header(full_text: str) -> str:
    """提取 CDLL 加载与 POSE/Joint 结构体定义"""
    header_parts = []
    cdll_match = re.search(
        r'(?:robot\s*=\s*)?(?:ctypes\.)?CDLL\s*\(\s*(?:[rR])?["\']([^"\']+)["\']\s*\)',
        full_text, re.IGNORECASE,
    )
    if cdll_match:
        dll_name = cdll_match.group(1)
        header_parts.append("import ctypes")
        header_parts.append(f"robot = ctypes.CDLL(r\"{dll_name}\")")
        
    for m in re.finditer(r'class\s+(?:POSE|Pose|Joint|RobJoint|RobPos|JNT)\s*\(Structure\)\s*:[\s\S]{0,800}?(?=\n(?:def |class |\Z|\d+\.))', full_text, re.IGNORECASE):
        header_parts.append(m.group(0).strip())

    return f"```python\n" + "\n".join(header_parts) + "\n```\n" if header_parts else ""

def _v4_extract_function_names(text: str) -> List[str]:
    """提取真实 API 函数名，剔除类型声明干扰"""
    funcs_raw = set()
    
    # 模式 1: "函数名称 | robot_movc(...)" 或 "robot_movc(POSE..."
    for m in re.finditer(r'(?:函数名称|函数名)[\s\|\:\：]*\s*([a-zA-Z_]\w+)\s*\(', text):
        funcs_raw.add(m.group(1).strip('_'))
        
    # 模式 2: 示例代码中的 robot.robot_xxx 调用
    for m in re.finditer(r'(?:robot|arm)\.([a-zA-Z_]\w+)', text):
        funcs_raw.add(m.group(1).strip('_'))
        
    # 模式 3: 具有下划线的 C-API 函数调用
    for m in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*_[a-zA-Z0-9_]+)\s*\(', text):
        fname = m.group(1).strip('_')
        if len(fname) >= 6 and '_' in fname:
            funcs_raw.add(fname)
            
    valid_funcs = [
        f for f in funcs_raw
        if f.lower() not in _CTYPES_BLACKLIST and not f.startswith("c_") and len(f) >= 4
    ]
    return sorted(valid_funcs, key=lambda x: x.lower())[:5]

def _v4_parse_sdk_state_machine(text: str) -> List[Tuple[int, int, str]]:
    """按章节精准切分"""
    matches = list(_SDK_CHAPTER_BOUNDARY_RE.finditer(text))
    if not matches:
        return [(0, len(text), "SDK 全文")]

    boundaries = []
    titles = []
    
    if matches[0].start() > 0:
        boundaries.append(0)
        titles.append("SDK 基础配置")

    for m in matches:
        boundaries.append(m.start())
        line_end = text.find('\n', m.start())
        raw_title = text[m.start():line_end].strip() if line_end != -1 else text[m.start():m.start()+40].strip()
        titles.append(raw_title)
        
    boundaries.append(len(text))
    titles.append("")

    blocks = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i+1]
        if s < e:
            blocks.append((s, e, titles[i]))
    return blocks

def load_single_sdk_pdf(file_path: str) -> Tuple[List[Document], List[Document]]:
    """统一 SDK PDF 加载与切片"""
    filename = os.path.basename(file_path)
    product_id = _resolve_product_id_from_filename(filename)
    
    # 🔴 关键改造：使用 PyMuPDF 版面排序提取
    full_text = _extract_text_with_fitz(file_path)
    full_text = _clean_sdk_pdf_text(full_text)
    
    sdk_header = _extract_sdk_header(full_text)
    sdk_blocks = _v4_parse_sdk_state_machine(full_text)
    
    parents = [Document(
        page_content=f"[文档: {filename}]\n\n{full_text[:1200]}",
        metadata={
            "source": filename,
            "product_id": product_id,
            "doc_type": "c_sdk",
            "chunk_type": "parent",
            "parent_id": None,
            "section_title": "SDK 全文总览",
            "section_level": 1,
        }
    )]
    
    children = []
    parent_id = f"parent_{product_id}_0"
    
    for s, e, title in sdk_blocks:
        block_text = full_text[s:e].strip()
        if len(block_text) < 20:
            continue
            
        func_names = _v4_extract_function_names(block_text)
        is_api = len(func_names) > 0
        clean_title = re.sub(r'^[ \t]*', '', title).strip() or "SDK 接口"
        
        prefix = f"[文档: {filename}]\n[章节: {clean_title}]\n"
        if func_names:
            prefix += f"[Functions: {', '.join(func_names)}]\n\n"
        else:
            prefix += "\n"
            
        children.append(Document(
            page_content=f"{prefix}{block_text}",
            metadata={
                "source": filename,
                "product_id": product_id,
                "doc_type": "c_sdk",
                "chunk_type": "child",
                "parent_id": parent_id,
                "section_title": clean_title,
                "section_level": 2,
                "function_names": ",".join(func_names),
                "api_atomic": is_api,
                "is_api": is_api,
                "sdk_header": sdk_header if is_api else "",
                "has_multimodal_data": False,
            }
        ))
        
    logger.info(f"✅ [SDK专轨] {filename}: {len(parents)} parents + {len(children)} children (product={product_id})")
    return parents, children

# ============================================================
# KV 属性提取与持久化
# ============================================================

_KV_STORE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kv_db")
_KV_STORE_FILE = os.path.join(_KV_STORE_DIR, "attribute_kv.json")

_RE_KV_PAIR = re.compile(
    r'('
    r'(?:端口号?|波特率|IP地址|速率|频率|超时|周期|间隔|'
    r'默认密码|管理员密码|操作员|技术员|密码|用户名|账号|'
    r'数据位|停止位|校验位|从站地址|从站节点号|节点号|站号|通道|低通滤波器)'
    r')[\s：:]*'
    r'([^\s，。,.\n]{1,80})',
    re.IGNORECASE,
)

_MANUAL_CALIBRATION = {
    "JAKA": {
        "Modbus TCP 端口号": "6502",
        "Modbus RTU 默认波特率": "9600",
        "管理员默认密码": "jakazuadmin",
        "技术员默认密码": "0000",
        "操作员默认密码": "0",
        "Modbus 默认从站地址": "1",
    }
}

def export_kv_attributes(all_children: List[Document]):
    os.makedirs(_KV_STORE_DIR, exist_ok=True)
    store = {}

    for doc in all_children:
        pid = doc.metadata.get("product_id", "General")
        if pid not in store:
            store[pid] = {}
        for m in _RE_KV_PAIR.finditer(doc.page_content):
            k, v = m.group(1).strip().rstrip("：:"), m.group(2).strip().rstrip("：:。，,;；")
            if k and v and len(v) <= 30:
                store[pid][k] = v

    for pid, attrs in _MANUAL_CALIBRATION.items():
        if pid not in store:
            store[pid] = {}
        store[pid].update(attrs)

    with open(_KV_STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)
    logger.info(f"🔑 KV 属性库已自动生成并保存至: {_KV_STORE_FILE}")

# ============================================================
# 第一阶段统一入口
# ============================================================

def load_all_documents_v4_dual(
    data_dir: str = "/home/kasm-user/rag_project/data",
    jaka_md_path: str = "/home/kasm-user/rag_project/data/jaka_markdown/JAKA_Manual/auto/JAKA_Manual.md",
) -> Tuple[List[Document], List[Document]]:
    all_parents = []
    all_children = []

    if os.path.exists(jaka_md_path):
        j_parents, j_children = load_jaka_mineru_dual(jaka_md_path)
        all_parents.extend(j_parents)
        all_children.extend(j_children)
    else:
        logger.warning(f"⚠️ 未找到 JAKA Markdown 文件: {jaka_md_path}")

    if os.path.exists(data_dir):
        pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf") and "jaka" not in f.lower() and "zu" not in f.lower()]
        for pdf in pdf_files:
            pdf_path = os.path.join(data_dir, pdf)
            s_parents, s_children = load_single_sdk_pdf(pdf_path)
            all_parents.extend(s_parents)
            all_children.extend(s_children)

    export_kv_attributes(all_children)
    logger.info(f"🌟 [第一阶段完成] Parent 切片: {len(all_parents)} 个 | Child 切片: {len(all_children)} 个")
    return all_parents, all_children

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parents, children = load_all_documents_v4_dual()
    print(f"\n🚀 第一阶段数据处理与切片全部就绪！Parent: {len(parents)}, Child: {len(children)}")