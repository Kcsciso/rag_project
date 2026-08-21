import os
import re
import base64
import requests
import json
from PIL import Image

# 复用 HTTP 连接，提升请求稳定性
session = requests.Session()

def clean_html_tables(content: str) -> str:
    """将 HTML <table> 完整转换为标准 Markdown 表格"""
    def replace_table(match):
        html = match.group(0)
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.I | re.DOTALL)
        if not rows:
            return html
        
        md_lines = []
        col_count = 0
        for i, row in enumerate(rows):
            cells = [re.sub(r'<[^>]+>', '', c).strip() for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, re.I | re.DOTALL)]
            if not cells:
                continue
            if i == 0:
                col_count = len(cells)
                md_lines.append("| " + " | ".join(cells) + " |")
                md_lines.append("| " + " | ".join(["---"] * col_count) + " |")
            else:
                if len(cells) < col_count:
                    cells.extend([""] * (col_count - len(cells)))
                md_lines.append("| " + " | ".join(cells[:col_count]) + " |")
                
        return "\n\n" + "\n".join(md_lines) + "\n\n"
        
    return re.sub(r'(?:<html>\s*)?<table[^>]*>.*?</table>(?:\s*</html>)?', replace_table, content, flags=re.I | re.DOTALL)

def is_toc_chunk(text: str) -> bool:
    """过滤目录引导页切片"""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return False
    toc_lines = sum(1 for line in lines if re.search(r'(?:(?:\.\s*){2,}|…)\s*\d+\s*$', line))
    return toc_lines >= max(2, len(lines) // 2)

def encode_image(image_path: str) -> str:
    """图片转 Base64 编码"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def call_vlm_for_image_extraction(image_path: str, context: str) -> str:
    """调用本地 Qwen2-VL-7B 提取技术参数（防模板废话版）"""
    try:
        base64_image = encode_image(image_path)
    except Exception as e:
        return f"图片读取失败: {e}"

    prompt = (
        f"你是一个工业机器人技术文档解析专家。图片上下文为：'{context}'。\n"
        f"【任务】：仅提取图中可见的具体参数事实（如 IP/端口/寄存器/坐标数值/UI路径/按键名称/错误码）。\n"
        f"【严格要求】：\n"
        f"1. 如果图中有参数，直接用简洁的项目列表输出提取到的数据。\n"
        f"2. 绝对不要输出任何'未提供'、'无'、'未包含'的占位项或空模板字段。\n"
        f"3. 若图中纯粹为示意图/结构图/外观图且无关键技术参数，只需且必须直接回复：'仅为UI示意图'。"
    )

    payload = {
        "model": "Qwen/Qwen2-VL-7B-Instruct",
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
        "temperature": 0.05
    }

    try:
        resp = session.post("http://localhost:8005/v1/chat/completions", json=payload, timeout=60)
        resp.raise_for_status()
        res = resp.json()['choices'][0]['message']['content'].strip()
        
        # 清理多余反引号与空行
        res = re.sub(r'^```[a-zA-Z]*\n?', '', res)
        res = re.sub(r'\n?```$', '', res).strip()
        
        return res if res else "仅为UI示意图"
    except Exception as e:
        print(f"⚠️ VLM 解析失败 ({os.path.basename(image_path)}): {e}")
        return "图片信息提取服务暂不可用。"

def process_and_filter_images(text: str, md_file_path: str) -> str:
    """多模态图片位置精确匹配与 VLM 提取注入"""
    base_dir = os.path.dirname(md_file_path)
    img_pattern = re.compile(r'(!\[(.*?)\]\((images/[^)]+)\))')
    
    output_parts = []
    last_end = 0
    
    for match in img_pattern.finditer(text):
        output_parts.append(text[last_end:match.start()])
        full_tag = match.group(1)
        rel_path = match.group(3)
        abs_path = os.path.join(base_dir, rel_path)
        last_end = match.end()
        
        if not os.path.exists(abs_path):
            continue
            
        # 1. 几何过滤：剔除小图标与细长分割条
        try:
            with Image.open(abs_path) as img:
                w, h = img.size
                if w < 80 or h < 80 or (w / max(h, 1) > 8) or (h / max(w, 1) > 8):
                    continue
        except Exception:
            continue
            
        # 2. 截取前后 100 字符上下文
        start_idx = max(0, match.start() - 100)
        end_idx = min(len(text), match.end() + 100)
        context_window = text[start_idx:end_idx].replace(full_tag, "").strip()
        
        # 3. 图注意图校验
        has_caption = bool(re.search(r'(图\s*\d+|表\s*\d+|Figure|Table|界面|设置|配置|网络|参数|如下|说明)', context_window, re.I))
        
        if has_caption:
            print(f"  🔍 [VLM提纯] 解析图片: {os.path.basename(rel_path)} ...")
            extracted_info = call_vlm_for_image_extraction(abs_path, context_window)
            output_parts.append(f"\n{full_tag}\n> **[图表参数智能提纯]**:\n> {extracted_info.replace(chr(10), chr(10)+'> ')}\n")
        else:
            output_parts.append(full_tag)
            
    output_parts.append(text[last_end:])
    return "".join(output_parts)

def parse_mineru_markdown(file_path: str, chunk_size: int = 1000):
    """按章节层级与长度阈值切片"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    content = clean_html_tables(content)
    paragraphs = re.split(r'\n{2,}', content)
    
    slots = ["", "", "", ""]
    current_path_str = "JAKA机器人文档"
    
    chunks = []
    current_buffer = []
    current_length = 0
    
    def flush_buffer():
        nonlocal current_buffer, current_length
        if current_buffer:
            text = "\n\n".join(current_buffer)
            if len(text.strip()) >= 20 and not is_toc_chunk(text):
                processed_text = process_and_filter_images(text, file_path)
                chunks.append({
                    "path": current_path_str,
                    "content": processed_text
                })
            current_buffer = []
            current_length = 0

    chapter_pattern = re.compile(r'^#+\s+(第[一二三四五六七八九十\d]+章|\d+(?:\.\d+)*)\s*(.*)')

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        chap_match = chapter_pattern.match(para)
        if chap_match:
            flush_buffer()
            sec_num = chap_match.group(1)
            sec_title = chap_match.group(2).strip()
            
            level = 0 if '章' in sec_num else sec_num.count('.')
            level = min(level, 3)
            
            slots[level] = f"{sec_num} {sec_title}".strip()
            for i in range(level + 1, 4):
                slots[i] = ""
                
            current_path_str = " > ".join([s for s in slots if s])
            current_buffer.append(para)
            current_length += len(para)
            continue
            
        if current_length + len(para) > chunk_size and current_length > 0:
            flush_buffer()
            
        current_buffer.append(para)
        current_length += len(para)
        
    flush_buffer()
    return chunks

if __name__ == "__main__":
    # 配置输入与输出路径
    input_file = "/home/kasm-user/rag_project/data/jaka_markdown/JAKA_Manual/auto/JAKA_Manual.md"
    output_file = "/home/kasm-user/rag_project/data/jaka_manual_chunks.json"

    if not os.path.exists(input_file):
        print(f"❌ 找不到输入文件: {input_file}")
        exit(1)

    print("🚀 开始执行 JAKA 手册智能切片与多模态提纯...")
    results = parse_mineru_markdown(input_file, chunk_size=1000)
    
    # 将切片持久化保存到磁盘
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ 切片全部完成并落盘！")
    print(f"• 输出文件: {output_file}")
    print(f"• 切片总数: {len(results)} 个\n")

    # 抽样打印统计与检查
    vlm_chunks = [c for c in results if "[图表参数智能提纯]" in c['content']]
    param_chunks = [c for c in vlm_chunks if "仅为UI示意图" not in c['content']]
    
    print("=" * 60)
    print("📊 【切片质量摘要】")
    print(f"• 包含图表提纯的切片数: {len(vlm_chunks)}")
    print(f"• 成功提取出结构化参数的切片数: {len(param_chunks)}")
    print("=" * 60)

    # 抽样展示 1 个提取了真实参数的切片
    if param_chunks:
        print("\n🌟 【高价值多模态切片样本展示】")
        sample = param_chunks[0]
        print(f"📍 路径: {sample['path']}")
        print(f"📄 内容:\n{sample['content']}")
        print("=" * 60)