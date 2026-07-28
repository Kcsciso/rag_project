import re
import sys
import os
from collections import Counter, defaultdict
from typing import List, Dict, Any

# 增加根目录路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pdf_loader import load_pdfs_v4_dual


def _resolve_chunk_id(doc) -> str:
    """兼容多种 metadata 字段名，确保能打印真实 ID。"""
    meta = doc.metadata if hasattr(doc, "metadata") else {}
    cid = meta.get("chunk_id") or meta.get("id") or ""
    if cid:
        return str(cid)
    # 🔴 v13: Fallback 使用内容中段做指纹（前缀 40 字符对所有 Chunk 相同导致 ID 碰撞）
    pid = meta.get("parent_id", "?")
    fp = doc.page_content[40:100] if hasattr(doc, "page_content") else ""  # 跳过后缀前缀
    return f"c_{pid}_{abs(hash(fp)) % 10000:04d}"


def audit_chunks():
    print("🔍 开始增强版自动化切片质量审计 (Enhanced Chunk Health Audit)...\n")
    data_dir = "data"
    
    try:
        parents, children = load_pdfs_v4_dual(data_dir)
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return

    total_children = len(children)
    if total_children == 0:
        print("⚠️ 未加载到任何 Child Chunk，请检查 data 目录。")
        return

    # 📊 异常分类收集器
    issues = {
        "skeleton_chunk": [],         # 1. 骨架/目录占位块
        "multi_api_sticky": [],       # 2. 多个 API 强行粘连
        "corrupted_title": [],        # 3. 标题脏化 (含 \n、前缀等)
        "ocr_code_artifact": [],      # 4. 代码/下划线空格断裂 (如 rob _ ip)
        "isolated_caption": [],       # 5. GUI 轨孤立图注块
        "corrupted_breadcrumb": [],   # 6. 面包屑路径重复
        "orphaned_sdk_code": [],      # 7. SDK 代码与描述身首异处
    }

    # 正则规则定义
    re_ocr_underscore = re.compile(r'\b\w+\s+_\s+\w+\b|_\s+_|\w+\s+\.\s+\w+')
    re_isolated_caption = re.compile(r'^\s*图\s*\d+[-\.]\d+.*$', re.MULTILINE)
    re_api_heading = re.compile(r'^\s*(?:\d+\.|\#+\s*|函数名称)\s*[\u4e00-\u9fa5A-Za-z0-9_]+', re.MULTILINE)

    for doc in children:
        cid = _resolve_chunk_id(doc)
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        p_id = meta.get("product_id", "")
        doc_type = meta.get("doc_type", "")
        content = doc.page_content or ""
        sec_title = str(meta.get("section_title", ""))
        func_names_raw = meta.get("function_names", "")
        # function_names 存储为逗号分隔字符串，需拆分为列表
        func_names = [f.strip() for f in func_names_raw.split(",") if f.strip()] if func_names_raw else []

        # ----------------------------------------------------
        # 1. 骨架目录块检测 (Skeleton / Index Chunk)
        # ----------------------------------------------------
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        # 如果长度很短，且主要由序号/标题构成，且没有具体代码实现/详细参数
        if len(content) < 180:
            skeleton_indicators = sum(1 for l in lines if re.match(r'^\d+\..*$', l) or "示例代码" in l)
            if skeleton_indicators >= 2 or (len(lines) <= 4 and "示例代码" in content and not ("def " in content or "(" in content)):
                issues["skeleton_chunk"].append((cid, p_id, content.replace("\n", " ")[:60]))

        # ----------------------------------------------------
        # 2. 多 API 强行粘连检测 (Multi-API Sticky Chunk)
        # ----------------------------------------------------
        # 🔴 v15: 以正文中的 API 边界标题数为准 — function_names 可能含交叉引用
        if doc_type == "c_sdk":
            _api_headings_in_text = len(re.findall(r'(?:^|\n)\s*(?:函数名称|\d{1,2}[\.\、\s])', content))
            if _api_headings_in_text >= 2:
                issues["multi_api_sticky"].append(
                    (cid, p_id, f"正文含 {_api_headings_in_text} 个 API 边界标题")
                )
            elif len(func_names) > 2:
                # 仅当 function_names > 2（远超交叉引用正常范围）才报
                issues["multi_api_sticky"].append(
                    (cid, p_id, f"function_names={len(func_names)}个: {func_names[:3]}")
                )

        # ----------------------------------------------------
        # 3. 标题与元数据脏化检测 (Title Noise)
        # ----------------------------------------------------
        # 🔴 v15: 豁免无标题的导言/过渡块 — 无 API 边界即非正式 API 块
        _is_preamble = (
            not re_api_heading.search(content)        # 不含数字标题或函数名称
            and "函数名称" not in content              # 不含中文函数表头
            and not re.search(r'(?:参数说明|返回值|功能描述)', content)  # 不含 API 描述关键词
        )
        if not _is_preamble:
            if not sec_title or sec_title.strip() in ["", "#", "##"]:
                issues["corrupted_title"].append((cid, p_id, "标题为空或仅为 #"))
            elif "\n" in sec_title or "函数名称" in sec_title:
                issues["corrupted_title"].append((cid, p_id, f"脏标题: {repr(sec_title)}"))

        # ----------------------------------------------------
        # 4. PDF 代码/符号断裂检测 (OCR Artifacts)
        # ----------------------------------------------------
        ocr_match = re_ocr_underscore.search(content)
        if ocr_match:
            issues["ocr_code_artifact"].append((cid, p_id, f"错误符号: '{ocr_match.group(0)}'"))

        # ----------------------------------------------------
        # 5. GUI 轨孤立图注检测 (Isolated Caption)
        # ----------------------------------------------------
        if doc_type == "gui_app":
            non_empty_lines = [l for l in lines if not l.startswith("[")]
            if len(non_empty_lines) <= 2 and any(re_isolated_caption.match(l) for l in non_empty_lines):
                issues["isolated_caption"].append((cid, p_id, content.replace("\n", " ")[:50]))

        # ----------------------------------------------------
        # 6. 面包屑路径循环重复检测
        # ----------------------------------------------------
        path_match = re.search(r'\[路径:\s*(.*?)\]', content)
        if path_match:
            breadcrumb = path_match.group(1)
            parts = [p.strip() for p in breadcrumb.split('>') if p.strip()]
            if len(parts) != len(set(parts)):
                issues["corrupted_breadcrumb"].append((cid, p_id, breadcrumb))

    # ====================================================
    # 📊 输出综合质量评估报告
    # ====================================================
    print("=" * 70)
    print(f"📋 工业机器人 RAG 切片质量健康度报告 (Total Child Chunks: {total_children})")
    print(f"   Parent Chunks 总数: {len(parents)}")
    print("=" * 70)

    health_score = 100.0
    
    def print_metric(title, issue_key, weight=1.0):
        nonlocal health_score
        items = issues[issue_key]
        count = len(items)
        ratio = (count / total_children) * 100
        health_score -= (count / total_children) * 100 * weight
        status = "✅ PASS" if count == 0 else ("⚠️ WARN" if ratio < 5 else "❌ FAIL")
        print(f"\n{title}: {count}/{total_children} ({ratio:.1f}%) [{status}]")
        if count > 0:
            for cid, pid, detail in items[:3]:
                print(f"   - [{pid}] {cid}: {detail}")
            if count > 3:
                print(f"   ... 剩余 {count - 3} 项未列出")

    print_metric("1. 骨架/目录占位块 (Skeleton Chunks)", "skeleton_chunk", weight=1.5)
    print_metric("2. 多 API 跨边界强行粘连 (Multi-API Sticky)", "multi_api_sticky", weight=2.0)
    print_metric("3. 标题/元数据脏化 (Corrupted Section Title)", "corrupted_title", weight=1.0)
    print_metric("4. PDF 代码下划线/空格断裂 (OCR Artifacts)", "ocr_code_artifact", weight=1.0)
    print_metric("5. GUI 轨孤立图注碎片 (Isolated Caption)", "isolated_caption", weight=0.8)
    print_metric("6. 面包屑路径死循环 (Corrupted Breadcrumb)", "corrupted_breadcrumb", weight=1.0)

    health_score = max(0.0, health_score)
    print("\n" + "=" * 70)
    print(f"📈 切片综合健康度得分 (Health Score): {health_score:.1f} / 100.0")
    
    # 按产品分布统计
    by_product = Counter(doc.metadata.get("product_id", "?") for doc in children)
    by_doc_type = Counter(doc.metadata.get("doc_type", "?") for doc in children)
    api_count = sum(1 for doc in children if doc.metadata.get("is_api"))
    print(f"📊 产品分布: {dict(by_product)}")
    print(f"📊 文档类型: {dict(by_doc_type)}")
    print(f"📊 强原子 API 块总数: {api_count}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    audit_chunks()