import re
import sys
import os
import json
from collections import Counter
from datetime import datetime

# 增加根目录路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pdf_loader import load_pdfs_v4_dual

def _resolve_chunk_id(doc) -> str:
    """兼容多种 metadata 字段名，确保能打印真实 ID。"""
    meta = doc.metadata if hasattr(doc, "metadata") else {}
    cid = meta.get("chunk_id") or meta.get("id") or ""
    if cid:
        return str(cid)
    pid = meta.get("parent_id", "?")
    fp = doc.page_content[40:100] if hasattr(doc, "page_content") else ""
    return f"c_{pid}_{abs(hash(fp)) % 10000:04d}"


def audit_chunks():
    print("🔍 开始工业级 RAG 架构切片质量审计 (Deep Semantic Audit)...\n")
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

    # 📊 8 大质量指标异常收集器
    issues = {
        "skeleton_chunk": [],         # 1. 骨架/目录占位块
        "multi_api_sticky": [],       # 2. 多个 API 强行粘连
        "corrupted_title": [],        # 3. 标题/元数据脏化
        "ocr_code_artifact": [],      # 4. PDF 代码下划线/空格断裂
        "isolated_caption": [],       # 5. GUI 轨孤立图注块
        "corrupted_breadcrumb": [],   # 6. 面包屑路径死循环/语义脱节
        "low_containment_sdk": [],    # 7. SDK 碎化与低自包含度 (裸代码/裸文档)
        "ast_collapse": [],           # 8. AST 章节序号倒挂与架构崩塌
    }

    re_ocr_underscore = re.compile(r'\b\w+\s+_\s+\w+\b|_\s+_|\w+\s+\.\s+\w+')
    re_isolated_caption = re.compile(r'^\s*图\s*\d+[-\.]\d+.*$', re.MULTILINE)
    re_sec_num = re.compile(r'^\s*(\d{1,2})[\.、\s]')

    sdk_chunk_count = 0

    for doc in children:
        cid = _resolve_chunk_id(doc)
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        p_id = meta.get("product_id", "")
        doc_type = meta.get("doc_type", "")
        content = doc.page_content or ""
        sec_title = str(meta.get("section_title", ""))

        if doc_type == "c_sdk":
            sdk_chunk_count += 1

        lines = [line.strip() for line in content.split("\n") if line.strip()]

        # ----------------------------------------------------
        # 1. 骨架目录块检测 (Skeleton Chunk)
        # ----------------------------------------------------
        if len(content) < 150:
            skeleton_indicators = sum(1 for l in lines if re.match(r'^\d+\..*$', l) or "示例代码" in l)
            if skeleton_indicators >= 2 or (len(lines) <= 4 and "示例代码" in content and not ("def " in content or "(" in content)):
                issues["skeleton_chunk"].append((cid, p_id, content.replace("\n", " ")[:60], content))

        # ----------------------------------------------------
        # 2. 多 API 强行粘连检测 (Multi-API Sticky Chunk)
        # ----------------------------------------------------
        if doc_type == "c_sdk":
            # 🔴 核心修改：废除愚蠢的正则扫描和函数数量惩罚。
            # 真正的粘连是：一个 Chunk 里出现了多套完整的 API 说明骨架
            param_count = content.count("参数说明")
            return_count = content.count("返回值")
            if param_count >= 2 and return_count >= 2:
                issues["multi_api_sticky"].append((cid, p_id, f"发现 {param_count} 个参数说明和 {return_count} 个返回值，发生结构粘连", content))

        # ----------------------------------------------------
        # 3. 标题与元数据脏化检测 (Corrupted Title)
        # ----------------------------------------------------
        if not sec_title or sec_title.strip() in ["", "#", "##"]:
            issues["corrupted_title"].append((cid, p_id, "标题为空或仅为 #", content))
        elif "\n" in sec_title or "函数名称" in sec_title:
            issues["corrupted_title"].append((cid, p_id, f"脏标题: {repr(sec_title)}", content))

        # ----------------------------------------------------
        # 4. PDF 代码/符号断裂检测 (OCR Artifacts)
        # ----------------------------------------------------
        ocr_match = re_ocr_underscore.search(content)
        if ocr_match:
            issues["ocr_code_artifact"].append((cid, p_id, f"错误符号: '{ocr_match.group(0)}'", content))

        # ----------------------------------------------------
        # 5. GUI 轨孤立图注检测 (Isolated Caption)
        # ----------------------------------------------------
        if doc_type == "gui_app":
            non_empty_lines = [l for l in lines if not l.startswith("[")]
            if len(non_empty_lines) <= 2 and any(re_isolated_caption.match(l) for l in non_empty_lines):
                issues["isolated_caption"].append((cid, p_id, content.replace("\n", " ")[:50], content))

        # ----------------------------------------------------
        # 6. 面包屑路径死循环与语义脱节检测 (Breadcrumb Noise)
        # ----------------------------------------------------
        path_match = re.search(r'\[路径:\s*(.*?)\]', content)
        if path_match:
            breadcrumb = path_match.group(1)
            parts = [p.strip() for p in breadcrumb.split('>') if p.strip()]
            if len(parts) != len(set(parts)):
                issues["corrupted_breadcrumb"].append((cid, p_id, f"死循环路径: {breadcrumb}", content))
            elif sec_title and ("上电" in sec_title or "emergency" in sec_title.lower()):
                if any(k in breadcrumb for k in ["等待", "延时", "命令发送"]):
                    issues["corrupted_breadcrumb"].append((cid, p_id, f"语义脱节: '{sec_title}' 挂载于 '{breadcrumb}'", content))

        # ----------------------------------------------------
        # 7. SDK 碎化与低自包含度检测 (Low-Containment SDK)
        # ----------------------------------------------------
        if doc_type == "c_sdk":
            # 🔴 核心修改：大幅放宽长度容忍度。物理拉直后，100字的 API 也是极品 API。
            has_code = bool(re.search(r'\b[a-zA-Z_]\w*\s*\(|robot\.|dll\.|ctypes|CDLL|c_int|POSE|Joint|\.restype|\.argtypes|import\s', content))
            has_doc = any(k in content for k in ["功能描述", "函数说明", "参数说明", "返回值", "功能说明"])
            
            if has_code and not has_doc and len(content) < 80:
                issues["low_containment_sdk"].append((cid, p_id, f"极短裸代码碎片(len={len(content)}): 缺失文档说明", content))
            elif has_doc and not has_code and len(content) < 80:
                issues["low_containment_sdk"].append((cid, p_id, f"极短裸文档碎片(len={len(content)}): 缺失代码特征", content))

        # ----------------------------------------------------
        # 8. AST 章节序号倒挂与架构崩塌检测 (AST Collapse)
        # ----------------------------------------------------
        sec_num_m = re_sec_num.search(sec_title) or re_sec_num.search(content)
        if sec_num_m and path_match:
            sec_num = int(sec_num_m.group(1))
            breadcrumb = path_match.group(1)
            parent_nums = [int(n) for n in re.findall(r'(?:^|>)\s*(\d{1,2})[\.、\s]', breadcrumb)]
            if parent_nums:
                max_parent_num = max(parent_nums)
                if sec_num >= max_parent_num + 3:
                    issues["ast_collapse"].append(
                        (cid, p_id, f"章节号 {sec_num} 倒挂于父级路径 '{breadcrumb}'", content)
                    )

    # ====================================================
    # 📊 输出综合 quality 评估报告
    # ====================================================
    print("=" * 70)
    print(f"📋 工业机器人 RAG 切片质量健康度报告 (Total Child Chunks: {total_children})")
    print(f"   Parent Chunks 总数: {len(parents)} | C-SDK 切片总数: {sdk_chunk_count}")
    print("=" * 70)

    health_score = 100.0
    sdk_health_score = 100.0
    
    report_dict = {}

    def process_metric(title, issue_key, weight=1.0, is_sdk_metric=False, max_deduct=20.0):
        nonlocal health_score, sdk_health_score
        items = issues[issue_key]
        count = len(items)
        ratio = (count / total_children) * 100 if total_children > 0 else 0
        
        deduction = min((count / total_children) * 100 * weight, max_deduct) if total_children > 0 else 0
        health_score -= deduction
        
        if is_sdk_metric and sdk_chunk_count > 0:
            sdk_deduction = min((count / sdk_chunk_count) * 100 * weight, max_deduct)
            sdk_health_score -= sdk_deduction

        status = "✅ PASS" if count == 0 else ("⚠️ WARN" if ratio < 3 else "❌ FAIL")
        print(f"\n{title}: {count}/{total_children} ({ratio:.1f}%) [{status}]")
        
        report_dict[title] = []

        if count > 0:
            for cid, pid, detail, raw_content in items[:3]:
                print(f"   - [{pid}] {cid}: {detail}")
            if count > 3:
                print(f"   ... 剩余 {count - 3} 项未列出，详情见完整日志")
            
            for cid, pid, detail, raw_content in items:
                report_dict[title].append({
                    "chunk_id": cid,
                    "product": pid,
                    "detail": detail,
                    "content_preview": raw_content[:150] + "..." if len(raw_content) > 150 else raw_content
                })

    process_metric("1. 骨架/目录占位块 (Skeleton Chunks)", "skeleton_chunk", weight=1.5)
    process_metric("2. 多 API 跨边界强行粘连 (Multi-API Sticky)", "multi_api_sticky", weight=2.0, is_sdk_metric=True)
    process_metric("3. 标题/元数据脏化 (Corrupted Section Title)", "corrupted_title", weight=1.0)
    process_metric("4. PDF 代码下划线/空格断裂 (OCR Artifacts)", "ocr_code_artifact", weight=1.0)
    process_metric("5. GUI 轨孤立图注碎片 (Isolated Caption)", "isolated_caption", weight=0.8)
    process_metric("6. 面包屑死循环与语义脱节 (Corrupted Breadcrumb)", "corrupted_breadcrumb", weight=1.5, is_sdk_metric=True)
    process_metric("7. SDK 碎化与低自包含度 (Low-Containment SDK)", "low_containment_sdk", weight=2.5, is_sdk_metric=True)
    process_metric("8. AST 章节序号倒挂 (AST Hierarchy Collapse)", "ast_collapse", weight=3.0, is_sdk_metric=True)

    health_score = max(0.0, health_score)
    sdk_health_score = max(0.0, sdk_health_score)

    print("\n" + "=" * 70)
    print(f"📈 全库综合健康度得分 (Overall Score): {health_score:.1f} / 100.0")
    print(f"🎯 C-SDK 核心轨健康度得分 (C-SDK Core Score): {sdk_health_score:.1f} / 100.0")
    
    by_product = Counter(doc.metadata.get("product_id", "?") for doc in children)
    by_doc_type = Counter(doc.metadata.get("doc_type", "?") for doc in children)
    api_count = sum(1 for doc in children if doc.metadata.get("is_api"))
    print(f"📊 产品分布: {dict(by_product)}")
    print(f"📊 文档类型: {dict(by_doc_type)}")
    print(f"📊 强原子 API 块总数: {api_count}")
    print("=" * 70 + "\n")

    report_file = f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)
    print(f"📁 完整报错切片已导出至: {report_file} (可用于白盒问题定位)\n")


if __name__ == "__main__":
    audit_chunks()