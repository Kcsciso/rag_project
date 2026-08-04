#!/usr/bin/env python3
"""
=============================================================================
比邻星 (ProximaRAG) — v4.4 向量库白盒质量审计脚本 (多模态与排版强化版)
=============================================================================

审计规则:
  规则一: 零切片探测 — 检查已知产品的 Child 切片数，为 0 则 ERROR
  规则二: 垃圾切片扫描 — 扫描 page_content < 20 字符的 Child 切片，WARNING
  规则三: 关键实体与疑难排版存活抽样 (SDK函数、OCR隐藏参数、复杂表格、特异符号)
  规则四: L3 大纲与 OCR 标记注水检查 — 验证 Parent大纲 与 OCR元信息 是否成功写入

用法:
  python tests/audit_ingestion.py
=============================================================================
"""

import logging
import sys
import os
import re

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("audit_ingestion")

KNOWN_PRODUCTS = ["OpenC3", "JAKA", "OpenR6"]
MIN_CHILD_CHARS = 20


def load_collections():
    """加载 v4 Parent + Child Collection。"""
    from src.vector_store import load_vector_store_from_name

    child_vs = load_vector_store_from_name("rag_v4_child")
    parent_vs = load_vector_store_from_name("rag_v4_parent")

    if child_vs is None:
        logger.critical("❌ rag_v4_child Collection 未找到，请先运行建库脚本")
        return None, None

    logger.info(f"✅ Child Collection 已加载: {child_vs._collection.count()} chunks")
    if parent_vs:
        logger.info(f"✅ Parent Collection 已加载: {parent_vs._collection.count()} chunks")
    else:
        logger.warning("⚠️  Parent Collection 未找到")
    return parent_vs, child_vs


def rule1_zero_chunks(child_vs) -> int:
    """规则一: 零切片探测。返回发现的零切片产品数。"""
    print("\n" + "=" * 60)
    print("🔍 规则一: 零切片探测")
    print("=" * 60)

    errors = 0
    all_data = child_vs._collection.get(include=["metadatas"])
    product_counts = {}
    for meta in all_data["metadatas"]:
        pid = (meta.get("product_id") or "unknown").strip()
        product_counts[pid] = product_counts.get(pid, 0) + 1

    for pid in KNOWN_PRODUCTS:
        count = product_counts.get(pid, 0)
        status = "✅" if count > 0 else "❌ ERROR"
        print(f"  {status} {pid}: {count} Child chunks")
        if count == 0:
            errors += 1
            logger.error(f"❌ 产品 '{pid}' Child 切片数为 0！请检查 PDF 文件是否存在。")

    return errors


def rule2_garbage_chunks(child_vs) -> int:
    """规则二: 垃圾切片扫描。返回发现的垃圾切片数。"""
    print("\n" + "=" * 60)
    print("🔍 规则二: 垃圾切片扫描 (page_content < {} 字符)".format(MIN_CHILD_CHARS))
    print("=" * 60)

    all_data = child_vs._collection.get(include=["documents", "metadatas"])
    garbage = []

    for i, (doc_text, meta) in enumerate(zip(all_data["documents"], all_data["metadatas"])):
        if len(doc_text.strip()) < MIN_CHILD_CHARS:
            garbage.append({
                "chunk_id": all_data["ids"][i],
                "content": doc_text.strip().replace("\n", " "),
                "length": len(doc_text.strip()),
                "product_id": meta.get("product_id", "?"),
            })

    if garbage:
        print(f"  ⚠️  发现 {len(garbage)} 个垃圾切片 (< {MIN_CHILD_CHARS} 字符):")
        for g in garbage[:5]:
            print(f"    [{g['chunk_id'][:10]}...] pid={g['product_id']} len={g['length']} content: {g['content']}")
        logger.warning(f"发现 {len(garbage)} 个垃圾切片")
    else:
        print(f"  ✅ 所有 Child 切片长度均 ≥ {MIN_CHILD_CHARS} 字符")

    return len(garbage)


def _check_entity(child_vs, name, query, filter_dict, required_keywords):
    """辅助函数：执行全库暴力扫描，绕过向量检索的盲区"""
    print(f"\n  ── {name} ──")
    try:
        # 1. 暴力全库扫描 (获取该产品下的所有切片)
        all_data = child_vs._collection.get(where=filter_dict, include=["documents", "metadatas"])
        
        valid_hits = []
        for doc_text, meta in zip(all_data["documents"], all_data["metadatas"]):
            # 检查是否同时包含所有关键字
            if all(kw.lower() in doc_text.lower() for kw in required_keywords):
                valid_hits.append((doc_text, meta))

        if valid_hits:
            print(f"    ✅ 全库扫描命中！成功验证包含: {required_keywords}")
            best_text, best_meta = valid_hits[0]
            print(f"      Section={best_meta.get('section_title', '未知')}")
            preview = best_text.replace('\n', ' ')
            print(f"      Preview: {preview[:150]}...")
            return 1
        else:
            print(f"    ❌ 全库扫描了 {len(all_data['documents'])} 条切片，无一包含完整关键字 {required_keywords}")
            
            # 2. 🔍 降级诊断模式：单关键字独立扫描
            print("       -> 🔍 降级诊断信息:")
            for kw in required_keywords:
                kw_hits = sum(1 for d in all_data["documents"] if kw.lower() in d.lower())
                print(f"          单独包含 '{kw}' 的切片数量: {kw_hits}")
            return 0
            
    except Exception as e:
        print(f"    ❌ 扫描异常: {e}")
        return 0


def rule3_entity_sampling(child_vs) -> dict:
    """规则三: 关键实体存活抽样 (覆盖高压测试点)"""
    print("\n" + "=" * 60)
    print("🔍 规则三: 高压疑难实体存活验证 (OCR / 表格 / 符号)")
    print("=" * 60)
    
    results = {}
    
    # 3a. 常规 SDK 函数验证 (大小写与防混淆)
    results["SDK_Power_on"] = _check_entity(
        child_vs, "3a: OpenC3 'robot_Power_on' 存活验证", 
        "robot_Power_on", {"product_id": "OpenC3"}, ["robot_Power_on"]
    )
    
    # 3b. 图片内参数 OCR 提取验证
    results["OCR_6502"] = _check_entity(
        child_vs, "3b: JAKA 图片隐式参数 '6502' 存活验证 (OCR)", 
        "JAKA 默认端口号 6502", {"product_id": "JAKA"}, ["6502"]
    )
    
    # 3c. 复杂表格物理坐标排序提取验证
    results["Table_Windows"] = _check_entity(
        child_vs, "3c: JAKA '运行环境' 表格存活验证 (物理排序)", 
        "运行环境 Windows Android", {"product_id": "JAKA"}, ["Windows", "Android"]
    )
    
    # 3d. 特异符号/分词防断裂验证
    results["Symbol_Ethernet"] = _check_entity(
        child_vs, "3d: JAKA 特殊符号 'Ethernet/IP' 存活验证", 
        "Ethernet/IP IO", {"product_id": "JAKA"}, ["Ethernet/IP"]
    )
    
    return results


def rule4_architecture_marks(parent_vs, child_vs) -> dict:
    """规则四: L3 大纲与 OCR 标记注入验证"""
    print("\n" + "=" * 60)
    print("🔍 规则四: 架构级元数据(大纲与OCR)注入查验")
    print("=" * 60)
    results = {}

    # 4a: 检查 Parent 中是否成功注入跨级大纲 [章节大纲参考]
    if parent_vs:
        print("\n  ── 4a: Parent 跨级大纲注入率 ──")
        all_parents = parent_vs._collection.get(include=["documents", "metadatas"])
        toc_count = sum(1 for doc in all_parents["documents"] if "[章节大纲参考]" in doc)
        total_parents = len(all_parents["documents"])
        pct = (toc_count / total_parents * 100) if total_parents else 0
        status = "✅" if toc_count > 0 else "❌"
        print(f"    {status} 发现 {toc_count}/{total_parents} ({pct:.1f}%) 的 Parent 包含微缩大纲")
        results["Parent_TOC_Injected"] = toc_count

    # 4b: 检查 Child 中是否有 JAKA 强制 OCR 的存活证明
    print("\n  ── 4b: JAKA 强制 OCR 启动证明 ──")
    all_children = child_vs._collection.get(include=["documents", "metadatas"])
    ocr_chunks = []
    for doc, meta in zip(all_children["documents"], all_children["metadatas"]):
        if meta.get("product_id") == "JAKA" and "[OCR补漏:" in doc:
            ocr_chunks.append(doc)
            
    if ocr_chunks:
        print(f"    ✅ 成功找到 {len(ocr_chunks)} 个带 OCR 标记的 JAKA 切片。")
        print(f"       示例: {ocr_chunks[0][:100].replace(chr(10), ' ')}...")
    else:
        print("    ⚠️  未找到任何 [OCR补漏:] 标记。若确认含有大量图片页面，说明 OCR 引擎可能未启动。")
    results["JAKA_OCR_Count"] = len(ocr_chunks)

    return results


def audit_summary(parent_vs, child_vs, errors, garbage, entities, marks):
    """输出最终审查报告"""
    print("\n" + "=" * 60)
    print("📊 终极审计汇总报告")
    print("=" * 60)

    c_count = child_vs._collection.count() if child_vs else 0

    print(f"\n  📦 数据规模: Child={c_count} 块")
    print(f"  📐 基础健康: 零切片错误={errors}, 垃圾切片={garbage}")

    print(f"\n  🔑 高压实体存活率 (4项):")
    passed_entities = sum(1 for v in entities.values() if v > 0)
    for k, v in entities.items():
        print(f"     {'✅' if v > 0 else '❌'} {k}")
    
    print(f"\n  🏷️  架构标记注入:")
    print(f"     {'✅' if marks.get('Parent_TOC_Injected', 0) > 0 else '❌'} Parent 大纲注入数: {marks.get('Parent_TOC_Injected', 0)}")
    print(f"     {'✅' if marks.get('JAKA_OCR_Count', 0) > 0 else '⚠️'} JAKA OCR 触发切片数: {marks.get('JAKA_OCR_Count', 0)}")

    # 综合判定
    all_pass = (errors == 0) and (garbage <= c_count * 0.05) and (passed_entities == len(entities))
    
    print(f"\n  🏁 综合判定:")
    if all_pass:
        print(f"     ✅ 审计通过 — 多模态版面分析与数据重构成功！")
    else:
        print(f"     ❌ 审计未通过 — 请检查上述打 ❌ 的核心问题！")


def main():
    print("=" * 60)
    print("  比邻星 (ProximaRAG) v4.4 向量库白盒质量审计 (多模态强化版)")
    print("=" * 60)

    parent_vs, child_vs = load_collections()
    if child_vs is None:
        sys.exit(1)

    errors = rule1_zero_chunks(child_vs)
    garbage = rule2_garbage_chunks(child_vs)
    entities = rule3_entity_sampling(child_vs)
    marks = rule4_architecture_marks(parent_vs, child_vs)

    audit_summary(parent_vs, child_vs, errors, garbage, entities, marks)

    if errors > 0 or any(v == 0 for v in entities.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()