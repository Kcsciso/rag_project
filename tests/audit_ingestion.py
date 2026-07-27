#!/usr/bin/env python3
"""
=============================================================================
比邻星 (ProximaRAG) — v4 向量库白盒质量审计脚本
=============================================================================

审计规则:
  规则一: 零切片探测 — 检查已知产品的 Child 切片数，为 0 则 ERROR
  规则二: 垃圾切片扫描 — 扫描 page_content < 20 字符的 Child 切片，WARNING
  规则三: 关键实体存活抽样 — 强制检索 "robot_Power_on"(OpenC3) 和 "6502"(JAKA)
          验证 metadata 中的产品归属、章节路径、function_names 原始大小写

用法:
  python tests/audit_ingestion.py
=============================================================================
"""

import logging
import sys
import os

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
        logger.critical("❌ rag_v4_child Collection 未找到，请先运行 rebuild_v4.py")
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

    # 报告未知产品
    unknowns = {k: v for k, v in product_counts.items() if k not in KNOWN_PRODUCTS}
    if unknowns:
        for pid, count in unknowns.items():
            print(f"  ⚠️  未知产品: {pid}: {count} Child chunks")

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
                "content": doc_text.strip(),
                "length": len(doc_text.strip()),
                "product_id": meta.get("product_id", "?"),
                "section_title": meta.get("section_title", ""),
            })

    if garbage:
        print(f"  ⚠️  发现 {len(garbage)} 个垃圾切片 (< {MIN_CHILD_CHARS} 字符):")
        for g in garbage[:10]:  # 最多显示 10 个
            print(f"    [{g['chunk_id'][:40]}] pid={g['product_id']} len={g['length']} "
                  f"section={g['section_title'][:40]}")
            if g["content"]:
                print(f"      content: {g['content'][:80]}")
        if len(garbage) > 10:
            print(f"    ... 还有 {len(garbage) - 10} 个垃圾切片未显示")
        logger.warning(f"发现 {len(garbage)} 个垃圾切片")
    else:
        print(f"  ✅ 所有 Child 切片长度均 ≥ {MIN_CHILD_CHARS} 字符")

    return len(garbage)


def rule3_entity_sampling(child_vs) -> dict:
    """规则三: 关键实体存活抽样。返回抽样结果摘要。"""
    print("\n" + "=" * 60)
    print("🔍 规则三: 关键实体存活抽样")
    print("=" * 60)

    results = {}

    # ── 3a: OpenC3 robot_Power_on ──
    print("\n  ── 3a: OpenC3 'robot_Power_on' 检索 ──")
    try:
        oc3_docs = child_vs.similarity_search_with_score(
            "robot_Power_on", k=5,
            filter={"product_id": "OpenC3"},
        )
        hits = []
        for doc, score in oc3_docs:
            fn = doc.metadata.get("function_names", "")
            section = doc.metadata.get("section_title", "")
            hits.append({
                "score": round(score, 4),
                "function_names": fn,
                "section_title": section,
                "product_id": doc.metadata.get("product_id", "?"),
                "text_preview": doc.page_content[:120],
            })

        if hits:
            print(f"    ✅ 找到 {len(hits)} 个匹配切片:")
            for h in hits:
                print(f"      score={h['score']} pid={h['product_id']} "
                      f"section={h['section_title'][:50]}")
                print(f"      function_names={h['function_names']}")
                # 验证原始大小写
                if "Power_on" in h["function_names"]:
                    print(f"      ✅ 原始大小写保留: 'Power_on' 存在")
                else:
                    print(f"      ⚠️  未检测到原始大小写 'Power_on'，当前 function_names={h['function_names']}")
                print(f"      preview: {h['text_preview'][:100]}")
        else:
            print(f"    ❌ 未找到 robot_Power_on 相关切片！")
            logger.error("OpenC3 robot_Power_on 检索为空 — 关键实体缺失")
        results["OpenC3_robot_Power_on"] = len(hits)
    except Exception as e:
        print(f"    ❌ 检索异常: {e}")
        results["OpenC3_robot_Power_on"] = 0

    # ── 3b: JAKA 端口号 6502 ──
    print("\n  ── 3b: JAKA '6502' 检索 ──")
    try:
        jaka_docs = child_vs.similarity_search_with_score(
            "6502 端口号 Modbus", k=5,
            filter={"product_id": "JAKA"},
        )
        hits = []
        for doc, score in jaka_docs:
            section = doc.metadata.get("section_title", "")
            hits.append({
                "score": round(score, 4),
                "section_title": section,
                "product_id": doc.metadata.get("product_id", "?"),
                "text_preview": doc.page_content[:120],
            })

        if hits:
            print(f"    ✅ 找到 {len(hits)} 个匹配切片:")
            for h in hits:
                print(f"      score={h['score']} pid={h['product_id']} "
                      f"section={h['section_title'][:60]}")
                print(f"      preview: {h['text_preview'][:100]}")
        else:
            print(f"    ❌ 未找到 6502 相关切片！")
            logger.error("JAKA 6502 检索为空 — 关键实体缺失")
        results["JAKA_6502"] = len(hits)
    except Exception as e:
        print(f"    ❌ 检索异常: {e}")
        results["JAKA_6502"] = 0

    # ── 3c: 面包屑路径抽样 (随机 3 个 Child 检查 [路径:] 前缀) ──
    print("\n  ── 3c: 面包屑路径抽样 ──")
    all_data = child_vs._collection.get(include=["documents", "metadatas"], limit=200)
    breadcrumb_count = 0
    path_count = 0
    section_count = 0
    sampled = 0
    for doc_text, meta in zip(all_data["documents"], all_data["metadatas"]):
        if meta.get("chunk_type") != "child":
            continue
        if "[文档:" in doc_text:
            breadcrumb_count += 1
        if "[路径:" in doc_text:
            path_count += 1
        if "[章节:" in doc_text:
            section_count += 1
        sampled += 1

    print(f"    ✅ 抽样 {sampled} 个 Child 切片:")
    print(f"       [文档:] 前缀: {breadcrumb_count}/{sampled} ({100*breadcrumb_count/max(sampled,1):.0f}%)")
    print(f"       [路径:] 前缀: {path_count}/{sampled} ({100*path_count/max(sampled,1):.0f}%)")
    print(f"       [章节:] 前缀: {section_count}/{sampled} ({100*section_count/max(sampled,1):.0f}%)")
    results["breadcrumb_coverage"] = {
        "sampled": sampled,
        "doc_pct": 100 * breadcrumb_count / max(sampled, 1),
        "path_pct": 100 * path_count / max(sampled, 1),
        "section_pct": 100 * section_count / max(sampled, 1),
    }

    return results


def audit_summary(parent_vs, child_vs, errors, garbage, entities):
    """输出汇总报告。"""
    print("\n" + "=" * 60)
    print("📊 审计汇总报告")
    print("=" * 60)

    p_count = parent_vs._collection.count() if parent_vs else 0
    c_count = child_vs._collection.count() if child_vs else 0

    # 产品分布
    all_data = child_vs._collection.get(include=["metadatas"])
    product_counts = {}
    for meta in all_data["metadatas"]:
        pid = (meta.get("product_id") or "unknown").strip()
        product_counts[pid] = product_counts.get(pid, 0) + 1

    # function_names 覆盖率
    fn_count = sum(1 for meta in all_data["metadatas"] if meta.get("function_names", "").strip())

    print(f"\n  📦 向量库规模:")
    print(f"     Parent: {p_count} chunks")
    print(f"     Child:  {c_count} chunks")

    print(f"\n  🏷️  产品分布:")
    for pid in sorted(product_counts.keys()):
        flag = "✅" if pid in KNOWN_PRODUCTS else "⚠️"
        print(f"     {flag} {pid}: {product_counts[pid]} Child chunks")

    print(f"\n  📐 质量指标:")
    print(f"     function_names 覆盖率: {fn_count}/{c_count} ({100*fn_count/max(c_count,1):.0f}%)")
    print(f"     零切片产品: {errors}")
    print(f"     垃圾切片: {garbage}")

    print(f"\n  🔑 关键实体:")
    for k, v in entities.items():
        if isinstance(v, dict):
            print(f"     {k}: 路径{int(v.get('path_pct',0))}% 章节{int(v.get('section_pct',0))}%")
        else:
            status = "✅" if v > 0 else "❌"
            print(f"     {status} {k}: {v} 命中")

    # 综合判定
    print(f"\n  🏁 综合判定:")
    all_pass = errors == 0 and garbage <= c_count * 0.02  # ≤2% 垃圾容忍度
    if all_pass:
        print(f"     ✅ 审计通过 — 向量库数据质量合格")
    else:
        print(f"     ❌ 审计未通过 — 请检查上述告警项")


def main():
    print("=" * 60)
    print("  比邻星 (ProximaRAG) v4 向量库白盒质量审计")
    print("=" * 60)

    parent_vs, child_vs = load_collections()
    if child_vs is None:
        sys.exit(1)

    errors = rule1_zero_chunks(child_vs)
    garbage = rule2_garbage_chunks(child_vs)
    entities = rule3_entity_sampling(child_vs)

    audit_summary(parent_vs, child_vs, errors, garbage, entities)

    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
