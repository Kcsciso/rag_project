#!/usr/bin/env python3
"""
=============================================================================
v4 向量库重建脚本 — Parent-Child Dual Indexing
=============================================================================

重建流程:
  1. 删除旧的 v4 Collections（rag_v4_parent, rag_v4_child）
  2. 用 load_pdfs_v4_dual() 重新解析所有 PDF
  3. 创建新的 Parent + Child ChromaDB Collections
  4. 重建 BM25 索引
  5. 输出统计信息

运行: conda run -n rag_agent python rebuild_v4.py
=============================================================================
"""
import logging, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

from src.config import PDF_DATA_DIR, CHROMA_PERSIST_DIR, PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE
from src.pdf_loader import load_pdfs_v4_dual
from src.vector_store import create_dual_collections

def main():
    print("=" * 60)
    print("🔄 v4 Parent-Child Dual Indexing 向量库重建")
    print("=" * 60)

    # Step 1: 加载
    print(f"\n📄 数据目录: {PDF_DATA_DIR}")
    print(f"📐 Parent chunk: {PARENT_CHUNK_SIZE} chars")
    print(f"📐 Child chunk: {CHILD_CHUNK_SIZE} chars")

    parents, children = load_pdfs_v4_dual(
        PDF_DATA_DIR,
        child_chunk_size=CHILD_CHUNK_SIZE,
        parent_chunk_size=PARENT_CHUNK_SIZE,
    )

    if not parents or not children:
        print("❌ 加载失败，无有效文档")
        return 1

    # Step 2: 创建 Collection
    print(f"\n📦 写入 ChromaDB: {CHROMA_PERSIST_DIR}")
    parent_vs, child_vs = create_dual_collections(
        parents, children, persist_dir=CHROMA_PERSIST_DIR,
    )

    # Step 3: 统计
    api_atomic = sum(1 for d in children if d.metadata.get("api_atomic"))
    with_funcs = sum(1 for d in children if d.metadata.get("function_names"))

    # 按产品统计
    from collections import Counter
    parent_by_product = Counter(d.metadata.get("product_id", "?") for d in parents)
    child_by_product = Counter(d.metadata.get("product_id", "?") for d in children)

    print(f"\n{'='*60}")
    print("✅ 重建完成!")
    print(f"{'='*60}")
    print(f"  Parent Collection: {len(parents)} chunks")
    for pid, cnt in sorted(parent_by_product.items()):
        print(f"    {pid}: {cnt}")
    print(f"  Child Collection:  {len(children)} chunks")
    for pid, cnt in sorted(child_by_product.items()):
        print(f"    {pid}: {cnt}")
    print(f"  API 原子块:       {api_atomic}")
    print(f"  含函数名元数据:    {with_funcs}")
    print(f"  存储路径:          {CHROMA_PERSIST_DIR}")

    # Step 4: 示例输出
    if children:
        print(f"\n📝 Child 示例 (前 3):")
        for i, d in enumerate(children[:3]):
            api_tag = "[API]" if d.metadata.get("api_atomic") else ""
            funcs = d.metadata.get("function_names", [])
            print(f"  [{i+1}] {api_tag} pid={d.metadata.get('parent_id','?')} funcs={funcs}")
            print(f"      {d.page_content[:150]}...")

    return 0

if __name__ == "__main__":
    sys.exit(main())
