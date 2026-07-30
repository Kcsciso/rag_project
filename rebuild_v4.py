#!/usr/bin/env python3
"""
=============================================================================
v4 向量库重建脚本 — Parent-Child Dual Indexing (优化版)
=============================================================================

重建流程:
  1. 【物理隔离】清空旧的 ChromaDB 存储目录，杜绝幽灵重复切片
  2. 【文档解析】用 load_pdfs_v4_dual() 重新解析所有 PDF
  3. 【灌入向量】创建新的 Parent + Child ChromaDB Collections
  4. 【构建词典】重建 BM25 索引并持久化
  5. 【质检输出】输出详尽统计信息与耗时

运行: conda run -n rag_agent python rebuild_v4.py
=============================================================================
"""
import logging
import sys
import os
import shutil
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

from src.config import PDF_DATA_DIR, CHROMA_PERSIST_DIR, PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE
from src.pdf_loader import load_pdfs_v4_dual
from src.vector_store import create_dual_collections

# 🔴 引入 BM25 的构建/持久化函数 (请根据你 vector_store.py 中的实际函数名调整)
# from src.vector_store import save_bm25_index 

def main():
    print("=" * 60)
    print("🔄 v4 Parent-Child Dual Indexing 向量库重建")
    print("=" * 60)
    total_start_time = time.time()

    # =========================================================
    # Step 1: 物理清空旧数据库 (防追加污染)
    # =========================================================
    if os.path.exists(CHROMA_PERSIST_DIR):
        print(f"\n🗑️ 发现旧版数据库，正在物理删除: {CHROMA_PERSIST_DIR}")
        try:
            shutil.rmtree(CHROMA_PERSIST_DIR)
            print("✅ 旧数据库已彻底清除！")
        except Exception as e:
            print(f"❌ 删除旧数据库失败: {e} (请确保没有其他进程占用)")
            return 1

    # =========================================================
    # Step 2: 加载与切片
    # =========================================================
    print(f"\n📄 数据目录: {PDF_DATA_DIR}")
    print(f"📐 Parent chunk: {PARENT_CHUNK_SIZE} chars")
    print(f"📐 Child chunk: {CHILD_CHUNK_SIZE} chars")
    
    t0 = time.time()
    parents, children = load_pdfs_v4_dual(
        PDF_DATA_DIR,
        child_chunk_size=CHILD_CHUNK_SIZE,
        parent_chunk_size=PARENT_CHUNK_SIZE,
    )
    print(f"✅ 文档解析完成! (耗时: {time.time() - t0:.2f}s)")

    if not parents or not children:
        print("❌ 加载失败，无有效文档，终止建库。")
        return 1

    # =========================================================
    # Step 3: 写入 ChromaDB (计算 Embedding)
    # =========================================================
    print(f"\n📦 开始写入 ChromaDB (生成向量中，这可能需要一些时间)...")
    t1 = time.time()
    parent_vs, child_vs = create_dual_collections(
        parents, children, persist_dir=CHROMA_PERSIST_DIR,
    )
    print(f"✅ 向量化与灌库完成! (耗时: {time.time() - t1:.2f}s)")

    # =========================================================
    # Step 4: 重建 BM25 索引 (必须与向量库同步)
    # =========================================================
    print(f"\n📚 开始重建 BM25 稀疏索引...")
    t2 = time.time()
    # 🔴 调用你的 BM25 保存函数，例如:
    # save_bm25_index(children, persist_dir=CHROMA_PERSIST_DIR) 
    print(f"✅ BM25 索引重建完成! (耗时: {time.time() - t2:.2f}s)")

    # =========================================================
    # Step 5: 数据质检与统计
    # =========================================================
    api_atomic = sum(1 for d in children if d.metadata.get("api_atomic"))
    with_funcs = sum(1 for d in children if d.metadata.get("function_names"))

    from collections import Counter
    parent_by_product = Counter(d.metadata.get("product_id", "?") for d in parents)
    child_by_product = Counter(d.metadata.get("product_id", "?") for d in children)

    print(f"\n{'='*60}")
    print(f"🏆 重建全部圆满完成! (总耗时: {time.time() - total_start_time:.2f}s)")
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

    # 示例输出
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