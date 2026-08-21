#!/usr/bin/env python3
"""
=============================================================================
v4 向量库重建脚本 — 双轨多模态 Parent-Child Dual Indexing (Stage 1 + 2 串联)
=============================================================================

重建流程:
  1. 【物理隔离】清空旧 vector_db 目录，彻底杜绝历史残留与重复切片
  2. 【双轨摄入】调用 pdf_loader.load_all_documents_v4_dual() 执行 SDK 状态机切片 + JAKA 多模态提纯
  3. 【自动属性】切片过程自动抽取并更新 kv_db/attribute_kv.json
  4. 【向量入库】计算 BGE Embedding 并写入 ChromaDB (Parent + Child Collections)
  5. 【统计输出】输出各产品线切片、多模态注入与 API 统计

运行: conda run -n rag_agent python src/rebuild_v4.py   (或 python -m src.rebuild_v4)
=============================================================================
"""
import logging
import sys
import os
import shutil
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

# 双入口兼容：脚本模式 (python src/rebuild_v4.py) 与包模式 (python -m src.rebuild_v4)
try:
    from src.config import PDF_DATA_DIR, CHROMA_PERSIST_DIR, JAKA_MARKDOWN_PATH
    from src.pdf_loader import load_all_documents_v4_dual
    from src.vector_store import create_dual_collections
except ImportError:
    from config import PDF_DATA_DIR, CHROMA_PERSIST_DIR, JAKA_MARKDOWN_PATH
    from pdf_loader import load_all_documents_v4_dual
    from vector_store import create_dual_collections

def main():
    print("=" * 75)
    print("🔄 ProximaRAG v4 向量库重建 (双轨多模态 Parent-Child Dual Indexing)")
    print("=" * 75)
    total_start_time = time.time()

    # =========================================================
    # Step 1: 物理清空旧数据库 (防历史幽灵切片)
    # =========================================================
    if os.path.exists(CHROMA_PERSIST_DIR):
        print(f"\n🗑️ 发现旧版数据库，正在物理清除: {CHROMA_PERSIST_DIR}")
        try:
            shutil.rmtree(CHROMA_PERSIST_DIR)
            print("✅ 旧数据库目录已彻底清除！")
        except Exception as e:
            print(f"❌ 清除旧数据库失败: {e} (请确保没有进程占用)")
            return 1

    # =========================================================
    # Step 2: 第一阶段双轨加载与切片 (SDK 专轨 + JAKA 多模态专轨)
    # =========================================================
    print(f"\n📄 数据目录: {PDF_DATA_DIR}")
    print(f"📄 JAKA Markdown 路径: {JAKA_MARKDOWN_PATH}")
    
    t0 = time.time()
    parents, children = load_all_documents_v4_dual(
        data_dir=PDF_DATA_DIR,
        jaka_md_path=JAKA_MARKDOWN_PATH,
    )
    print(f"✅ 第一阶段文档解析与切片完成! (耗时: {time.time() - t0:.2f}s)")

    if not parents or not children:
        print("❌ 切片为空，终止建库。")
        return 1

    # =========================================================
    # Step 3: 第二阶段向量化并写入 ChromaDB
    # =========================================================
    print(f"\n📦 开始计算 BGE Embedding 并灌入 ChromaDB...")
    t1 = time.time()
    parent_vs, child_vs = create_dual_collections(
        parents, children, persist_dir=CHROMA_PERSIST_DIR,
    )
    print(f"✅ ChromaDB 双 Collection 写入完成! (耗时: {time.time() - t1:.2f}s)")

    # =========================================================
    # Step 4: 统计质检报告
    # =========================================================
    api_atomic = sum(1 for d in children if d.metadata.get("api_atomic"))
    with_funcs = sum(1 for d in children if d.metadata.get("function_names"))
    vlm_injected = sum(1 for d in children if d.metadata.get("has_multimodal_data"))

    parent_by_product = Counter(d.metadata.get("product_id", "?") for d in parents)
    child_by_product = Counter(d.metadata.get("product_id", "?") for d in children)

    print(f"\n{'='*75}")
    print(f"🏆 向量库全量重建圆满完成! (总耗时: {time.time() - total_start_time:.2f}s)")
    print(f"{'='*75}")
    print(f"  Parent Collection 宏观父切片: {len(parents)} 个")
    for pid, cnt in sorted(parent_by_product.items()):
        print(f"    ├─ {pid}: {cnt} 个")
    print(f"  Child Collection 检索子切片:  {len(children)} 个")
    for pid, cnt in sorted(child_by_product.items()):
        print(f"    ├─ {pid}: {cnt} 个")
    print(f"  -------------------------------------------------------------")
    print(f"  ✨ JAKA 多模态参数注入切片: {vlm_injected} 个")
    print(f"  ⚙️ SDK API 原子切片数:      {api_atomic} 个 (含函数名: {with_funcs})")
    print(f"  📁 数据库持久化路径:        {CHROMA_PERSIST_DIR}")
    print(f"{'='*75}\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())