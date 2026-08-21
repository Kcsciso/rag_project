#!/usr/bin/env python3
"""
=============================================================================
Stage 1 离线冒烟测试 — 数据摄入、双轨切片与多模态提纯 (2026-08-21)
=============================================================================

覆盖（全部离线，不依赖 GPU / vLLM / VLM 服务）:
  1. SDK 专轨:  PyMuPDF 版面排序提取 + 章节原子切片 (OpenC3=27 章 / OpenR6=30 章)
                + function_names / api_atomic 提取 + Ctypes 类型名黑名单
  2. JAKA 专轨: MinerU Markdown 解析 — HTML 表格转 Markdown 零残留
                + 章节 Parent 切片 + 软装箱 Child 切片容量
  3. KV 属性库: 人工校准键值 (6502/9600) 必在导出结果中
  4. 统一入口:  load_all_documents_v4_dual 组合正确 (双轨 + KV 导出)

运行: python tests/test_stage1.py
说明: JAKA 多模态 VLM 注入依赖 :8005 服务与持久化缓存，本测试仅验证结构层；
      若 data/jaka_manual_chunks.json 存在，会自动走秒级缓存路径。
=============================================================================
"""
import json
import logging
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from config import PDF_DATA_DIR, JAKA_MARKDOWN_PATH  # noqa: E402
from pdf_loader import (  # noqa: E402
    clean_html_tables,
    export_kv_attributes,
    load_all_documents_v4_dual,
    load_jaka_mineru_dual,
    load_single_sdk_pdf,
)

logging.disable(logging.CRITICAL)

_PASSED = 0
_FAILED = 0


def check(name, cond, detail=""):
    global _PASSED, _FAILED
    if cond:
        _PASSED += 1
        print(f"  ✅ {name}" + (f" — {detail}" if detail else ""))
    else:
        _FAILED += 1
        print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))


def test_sdk_track():
    print("\n【1】SDK 专轨 — fitz 章节原子切片")
    openc3 = "OpenC3六轴机械臂SDK说明文档_win.pdf"
    openr6 = "windows系统OpenR6_sdk使用文档.pdf"

    p3, c3 = load_single_sdk_pdf(os.path.join(PDF_DATA_DIR, openc3))
    p6, c6 = load_single_sdk_pdf(os.path.join(PDF_DATA_DIR, openr6))

    c3_chapters = [c for c in c3 if re.match(r"\d{1,2}\s*\.", c.metadata.get("section_title", ""))]
    c6_chapters = [c for c in c6 if re.match(r"\d{1,2}\s*\.", c.metadata.get("section_title", ""))]

    check("OpenC3 章节原子切片 = 27", len(c3_chapters) == 27, f"实测 {len(c3_chapters)}")
    check("OpenR6 章节原子切片 ≥ 30 (含 TOC 噪声块时 ≤32)", 30 <= len(c6_chapters) <= 32,
          f"实测 {len(c6_chapters)}")
    check("OpenC3 Parent 存在", len(p3) == 1, f"{len(p3)}")
    check("OpenR6 Parent 存在", len(p6) == 1, f"{len(p6)}")

    for name, children in [(openc3, c3), (openr6, c6)]:
        api = [c for c in children if c.metadata.get("api_atomic")]
        missing = [c.metadata.get("section_title") for c in api if not c.metadata.get("function_names")]
        polluted = [
            f for c in children
            for f in c.metadata.get("function_names", "").split(",") if f
            if f.startswith("c_") or f.lower() in ("restype", "argtypes", "pointer", "byref", "cast")
        ]
        check(f"{name[:12]} api_atomic 全带函数名", len(api) > 0 and not missing,
              f"{len(api)} 个原子块" + (f"，缺失: {missing}" if missing else ""))
        check(f"{name[:12]} Ctypes 类型名零污染", not polluted, f"{polluted}" if polluted else "0 条")

    sizes = [len(c.page_content) for c in c3_chapters + c6_chapters]
    check("SDK 切片 150~900 字符区间", all(150 <= s <= 900 for s in sizes),
          f"min={min(sizes)} max={max(sizes)}")


def test_jaka_track():
    print("\n【2】JAKA 专轨 — MinerU Markdown + 表格规整 + 软装箱")
    parents, children = load_jaka_mineru_dual(JAKA_MARKDOWN_PATH)

    check("JAKA Parent 章节 = 9 (5 章 + 4 附录)", len(parents) == 9, f"实测 {len(parents)}")
    check("JAKA Child 切片 ≥ 200", len(children) >= 200, f"实测 {len(children)}")

    h1 = [p.metadata.get("section_title", "") for p in parents]
    check("章节标题含 '第 1 章'", any("第 1 章" in t for t in h1), "")
    check("附录四 (Ethernet/IP) 在库", any("附录四" in t for t in h1), "")

    # HTML 残留与表格转换
    with open(JAKA_MARKDOWN_PATH, encoding="utf-8") as f:
        raw = f.read()
    cleaned = clean_html_tables(raw)
    residue = re.findall(r"</?(html|body|p|div|span|thead|tbody|tr|td|th|table)[^>]*>", cleaned, re.I)
    check("HTML 表格标签零残留", not residue, f"{residue}" if residue else "0 条")
    check("Markdown 表格行已生成", cleaned.count("\n| ") > 300, f"{cleaned.count(chr(10)+'| ')} 行")


def test_kv_store():
    print("\n【3】KV 属性库导出")
    _, children = load_jaka_mineru_dual(JAKA_MARKDOWN_PATH)
    with tempfile.TemporaryDirectory() as tmp:
        import pdf_loader as pl
        saved = pl._KV_STORE_FILE
        pl._KV_STORE_FILE = os.path.join(tmp, "attribute_kv.json")
        try:
            export_kv_attributes(children)
            with open(pl._KV_STORE_FILE, encoding="utf-8") as f:
                store = json.load(f)
        finally:
            pl._KV_STORE_FILE = saved

    jaka = store.get("JAKA", {})
    check("JAKA 人工校准值存在", jaka.get("Modbus TCP 端口号") == "6502"
          and jaka.get("Modbus RTU 默认波特率") == "9600", "")


def test_unified_entry():
    print("\n【4】统一入口 load_all_documents_v4_dual")
    # KV 导出重定向到临时目录，避免测试污染 kv_db/attribute_kv.json
    import pdf_loader as pl
    saved = pl._KV_STORE_FILE
    pl._KV_STORE_FILE = os.path.join(tempfile.mkdtemp(), "attribute_kv.json")
    try:
        parents, children = load_all_documents_v4_dual(
            data_dir=PDF_DATA_DIR, jaka_md_path=JAKA_MARKDOWN_PATH,
        )
    finally:
        pl._KV_STORE_FILE = saved
    check("统一入口产出非空", len(parents) > 0 and len(children) > 200,
          f"parents={len(parents)} children={len(children)}")
    pids = {c.metadata.get("product_id") for c in children}
    check("三产品线齐全 (JAKA/OpenC3/OpenR6)", {"JAKA", "OpenC3", "OpenR6"} <= pids, f"{sorted(pids)}")


if __name__ == "__main__":
    print("=" * 62)
    print("🧪 ProximaRAG Stage 1 离线冒烟测试")
    print("=" * 62)
    try:
        test_sdk_track()
    except Exception as e:
        print(f"  ❌ SDK 轨测试异常: {e}")
        _FAILED += 1
    try:
        test_jaka_track()
    except Exception as e:
        print(f"  ❌ JAKA 轨测试异常: {e}")
        _FAILED += 1
    try:
        test_kv_store()
    except Exception as e:
        print(f"  ❌ KV 测试异常: {e}")
        _FAILED += 1
    try:
        test_unified_entry()
    except Exception as e:
        print(f"  ❌ 统一入口测试异常: {e}")
        _FAILED += 1

    print("\n" + "=" * 62)
    print(f"🏁 结果: {_PASSED} 通过 / {_FAILED} 失败")
    print("=" * 62)
    sys.exit(1 if _FAILED else 0)
