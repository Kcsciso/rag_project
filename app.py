"""
=============================================================================
比邻星 (ProximaRAG) — RAG 知识库对话系统 FastAPI 主入口 (v32 定版)
=============================================================================

启动方式：
  # 开发模式（热重载）
  uvicorn app:app --reload --host 0.0.0.0 --port 8000

  # 生产模式
  python app.py

API 路由一览：
  GET  /                → 渲染 比邻星 主页面
  POST /api/chat        → RAG 对话（支持流式 SSE）
  POST /api/upload      → 上传 PDF 并增量更新向量库
  GET  /api/status      → 获取向量库状态
  GET  /api/products    → 获取已注册产品列表
  GET  /api/debug/*     → 检索沙盒与切片检查
=============================================================================
"""

import asyncio
import json
import logging
import os
import re
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import (
    PDF_DATA_DIR,
    CHROMA_PERSIST_DIR,
    CHILD_CHUNK_SIZE,
    PARENT_CHUNK_SIZE,
    RETRIEVAL_K,
    HOST,
    PORT,
    MAX_UPLOAD_SIZE,
)
from src.pdf_loader import load_all_documents_v4_dual
from src.vector_store import (
    create_vector_store,
    load_vector_store,
    get_vector_store_info,
    clear_vector_store,
    get_registered_products,
    cleanup_vector_store,
)
from src.rag_chain import LLMServiceError, shutdown_clients
from src.graph_rag import (
    run_graph,
    run_graph_stream,
    set_graph_vector_store,
)

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app")

# ============================================================
# 输入安全 — 查询与文件名清洗
# ============================================================

MAX_QUERY_LENGTH = 2000
MAX_HISTORY_ITEMS = 100
SSE_QUEUE_MAXSIZE = 256
ALLOWED_CHAT_ROLES = {"user", "assistant"}

_NULL_OR_CONTROL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')


def sanitize_query(query: str) -> str:
    """清洗用户查询字符串"""
    query = _NULL_OR_CONTROL_RE.sub('', query)
    query = query.replace('\r\n', '\n').replace('\r', '\n')
    return query.strip()


def sanitize_filename(filename: str) -> str:
    """清洗上传文件名 — 防路径遍历 + 防 null 字节注入"""
    filename = os.path.basename(filename)
    filename = _NULL_OR_CONTROL_RE.sub('', filename)
    filename = filename.strip()
    if not filename:
        filename = "uploaded_document.pdf"
    return filename


def validate_chat_history(history: list) -> list:
    """校验并清洗对话历史"""
    if not isinstance(history, list):
        raise HTTPException(status_code=400, detail="history 必须是 JSON 数组")

    if len(history) > MAX_HISTORY_ITEMS:
        logger.warning(
            f"对话历史过长 ({len(history)} 条)，截断至最近 {MAX_HISTORY_ITEMS} 条"
        )
        history = history[-MAX_HISTORY_ITEMS:]

    cleaned = []
    for i, item in enumerate(history):
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail=f"history[{i}] 必须是对象")
        role = item.get("role", "")
        content = item.get("content", "")
        if role not in ALLOWED_CHAT_ROLES:
            raise HTTPException(
                status_code=400,
                detail=f"history[{i}] 角色无效: '{role}'，仅允许 user/assistant"
            )
        if not content or not isinstance(content, str):
            raise HTTPException(status_code=400, detail=f"history[{i}] content 不能为空")
        if len(content) > 4000:
            content = content[:4000]
        cleaned.append({"role": role, "content": sanitize_query(content)})
    return cleaned


# ============================================================
# FastAPI 应用初始化
# ============================================================
app = FastAPI(
    title="比邻星 (ProximaRAG)",
    description="湖南比邻星科技 — 官方开发与使用文档智能问答系统",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 全局向量库实例
vector_store = None


@app.on_event("startup")
async def startup_event():
    """应用启动时：加载已有向量库，初始化 LangGraph 状态图引擎"""
    global vector_store
    logger.info("🚀 比邻星 (ProximaRAG) 后端服务正在启动...")

    # ---- 配置校验：降级通道与主通道状态 ----
    from src.config import DEEPSEEK_API_KEY, BASE_URL
    if "localhost" in BASE_URL or "127.0.0.1" in BASE_URL:
        logger.info(f"📋 当前主 LLM 通道: 本地 vLLM ({BASE_URL})")
    else:
        logger.info(f"📋 当前主 LLM 通道: 云端 API ({BASE_URL})")

    # ---- 加载向量库 ----
    vector_store = load_vector_store(CHROMA_PERSIST_DIR)
    if vector_store:
        info = get_vector_store_info(vector_store)
        logger.info(f"📚 已加载向量库：{info['document_count']} 个文档片段")

        # 重建 BM25 内存稀疏索引
        from src.vector_store import build_bm25_from_chromadb
        build_bm25_from_chromadb(vector_store)

        # 注入 LangGraph 引擎
        set_graph_vector_store(vector_store)
        logger.info("📐 LangGraph 状态图引擎已就绪")
    else:
        logger.info("📭 向量库为空，请先运行 rebuild_v4.py 进行初始化建库")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时：释放 LLM 连接池与向量库资源"""
    logger.info("🛑 比邻星 (ProximaRAG) 正在关闭...")
    try:
        shutdown_clients()
        logger.info("✅ LLM 客户端连接池已释放")
    except Exception as e:
        logger.warning(f"关闭 LLM 客户端时出错: {e}")
    try:
        cleanup_vector_store()
        logger.info("✅ 向量库资源已释放")
    except Exception as e:
        logger.warning(f"释放向量库资源时出错: {e}")


# ============================================================
# 页面路由
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """渲染比邻星对话主页面"""
    return templates.TemplateResponse(request, "index.html", {"request": request})


# ============================================================
# 对话 API
# ============================================================

@app.post("/api/chat")
async def chat(
    query: str = Form(..., description="用户问题"),
    history: Optional[str] = Form(None, description="JSON 格式的对话历史"),
    stream: bool = Form(True, description="是否使用流式输出"),
    product_id: Optional[str] = Form(None, description="产品标识（可选）"),
):
    """RAG 对话接口 — 支持 SSE 流式与完整 JSON 返回"""
    global vector_store

    if vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="向量库尚未初始化，请先构建向量库。"
        )

    query = sanitize_query(query)
    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")

    if len(query) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"查询内容过长，请控制在 {MAX_QUERY_LENGTH} 字符以内"
        )

    if product_id is not None:
        product_id = sanitize_query(product_id)
        if not product_id:
            product_id = None

    chat_history = None
    if history:
        try:
            raw_history = json.loads(history)
            chat_history = validate_chat_history(raw_history)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="history 参数 JSON 格式无效")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"history 参数无效: {e}")

    # ---- 流式 SSE 响应 ----
    if stream:
        async def generate_sse():
            loop = asyncio.get_event_loop()
            queue: asyncio.Queue = asyncio.Queue(maxsize=SSE_QUEUE_MAXSIZE)
            cancelled = False

            def _run_blocking_stream():
                try:
                    for token in run_graph_stream(
                        query, chat_history, product_id=product_id,
                    ):
                        if cancelled:
                            break
                        loop.call_soon_threadsafe(queue.put_nowait, ("delta", token))
                    if not cancelled:
                        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
                except Exception as exc:
                    logger.error(f"流式对话错误: {exc}")
                    if not cancelled:
                        friendly_error = "抱歉，系统处理您的请求时遇到了问题，请稍后重试。"
                        loop.call_soon_threadsafe(queue.put_nowait, ("error", friendly_error))

            loop.run_in_executor(None, _run_blocking_stream)

            try:
                while True:
                    msg_type, payload = await queue.get()
                    if msg_type == "delta":
                        yield f"data: {json.dumps({'delta': payload}, ensure_ascii=False)}\n\n"
                    elif msg_type == "done":
                        yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
                        break
                    elif msg_type == "error":
                        yield f"data: {json.dumps({'error': payload}, ensure_ascii=False)}\n\n"
                        break
            except asyncio.CancelledError:
                cancelled = True
                logger.info("🔌 SSE 连接断开，后台生成已中止")

        return StreamingResponse(
            generate_sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ---- 非流式 JSON 响应 ----
    else:
        try:
            result = run_graph(query, chat_history, product_id=product_id)
            return JSONResponse(content=result)
        except LLMServiceError as e:
            logger.error(f"LLM 服务不可用: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "error": str(e),
                    "error_type": "llm_unavailable",
                    "message": "大模型通道当前不可用，请检查本地 vLLM 服务后重试。",
                },
            )
        except Exception as e:
            logger.error(f"对话错误: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# PDF 上传与增量更新 API
# ============================================================

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(..., description="PDF 文件")):
    """PDF 上传接口 — 增量更新入库"""
    global vector_store

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件格式")

    safe_filename = sanitize_filename(file.filename)
    if not safe_filename.lower().endswith(".pdf"):
        safe_filename += ".pdf"

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件过大，上限为 {MAX_UPLOAD_SIZE // (1024*1024)}MB"
        )

    os.makedirs(PDF_DATA_DIR, exist_ok=True)
    save_path = os.path.join(PDF_DATA_DIR, safe_filename)

    with open(save_path, "wb") as f:
        f.write(content)

    logger.info(f"📄 已保存 PDF: {save_path} ({len(content)} 字节)")

    try:
        from src.vector_store import upsert_product_documents

        result = upsert_product_documents(
            save_path,
            product_id="",
            child_chunk_size=CHILD_CHUNK_SIZE,
            parent_chunk_size=PARENT_CHUNK_SIZE,
        )

        # 重新同步内存句柄与 Graph 引擎
        vector_store = load_vector_store(CHROMA_PERSIST_DIR)
        if vector_store:
            set_graph_vector_store(vector_store)

        return JSONResponse({
            "success": True,
            "message": (
                f"文件已增量处理: {result.get('status','?')}"
                if result.get("status") != "skipped"
                else f"文件已跳过 (MD5 相同): {safe_filename}"
            ),
            "file_name": safe_filename,
            "product_id": result.get("product_id", "unknown"),
            "parents": result.get("parents", 0),
            "children": result.get("children", 0),
            "status": result.get("status", "?"),
        })

    except Exception as e:
        logger.error(f"向量库增量更新失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新失败: {e}")


# ============================================================
# 状态查询 API
# ============================================================

@app.get("/api/status")
async def status():
    """向量库状态查询"""
    global vector_store
    info = get_vector_store_info(vector_store)
    return JSONResponse(content={
        "ready": vector_store is not None and info["document_count"] > 0,
        "document_count": info["document_count"],
    })


@app.get("/api/products")
async def list_products():
    """获取已注册产品列表"""
    products = get_registered_products()
    return JSONResponse(content={
        "products": products,
        "count": len(products),
    })


# ============================================================
# 调试端点 — Retrieval Debugger
# ============================================================

@app.get("/api/debug/inspect_chunks")
async def inspect_chunks(
    product_id: Optional[str] = None,
    keyword: Optional[str] = None,
    limit: int = 10,
):
    """切片检查器 — 获取磁盘上最新的 Collection"""
    import chromadb
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        coll = client.get_collection("rag_v4_child")
    except Exception:
        try:
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            coll = client.get_collection("rag_documents")
        except Exception as e:
            return JSONResponse({"error": f"Collection 访问失败: {e}"}, status_code=404)

    where_filter = {}
    if product_id:
        where_filter["product_id"] = product_id

    try:
        all_data = coll.get(
            where=where_filter if where_filter else None,
            include=["documents", "metadatas"],
        )
    except Exception as e:
        return JSONResponse({"error": f"查询切片失败: {e}"}, status_code=500)

    chunks = []
    for doc_id, doc_text, meta in zip(
        all_data["ids"], all_data["documents"], all_data["metadatas"]
    ):
        if keyword and keyword.lower() not in doc_text.lower():
            continue
        fn_str = meta.get("function_names", "")
        chunks.append({
            "chunk_id": doc_id,
            "product_id": meta.get("product_id", "?"),
            "section": meta.get("section_title", "") or meta.get("section_id", ""),
            "source": meta.get("source", "?"),
            "api_atomic": meta.get("api_atomic", False),
            "function_names": [f.strip() for f in fn_str.split(",") if f.strip()],
            "parent_id": meta.get("parent_id", ""),
            "text_preview": doc_text if limit <= 5 else doc_text[:500],
            "text_length": len(doc_text),
        })
        if len(chunks) >= limit:
            break

    return JSONResponse({
        "total_in_collection": coll.count(),
        "filtered_count": len(chunks),
        "filters": {"product_id": product_id, "keyword": keyword},
        "chunks": chunks,
    })


@app.post("/api/debug/retrieve")
async def debug_retrieve(
    query: str = Form(...),
    product_id: Optional[str] = Form(None),
    k: int = Form(8),
):
    """检索沙盒 — 仅输出检索管线中间结果，不调用大模型生成"""
    global vector_store
    if not vector_store:
        return JSONResponse({"error": "vector store not loaded"}, status_code=503)

    from src.vector_store import (
        search_similar_with_threshold, bm25_search,
        _match_function_names, _extract_query_code_entities,
    )
    from src.rag_chain import (
        _hybrid_retrieve, _score_chunk_for_query, _autocut_knee,
        _is_noise_chunk, _extract_query_code_entities as _eqc,
        _is_sdk_code_query, _match_function_names as _mfn,
    )

    fetch_k = k * 5
    try:
        vec_results = search_similar_with_threshold(
            vector_store, query, k=fetch_k, threshold=None, product_id=product_id,
        )
    except Exception:
        vec_results = []
    
    vec_scores = []
    for doc in vec_results:
        try:
            scored = vector_store.similarity_search_with_score(
                doc.page_content[:80], k=1,
                filter={"product_id": product_id} if product_id else None,
            )
            vec_scores.append(round(scored[0][1], 4) if scored else 0)
        except Exception:
            vec_scores.append(0)

    vector_top = [
        {"score": s, "text": d.page_content[:200], "source": d.metadata.get("source", "?"),
         "function_names": d.metadata.get("function_names", "")}
        for d, s in zip(vec_results[:k], vec_scores[:k])
    ]

    bm25_results = []
    if product_id:
        try:
            bm25_results = bm25_search(query, product_id, k=k)
        except Exception:
            pass
    bm25_top = [
        {"score": round(s, 4), "text": d.page_content[:200],
         "source": d.metadata.get("source", "?")}
        for d, s in bm25_results[:k]
    ]

    try:
        final_docs = _hybrid_retrieve(
            vector_store, query, k=k, product_id=product_id,
        )
    except Exception:
        final_docs = []
        
    final_context = [
        {"text": d.page_content[:250], "source": d.metadata.get("source", "?"),
         "function_names": d.metadata.get("function_names", ""),
         "api_atomic": d.metadata.get("api_atomic", False)}
        for d in final_docs[:k]
    ]

    code_entities = _extract_query_code_entities(query)
    is_sdk = _is_sdk_code_query(query)
    has_func = any(
        _match_function_names(d.metadata.get("function_names", ""), code_entities)
        for d in final_docs if hasattr(d, 'metadata')
    )
    force_no_code = is_sdk and not has_func and not any(
        d.page_content and ("点击" in d.page_content or "设置" in d.page_content)
        for d in final_docs if hasattr(d, 'page_content')
    )

    noise_count = sum(1 for d in vec_results if _is_noise_chunk(d.page_content))

    return JSONResponse({
        "query": query,
        "product_id": product_id,
        "pipeline": {
            "vector_recall": {"total_fetched": len(vec_results), "top_k": vector_top},
            "bm25_recall": {"total_fetched": len(bm25_results), "top_k": bm25_top},
            "rrf_final_context": final_context,
        },
        "guards": {
            "force_no_code": force_no_code,
            "is_sdk_query": is_sdk,
            "code_entities_in_query": code_entities,
            "function_names_matched_in_context": has_func,
        },
        "stats": {
            "candidate_pool": len(vec_results),
            "noise_filtered": noise_count,
            "final_kept": len(final_docs),
        },
    })


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=HOST,
        port=PORT,
        reload=False,
    )