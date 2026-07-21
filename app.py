"""
=============================================================================
NewsPage — RAG 知识库对话系统 FastAPI 主入口
=============================================================================

启动方式：
  # 开发模式（热重载）
  uvicorn app:app --reload --host 0.0.0.0 --port 8000

  # 生产模式
  python app.py

API 路由一览：
  GET  /                → 渲染 NewsPage 主页面
  POST /api/chat        → RAG 对话（支持流式 SSE）
  POST /api/upload      → 上传 PDF 并重建向量库
  GET  /api/status      → 获取向量库状态
=============================================================================
"""

import json
import logging
import os
import shutil
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import (
    PDF_DATA_DIR,
    CHROMA_PERSIST_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVAL_K,
    HOST,
    PORT,
    MAX_UPLOAD_SIZE,
)
from src.pdf_loader import load_pdfs_from_directory
from src.vector_store import (
    create_vector_store,
    load_vector_store,
    search_similar,
    get_vector_store_info,
)
from src.rag_chain import rag_chat, rag_chat_stream, LLMServiceError

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app")

# ============================================================
# FastAPI 应用初始化
# ============================================================
app = FastAPI(
    title="NewsPage",
    description="湖南比邻星科技 — 官方开发与使用文档智能问答系统",
    version="1.0.0",
)

# 挂载静态文件目录（CSS / JS）
app.mount("/static", StaticFiles(directory="static"), name="static")

# 模板引擎配置
templates = Jinja2Templates(directory="templates")

# ============================================================
# 全局状态：向量库实例
# ============================================================
# 应用启动时加载已有的向量库，如果没有则为 None
vector_store = None


@app.on_event("startup")
async def startup_event():
    """应用启动时：加载已有向量库，并进行配置校验"""
    global vector_store
    logger.info("🚀 NewsPage — 湖南比邻星科技文档智能问答系统 正在启动...")

    # ---- 配置校验：DeepSeek API Key 降级通道可用性 ----
    from src.config import DEEPSEEK_API_KEY, BASE_URL
    if DEEPSEEK_API_KEY == "sk-your-deepseek-key-here":
        logger.warning(
            "⚠️  DEEPSEEK_API_KEY 仍为默认占位符 'sk-your-deepseek-key-here'，"
            "DeepSeek 降级通道（第 2 层容灾）将不可用！"
        )
        logger.warning(
            "   请设置环境变量: export DEEPSEEK_API_KEY=<your-deepseek-key>"
        )
        logger.warning(
            "   获取 Key: https://platform.deepseek.com/api_keys"
        )
    else:
        logger.info("✅ DEEPSEEK_API_KEY 已配置，第 2 层 DeepSeek 降级通道可用")

    if BASE_URL == "http://localhost:8000/v1":
        logger.info("📋 当前主 LLM 通道: 本地 vLLM (http://localhost:8000/v1)")
    else:
        logger.info(f"📋 当前主 LLM 通道: {BASE_URL}")

    # ---- 加载向量库 ----
    vector_store = load_vector_store(CHROMA_PERSIST_DIR)
    if vector_store:
        info = get_vector_store_info(vector_store)
        logger.info(f"📚 已加载向量库：{info['document_count']} 个文档片段")
    else:
        logger.info("📭 向量库为空，请通过 WebUI 上传 PDF 文件")


# ============================================================
# 页面路由
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    渲染 NewsPage 主页面（湖南比邻星科技文档智能问答系统）。

    使用 Jinja2 模板引擎将 index.html 渲染为完整 HTML 页面。
    页面标题在 <title> 和 header 中均设为 "NewsPage"。
    """
    return templates.TemplateResponse(request, "index.html", {"request": request})


# ============================================================
# 对话 API
# ============================================================

@app.post("/api/chat")
async def chat(
    query: str = Form(..., description="用户问题"),
    history: Optional[str] = Form(None, description="JSON 格式的对话历史"),
    stream: bool = Form(True, description="是否使用流式输出"),
):
    """
    RAG 对话接口 — POST /api/chat

    【请求参数（表单格式）】
    - query (必填): 用户输入的问题
    - history (可选): JSON 字符串，格式 [{"role":"user","content":"..."}, ...]
    - stream (可选): 是否流式输出，默认 True

    【响应格式】
    - 流式 (stream=true): SSE (Server-Sent Events) 事件流
        格式: data: {"delta": "文本增量"}\n\n
        结束: data: {"sources": [...], "done": true}\n\n
    - 非流式 (stream=false): JSON
        格式: {"answer": "完整回答", "sources": [...], "model": "..."}
    """
    global vector_store

    if vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="向量库尚未初始化，请先上传 PDF 文件。"
        )

    # 解析历史对话
    chat_history = None
    if history:
        try:
            chat_history = json.loads(history)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="history 参数 JSON 格式无效")

    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")

    # ---- 流式 SSE 响应 ----
    if stream:
        async def generate_sse():
            """生成 SSE 事件流"""
            try:
                # rag_chat_stream 返回一个生成器，逐 token 产出
                for token in rag_chat_stream(
                    vector_store, query, chat_history, k=RETRIEVAL_K
                ):
                    # SSE 格式: "data: <json>\n\n"
                    yield f"data: {json.dumps({'delta': token}, ensure_ascii=False)}\n\n"

                # 流式结束后发送完成信号
                yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"

            except Exception as e:
                logger.error(f"流式对话错误: {e}")
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            generate_sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲（如果有反向代理）
            },
        )

    # ---- 非流式 JSON 响应 ----
    else:
        try:
            result = rag_chat(vector_store, query, chat_history, k=RETRIEVAL_K)
            return JSONResponse(content=result)
        except LLMServiceError as e:
            # 第 4 层兜底：返回 503 + 结构化中文错误
            logger.error(f"LLM 服务不可用（四层容灾已耗尽）: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "error": str(e),
                    "error_type": "llm_unavailable",
                    "message": "所有大模型通道（本地 vLLM / 云端 API）当前均不可用，"
                               "纯文档检索模式也无法完成。请检查服务状态后重试。",
                },
            )
        except Exception as e:
            logger.error(f"对话错误: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# PDF 上传 API
# ============================================================

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(..., description="PDF 文件")):
    """
    PDF 上传接口 — POST /api/upload

    【处理流程】
    1. 校验文件类型（只允许 .pdf）
    2. 校验文件大小（默认上限 50MB）
    3. 保存文件到 data/ 目录
    4. 重新扫描 data/ 下所有 PDF 并重建向量库
    5. 返回重建结果

    【注意】
    - 每次上传都会触发全量重建（简单可靠，适合文档量不大的场景）
    - 如需增量更新，可后续优化为仅索引新增文档
    """
    global vector_store

    # ---- 校验 ----
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件格式")

    # 读取文件内容并检查大小
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件太大，最大允许 {MAX_UPLOAD_SIZE // (1024*1024)}MB"
        )

    # ---- 保存文件 ----
    os.makedirs(PDF_DATA_DIR, exist_ok=True)
    save_path = os.path.join(PDF_DATA_DIR, file.filename)

    with open(save_path, "wb") as f:
        f.write(content)

    logger.info(f"📄 已保存 PDF: {save_path} ({len(content)} 字节)")

    # ---- 重建向量库 ----
    try:
        # 如果存在旧向量库，先清空（ChromaDB 的 from_documents 会自动覆盖）
        documents = load_pdfs_from_directory(PDF_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP)

        if not documents:
            return JSONResponse({
                "success": True,
                "message": f"文件 {file.filename} 已保存，但未提取到有效文本。",
                "document_count": 0,
            })

        vector_store = create_vector_store(documents, CHROMA_PERSIST_DIR)
        info = get_vector_store_info(vector_store)

        return JSONResponse({
            "success": True,
            "message": f"文件 {file.filename} 已上传，向量库已重建",
            "document_count": info["document_count"],
            "file_name": file.filename,
        })

    except Exception as e:
        logger.error(f"向量库重建失败: {e}")
        raise HTTPException(status_code=500, detail=f"向量库重建失败: {e}")


# ============================================================
# 状态查询 API
# ============================================================

@app.get("/api/status")
async def status():
    """
    向量库状态查询 — GET /api/status

    返回向量库的基本信息，供前端展示。
    """
    global vector_store
    info = get_vector_store_info(vector_store)
    return JSONResponse(content={
        "ready": vector_store is not None and info["document_count"] > 0,
        "document_count": info["document_count"],
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
        reload=True,  # 开发模式：代码变更时自动重启
    )
