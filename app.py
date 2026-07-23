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
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVAL_K,
    HOST,
    PORT,
    MAX_UPLOAD_SIZE,
)
from src.pdf_loader import load_pdfs_from_directory
from src.multimodal_loader import load_enhanced_documents  # 🔴 阶段二：多模态解析
from src.vector_store import (
    create_vector_store,
    load_vector_store,
    get_vector_store_info,
    clear_vector_store,
    get_registered_products,
    cleanup_vector_store,
)
from src.rag_chain import LLMServiceError, shutdown_clients

# 🔴 第一阶段架构升级：LangGraph 状态图引擎（平滑切换）
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

# 查询长度上限（字符）
MAX_QUERY_LENGTH = 2000
# 对话历史最大条数（防 JSON 深度炸弹）
MAX_HISTORY_ITEMS = 100
# SSE 队列最大容量（防内存耗尽）
SSE_QUEUE_MAXSIZE = 256
# 允许的聊天角色
ALLOWED_CHAT_ROLES = {"user", "assistant"}

# null 字节和控制字符的正则（禁止出现在查询和文件名中）
_NULL_OR_CONTROL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')


def sanitize_query(query: str) -> str:
    """
    清洗用户查询字符串。

    处理：
      - 删除 null 字节（\x00）→ 防止 SQLite/chroma 截断
      - 删除控制字符（\x01-\x1f 除 \t \n）→ 防止日志/终端注入
      - 规范化换行：\r\n → \n
      - 去首尾空白
    """
    query = _NULL_OR_CONTROL_RE.sub('', query)
    query = query.replace('\r\n', '\n').replace('\r', '\n')
    return query.strip()


def sanitize_filename(filename: str) -> str:
    """
    清洗上传文件名 — 防路径遍历 + 防 null 字节注入。

    处理：
      1. os.path.basename() → 去除 ../../ 等路径遍历
      2. 删除 null 字节和控制字符
      3. 如果清洗后为空，返回安全的默认名称
    """
    filename = os.path.basename(filename)
    filename = _NULL_OR_CONTROL_RE.sub('', filename)
    filename = filename.strip()
    if not filename:
        filename = "uploaded_document.pdf"
    return filename


def validate_chat_history(history: list) -> list:
    """
    校验并清洗对话历史。

    校验规则：
      - 必须是 list 类型
      - 最多允许 MAX_HISTORY_ITEMS 条记录
      - 每条记录中的 role 必须是 "user" 或 "assistant"
      - 每条 content 不能为空
      - 每条 content 长度上限 4000 字符
    """
    if not isinstance(history, list):
        raise HTTPException(status_code=400, detail="history 必须是 JSON 数组")

    if len(history) > MAX_HISTORY_ITEMS:
        # 截断 + 警告（而非直接拒绝）
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

    # ---- 配置校验：智谱 GLM-4.7-Flash API Key 降级通道可用性 ----
    from src.config import DEEPSEEK_API_KEY, BASE_URL
    if DEEPSEEK_API_KEY == "1fe4c37fd3264ffa9f535fec9d0fc96b.UtiuwWTVuFofYHnB":
        logger.info(
            "✅ 智谱 GLM-4.7-Flash API Key 已使用默认值，"
            "第 2 层智谱降级通道可用 (glm-4.7-flash)"
        )
        logger.info(
            "   如需自定义 Key，请设置环境变量: export ZHIPU_API_KEY=<your-key>"
        )
    else:
        logger.info("✅ ZHIPU_API_KEY 已从环境变量加载，第 2 层智谱降级通道可用")

    if "localhost" in BASE_URL or "127.0.0.1" in BASE_URL:
        logger.info(f"📋 当前主 LLM 通道: 本地 vLLM ({BASE_URL})")
    else:
        logger.info(f"📋 当前主 LLM 通道: 云端 API ({BASE_URL})")

    # ---- 加载向量库 ----
    vector_store = load_vector_store(CHROMA_PERSIST_DIR)
    if vector_store:
        info = get_vector_store_info(vector_store)
        logger.info(f"📚 已加载向量库：{info['document_count']} 个文档片段")

        # 🔴 重建 BM25 索引（内存索引，重启后需从 ChromaDB 恢复）
        from src.vector_store import build_bm25_from_chromadb
        build_bm25_from_chromadb(vector_store)

        # 🔴 LangGraph 引擎注入向量库实例
        set_graph_vector_store(vector_store)
        logger.info("📐 LangGraph 状态图引擎已就绪")
    else:
        logger.info("📭 向量库为空，请通过 WebUI 上传 PDF 文件")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时：释放 LLM 客户端连接池、BM25 索引、嵌入模型等资源"""
    logger.info("🛑 NewsPage 正在关闭...")
    try:
        shutdown_clients()
        logger.info("✅ LLM 客户端连接池已释放")
    except Exception as e:
        logger.warning(f"关闭 LLM 客户端时出错: {e}")
    try:
        cleanup_vector_store()
        logger.info("✅ 向量库资源已释放（嵌入函数 + BM25 索引）")
    except Exception as e:
        logger.warning(f"释放向量库资源时出错: {e}")


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
    product_id: Optional[str] = Form(None, description="产品标识（如 OpenR6 / OpenC3），可选，不传则自动识别"),
):
    """
    RAG 对话接口 — POST /api/chat

    【安全措施】
      - query 清洗：删除 null 字节、控制字符
      - history 校验：仅允许 user/assistant 角色、截断超长内容
      - 长度上限：query ≤ 2000 字符

    【产品路由 — 新增】
      - product_id 参数：前端可通过下拉框强指定产品范围
      - 若未提供 product_id，后端自动运行 Product Router 识别
      - 若无法识别产品，返回主动澄清反问（needs_clarification=True）

    【请求参数（表单格式）】
    - query (必填): 用户输入的问题
    - history (可选): JSON 字符串，格式 [{"role":"user","content":"..."}, ...]
    - stream (可选): 是否流式输出，默认 True
    - product_id (可选): 产品标识，如 "OpenR6" 或 "OpenC3"

    【响应格式】
    - 流式 (stream=true): SSE (Server-Sent Events) 事件流
        格式: data: {"delta": "文本增量"}\n\n
        结束: data: {"sources": [...], "done": true}\n\n
        澄清: data: {"delta": "请问您询问的是..."}\n\n (needs_clarification=true)
    - 非流式 (stream=false): JSON
        格式: {"answer": "完整回答", "sources": [...], "model": "...",
                "needs_clarification": true/false}
    """
    global vector_store

    if vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="向量库尚未初始化，请先上传 PDF 文件。"
        )

    # ---- 查询清洗 ----
    query = sanitize_query(query)
    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")

    if len(query) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"查询内容过长，请控制在 {MAX_QUERY_LENGTH} 字符以内"
        )

    # ---- product_id 清洗 ----
    if product_id is not None:
        product_id = sanitize_query(product_id)
        if not product_id:
            product_id = None

    # ---- 历史对话校验 ----
    chat_history = None
    if history:
        try:
            raw_history = json.loads(history)
            chat_history = validate_chat_history(raw_history)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="history 参数 JSON 格式无效")
        except HTTPException:
            raise  # 透传 validate_chat_history 中的 HTTPException
        except Exception as e:
            logger.warning(f"history 校验异常: {e}")
            raise HTTPException(status_code=400, detail=f"history 参数无效: {e}")

    # ---- 流式 SSE 响应 ----
    if stream:
        async def generate_sse():
            """
            异步 SSE 事件生成器 — 防泄露增强版。

            安全特性：
              - bounded queue (maxsize=SSE_QUEUE_MAXSIZE)：防止内存耗尽
              - asyncio.CancelledError 捕获：客户端断开时停止消费
              - 线程池 Future 追踪：可取消阻塞调用
            """
            loop = asyncio.get_event_loop()
            queue: asyncio.Queue = asyncio.Queue(maxsize=SSE_QUEUE_MAXSIZE)
            cancelled = False

            def _run_blocking_stream():
                """在线程池中运行 LangGraph 引擎的流式生成器"""
                try:
                    for token in run_graph_stream(
                        query, chat_history, product_id=product_id,
                    ):
                        if cancelled:
                            # 客户端已断开 → 不再往队列投递，退出生成循环
                            break
                        loop.call_soon_threadsafe(queue.put_nowait, ("delta", token))
                    if not cancelled:
                        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
                except Exception as exc:
                    logger.error(f"流式对话错误: {exc}")
                    if not cancelled:
                        # 🔴 发送用户友好的错误提示，而非原始 Python 异常堆栈
                        friendly_error = "抱歉，系统处理您的请求时遇到了问题，请稍后重试。"
                        loop.call_soon_threadsafe(queue.put_nowait, ("error", friendly_error))

            # 将阻塞调用卸载到默认线程池
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
                # 客户端断开连接（关闭浏览器 / 网络中断）
                # 设置取消标志 → 线程池中的生成器在下一个 token 时退出
                cancelled = True
                logger.info("🔌 SSE 连接已断开（客户端取消），生成器将退出")
                # 不再 yield — 优雅退出

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
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件格式")

    # 🔴 文件名清洗：防路径遍历 (../../etc/passwd) + 防 null 字节注入
    safe_filename = sanitize_filename(file.filename)
    if not safe_filename.lower().endswith(".pdf"):
        safe_filename += ".pdf"

    # 读取文件内容并检查大小
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件太大，最大允许 {MAX_UPLOAD_SIZE // (1024*1024)}MB"
        )

    # ---- 保存文件 ----
    os.makedirs(PDF_DATA_DIR, exist_ok=True)
    save_path = os.path.join(PDF_DATA_DIR, safe_filename)

    with open(save_path, "wb") as f:
        f.write(content)

    logger.info(f"📄 已保存 PDF: {save_path} ({len(content)} 字节)")

    # ---- 重建向量库（先清空旧数据，再入库新产品切片） ----
    try:
        # 🔴 清空旧向量库：确保不同产品的旧切片不会与新切片混合
        clear_vector_store(CHROMA_PERSIST_DIR)
        logger.info("🧹 旧向量库已清空，开始重新索引...")

        # 🔴 阶段二：智能加载 — 有表格/图片时用增强解析，纯文本用标准解析
        from src.multimodal_loader import _extract_tables_from_page, _extract_images_info
        has_rich_content = False
        for f in os.listdir(PDF_DATA_DIR):
            if not f.lower().endswith('.pdf'):
                continue
            fp = os.path.join(PDF_DATA_DIR, f)
            try:
                imgs = _extract_images_info(fp)
                if imgs:
                    has_rich_content = True
                    break
            except Exception:
                pass
        if has_rich_content:
            logger.info("📊 检测到含表格/图片的复杂文档，启用多模态解析")
            documents = load_enhanced_documents(PDF_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP)
        else:
            documents = load_pdfs_from_directory(PDF_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP)

        if not documents:
            return JSONResponse({
                "success": True,
                "message": f"文件 {safe_filename} 已保存，但未提取到有效文本。",
                "document_count": 0,
            })

        vector_store = create_vector_store(documents, CHROMA_PERSIST_DIR)
        info = get_vector_store_info(vector_store)

        # 🔴 重建后重新注入 Graph 引擎
        set_graph_vector_store(vector_store)
        logger.info("📐 LangGraph 引擎已更新（向量库重建后）")

        # 统计各产品的切片数量
        product_counts = {}
        for doc in documents:
            pid = doc.metadata.get("product_id", "unknown")
            product_counts[pid] = product_counts.get(pid, 0) + 1

        return JSONResponse({
            "success": True,
            "message": f"文件 {safe_filename} 已上传，向量库已重建",
            "document_count": info["document_count"],
            "file_name": safe_filename,
            "product_id": documents[0].metadata.get("product_id", "unknown") if documents else "unknown",
            "product_distribution": product_counts,
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


@app.get("/api/products")
async def list_products():
    """
    获取已注册产品列表 — GET /api/products

    返回当前向量库中已入库的产品 ID 列表，
    供前端渲染产品选择下拉框。

    若向量库为空，返回空列表。
    """
    products = get_registered_products()
    return JSONResponse(content={
        "products": products,
        "count": len(products),
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
