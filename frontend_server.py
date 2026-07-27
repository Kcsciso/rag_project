"""
=============================================================================
比邻星 (ProximaRAG) 前端 UI 服务 — 端口 8501（HTML 模板渲染 + API 反向代理）
=============================================================================
"""
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

API_BACKEND = "http://localhost:7860"

app = FastAPI(title="比邻星 (ProximaRAG) Frontend")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.api_route("/api/{rest_of_path:path}", methods=["GET", "POST"])
async def proxy_api(request: Request):
    """反向代理 /api/* → 后端 7860"""
    path = f"/api/{request.path_params['rest_of_path']}"
    async with httpx.AsyncClient(timeout=httpx.Timeout(90.0)) as client:
        if request.method == "GET":
            resp = await client.get(f"{API_BACKEND}{path}", params=str(request.url.query))
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
        else:
            form = await request.form()
            resp = await client.post(f"{API_BACKEND}{path}", data=dict(form), timeout=90.0)
            if "text/event-stream" in resp.headers.get("content-type", ""):
                return StreamingResponse(
                    resp.aiter_bytes(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )
            return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)
