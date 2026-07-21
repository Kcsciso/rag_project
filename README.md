# 📰 NewsPage — 湖南比邻星科技文档智能问答系统

基于 **RAG（Retrieval-Augmented Generation）** 架构的官方技术文档与使用手册智能问答系统。专为**湖南比邻星科技有限公司**的开发者和用户打造，采用双 A100 GPU 算力底座，底层搭载 **vLLM + 开源大模型**实现完全私有化、低延迟的本地推理。

---

## 🚀 核心特性

### 🤖 智能推理引擎
- **四层金字塔容灾**：本地 vLLM（Layer 1）→ 智谱 GLM-4.7-Flash 云端 API（Layer 2）→ 智能结构化纯检索直出（Layer 3）→ 优雅错误提示（Layer 4），极端故障下仍可服务。
- **显卡智能自适应部署**：`start_services.sh` 通过 `nvidia-smi` 实时扫描所有 GPU 空闲显存，自动绑定剩余空间最大的 GPU（`CUDA_VISIBLE_DEVICES`），避免硬编码导致的 OOM 崩溃。
- **毫秒级流式秒回**：FastAPI SSE 异步非阻塞线程池隔离 + 前端 50ms 节流渲染（Throttle），LLM 读取超时激进缩短至 12s（最坏等待降 60%）。

### 📄 比邻星文档深度解析
- 多份 PDF 技术文档的批量加载，递归字符级文本分块（`chunk_size=600 / chunk_overlap=100`），API 规范切片不失真。
- Layer 3 降级时自动进行**行级归一化去重**（`_normalize_code_line` + 全局 `_global_seen_lines`），彻底消除 chunk_overlap 导致的代码块重复输出。

### 🔒 企业级安全与稳定性
- **全栈输入防御**：防路径遍历（`sanitize_filename`）、Null 字节与控制字符清洗（`sanitize_query`）、Prompt 注入过滤（`_contains_injection_pattern`）、历史消息角色白名单（`validate_chat_history`）。
- **滑动窗口记忆**：多轮对话最多保留 3 轮历史，防止上下文超出 4096 Token 限制。
- **全链路异常自动降级**：OOM、超时、限流等异常自动跌落至纯检索直出，覆盖 8 种故障场景。
- **SSE 断连优雅清理**：客户端断开时线程池生成器自动退出，`asyncio.Queue` 限界防内存耗尽（`maxsize=256`）。
- **资源泄露防范**：`shutdown_clients()` 释放 LLM 连接池 + `cleanup_vector_store()` 释放嵌入模型显存，FastAPI `shutdown` 事件自动触发。

### 🎨 现代化 Web 体验
- **NewsPage** 科技蓝深色主题，双栏布局（对话 + 上传/状态面板）。
- SSE 流式打字机效果 + Markdown 实时渲染 + `highlight.js` 代码高亮。
- PDF 拖拽上传、一键重建知识库、实时状态指示器。

---

## 📁 项目目录结构

```text
rag_project/
├── src/
│   ├── config.py              # 全局配置中心 + GPU 智能探测 API
│   ├── pdf_loader.py          # PDF 解析与递归字符级文本分块
│   ├── vector_store.py        # ChromaDB 向量库（HF→ONNX 双轨嵌入）
│   └── rag_chain.py           # RAG 四层容灾管线 + 安全防御 + 资源清理
├── templates/
│   └── index.html             # NewsPage 聊天与文档交互主页面
├── static/
│   ├── style.css              # 科技蓝深色主题样式
│   └── app.js                 # SSE 流式通信 + 50ms 节流渲染
├── data/                      # 用户上传的 PDF 文档目录
├── vector_db/                 # ChromaDB 向量数据持久化目录
├── app.py                     # FastAPI 异步应用入口（含安全中间件）
├── tunnel.py                  # ngrok 公网穿透脚本
├── check_status.py            # 统一服务健康检查（GPU 实时监测）
├── start_services.sh          # 一键自适应启动脚本
├── test_robot_rag.py          # 核心 RAG 功能自动化回归测试
├── test_stability.py          # 多轮对话 + 并发 + 异常降级压力测试
├── dev_log.md                 # 完整开发与迭代演进日志
├── CLAUDE.md                  # AI 协同开发规范与系统红线
└── README.md                  # 本文件
```

---

## ⚙️ 系统环境与约束

| 项目 | 说明 |
|------|------|
| **硬件底座** | 2 × NVIDIA A100-PCIE-40GB（CUDA 12.4），支持多卡隔离 |
| **环境管理器** | Conda（`rag_agent`，Python 3.10） |
| **推理引擎** | vLLM 0.16.0（OpenAI 兼容 API，端口 **8001**） |
| **默认模型** | `Qwen/Qwen2.5-1.5B-Instruct`（约 3.7 GB 显存，GPU 自适应部署） |
| **云端降级** | 智谱 GLM-4.7-Flash（免费模型，`open.bigmodel.cn`） |
| **嵌入模型** | `all-MiniLM-L6-v2`（384 维）→ ONNX 自动回退 |
| **Web 框架** | FastAPI + Jinja2（端口 **8000**） |

**🔴 核心锁定依赖（严禁升级）**：
- `torch==2.6.0+cu124` / `torchvision==0.21.0+cu124` / `torchaudio==2.6.0+cu124`
- `vllm==0.16.0`（`--no-deps` 隔离安装）
- `sentence-transformers==2.7.0`

---

## 🚀 部署与启动指南

### 1. 准备文档

将湖南比邻星科技有限公司的开发文档、API 规范或产品使用手册（PDF 格式）放入 **`data/`** 目录。

### 2. 一键启动（推荐）

```bash
chmod +x start_services.sh

# 完整启动（智能 GPU 检测 → vLLM → FastAPI）
./start_services.sh

# 仅启动 vLLM 推理服务
./start_services.sh --vllm-only

# 仅启动 FastAPI 后端（vLLM 已运行）
./start_services.sh --fastapi-only

# 手动指定 GPU（覆盖自动检测）
./start_services.sh --gpu 0
```

脚本自动完成：Conda 环境激活 → GPU 空闲显存扫描 → 端口占用检测 → vLLM 后台拉起 → 就绪轮询 → FastAPI 启动 → `Ctrl+C` 优雅退出。

### 3. 手动启动（终端 A + B）

**终端 A — vLLM 推理服务**：
```bash
conda activate rag_agent
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --served-model-name Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 4096 \
    --port 8001 \
    --gpu-memory-utilization 0.20 \
    --trust-remote-code \
    --enforce-eager
```

> 💡 使用 `start_services.sh` 可自动选择空闲最大的 GPU，无需手工指定 `CUDA_VISIBLE_DEVICES`。

**终端 B — NewsPage FastAPI 后端**：
```bash
conda activate rag_agent
export HF_ENDPOINT=https://hf-mirror.com
python app.py
```

服务启动后访问：**`http://localhost:8000`**（页面标题：**NewsPage**）  
API 文档：`http://localhost:8000/docs`

### 4. 系统健康检查

```bash
python check_status.py                # 一次性完整报告
python check_status.py --watch 10     # 每 10 秒自动刷新
```

报告覆盖：vLLM 在线状态 + 模型名 + 部署 GPU、NewsPage 后端状态 + 向量库文档数、GPU 0/1 实时显存/温度/功率、四层容灾可用性。

### 5. 环境变量覆盖

```bash
# 切回本地 vLLM（默认）
export LLM_BASE_URL="http://localhost:8001/v1"
export LLM_MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"

# 主通道直连智谱云端 API
export LLM_BASE_URL="https://open.bigmodel.cn/api/paas/v4"
export LLM_API_KEY="<your-zhipu-key>"
export LLM_MODEL_NAME="glm-4.7-flash"

# 手动指定 vLLM GPU
export VLLM_GPU_ID=0
```

### 6. 公网隧道（可选）

```bash
conda run -n rag_agent python tunnel.py --token <YOUR_NGROK_AUTHTOKEN>
```

---

## 🛠️ 运维工具清单

| 文件 | 用途 |
|------|------|
| `check_status.py` | 统一服务健康检查 — vLLM、FastAPI、GPU 实时显存、四层容灾可用性 |
| `start_services.sh` | 一键自适应启动 — GPU 智能选择、端口检测、vLLM 后台拉起、优雅退出 |
| `test_robot_rag.py` | 核心 RAG 功能回归测试（4 题 × 流式/非流式双模式） |
| `test_stability.py` | 稳定性压力测试（多轮对话、并发保护、7 种异常降级场景） |
| `tunnel.py` | ngrok 公网穿透，支持 authtoken 认证 |

---

## 📡 API 接口文档

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 渲染 **NewsPage** 主页面 |
| `POST` | `/api/chat` | RAG 对话（支持 SSE 流式输出）。参数：`query`（必填）、`history`（可选 JSON）、`stream`（默认 true） |
| `POST` | `/api/upload` | 上传 PDF 并自动重建向量库 |
| `GET` | `/api/status` | 返回向量库就绪状态与已索引文档片段数 |

---

## 📝 开发与排错日志

有关环境排查、兼容补丁、四层容灾、GPU 自适应、安全加固等 19 个章节的详细开发记录与架构决策（ADR），请参阅 [dev_log.md](./dev_log.md)。
