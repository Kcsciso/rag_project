# 📰 NewsPage — 湖南比邻星科技文档智能问答系统

基于 **RAG（Retrieval-Augmented Generation）** 架构的官方技术文档与使用手册智能问答系统。专为**湖南比邻星科技有限公司**的开发者和用户打造，采用双 A100 GPU 算力底座，底层搭载 **vLLM + 开源大模型**实现完全私有化、低延迟的本地推理。

---

## 🚀 核心特性

### 🤖 智能推理引擎
- **四层金字塔容灾**：本地 vLLM → 智谱 GLM-4.7-Flash 云端 API → 智能结构化纯检索直出（行级归一化去重）→ 优雅错误提示，极端故障下仍可服务。
- **显卡智能自适应部署**：`start_services.sh` 通过 `nvidia-smi` 实时扫描所有 GPU 空闲显存，自动绑定剩余空间最大的 GPU，避免硬编码导致的 OOM 崩溃。`detect_best_gpu()` 函数的 stdout/stderr 已严格隔离，杜绝日志污染变量。
- **毫秒级流式秒回**：FastAPI SSE 异步非阻塞线程池隔离 + 前端 50ms 节流渲染，LLM 读取超时激进缩短至 12s。

### 🔍 智能检索优化
- **Query 预处理**（`_preprocess_query`）：多层迭代剥离口语化噪音（"那个啥"、"你给我整一个"、"呗"等 25+ 模式），提取核心检索实体。
- **混合检索**（`_hybrid_retrieve`）：向量召回 4 倍候选池 → 相似度阈值 0.78 过滤 → 43 个中文领域操作词 + 20 个 SDK 函数名三层加权重排序 → 返回精准 Top-K。
- **行级归一化去重**：Layer 3 降级时自动归一化代码行指纹，全局集合 `_global_seen_lines` 彻底消除 chunk_overlap 导致的代码块重复。

### 🔒 企业级安全与稳定性
- **全栈输入防御**：防路径遍历（`sanitize_filename`）、Null 字节与控制字符清洗（`sanitize_query`）、Prompt 注入过滤（`_contains_injection_pattern`）、历史消息角色白名单（`validate_chat_history`）。
- **滑动窗口记忆**：多轮对话最多保留 3 轮历史，防止上下文超出 4096 Token 限制。
- **全链路异常自动降级**：覆盖 9 种故障场景（含 SSE 客户端断开），OOM/超时/限流自动跌落至纯检索直出。
- **资源泄露防范**：`shutdown_clients()` 释放 LLM 连接池 + `cleanup_vector_store()` 释放嵌入模型显存，FastAPI `shutdown` 事件自动触发。

### 🎨 现代化 Web 体验
- **NewsPage** 科技蓝深色主题，双栏布局（对话 + 上传/状态面板）。
- SSE 流式打字机效果 + Markdown 实时渲染 + `highlight.js` 代码高亮。

---

## 📁 项目目录结构

```text
rag_project/
├── src/
│   ├── config.py              # 全局配置中心 + GPU 智能探测 API
│   ├── pdf_loader.py          # PDF 解析与递归字符级文本分块
│   ├── vector_store.py        # ChromaDB 向量库（HF→ONNX 双轨嵌入）
│   └── rag_chain.py           # RAG 四层容灾 + 混合检索 + 口语化预处理 + 安全防御
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
├── start_services.sh          # 一键自适应启动脚本（GPU 智能选择）
├── test_robot_rag.py          # 核心 RAG 功能自动化回归测试
├── test_stability.py          # 多轮对话 + 并发 + 异常降级压力测试
├── test_human_simulation.py   # 全场景人类模拟测试（14 用例 × 5 类别）
├── dev_log.md                 # 完整开发与迭代演进日志（20 章）
├── CLAUDE.md                  # AI 协同开发规范与系统红线
└── README.md                  # 本文件
```

---

## ⚙️ 系统环境与约束

| 项目 | 说明 |
|------|------|
| **硬件底座** | 2 × NVIDIA A100-PCIE-40GB（CUDA 12.4） |
| **环境管理器** | Conda（`rag_agent`，Python 3.10） |
| **推理引擎** | vLLM 0.16.0（OpenAI 兼容 API，端口 **8001**） |
| **默认模型** | `Qwen/Qwen2.5-1.5B-Instruct`（~3.7 GB，GPU 自适应部署） |
| **云端降级** | 智谱 GLM-4.7-Flash（免费模型，`open.bigmodel.cn`） |
| **嵌入模型** | `all-MiniLM-L6-v2`（384 维）→ ONNX 自动回退 |
| **相似度阈值** | 0.78（cosine distance，含 5% 混合检索放宽） |
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

**终端 B — NewsPage FastAPI 后端**：
```bash
conda activate rag_agent
export HF_ENDPOINT=https://hf-mirror.com
python app.py
```

访问：**`http://localhost:8000`**（页面标题：**NewsPage**）｜API 文档：`http://localhost:8000/docs`

### 4. 一键停止所有服务

```bash
# 快速停止所有 NewsPage 相关进程
pkill -f "app.py" 2>/dev/null
pkill -f "vllm.entrypoints" 2>/dev/null
# 或定义快捷别名:
alias stoprag='pkill -f "app.py"; pkill -f "vllm.entrypoints"; echo \"NewsPage 已停止\"'
```

### 5. 系统健康检查

```bash
python check_status.py                # 一次性完整报告
python check_status.py --watch 10     # 每 10 秒自动刷新
```

### 6. 环境变量覆盖

```bash
export LLM_BASE_URL="http://localhost:8001/v1"           # 本地 vLLM
export LLM_MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
export VLLM_GPU_ID=0                                      # 手动指定 GPU
```

---

## 🧪 自动化测试

| 脚本 | 覆盖范围 | 命令 |
|------|---------|------|
| `test_human_simulation.py` | 5 类 14 用例（口语噪音、错别字、多轮指代、长文本组合、边界攻击） | `python test_human_simulation.py` |
| `test_robot_rag.py` | 核心 RAG 功能回归（4 题 × 双模式） | `conda run -n rag_agent python test_robot_rag.py` |
| `test_stability.py` | 多轮对话 + 并发保护 + 7 种异常降级 | `conda run -n rag_agent python test_stability.py` |

---

## 📡 API 接口文档

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 渲染 **NewsPage** 主页面 |
| `POST` | `/api/chat` | RAG 对话（SSE 流式）。参数：`query`（必填）、`history`（可选 JSON）、`stream`（默认 true） |
| `POST` | `/api/upload` | 上传 PDF 并自动重建向量库 |
| `GET` | `/api/status` | 返回向量库就绪状态与已索引文档片段数 |

---

## 📝 开发与排错日志

有关环境排查、兼容补丁、四层容灾、GPU 自适应、安全加固、混合检索、人类模拟测试等 20 个章节的详细开发记录与架构决策（ADR），请参阅 [dev_log.md](./dev_log.md)。
