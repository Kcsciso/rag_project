# 🔴 系统红线与开发规则 (STRICT CONSTRAINTS)

## 1. 硬件与 GPU 管理（双 A100 分离策略）

- **算力底座**: 2 × NVIDIA A100-PCIE-40GB（CUDA 12.4）。
- **显卡分配规则**:
  - `CUDA_VISIBLE_DEVICES=1`（GPU 1，端口 **8001**）：专用于本地 `vllm` 大模型推理服务，当前加载 **Qwen2.5-1.5B-Instruct**（轻量模型，~3.7 GB 显存占用）。
  - `CUDA_VISIBLE_DEVICES=0`（GPU 0）：留给向量检索引擎（ChromaDB / PyTorch 嵌入计算）及其他后台任务。**注意**：GPU 0 为多人共享，空闲显存极有限（高峰期仅 ~16 MB）。
- **GPU 资源充裕时的升级路径**：若 GPU 0/1 空闲 ≥15 GB，可将 Layer 1 升级为 `Qwen/Qwen2.5-7B-Instruct`（通过环境变量 `LLM_MODEL_NAME` 切换，vLLM 启动时调整 `--gpu-memory-utilization 0.40`）。
- **核心操作**：启动任何推理或训练脚本前必须显式指定 `CUDA_VISIBLE_DEVICES`。

## 2. 核心依赖红线（严禁升级）

环境管理器为 Conda（`rag_agent`，Python 3.10）。以下基础依赖已被**严格锁定**，**绝不允许执行 `pip install --upgrade`**：
- `torch==2.6.0+cu124`
- `torchvision==0.21.0+cu124`
- `torchaudio==2.6.0+cu124`
- `vllm==0.16.0`（通过 `--no-deps` 隔离安装）
- `sentence-transformers==2.7.0`

允许新增的辅助包（需确认与锁定依赖无冲突）：
- `pypdf`（PDF 文本提取，纯 Python）
- `langchain-chroma`（LangChain × ChromaDB 集成）
- `rank-bm25`（混合检索中 BM25 关键词召回，计划引入）

## 3. RAG 架构与 AI 生态

- **可用框架**: LangChain、LangGraph、ChromaDB、faiss-gpu。
- **LLM 推理引擎**: 本地部署的 `vllm` OpenAI 兼容服务。
  - **主模型（Layer 1）**: `Qwen/Qwen2.5-1.5B-Instruct`，运行在 `http://localhost:8001/v1`，GPU 1，`--gpu-memory-utilization 0.20`，`--max-model-len 4096`，`--enforce-eager`。
  - **降级模型（Layer 2）**: `glm-4.7-flash`（智谱 GLM-4.7-Flash 免费模型），端点 `https://open.bigmodel.cn/api/paas/v4`，通过 `ZHIPU_API_KEY` 环境变量认证。
- **嵌入模型**: `all-MiniLM-L6-v2`（384 维，英文优化），可通过配置切换为 `BAAI/bge-small-zh-v1.5`（512 维，中文专优）。支持 HuggingFace → ONNX 自动回退。
- **UI 命名规范**: 前端界面或网页标题（HTML `<title>` 与 header）**必须**命名为 **NewsPage**。

## 4. 操作指令

- 在执行破坏性 Bash 命令或安装软件包之前，必须征得用户的明确授权。
- 所有重大架构调整、Ablation 实验及 Git 提交必须记录在 `dev_log.md` 中。
- 严禁删除项目根目录下的 `pyairports/` 目录（若存在）或 Conda 环境 `site-packages/pyairports/` 下的 Shim 适配层。

---

# 🚀 本地服务启动顺序（关键）

## 第一步：启动本地 vLLM 大模型推理服务（终端 A）

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

**参数说明**：
| 参数 | 值 | 理由 |
|------|-----|------|
| `CUDA_VISIBLE_DEVICES` | `1` | 隔离 GPU 1，避免与 GPU 0 上其他用户的进程冲突 |
| `--port` | **8001** | FastAPI 占用 8000，vLLM 使用独立端口 |
| `--gpu-memory-utilization` | `0.20` | 1.5B 模型仅需 ~3.7 GB，20% 足够容纳权重 + KV Cache |
| `--max-model-len` | `4096` | RAG 5-chunk 上下文 + 系统 Prompt ≈ 1500 tokens，4096 安全裕量 |
| `--enforce-eager` | 启用 | 跳过 CUDA Graph 编译，加速冷启动，避免假死 |
| `PYTHONUNBUFFERED` | `1` | 确保 vLLM 日志实时输出，便于排查启动卡死 |

**GPU 资源充裕时升级至 7B 模型**：
```bash
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8001 \
    --gpu-memory-utilization 0.40 \
    --max-model-len 8192 \
    --trust-remote-code \
    --enforce-eager
```
并设置环境变量 `export LLM_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"`。

## 第二步：启动 RAG 后端应用（终端 B）

```bash
conda activate rag_agent
export HF_ENDPOINT=https://hf-mirror.com
python app.py
```

- 服务地址：`http://localhost:8000`
- 页面标题：**NewsPage**
- API 文档：`http://localhost:8000/docs`

如需从云端 API 切回本地 vLLM（或反向切换），通过环境变量覆盖：
```bash
# 切回本地 vLLM（默认值，通常无需手动设置）
export LLM_BASE_URL="http://localhost:8001/v1"
export LLM_API_KEY="EMPTY"
export LLM_MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"

# 或强制主通道使用云端智谱 API
export LLM_BASE_URL="https://open.bigmodel.cn/api/paas/v4"
export LLM_API_KEY="<your-zhipu-key>"
export LLM_MODEL_NAME="glm-4.7-flash"
```

## 第三步：启动公网隧道（终端 C — 可选）

```bash
conda run -n rag_agent python tunnel.py
# 或带认证:
conda run -n rag_agent python tunnel.py --token <YOUR_NGROK_AUTHTOKEN>
```

---

# 🏗️ 项目架构与模块

| 文件 | 行数 | 功能描述 |
|------|------|----------|
| `src/config.py` | ~140 | **全局配置中心** — 双通道 LLM 端点（本地 vLLM :8001 / 智谱 GLM-4.7-Flash）、ChromaDB 路径、嵌入模型（HF → ONNX 双轨回退）、PDF 分块参数（600/100）、相似度阈值（0.75）、Web 端口（8000） |
| `src/pdf_loader.py` | ~210 | **PDF 加载模块** — pypdf 逐页提取 → RecursiveCharacterTextSplitter 13 级递归分块（段落→换行→中英文标点→空格），chunk_size=600 / chunk_overlap=100 |
| `src/vector_store.py` | ~530 | **向量知识库模块** — HuggingFaceEmbeddings（优先）+ ONNXMiniLM_L6_V2（自动回退）双轨嵌入；ChromaDB 持久化（cosine 距离）；`search_similar_with_threshold()` 带阈值过滤检索 |
| `src/rag_chain.py` | ~1000 | **RAG 核心管线** — 四层金字塔容灾（本地 vLLM → 智谱 API → 智能结构化纯检索直出 → 优雅错误提示）；流式/非流式双模式；滑动窗口（3 轮）；并发互斥锁；全链路异常自动降级 |
| `app.py` | ~340 | **FastAPI 主入口** — 4 条路由：`GET /`（NewsPage 主页）、`POST /api/chat`（SSE 流式 RAG 对话）、`POST /api/upload`（PDF 上传+全量重建向量库）、`GET /api/status`（知识库状态）；启动期配置校验 |
| `tunnel.py` | ~115 | **ngrok 公网隧道** — 将本地 8000 端口暴露到公网，支持 authtoken 认证 |
| `templates/index.html` | ~145 | **NewsPage 主页面** — 双栏布局（左侧对话区 + 右侧上传/状态面板），集成 highlight.js + marked.js |
| `static/style.css` | ~455 | **UI 样式** — 科技蓝深色主题，CSS 变量体系，响应式布局 |
| `static/app.js` | ~305 | **前端交互** — SSE 流式消息接收、Markdown 实时渲染、PDF 拖拽上传、知识库状态轮询、Enter 发送 |
| `test_robot_rag.py` | ~390 | **RAG 自动化测试** — 4 题测试集（机械臂 SDK），非流式+流式双模式验证 |
| `test_stability.py` | ~400 | **稳定性压力测试** — 多轮对话、滑动窗口、并发保护、7 种故障降级场景覆盖 |
| `dev_log.md` | — | **详细开发与排错日志**（17 个章节，含 ADR 架构决策记录） |

## 四层金字塔容灾架构（ADR-5）

```
用户提问
  │
  ▼
ChromaDB 向量检索（带相似度阈值 0.75 过滤）
  │
  ├── Layer 1: 本地 vLLM (Qwen2.5-1.5B-Instruct, GPU 1, :8001)
  │     • 超时: connect=3s / read=30s
  │     • 并发保护: threading.Lock（串行化，超时 30s 后降级）
  │     └── 失败 → Layer 2
  │
  ├── Layer 2: 智谱 GLM-4.7-Flash (open.bigmodel.cn)
  │     • 当主通道已是智谱时自动跳过（_FALLBACK_ENABLED 判断）
  │     └── 失败 → Layer 3
  │
  ├── Layer 3: 纯向量检索智能直出 (CPU-only, 零显存/零API)
  │     • 关键词匹配排序 → 提取函数名/描述/参数/返回值/代码
  │     • 结构化输出，去重，Top-2 精准提取
  │     • 流式模式: ~15 字符/块分段 yield 模拟打字机
  │     └── 失败 → Layer 4
  │
  └── Layer 4: 优雅中文错误提示
        • "大模型服务暂时不可用，请稍后重试"
        • HTTP 503 + 结构化 JSON
```

## 全链路异常降级覆盖矩阵

| 故障类型 | 触发异常 | 降级路径 |
|----------|----------|----------|
| 向量检索异常 | 任意 Exception | 空上下文 → Layer 3 智能直出 |
| Prompt 构建异常 | 任意 Exception | 直接跳转 Layer 3 |
| vLLM 网络超时/连接失败 | `httpx.TimeoutException` / `NetworkError` | Layer 2（云端智谱） |
| vLLM OOM / CUDA 错误 | 任意 Exception（非网络类） | Layer 2 |
| 并发锁获取超时（30s） | — | 跳过 Layer 1 → Layer 2 |
| 云端 API 超时/限流 (429) | `httpx.TimeoutException` / `APIStatusError` | Layer 3 |
| Layer 3 内部异常 | 任意 Exception | Layer 4（友好提示） |
| 主通道已是智谱 API | — | 跳过 Layer 2（`_FALLBACK_ENABLED = False`） |

---

# ⚠️ 已知兼容补丁

## `pyairports` Shim 适配层

- **位置**: `site-packages/pyairports/`（Conda 环境内，非项目根目录）
- **背景**: `vllm` 依赖链 `outlines → pyairports.airports` 需要 `AIRPORT_LIST` 常量。PyPI 上的 `pyairports==0.0.1` 是恶意占位包（仅含 `sample` 模块），真实源码位于 `GitHub: NICTA/pyairports`。
- **修复**: 在 site-packages 下创建本地 Shim，包含 111 条全球主要机场数据，与 NICTA 原始接口完全兼容。
- **规则**: 严禁删除此 Shim 目录，严禁执行 `pip install pyairports`（会覆盖本地适配层导致 vLLM 导入失败）。

## `sentence-transformers` 与 `torchcodec` 冲突

- **症状**: `import sentence_transformers` → `torchcodec.decoders` → `libnvrtc.so.13` 缺失
- **影响范围**: 仅影响音视频模态加载路径，**文本嵌入场景不会触发**
- **处理**: 随 pyairports 修复而连带解决（import 链不再中断）。若嵌入加载失败，自动回退到 ONNX Runtime 方案。

## ChromaDB 距离度量校准

- `create_vector_store()` 中显式指定 `collection_metadata={"hnsw:space": "cosine"}`，强制使用余弦距离（0=完全相同, 1=正交, 2=完全相反）。
- `SIMILARITY_THRESHOLD = 0.75`：基于 Qwen2.5-7B-Instruct + 机械臂 SDK 文档实测校准，能召回 Q3 `get_robot_pose`（distance=0.72）同时过滤大部分无关切片。

---

# 📋 当前生产配置摘要

```python
# Layer 1: 本地 vLLM
BASE_URL     = "http://localhost:8001/v1"
MODEL_NAME   = "Qwen/Qwen2.5-1.5B-Instruct"

# Layer 2: 智谱 GLM-4.7-Flash（云端降级）
DEEPSEEK_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEEPSEEK_MODEL    = "glm-4.7-flash"

# 检索参数
CHUNK_SIZE           = 600
CHUNK_OVERLAP        = 100
RETRIEVAL_K          = 5
SIMILARITY_THRESHOLD = 0.75
DIRECT_RETRIEVAL_K   = 2

# Web 服务
HOST = "0.0.0.0"
PORT = 8000

# 嵌入模型
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # 可切换 BAAI/bge-small-zh-v1.5
FALLBACK_TO_ONNX     = True
```
