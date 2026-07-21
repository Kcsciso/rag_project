# 🔴 系统红线与开发规则（STRICT CONSTRAINTS）

## 1. 硬件与 GPU 管理（双 A100 智能自适应）

- **算力底座**: 2 × NVIDIA A100-PCIE-40GB（CUDA 12.4）。
- **GPU 自适应策略**（Dynamic GPU Detection）:
  - **禁止硬编码 `CUDA_VISIBLE_DEVICES`**。启动脚本 `start_services.sh` 和 `src/config.py` 已内置 `nvidia-smi` 空闲显存扫描。
  - 自动选择**剩余显存最大的 GPU**（过滤空闲 < 5 GB 的 GPU）。
  - 手动覆盖方式：`--gpu <id>` 参数 或 `VLLM_GPU_ID` 环境变量。
- **默认分配**（自动检测无结果时的回退）:
  - GPU 1（端口 **8001**）：本地 vLLM 推理服务，当前模型 **Qwen2.5-1.5B-Instruct**（~3.7 GB）。
  - GPU 0：向量检索引擎（ChromaDB / PyTorch 嵌入计算）。**注意**：GPU 0 为多人共享，高峰期空闲仅 ~16 MB。
- **GPU 升级路径**：空闲 ≥15 GB 时，通过 `LLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct` + `--gpu-memory-utilization 0.40` 升级。
- **核心操作**：启动推理或训练脚本前必须显式指定 `CUDA_VISIBLE_DEVICES`（优先使用自动检测结果）。

## 2. 核心依赖红线（严禁升级）

环境管理器为 Conda（`rag_agent`，Python 3.10）。以下依赖被**严格锁定**，**绝不允许执行 `pip install --upgrade`**：
- `torch==2.6.0+cu124`
- `torchvision==0.21.0+cu124`
- `torchaudio==2.6.0+cu124`
- `vllm==0.16.0`（通过 `--no-deps` 隔离安装）
- `sentence-transformers==2.7.0`

允许新增的辅助包（需确认与锁定依赖无冲突）：
- `pypdf`、`langchain-chroma`（已安装）
- `rank-bm25`（混合检索 BM25 关键词召回，计划引入）

## 3. RAG 架构与 AI 生态

- **可用框架**: LangChain、LangGraph、ChromaDB、faiss-gpu。
- **LLM 推理引擎**: 本地 `vllm` OpenAI 兼容服务。
  - **Layer 1（主模型）**: `Qwen/Qwen2.5-1.5B-Instruct`，`http://localhost:8001/v1`，GPU 自适应，`--gpu-memory-utilization 0.20`，`--max-model-len 4096`，`--enforce-eager`。
  - **Layer 2（云端降级）**: `glm-4.7-flash`（智谱 GLM-4.7-Flash），端点 `https://open.bigmodel.cn/api/paas/v4`，认证 `ZHIPU_API_KEY`。
  - **超时策略**: `connect=2.0s / read=12.0s / write=12.0s / pool=2.0s`（激进失败 → 快速降级）。
- **嵌入模型**: `all-MiniLM-L6-v2`（384 维），可切换 `BAAI/bge-small-zh-v1.5`（512 维，中文专优）。HuggingFace → ONNX 自动回退。
- **UI 命名规范**: 前端界面或网页标题（HTML `<title>` 与 header）**必须**命名为 **NewsPage**。

## 4. 安全开发红线（新增）

- **输入清洗**：所有用户输入的 query 必须经 `sanitize_query()` 清洗（去 null 字节、控制字符、规范化换行）。
- **文件名安全**：所有上传文件名必须经 `sanitize_filename()` 清洗（`os.path.basename` 防路径遍历 + null 字节删除）。
- **Prompt 注入防御**：`_build_messages()` 中 chat_history 的 role 必须为 `user` / `assistant`（白名单），非法 role 自动丢弃并记录 WARNING。
- **历史长度限制**：对话历史最多 100 条（`MAX_HISTORY_ITEMS`），超出截断；每条 content 上限 4000 字符。
- **查询长度限制**：`MAX_QUERY_LENGTH=2000` 字符。
- **SSE 资源管理**：队列限界 `maxsize=256`；客户端断开时必须设置取消标志让线程池生成器退出。
- **资源清理**：应用关闭时必须调用 `shutdown_clients()`（释放 LLM 连接池），嵌入模型引用在 `shutdown` 事件中释放。

## 5. 操作指令

- 在执行破坏性 Bash 命令或安装软件包之前，必须征得用户的明确授权。
- 所有重大架构调整、Ablation 实验及 Git 提交必须记录在 `dev_log.md` 中。
- 严禁删除 Conda 环境 `site-packages/pyairports/` 下的 Shim 适配层。
- 保持 `README.md`、`CLAUDE.md`、`dev_log.md` 三份文档与代码库同步。

---

# 🚀 本地服务启动顺序（关键）

## 方式一：一键启动（推荐）

```bash
chmod +x start_services.sh
./start_services.sh                    # 智能 GPU 检测 → vLLM → FastAPI
./start_services.sh --vllm-only        # 仅启动 vLLM
./start_services.sh --fastapi-only     # 仅启动 FastAPI（vLLM 已运行）
./start_services.sh --gpu 0            # 手动指定 GPU 0
```

脚本自动完成 GPU 空闲扫描、端口检测、vLLM 后台拉起、就绪轮询、`Ctrl+C` 优雅退出。

## 方式二：手动分步启动

### 第一步：启动本地 vLLM 推理服务（终端 A）

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

| 参数 | 值 | 理由 |
|------|-----|------|
| `CUDA_VISIBLE_DEVICES` | `1`（或自动检测结果） | 隔离 GPU，避免与其他用户进程冲突 |
| `--port` | **8001** | FastAPI 占用 8000，vLLM 使用独立端口 |
| `--gpu-memory-utilization` | `0.20` | 1.5B 模型仅需 ~3.7 GB，20% 安全裕量 |
| `--max-model-len` | `4096` | RAG 5-chunk + system prompt ≈ 1500 tokens |
| `--enforce-eager` | 启用 | 跳过 CUDA Graph 编译，加速冷启动 |
| `PYTHONUNBUFFERED` | `1` | vLLM 日志实时输出 |

### 第二步：启动 RAG 后端应用（终端 B）

```bash
conda activate rag_agent
export HF_ENDPOINT=https://hf-mirror.com
python app.py
```

- 服务地址：`http://localhost:8000`（页面标题：**NewsPage**）
- API 文档：`http://localhost:8000/docs`

**环境变量覆盖**：
```bash
# 切回本地 vLLM（默认）
export LLM_BASE_URL="http://localhost:8001/v1"
export LLM_MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"

# 主通道直连智谱云端 API
export LLM_BASE_URL="https://open.bigmodel.cn/api/paas/v4"
export LLM_MODEL_NAME="glm-4.7-flash"

# 手动指定 vLLM GPU
export VLLM_GPU_ID=0
```

### 第三步：启动公网隧道（终端 C — 可选）

```bash
conda run -n rag_agent python tunnel.py --token <YOUR_NGROK_AUTHTOKEN>
```

---

# 🏗️ 项目架构与模块

| 文件 | 功能描述 |
|------|----------|
| `src/config.py` | **全局配置中心** — 双通道 LLM（vLLM :8001 / 智谱 GLM-4.7-Flash）、GPU 智能探测 API（`detect_best_gpu` / `get_all_gpu_info` / `VLLM_GPU_ID`）、ChromaDB 路径、嵌入模型双轨回退、检索参数（600/100/5/0.75） |
| `src/pdf_loader.py` | **PDF 加载** — pypdf 逐页提取 → RecursiveCharacterTextSplitter 13 级递归分块 |
| `src/vector_store.py` | **向量知识库** — HuggingFaceEmbeddings（优先）+ ONNXMiniLM_L6_V2（回退）双轨嵌入；`search_similar_with_threshold()` 阈值过滤；`cleanup_vector_store()` 资源释放 |
| `src/rag_chain.py` | **RAG 核心管线** — 四层金字塔容灾（vLLM → 智谱 → 纯检索直出 → 优雅错误）；Prompt 注入防御（role 白名单 + 注入检测）；Layer 3 行级归一化去重；`shutdown_clients()`；超时 2s/12s |
| `app.py` | **FastAPI 主入口** — 4 条路由 + 安全中间件（`sanitize_query` / `sanitize_filename` / `validate_chat_history`）；SSE 防泄露（`cancelled` + `CancelledError` + 限界队列）；`shutdown` 事件 |
| `check_status.py` | **健康检查** — vLLM + FastAPI + GPU 显存/温度/功率 + 四层容灾可用性 + vLLM 部署 GPU 识别 |
| `start_services.sh` | **一键启动** — GPU 智能选择 + 端口检测 + vLLM 后台拉起 + FastAPI 启动 + 优雅退出 |
| `tunnel.py` | **ngrok 隧道** — 公网穿透，authtoken 认证 |
| `test_robot_rag.py` | **功能回归测试** — 4 题 × 流式/非流式双模式 |
| `test_stability.py` | **稳定性压力测试** — 多轮对话 + 并发 + 7 种异常降级场景 |

## 四层金字塔容灾架构（ADR-5）

```
用户提问
  │
  ▼
ChromaDB 向量检索（相似度阈值 0.75 过滤）
  │
  ├── Layer 1: 本地 vLLM (Qwen2.5-1.5B-Instruct, GPU 自适应, :8001)
  │     • 超时: connect=2s / read=12s
  │     • 并发保护: threading.Lock（串行化，超时 30s 后降级）
  │     └── 失败 → Layer 2
  │
  ├── Layer 2: 智谱 GLM-4.7-Flash (open.bigmodel.cn)
  │     • 主通道已是智谱时自动跳过（_FALLBACK_ENABLED）
  │     └── 失败 → Layer 3
  │
  ├── Layer 3: 纯向量检索智能直出 (CPU-only, 零显存/零API)
  │     • 行级归一化去重 → 函数/描述/参数/返回值/代码提取
  │     • 流式: ~15 字符/块分段 yield 模拟打字机
  │     └── 失败 → Layer 4
  │
  └── Layer 4: 优雅中文错误提示
        • "大模型服务暂时不可用，请稍后重试"
        • HTTP 503 + 结构化 JSON
```

## 全链路异常降级覆盖矩阵

| 故障类型 | 降级路径 |
|----------|----------|
| 向量检索异常 | 空上下文 → Layer 3 智能直出 |
| Prompt 构建异常 | 直接跳转 Layer 3 |
| vLLM 网络超时/连接失败 (2s/12s) | Layer 2（云端智谱） |
| vLLM OOM / CUDA 错误 | Layer 2 |
| 并发锁获取超时（30s） | 跳过 Layer 1 → Layer 2 |
| 云端 API 超时/限流 (429) | Layer 3 |
| Layer 3 内部异常 | Layer 4（友好提示） |
| 主通道已是智谱 API | 跳过 Layer 2（同源去重） |
| SSE 客户端断开 | cancelled 标志 → 线程池生成器退出 |

---

# ⚠️ 已知兼容补丁

## `pyairports` Shim 适配层

- **位置**: `site-packages/pyairports/`（Conda 环境内，非项目根目录）
- **背景**: `vllm` 依赖链 `outlines → pyairports.airports`。PyPI 上 `pyairports==0.0.1` 为恶意占位包。
- **修复**: 在 site-packages 下创建本地 Shim（111 条机场数据，与 NICTA 接口兼容）。
- **规则**: 严禁删除 Shim，严禁 `pip install pyairports`。

## `sentence-transformers` 与 `torchcodec` 冲突

- **症状**: `torchcodec.decoders → libnvrtc.so.13` 缺失
- **影响**: 仅音视频模态加载路径，文本嵌入不受影响。失败时自动回退 ONNX Runtime。

## ChromaDB 距离度量

- `collection_metadata={"hnsw:space": "cosine"}` 强制余弦距离。
- `SIMILARITY_THRESHOLD=0.75`：实测校准值，可召回 `get_robot_pose` 同时过滤无关切片。

---

# 🔧 运维与诊断

```bash
# 健康检查
python check_status.py                # 一次性完整报告
python check_status.py --watch 10     # 每 10 秒刷新

# 自动化测试
conda run -n rag_agent python test_robot_rag.py      # RAG 功能回归
conda run -n rag_agent python test_stability.py       # 稳定性压力测试
```

---

# 📋 当前生产配置摘要

```python
# Layer 1: 本地 vLLM
BASE_URL     = "http://localhost:8001/v1"
MODEL_NAME   = "Qwen/Qwen2.5-1.5B-Instruct"

# Layer 2: 智谱 GLM-4.7-Flash（云端降级）
DEEPSEEK_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEEPSEEK_MODEL    = "glm-4.7-flash"

# LLM 超时
LLM_TIMEOUT = httpx.Timeout(connect=2.0, read=12.0, write=12.0, pool=2.0)

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
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FALLBACK_TO_ONNX     = True

# GPU 自适应
VLLM_GPU_ID = <auto-detected>  # 环境变量覆盖: export VLLM_GPU_ID=1
MIN_FREE_MEMORY_MIB = 5120     # 最低空闲显存门槛（5 GB）
```
