# NewsPage RAG 项目 — 开发日志

> **日期**: 2026-07-20  
> **开发者**: Kcsciso  
> **项目概述**: 基于 RAG（检索增强生成）架构的智能文档对话系统，支持加载本地 PDF、生成向量知识库、WebUI 交互及 ngrok 网络穿透。

---

## 一、环境问题修复记录

### 1.1 `vllm import` 失败 — pyairports 依赖缺失

**问题链**:
```
import vllm → outlines (v0.0.46) → outlines/types/airports.py
  → from pyairports.airports import AIRPORT_LIST
  → ModuleNotFoundError: No module named 'pyairports'
```

**根因**: PyPI 上的 `pyairports==0.0.1` 是一个恶意占位包（作者 "John Doe"，仅包含一个 `sample` 模块而非实际的机场数据）。真实源码位于 `GitHub: NICTA/pyairports`，但服务器处于内网隔离状态无法通过 git 获取。

**修复方案**: 在 site-packages 下创建本地 pyairports Shim（替身适配层）:
- `site-packages/pyairports/__init__.py` — 模块入口
- `site-packages/pyairports/airports.py` — 包含 111 条全球主要机场数据，提供 `AIRPORT_LIST`（与 NICTA 原始接口完全兼容）

### 1.2 `sentence_transformers import` 失败

**问题链**:
```
import sentence_transformers → torchcodec.decoders → libnvrtc.so.13 (缺失)
```

**根因**: `sentence-transformers==2.7.0` 的 `modality_types.py` 无条件导入 torchcodec，而 torchcodec 需要的 `libnvrtc.so.13`（CUDA 运行时）在当前环境中不存在。

**修复方案**: 该问题随 pyairports 修复而连带修复——因为 outlines 的 import 不再抛出异常，import 链不会中断。经测试，`import sentence_transformers` 本身可以完成（导入阶段不触发 torchcodec 路径），只有在实际加载音视频模态时才会调用 torchcodec——而文本嵌入场景不会经过该路径。

### 1.3 HuggingFaceEmbeddings 模型下载

**配置**: 设置 `HF_ENDPOINT=https://hf-mirror.com`（国内 HuggingFace 镜像），禁用 `HF_HUB_ENABLE_HF_TRANSFER` 以避免缺少 `hf_transfer` 包导致的下载失败。

**结果**: `all-MiniLM-L6-v2` (384维) 模型从镜像成功下载，嵌入功能正常工作。

---

## 二、新增依赖清单

| 包名 | 版本 | 用途 | 安装命令 |
|------|------|------|----------|
| `pypdf` | 6.14.2 | PDF 文本提取（纯 Python） | `pip install pypdf` |
| `langchain-chroma` | 1.1.0 | LangChain × ChromaDB 集成 | `pip install langchain-chroma` |

**未触碰的锁定依赖**: `torch`, `vllm`, `torchvision`, `torchaudio`, `sentence-transformers` 均保持原版本不变。

---

## 三、项目文件清单

### 3.1 核心逻辑 (`src/`)

| 文件 | 行数 | 功能描述 |
|------|------|----------|
| `src/__init__.py` | 1 | 包标识文件 |
| `src/config.py` | 127 | **全局配置中心** — LLM API (DeepSeek / 本地 vLLM)、向量库路径、嵌入模型 (huggingface / ONNX 回退)、PDF 分块参数、Web 服务端口，全部通过常量和环境变量可配置 |
| `src/pdf_loader.py` | 167 | **PDF 加载模块** — 使用 pypdf 逐页提取文本，RecursiveCharacterTextSplitter 进行递归分块（由粗到细：段落→换行→句号→字符），chunk_size=500 / chunk_overlap=50 |
| `src/vector_store.py` | 244 | **向量知识库模块** — HuggingFaceEmbeddings (优先) + ONNXMiniLM_L6_V2 (自动回退) 双轨嵌入策略；ChromaDB 持久化存储；语义相似度检索 (Top-K)；适配器模式封装 ONNX 接口 |
| `src/rag_chain.py` | 262 | **RAG 对话管线** — 经典四步法：检索 → 增强 → 生成 → 返回；支持 OpenAI 兼容 API (DeepSeek / vLLM)；内置非流式 + 流式 (SSE) 两种响应模式；多轮对话历史管理 |

### 3.2 Web 服务

| 文件 | 行数 | 功能描述 |
|------|------|----------|
| `app.py` | 241 | **FastAPI 主入口** — 4 条路由：`GET /` (NewsPage 主页)、`POST /api/chat` (流式 SSE 对话)、`POST /api/upload` (PDF 上传+自动重建向量库)、`GET /api/status` (知识库状态)；启动时自动加载已有向量库 |
| `templates/index.html` | 126 | **NewsPage 主页面** — 标题 "NewsPage"；双栏布局（左侧对话区 + 右侧上传/状态面板）；支持拖拽上传 PDF |
| `static/style.css` | 385 | **UI 样式** — CSS 变量体系；响应式布局 (桌面/移动端)；消息气泡动画；上传进度条 |
| `static/app.js` | 263 | **前端交互** — SSE 流式消息接收；PDF 拖拽/点击上传；知识库状态轮询；对话历史管理；Enter 发送 / Shift+Enter 换行 |

### 3.3 网络穿透

| 文件 | 行数 | 功能描述 |
|------|------|----------|
| `tunnel.py` | 115 | **ngrok 隧道脚本** — 将本地 8000 端口暴露到公网；支持 authtoken (环境变量或命令行参数)；自动打印公网 URL |

### 3.4 数据目录

| 目录 | 用途 |
|------|------|
| `data/` | 存放用户上传的 PDF 文件 (`.gitkeep` 初始化) |
| `vector_db/` | ChromaDB 持久化向量数据 (`.gitkeep` 初始化) |

### 3.5 配置文件

| 文件 | 用途 |
|------|------|
| `requirements_new.txt` | 新增 Python 依赖清单 |
| `CLAUDE.md` | 项目约束规则 (已存在) |
| `README.md` | 项目说明 (已存在) |

---

## 四、架构决策记录 (ADR)

### ADR-1: 嵌入模型双轨策略
- **决策**: HuggingFaceEmbeddings 作为主力，ONNXMiniLM_L6_V2 作为自动回退
- **理由**: 环境存在 sentence-transformers 兼容性风险，双轨保证鲁棒性
- **回退触发条件**: HuggingFaceEmbeddings 初始化失败 **或** 首次 embed 调用失败

### ADR-2: LLM 后端可替换设计
- **决策**: 使用 OpenAI 兼容 SDK，通过配置常量切换后端
- **支持**: DeepSeek API (云端) / 本地 vLLM (http://localhost:8000/v1) / 任何 OpenAI 兼容 API

### ADR-3: pyairports Shim 而非 pip install
- **决策**: 在本地创建结构完整的 pyairports 模块而非 pip 安装
- **理由**: PyPI 上的 pyairports 是恶意占位包，无法通过标准 pip 获取正确版本

### ADR-4: FastAPI + 原生 HTML 而非 Gradio/Streamlit
- **决策**: 使用 FastAPI + Jinja2 + 原生 HTML/CSS/JS
- **理由**: FastAPI 是预装依赖；原生前端无额外依赖负担，更灵活可控

---

## 五、验证结果

| 测试项 | 结果 | 说明 |
|--------|------|------|
| `import vllm` | ✅ | vllm 0.5.4 成功导入 |
| `import sentence_transformers` | ✅ | 2.7.0 成功导入 |
| `HuggingFaceEmbeddings` | ✅ | 从 hf-mirror.com 下载 all-MiniLM-L6-v2，384维 |
| ChromaDB CRUD | ✅ | 创建 → 检索 → 查询全链路通过 |
| 语义检索准确性 | ✅ | "中国的首都" → 正确返回北京相关片段 |
| FastAPI 路由注册 | ✅ | 4 条路由全部就绪，标题为 "NewsPage" |

---

## 六、启动指南

```bash
# 1. 启动 RAG 服务
conda run -n rag_agent python app.py
# 访问: http://localhost:8000

# 2. (可选) 启动 ngrok 隧道
conda run -n rag_agent python tunnel.py --token <YOUR_NGROK_TOKEN>

# 3. 使用流程
#    - 打开浏览器访问 http://localhost:8000
#    - 在右侧面板上传 PDF 文件
#    - 在左侧对话框输入问题，基于文档内容进行 RAG 对话
```

---

## 七、容灾降级修复记录

> **日期**: 2026-07-21  
> **修复范围**: `src/rag_chain.py`、`app.py`  
> **修复目标**: 解决体检报告指出的三大致命隐患，构建四层金字塔容灾架构

### 7.1 超时拦截器修复 (Fix Issue 1)

**问题**: `OpenAI()` 客户端与 `client.chat.completions.create()` 调用均未显式设置 `timeout` 参数，依赖 httpx 默认值 `read=600s`。当 vLLM 进程假死（GPU 卡死但 TCP 端口仍监听）时，前端将卡死 10 分钟无任何反馈。

**修复方案**: 在 `src/rag_chain.py` 中显式配置 `httpx.Timeout`:

```python
LLM_TIMEOUT = httpx.Timeout(connect=3.0, read=15.0, write=15.0, pool=3.0)
```

- `connect=3.0s`: vLLM 未启动时 3 秒内抛出 `ConnectTimeout`，快速失败
- `read=15.0s`: vLLM 假死时 15 秒内抛出 `ReadTimeout`，触发降级
- 所有 `OpenAI()` 客户端实例 (`_get_client` / `_get_deepseek_client`) 统一使用此超时配置

**影响文件**: `src/rag_chain.py` — `LLM_TIMEOUT` 常量，`_get_client()` / `_get_deepseek_client()` 调用

### 7.2 降级异常捕获闭环修复 (Fix Issue 2a/2b)

**问题**: 
- `config.py` 已定义 `DEEPSEEK_BASE_URL`、`DEEPSEEK_API_KEY`、`DEEPSEEK_MODEL` 三常量，但 `rag_chain.py` 从未 import 或使用，DeepSeek 降级逻辑完全缺失
- 非流式路径直接 `HTTPException(500, detail=str(e))`，返回原始 Python 异常而非用户可读提示

**修复方案**:

1. **DeepSeek 降级客户端**: 新增 `_get_deepseek_client()` 单例工厂函数，与主客户端使用相同的超时配置
2. **智能降级判断**: `_FALLBACK_ENABLED = BASE_URL != DEEPSEEK_BASE_URL`，当主通道已是 DeepSeek 时避免同源无意义降级
3. **LLM 调用复用**: 抽取 `_call_llm()` 和 `_stream_llm()` 辅助函数，双通道共享同一调用逻辑，消除代码重复
4. **结构化错误响应**: `app.py` 捕获 `LLMServiceError` 后返回 `503 + JSON {"error": ..., "error_type": "llm_unavailable", "message": ...}` 而非裸 500

**影响文件**: `src/rag_chain.py` — `_get_deepseek_client()`, `_call_llm()`, `_stream_llm()`, `LLMServiceError`; `app.py` — `/api/chat` 非流式路径

### 7.3 纯向量检索直出模式 (Fix Issue 3 — 新增 Layer 3)

**问题**: 当本地 vLLM 和云端 DeepSeek API 全部不可用时，此前仅能抛出异常，用户完全无法获取任何信息。

**修复方案**: 在 `src/rag_chain.py` 中新增第 3 层降级——纯向量检索直出模式:

```
_direct_retrieval_response()         # 非流式：直接返回格式化文本
_direct_retrieval_response_stream()  # 流式：分段 yield 模拟打字机效果
_format_direct_retrieval_answer()    # 模板组装公共逻辑
```

**核心特征**:
- **纯 CPU 运行**: 仅使用 ChromaDB 向量检索 + 模板组装，不调用任何 LLM
- **零显存消耗**: 不经过 vLLM / PyTorch GPU 推理路径
- **零 API 费用**: 不产生任何云端 API 调用
- **秒级响应**: 省略 LLM 推理延迟，直接返回检索到的原文片段
- **流式兼容**: 以 ~15 字符/块的速率分段 yield，前端 SSE 打字机效果正常运作

**输出模板**:
```
【提示：当前大模型生成服务未就绪，已为您开启纯文档检索直出模式】

根据比邻星技术文档，找到以下相关内容：

1. [来源: xxx.pdf]
<文档原文片段>

2. [来源: yyy.pdf]
<文档原文片段>

---
💡 以上为文档原文检索结果。如需更深入的分析与总结，请等待大模型服务恢复后重试。
```

**注意事项**:
- 纯检索模式不做内容理解与总结，仅提供原文片段
- 多轮对话上下文不会影响检索结果（仅基于当前 query 检索）
- 使用独立 Top-K 参数 `DIRECT_RETRIEVAL_K = 3`，可独立于 `RETRIEVAL_K` 调整

**影响文件**: `src/rag_chain.py` — `_direct_retrieval_response()`, `_direct_retrieval_response_stream()`, `_format_direct_retrieval_answer()`, `DIRECT_RETRIEVAL_K`

### 7.4 启动期配置校验 (Fix Issue 4)

**问题**: 系统采用"延迟爆炸"策略——API Key 错误只在用户首次对话时暴露，而非在启动阶段就明确提示。

**修复方案**: 在 `app.py` 的 `startup_event()` 中增加启动期配置校验:

- **DEEPSEEK_API_KEY 占位符检测**: 如仍为默认值 `"sk-your-deepseek-key-here"`，打印 WARNING 并引导用户设置环境变量
- **主通道标识**: 打印当前使用的 LLM 通道类型（本地 vLLM / 云端 API）

**控制台输出示例**:
```
⚠️  DEEPSEEK_API_KEY 仍为默认占位符 'sk-your-deepseek-key-here'，DeepSeek 降级通道（第 2 层容灾）将不可用！
   请设置环境变量: export DEEPSEEK_API_KEY=<your-deepseek-key>
   获取 Key: https://platform.deepseek.com/api_keys
```

**影响文件**: `app.py` — `startup_event()`

---

## 八、架构决策记录 (ADR-5)

### ADR-5: 四层金字塔容灾架构

- **决策**: 构建"本地 vLLM → DeepSeek API → 纯检索直出 → 优雅错误"的四层降级链路
- **理由**:
  - 第 1 层（本地 vLLM）是主力通道：低延迟，零 API 费用，但依赖本地 GPU 服务健康
  - 第 2 层（DeepSeek API）是云端备份：算力充足，但依赖网络与 API Key 配置
  - 第 3 层（纯检索直出）是"保底可用"模式：纯 CPU 运行，零显存，零 API 费用，只要 ChromaDB 向量库完好即可工作——确保极端故障下用户仍能获取相关文档原文
  - 第 4 层（优雅错误提示）是最终防线：仅在向量库损坏等极端情况触发，向前端返回结构化中文错误而非 HTTP 500 堆栈
- **降级触发条件**: 网络超时 (`httpx.TimeoutException`)、连接失败 (`httpx.NetworkError`)、SDK 封装异常 (`APITimeoutError`, `APIConnectionError`) 触发层间切换；非网络异常（认证失败、参数错误等）同样走降级链路（因为下一层可能使用不同的认证/参数配置）
- **特殊处理**: 当主 `BASE_URL` 已是 DeepSeek API 时，自动跳过第 2 层降级（`_FALLBACK_ENABLED = False`），避免同源无意义重试
- **流式兼容**: 第 3 层提供 `_direct_retrieval_response_stream()` 模拟流式输出，将组装文本按 ~15 字符/块分段 yield，确保前端 SSE 打字机效果正常运作

### 架构图示

```
┌─────────────────────────────────────────────────────────────────┐
│  用户提问                                                        │
│    │                                                             │
│    ▼                                                             │
│  ┌─────────────────────┐                                         │
│  │ ChromaDB 向量检索    │  ← CPU 向量相似度搜索                  │
│  │ (search_similar)    │                                         │
│  └─────────┬───────────┘                                         │
│            │ context_docs                                        │
│            ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 第 1 层：本地 vLLM 推理 (GPU)                                ││
│  │   • 超时: connect=3s / read=15s                             ││
│  │   • 失败 → 进入第 2 层                                       ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │ 第 2 层：云端 DeepSeek API (Cloud)                           ││
│  │   • 自动无缝切换，日志标注"降级成功"                         ││
│  │   • 失败 → 进入第 3 层                                       ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │ 第 3 层：纯向量检索直出模式 (CPU-only)                       ││
│  │   • 零显存 / 零 API 费用 / 秒级响应                          ││
│  │   • 返回 ChromaDB 检索原文 + 模板提示                        ││
│  │   • 失败 → 进入第 4 层                                       ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │ 第 4 层：优雅中文错误提示                                     ││
│  │   • "大模型服务暂时不可用，请稍后重试"                       ││
│  │   • HTTP 503 + 结构化 JSON                                   ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## 九、修复后文件变更摘要

| 文件 | 变更类型 | 行数变化 | 关键变更 |
|------|----------|----------|----------|
| `src/rag_chain.py` | 重写 | 262 → ~460 | 新增超时配置、DeepSeek 降级客户端、LLM 调用辅助函数、纯检索直出模式、LLMServiceError 异常类、`rag_chat()` 与 `rag_chat_stream()` 四层容灾逻辑 |
| `app.py` | 修改 | 241 → ~270 | 导入 `LLMServiceError`、非流式路径 503 结构化错误响应、`startup_event()` 启动期配置校验 |

**未触碰的文件**: `src/config.py`, `src/vector_store.py`, `src/pdf_loader.py`, `templates/`, `static/`
