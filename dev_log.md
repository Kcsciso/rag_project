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
