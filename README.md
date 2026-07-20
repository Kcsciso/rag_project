# 📰 NewsPage - 湖南比邻星科技文档智能问答系统 (Local vLLM 架构)

> 基于 **RAG (Retrieval-Augmented Generation)** 架构的官方技术文档与使用手册问答系统。专为**湖南比邻星科技有限公司**的开发者和用户打造，采用双 A100 算力底座，底层搭载 **vLLM + 开源大模型** 实现完全私有化、高性能的本地推理。

---

## 🚀 核心特性

- **本地高性能推理底座 (`vllm`)**: 依托 vLLM 的 `PagedAttention` 技术与双 A100 硬件支持，实现大模型高并发、低延迟的本地化推理，保障数据隐私与稳定运行。
- **比邻星文档专属解析 (`src/pdf_loader.py`)**: 支持多份比邻星科技开发文档与使用手册的批量加载，采用递归字符级文本分块，确保 API 规范与技术参数切分不失真。
- **高精度稠密向量检索 (`src/vector_store.py`)**: 深度结合 ChromaDB 与 `sentence-transformers`（通过 `hf-mirror.com` 国内镜像加速），实现海量技术文档的秒级语义召回。
- **现代化 Web 交互界面**: 提供清爽直观的 **NewsPage** 聊天与文档管理交互体验。
- **一键公网穿透 (`tunnel.py`)**: 内置 ngrok 隧道集成，支持快速将本地系统发布至公网进行内部演示。

---

## 📁 项目目录结构

```text
rag_project/
├── src/
│   ├── config.py          # 全局配置中心（本地 API 路由、模型、路径管理）
│   ├── pdf_loader.py      # PDF 解析与文本切分模块
│   ├── vector_store.py    # ChromaDB 向量库与嵌入管理模块
│   └── rag_chain.py       # RAG 四步核心管线与本地 vLLM 交互
├── templates/
│   └── index.html         # NewsPage 聊天与文档交互主页面
├── static/
│   ├── style.css          # 页面样式文件
│   └── app.js             # 前端异步通信与打字机效果逻辑
├── data/                  # 存放比邻星科技 PDF 开发与使用文档目录
├── vector_db/             # ChromaDB 向量本地持久化目录
├── pyairports/            # 离线环境 Shim 适配层（vllm 依赖对齐）
├── app.py                 # FastAPI 异步应用入口
├── tunnel.py              # ngrok 公网穿透脚本
├── dev_log.md             # 详细开发与排错日志
├── CLAUDE.md              # AI 协同开发规范与红线
└── README.md              # 项目说明文档

```

---

## ⚙️ 系统环境与约束

* **硬件底座**: 2 × NVIDIA A100-PCIE-40GB (CUDA 12.4)
* **环境管理器**: Conda (`rag_agent`, Python 3.10)
* **核心锁定依赖 (严禁升级)**:
* `torch==2.6.0+cu124`
* `torchvision==0.21.0+cu124`
* `torchaudio==2.6.0+cu124`
* `vllm==0.16.0`（通过 `--no-deps` 隔离安装）



---

## 🚀 部署与启动指南

### 1. 准备比邻星科技文档

将湖南比邻星科技有限公司的开发文档、API 规范或产品使用手册（PDF 格式）放入 **`data/`** 目录中。

### 2. 启动本地 vLLM 大模型推理服务 (终端 A)

激活 Conda 环境，指定显卡 1 运行 vLLM：

```bash
conda activate rag_agent
export CUDA_VISIBLE_DEVICES=1
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --served-model-name deepseek-v4-pro \
    --max-model-len 8192 \
    --port 8000 \
    --gpu-memory-utilization 0.8

```

### 3. 启动 RAG 后端与 Web 服务 (终端 B)

在另一个终端中，配置镜像并启动 FastAPI 应用：

```bash
conda activate rag_agent
export HF_ENDPOINT=[https://hf-mirror.com](https://hf-mirror.com)
python app.py

```

服务启动成功后，可在浏览器访问：`http://localhost:8000`（系统界面标题为 **NewsPage**）。

### 4. 启动公网隧道 (终端 C - 可选)

```bash
python tunnel.py

```

---

## 📝 开发与排错日志

有关底层环境排查、`vllm`/`pyairports` 离线兼容补丁以及模型镜像调优的详细过程，请参考 [dev_log.md](https://www.google.com/search?q=./dev_log.md)。

```

```