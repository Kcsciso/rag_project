# 📰 NewsPage RAG System

> 基于 **RAG (Retrieval-Augmented Generation)** 架构的智能文档问答与检索系统，集成本地 PDF 知识库构建、语义向量检索、DeepSeek 动态大模型调用、FastAPI 异步后端、**NewsPage** Web 交互界面及公网隧道穿透。

---

## 🚀 核心特性

- **高效文档解析 (`src/pdf_loader.py`)**: 支持多 PDF 批量加载与递归字符级文本分块 (`RecursiveCharacterTextSplitter`)，保障上下文语义的连贯性。
- **双轨智能嵌入 (`src/vector_store.py`)**: 基于 ChromaDB 与 `sentence-transformers`（完美适配 `hf-mirror.com` 国内高速镜像），实现高维稠密向量的高效持久化与精准语义检索。
- **鲁棒 RAG 管线 (`src/rag_chain.py`)**: 严格执行四步标准化 RAG 闭环（检索上下文 ➔ 构造 Prompt ➔ 调用大模型 ➔ 流式/非流式输出）。
- **DeepSeek API 深度集成**: 搭载预配置的 DeepSeek 核心路由模型 (`deepseek-v4-pro`) 与子代理模型 (`deepseek-v4-flash`)。
- **现代化前端交互**: 提供清爽直观的 **NewsPage** 聊天与文档管理界面。
- **公网穿透支持 (`tunnel.py`)**: 内置 ngrok 隧道脚本，支持一键将本地服务发布至公网演示。

---

## 📁 项目目录结构

```text
rag_project/
├── src/
│   ├── config.py          # 全局配置中心（API 路由、模型、路径管理）
│   ├── pdf_loader.py      # PDF 解析与文本切分模块
│   ├── vector_store.py    # ChromaDB 向量库与嵌入管理模块
│   └── rag_chain.py       # RAG 四步核心管线与大模型交互
├── templates/
│   └── index.html         # NewsPage 聊天主页面
├── static/
│   ├── style.css          # 页面样式文件
│   └── app.js             # 前端交互与异步通信逻辑
├── data/                  # 本地 PDF 原始文档存储目录
├── vector_db/             # ChromaDB 向量持久化目录
├── pyairports/            # 离线环境 Shim 适配层（vllm 依赖对齐）
├── app.py                 # FastAPI 异步应用入口
├── tunnel.py              # ngrok 公网穿透脚本
├── dev_log.md             # 详细开发、排错与演进日志
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
* `vllm==0.16.0` (通过 `--no-deps` 隔离安装)



---

## 🚀 快速上手指南

### 1. 激活环境与镜像配置

在受限/隔离网络环境中，请先激活 Conda 环境并配置国内模型镜像：

```bash
conda activate rag_agent
export HF_ENDPOINT=[https://hf-mirror.com](https://hf-mirror.com)

```

### 2. 启动后端服务

运行 FastAPI 服务：

```bash
conda run -n rag_agent python app.py

```

服务启动成功后，可在浏览器访问：`http://localhost:8000`（界面标题为 **NewsPage**）。

### 3. 启动公网隧道 (可选)

如果需要将本地服务暴露到公网进行演示：

```bash
conda run -n rag_agent python tunnel.py

```

---

## 📝 开发与排错日志

有关底层环境排查、`vllm`/`pyairports` 离线兼容补丁以及模型镜像调优的详细过程，请参考 [dev_log.md](https://www.google.com/search?q=./dev_log.md)。

```

```