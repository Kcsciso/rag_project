## 🔴 系统红线与开发规则 (STRICT CONSTRAINTS)

### 1. 硬件与 GPU 管理 (双 A100 分离策略)
- **算力底座**: 2 × NVIDIA A100-PCIE-40GB (CUDA 12.4)。
- **显卡分配规则**: 
  - `CUDA_VISIBLE_DEVICES=1`: 专用于后台本地 `vllm` 大模型推理服务（加载开源基座）。
  - `CUDA_VISIBLE_DEVICES=0`: 留给向量检索引擎（ChromaDB/PyTorch 嵌入计算）及其他后台任务。
- **核心操作**: 启动任何推理或训练脚本前必须显式指定环境变量。

### 2. 核心依赖红线（严禁升级）
环境管理器为 Conda (`rag_agent`, Python 3.10)。以下基础依赖已被**严格锁定**，**绝不允许执行 `pip install --upgrade`**：
- `torch==2.6.0+cu124`
- `torchvision==0.21.0+cu124`
- `torchaudio==2.6.0+cu124`
- `vllm==0.16.0`（通过 `--no-deps` 隔离安装）。

### 3. RAG 架构与 AI 生态
- **可用框架**: LangChain, LangGraph, ChromaDB, faiss-gpu。
- **LLM 推理引擎**: 本地部署的 `vllm` OpenAI 兼容服务 (`http://localhost:8000/v1`)。
  - 核心服务模型: `Qwen/Qwen2.5-7B-Instruct`
  - 路由别名: `deepseek-v4-pro`
- **UI 命名规范**: 前端界面或网页标题（HTML `<title>`）**必须**命名为 **NewsPage**。

### 4. 操作指令
- 在执行破坏性 Bash 命令或安装软件包之前，必须征得用户的明确授权。
- 所有重大架构调整、Ablation 实验及 Git 提交必须记录在 `dev_log.md` 中。

---

## 🚀 本地服务启动顺序 (关键)

1. **第一步：启动本地 vLLM 大模型推理服务 (终端 A)**
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

2. **第二步：启动 RAG 后端应用 (终端 B)**
```bash
conda activate rag_agent
python app.py

```


3. **第三步：启动公网隧道 (终端 C - 可选)**
```bash
conda run -n rag_agent python tunnel.py

```



---

## 🏗️ 项目架构与模块

* `src/config.py`: 全局配置中心（本地 vLLM 接口、ChromaDB 路径管理）。
* `src/pdf_loader.py`: 比邻星科技开发/使用文档 PDF 加载与递归字符级文本分块。
* `src/vector_store.py`: ChromaDB 初始化与嵌入向量管理。
* `src/rag_chain.py`: RAG 四步核心管线（检索上下文 ➔ 构造比邻星专属 Prompt ➔ 调用本地 vLLM ➔ 流式对话）。
* `app.py`: FastAPI 应用程序，支撑 **NewsPage** Web 交互界面与 API 路由。
* `tunnel.py`: ngrok 公网安全隧道集成。
* `dev_log.md`: 详细开发与排错日志。

## ⚠️ 已知兼容补丁

* **`pyairports` Stub**: 为解决隔离环境中 `vllm` 对 `outlines` 的依赖缺失，根目录下部署了本地 Shim 适配层。严禁删除 `pyairports/` 目录。

```

```