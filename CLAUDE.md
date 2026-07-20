# CLAUDE.md - NewsPage RAG Project Guidelines

## 🔴 SYSTEM CONSTRAINTS & DEVELOPMENT RULES (STRICT)

### 1. Hardware & GPU Management
- **Compute Base:** 2 × NVIDIA A100-PCIE-40GB (CUDA 12.4 max).
- **Action Required:** Before running any training or inference scripts, explicitly check GPU availability and set `os.environ["CUDA_VISIBLE_DEVICES"]` to avoid conflicting with background tasks.

### 2. STRICT Dependency Red Lines (DO NOT UPGRADE)
The environment manager is Conda (`rag_agent`, Python 3.10). The following base dependencies are **LOCKED**. 
**NEVER execute `pip install --upgrade` for these packages:**
- `torch==2.6.0+cu124`
- `torchvision==0.21.0+cu124`
- `torchaudio==2.6.0+cu124`
- `vllm==0.16.0` (Installed via `--no-deps`. The dependency chain with PyTorch is severed to prevent conflicts).

If new packages need to be installed, install them individually without upgrading existing core libraries.

### 3. RAG Architecture & AI Ecosystem
- **Frameworks Available:** LangChain, LangGraph, ChromaDB, faiss-gpu.
- **LLM Engine (DeepSeek API):**
  - Base URL: `https://api.deepseek.com/anthropic`
  - Core Router Model: `deepseek-v4-pro`
  - Sub-Agent Model: `deepseek-v4-flash`
- **UI Naming Convention:** Any front-end interface or web page title **MUST** be named **NewsPage** (no spaces).

### 4. Operational Directives
- Always ask for explicit user permission before running destructive bash commands or installing packages.
- Always log significant architecture changes and progress in `dev_log.md` for Git commits.

---

## 🌐 Network & Mirror Configuration
- **HF Mirror:** For model downloading in restricted/air-gapped network environments, use:
  ```python
  import os
  os.environ['HF_ENDPOINT'] = '[https://hf-mirror.com](https://hf-mirror.com)'

## 🚀 Quick Execution Commands

* **Run Backend**:
```bash
conda run -n rag_agent python app.py

```


* **Run Tunnel**:
```bash
conda run -n rag_agent python tunnel.py

```



---

## 🏗️ Project Architecture & Modules

* **`src/config.py`**: Centralized configuration (DeepSeek API endpoints, model routers, ChromaDB paths).
* **`src/pdf_loader.py`**: PDF text extraction and recursive character text splitting.
* **`src/vector_store.py`**: ChromaDB initialization and embedding management.
* **`src/rag_chain.py`**: Four-step RAG pipeline (Retrieve, Prompt, Call LLM, Stream/Chat).
* **`app.py`**: FastAPI application serving the **NewsPage** web interface and API routes.
* **`tunnel.py`**: ngrok public tunnel integration.
* **`dev_log.md`**: Detailed development and troubleshooting logs.

---

## ⚠️ Known Workarounds

* **`pyairports` Stub**: A local shim is placed in `pyairports/` to satisfy `vllm`'s `outlines` dependency in restricted environments. Do not delete this directory.