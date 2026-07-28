# 🔴 系统红线与开发规则（STRICT CONSTRAINTS）

## 1. 硬件与 GPU 管理（双 A100 智能自适应）

- **算力底座**: 2 × NVIDIA A100-PCIE-40GB（CUDA 12.4）。
- **GPU 自适应**: 禁止硬编码 `CUDA_VISIBLE_DEVICES`。`start_services.sh` + `src/config.py` 内置 `nvidia-smi` 扫描，自动选择空闲显存最大的 GPU（过滤 <5GB）。手动覆盖：`--gpu <id>` / `VLLM_GPU_ID`。
- **默认**: GPU 1（:8001）→ vLLM Qwen2.5-7B-Instruct-AWQ (4-bit ~8GB)；GPU 0 → ChromaDB/嵌入。
- **降级**: 空闲<5GB → 自动降级 1.5B。

## 2. 核心依赖红线（严禁升级）

Conda `rag_agent` (Python 3.10)。**严禁 `pip install --upgrade`**：
`torch==2.6.0+cu124` `vllm==0.16.0` `sentence-transformers==2.7.0`
允许新增：`pypdf` `langchain-chroma` `rank-bm25`

## 3. RAG 架构与 AI 生态

- **框架**: LangChain、LangGraph、ChromaDB。**LLM**: vLLM Qwen2.5-7B-Instruct-AWQ @ :8001；云端降级 glm-4.7-flash。
- **嵌入**: BAAI/bge-small-zh-v1.5 (512维) → ONNX 自动回退。
- **UI 命名**: **比邻星 (ProximaRAG)**。
- **测试红线**: 修改 `rag_chain.py`/`graph_rag.py` 后必须运行 `python tests/run_eval.py --verbose`，8 项硬断言全部通过方为合格。

## 4. 安全开发红线

- 输入/文件名清洗、Prompt 注入防御（role 白名单）、历史≤100条、查询≤2000字符、SSE 资源管理、shutdown_clients()。
- 严禁删除 `site-packages/pyairports/` Shim 适配层。保持 `README.md` `CLAUDE.md` `dev_log.md` 与代码库同步。

## 5. 架构演进摘要（ADR-6~ADR-17 → v15）

| ADR | 版本 | 核心内容 | 关键函数/文件 |
|-----|------|---------|-------------|
| 6 | v1 | 产品物理隔离 | `_resolve_product_from_query()`, `PRODUCT_ROUTER_RULES` |
| 7-8 | v1 | BM25+向量 RRF 混合检索 + Autocut 动态截断 | `_hybrid_retrieve()`, `_autocut_knee()`, `_AUTOCUT_MIN_K=3` |
| 9-10 | v2 | 柔性 Grounding + 父子切片扩展 + Citation 清洗 | `_expand_parent_sections()`, `_build_messages()` |
| 11 | v2 | LangGraph 后处理 (ExtractAlign + SDK_Verify 自纠错) | `extract_align_node`, `sdk_verify_node` (graph_rag.py), retry 硬熔断 |
| 14 | v3 | Plan-Execute-Synthesize 三层架构 | `subgoal_planner_node`, `cross_product_retrieval_node` |
| 15 | v4 | API 原子切分 + Header 感知 + Parent-Child 双层索引 | `load_pdfs_v4_dual()`, `create_dual_collections()` |
| 16 | v4 | 增量 Upsert + OCR 图文抽取 + GPU 批量加速 | `upsert_product_documents()`, `_v4_get_ocr_engine()` |
| 17 | v5 | 双轨制 (gui_app/c_sdk) + 8 项硬断言 | `_resolve_doc_type()`, `_extract_sdk_header()` |
| v9-11 | v9-11 | 切片架构重构: I/O归一化/面包屑4槽/状态机/sdk_header解耦/GUI完整保留/复合拆解/400拦截/历史净化 | `_v4_build_breadcrumb()`, `_v4_parse_sdk_state_machine()`, `sanitize_chat_history()` |
| v12-13 | v12-13 | 裁Context保输出/骨架过滤/标题清洗/下划线归一化 | `_is_skeleton_chunk()`, `_sanitize_section_title()`, `_clean_pdf_text()` Step 4.3 |
| v14-15 | v14-15 | 历史沉渣净化+反跨产品门控+边界合并+Health Score 91.8 | `_anti_bleed_prefix`, `_MIN_BLOCK_GAP=30` |

### 当前关键配置

| 参数 | 值 | 说明 |
|------|-----|------|
| max_tokens | 2200 | v13: 保代码完整，裁Context不减输出 |
| _AUTOCUT_MIN_K | 3 | v11: 硬下限3，多步骤SDK不丢切片 |
| _MAX_CONTEXT_CHARS | 2000 | v11: 总Context字符硬上限 |
| CHILD_CHUNK_SIZE | 400 | H3/H4 函数级子层 |
| PARENT_CHUNK_SIZE | 1000 | H2 章节级父层 |
| CHUNK_MODE | v4_dual | Parent+Child 双层索引 |
| SIMILARITY_THRESHOLD | 0.68 | 向量检索阈值 |
| MAX_HISTORY_TURNS | 3 | 滑动窗口 |

---

# 🚀 本地服务启动

```bash
./start_services.sh                    # 一键启动 (GPU 智能检测 → vLLM → FastAPI)
./start_services.sh --vllm-only        # 仅 vLLM
./start_services.sh --fastapi-only     # 仅 FastAPI
```

**手动**:
```bash
# 终端 A: vLLM
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ --port 8001 --gpu-memory-utilization 0.25 \
    --max-model-len 8192 --enforce-eager --quantization awq

# 终端 B: FastAPI
python app.py   # → http://localhost:8000 (比邻星 ProximaRAG) · API: /docs
```

---

# 🏗️ 项目架构

| 文件 | 功能 |
|------|------|
| `src/config.py` | 全局配置 — 双通道 LLM、GPU 探测、ChromaDB 路径、嵌入模型、检索参数 |
| `src/pdf_loader.py` | PDF 加载(v4) — 状态机 SDK 解析、GUI Heading-to-Heading、Parent-Child 双层、OCR、下划线归一化、骨架过滤 |
| `src/vector_store.py` | 向量知识库 — bge-small-zh-v1.5 + ONNX 回退、BM25 混合检索、增量 Upsert |
| `src/rag_chain.py` | RAG 核心管线 — 四层容灾、混合检索、HyDE、双轨 Prompt、反泄露门控、历史净化 |
| `src/graph_rag.py` | LangGraph 状态图引擎 — 9 节点 + 条件边 + SDK 自纠错 + 硬熔断 |
| `src/agent_state.py` | RAGState TypedDict (21 字段) |
| `src/attribute_tool.py` | 动态属性意图 — LLM 提取→BM25→正则 KV |
| `app.py` | FastAPI (:8000) — /api/chat, /api/upload, /api/status, /api/products |
| `frontend_server.py` | 前端 UI (:8501) — Jinja2 + /api/* 反向代理 |
| `rebuild_v4.py` | v4 向量库重建脚本 |
| `check_status.py` | 健康检查 — vLLM + FastAPI + GPU |
| `start_services.sh` | 一键启动 — GPU 智能选择 + 就绪轮询 + 优雅退出 |
| `audit_chunks.py` | 切片健康度审计 (Health Score) |

### 关键函数索引

| 函数 | 位置 | 用途 |
|------|------|------|
| `_v4_parse_sdk_state_machine()` | pdf_loader.py | SDK 轨 4 类边界状态机 |
| `_sanitize_section_title()` | pdf_loader.py | 标题清洗器 |
| `_is_skeleton_chunk()` | pdf_loader.py | 离线骨架过滤 |
| `_clean_pdf_text()` | pdf_loader.py | 7 步通用文本清洗 |
| `_hybrid_retrieve()` | rag_chain.py | BM25+向量 RRF 混合检索 |
| `_decompose_compound_query()` | rag_chain.py | 复合查询拆解 (顺序连接词) |
| `_build_messages()` | rag_chain.py | Prompt 组装 + 双轨控制 + 反泄露门控 |
| `sanitize_chat_history()` | rag_chain.py | 历史沉渣净化中间件 |
| `_ensure_code_blocks_closed()` | rag_chain.py | 代码块自动闭合 Guardrail |
| `_call_llm()` / `_stream_llm()` | rag_chain.py | LLM 调用 + 400 拦截 + Context 裁切 |

---

# 📋 当前生产配置

```python
# LLM
BASE_URL     = "http://localhost:8001/v1"
MODEL_NAME   = "Qwen/Qwen2.5-7B-Instruct-AWQ"
DEEPSEEK_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEEPSEEK_MODEL    = "glm-4.7-flash"
# max_tokens=2200 (v13: 裁Context保输出)
LLM_INFERENCE_TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=15.0, pool=5.0)

# 检索
CHUNK_SIZE=300 / CHUNK_OVERLAP=50 / RETRIEVAL_K=10 / SIMILARITY_THRESHOLD=0.68
_AUTOCUT_MIN_K=3 / _AUTOCUT_MAX_K=5
CHUNK_MODE = "v4_dual"  # Parent(1000) + Child(400)

# 嵌入
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"  # 512维, HF→ONNX 回退

# 端口
# FastAPI :8000 | vLLM :8001 | Frontend :8501
```

### 运维命令

```bash
./start_services.sh              # 一键启动
pkill -f "app.py"; pkill -f "vllm"  # 一键停止
python check_status.py           # 健康检查
python tests/run_eval.py --verbose  # 回归评测
python audit_chunks.py           # 切片健康度审计
```
