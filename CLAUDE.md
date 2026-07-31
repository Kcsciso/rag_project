# 🔴 系统红线与开发规则（STRICT CONSTRAINTS）

## 1. 硬件与 GPU 管理（双 A100 智能自适应）

- **算力底座**: 2 × NVIDIA A100-PCIE-40GB（CUDA 12.4）。
- **GPU 自适应**: 禁止硬编码 `CUDA_VISIBLE_DEVICES`。`start_services.sh` + `src/config.py` 内置 `nvidia-smi` 扫描，自动选择空闲显存最大的 GPU（过滤 <5GB）。手动覆盖：`--gpu <id>` / `VLLM_GPU_ID`。
- **默认**: GPU 1（:8001）→ vLLM Qwen2.5-7B-Instruct-AWQ (4-bit ~8GB)；GPU 0 → ChromaDB/嵌入。
- **降级**: 空闲<5GB → 自动降级 1.5B。

## 0. 🔴 四层架构排雷法（AI 辅助开发思想钢印 — v21 新增）

**任何代码修改前，必须先声明该修改属于哪一层，并自检是否会破坏该层的核心特性。**

### L1 — 数据摄入与切片层 (pdf_loader.py)
| 核心特性 | 严禁破坏 |
|----------|---------|
| `_v4_extract_headings()` 代码注释拦截 | 8 特征词 ±120 字符上下文校验 |
| `_sanitize_section_title()` 伪标题黑名单 | frozenset 10 项 + 触发父级 H2 继承 |
| `_SDK_BLOCK_BOUNDARY_RE` 两类可验证边界 | 数字标题 + 函数名称/函数说明 |
| `_is_skeleton_chunk()` 骨架过滤 | 150 字符 + 代码特征词 + 实质参数三重判定 |
| `_clean_pdf_text()` 7 步清洗 | Step 6 SDK 代码换行修复不可省略 |
| Micro-Chunk Auto-Merge API 排他锁 | `_extract_primary_api_name()` 提取后不同 API 不合并 |
| 4 级 Title Fallback 链 | L1 状态机标题 → L2 面包屑 → L3 父级 H2 → L4 硬兜底 |
| 受保护区域 | 代码块 (```) + Markdown 表格 绝不拦腰切断 |

### L2 — 检索与重排层 (vector_store.py + rag_chain.py `_hybrid_retrieve`)
| 核心特性 | 严禁破坏 |
|----------|---------|
| RRF 四大提权引擎 | Entity Anchor (+0.05) / Function Names (+0.08) / Text Rebalance (+0.03) / CODE BM25 三倍写入 |
| 三层保底召回 | 阈值空 → 原始 Top-3 → kept_docs 恢复 |
| Autocut 断崖检测 | `_autocut_knee()` — RRF 分数相邻差值 Knee Point; `_AUTOCUT_MIN_K=8` (SDK=10), `_AUTOCUT_MAX_K=15` |
| 复合查询拆解 | `_decompose_compound_query()` + `_MIN_SUB_QUERY_LEN=2` — 两字核心动词不丢弃 |
| LLM 意图重写 (ADR-19) | `_rewrite_query_with_llm()` 代词消解+产品补全 — 禁止回退正则缝合 |
| BM25 标识符保护 | `_IDENTIFIER_RE` 正则预提取 → jieba 不拆蛇形函数名 |
| HyDE 防毒化 | SDK 轨 (OpenC3/OpenR6) + 短 Query (<6ch) + 精确 API 签名 → 全部禁用 |
| `[CODE:xxx]` 标签 | BM25 tokenizer 三倍写入实现 Boost=3.0 |
| 跨产品检索阈值一致性 | 禁止在 `cross_product_retrieval_node` 中硬编码不同于全局 `SIMILARITY_THRESHOLD` 的值 |

### L3 — 上下文组装与指令层 (rag_chain.py `_build_messages` + `RAG_SYSTEM_PROMPT`)
| 核心特性 | 严禁破坏 |
|----------|---------|
| 双轨制 Prompt | gui_app: 首句强制红线 + 绝对禁止代码 / c_sdk: SDK 两段式排版铁律 (首句出处+唯一代码块) |
| `_term_alignment_prefix` 动态术语对齐 | 仅在命中特定产品+同义词对时按需注入 (如 OpenR6 "使能"→`set_robot_arm_init`)，零全局 Token 损耗 |
| `_anti_bleed_prefix` 反跨产品泄露 | metadata function_names + 正文双重确认 → 仅目标缺失 + 非目标有 API 时注入 |
| Context Cap 整块剔除 | 从末尾 Parent 优先丢弃，不切割任何 Chunk 内部正文 |
| 历史沉渣净化 | `sanitize_chat_history()` + Citation 前缀清洗 + 代码块替换 + 尾部拒答剥离 |
| 柔性 Grounding 提示 | `_NUMERIC_QUERY_RE` 动态检测 → Context 无数值时追加诚实提示 |
| System Prompt 篇幅约束 | 新增规则时必须评估 Token 成本，当前上限 ~2000 tokens; 术语对齐规则已剥离至动态注入，严禁回填至全局 Prompt |
| `_last_numeric_context_missing` 线程安全 | 禁止在非请求作用域外读写此变量（已知并发 unsafe，待修复为 State 字段） |

### L4 — 生成控制与后处理层 (graph_rag.py 后处理节点 + rag_chain.py LLM 调用)
| 核心特性 | 严禁破坏 |
|----------|---------|
| SDK 两段式排版铁律 (ADR-22) | `_dual_track_prefix` 强制 "首句出处说明 + 唯一整合代码块"，`_dll_name` 基于 product_id 精确推断 |
| 静默斩尾 `_strip_hedging_tail()` | 8 模式 regex — "上述代码假设存在"/"参考文档未包含详细步骤" 等 |
| `_fix_and_close_sdk_code()` | Markdown 反引号闭合 + CDLL 智能补全（需 product_id 精确判定 DLL） |
| `extract_align_node` 属性词硬改写 | 50+ 领域属性词库 + 数值前后 12+8 字符窗口 |
| SDK 自纠错硬熔断 | `retry_count >= 2 → skip`（入口检测 + 循环检测 双保险） |
| `_stream_guardrail()` 伪流式 | 已知问题：全量缓冲导致 TTFB 退化，待修复为增量检查 |
| NEVER-EMPTY 保证 | 所有 4 层 + 流式/非流式双路径均覆盖终极兜底 |
| Temperature 策略 | 非流式 t=0.2 / 流式 t=0.01（代码近确定性输出） |

### 跨层数据流约束
- **LangGraph 管线优先**: `app.py` → `run_graph`/`run_graph_stream`，`rag_chat`/`rag_chat_stream` 为废弃内部 fallback
- **并发安全**: 模块级可变全局变量 (`_last_numeric_context_missing`, `_HYDE_CACHE`) 不保证线程安全，新逻辑优先使用 State 字段或请求作用域局部变量
- **Vector Store 注入**: 通过 `set_graph_vector_store()` 统一注入，禁止节点内直接 import ChromaDB 客户端绕过

---

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
| 7-8 | v1 | BM25+向量 RRF 混合检索 + Autocut 动态截断 | `_hybrid_retrieve()`, `_autocut_knee()`, `_AUTOCUT_MIN_K=4`, `_AUTOCUT_MAX_K=10` |
| 9-10 | v2 | 柔性 Grounding + 父子切片扩展 + Citation 清洗 | `_expand_parent_sections()`, `_build_messages()` |
| 11 | v2 | LangGraph 后处理 (ExtractAlign + SDK_Verify 自纠错) | `extract_align_node`, `sdk_verify_node` (graph_rag.py), retry 硬熔断 |
| 14 | v3 | Plan-Execute-Synthesize 三层架构 | `subgoal_planner_node`, `cross_product_retrieval_node` |
| 15 | v4 | API 原子切分 + Header 感知 + Parent-Child 双层索引 | `load_pdfs_v4_dual()`, `create_dual_collections()` |
| 16 | v4 | 增量 Upsert + OCR 图文抽取 + GPU 批量加速 | `upsert_product_documents()`, `_v4_get_ocr_engine()` |
| 17 | v5 | 双轨制 (gui_app/c_sdk) + 8 项硬断言 | `_resolve_doc_type()`, `_extract_sdk_header()` |
| v9-11 | v9-11 | 切片架构重构: I/O归一化/面包屑4槽/状态机/sdk_header解耦/GUI完整保留/复合拆解/400拦截/历史净化 | `_v4_build_breadcrumb()`, `_v4_parse_sdk_state_machine()`, `sanitize_chat_history()` |
| v12-13 | v12-13 | 裁Context保输出/骨架过滤/标题清洗/下划线归一化 | `_is_skeleton_chunk()`, `_sanitize_section_title()`, `_clean_pdf_text()` Step 4.3 |
| v16-17 | v16-17 | QueryFusion指代词门控/HyDE防毒化/Search-First软路由/确定性反问/首句章节Python注入/套话擦除 | `_search_first_soft_route()`, `build_product_clarification_response()`, `_strip_hedging_tail()` |
| v18 | v18 | 🔴 代码注释污染治理: Heading上下文拦截/伪标题黑名单/状态机净化/Golden TOC预留 | `_v4_extract_headings()` 代码注释拦截, `_sanitize_section_title()` 伪标题黑名单, `_v4_extract_sdk_toc()` |
| v19 | v19 | LLM Query Rewriting 意图重写引擎 (ADR-19) | `_rewrite_query_with_llm()`, `REWRITE_SYSTEM_PROMPT`, 废弃 `_fuse_short_query`/`_resolve_clarification_followup`/`_has_business_intent` |
| v20-22 | v20-22 | 四轮闭环重构 (ADR-20/21/22): 复合查询子任务阈值 4→2 / Autocut SDK 防误杀 / 动态术语对齐 / SDK 两段式排版铁律 | `_MIN_SUB_QUERY_LEN=2`, `_AUTOCUT_MIN_K=8`(SDK=10), `_AUTOCUT_MAX_K=15`, `_term_alignment_prefix`, `_dual_track_prefix` 两段式 |
| **当前** | **v22** | max_tokens=1024, MAX_HISTORY_TURNS=2, _MAX_CONTEXT_CHARS=4000(SDK=8000), _AUTOCUT_MIN_K=8(SDK=10), _AUTOCUT_MAX_K=15, _MIN_SUB_QUERY_LEN=2 | — |

### 当前关键配置

| 参数 | 值 | 说明 |
|------|-----|------|
| max_tokens | 1024 | v17: 代码+步骤完全充裕，从源头消解 vLLM 400 |
| _AUTOCUT_MIN_K | 8 | v22: 硬下限8，SDK 检索动态提升至 10 |
| _AUTOCUT_MAX_K | 15 | v22: 上限15，承载多参数/多步骤 SDK 切片 |
| _MIN_SUB_QUERY_LEN | 2 | v22: 复合查询最小子句长度，两字动词不丢弃 |
| _MAX_CONTEXT_CHARS | 4000 / 8000(SDK) | v20: 非SDK 4000 / SDK 8000，配合 Autocut 满载 |
| CHILD_CHUNK_SIZE | 400 | H3/H4 函数级子层 |
| PARENT_CHUNK_SIZE | 1000 | H2 章节级父层 |
| CHUNK_MODE | v4_dual | Parent+Child 双层索引 |
| SIMILARITY_THRESHOLD | 0.68 | 向量检索阈值 |
| RETRIEVAL_K | 10 | 单次检索召回数 |
| MAX_HISTORY_TURNS | 2 | v16: 滑动窗口 2 轮=4 条消息 |
| LLM_INFERENCE_TIMEOUT | connect=10.0, read=120.0, write=15.0, pool=5.0 | v20: 匹配 7B AWQ 多切片推理 |
| _VLLM_LOCK_TIMEOUT | 120.0s | v20: 对齐 inference read timeout |
| _temperature (stream) | 0.01 | v20: 极紧温度，代码近确定性输出 |
| _temperature (non-stream) | 0.2 | 非流式保持低随机性 |

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
| `_v4_parse_sdk_state_machine()` | pdf_loader.py | SDK 轨状态机 API 块解析器（`数字标题` + `函数名称/函数说明` 两类边界） |
| `_v4_extract_headings()` | pdf_loader.py | 标题提取 + 🔴 代码注释拦截（8 特征词上下文校验） |
| `_sanitize_section_title()` | pdf_loader.py | 标题清洗器 + 🔴 伪标题黑名单（10 项 frozenset） |
| `_v4_extract_sdk_toc()` | pdf_loader.py | 🔴 Golden TOC 目录树预解析（预留回退基础设施） |
| `_is_skeleton_chunk()` | pdf_loader.py | 离线骨架过滤 |
| `_clean_pdf_text()` | pdf_loader.py | 7 步通用文本清洗 + 🔴 Step 6 SDK 代码换行修复 |
| `_hybrid_retrieve()` | rag_chain.py | BM25+向量 RRF 混合检索 |
| `_decompose_compound_query()` | rag_chain.py | 复合查询拆解 (顺序连接词) |
| `_build_messages()` | rag_chain.py | Prompt 组装 + 双轨控制 + 反泄露门控 + 🔴 SDK Context Cap 4000 |
| `sanitize_chat_history()` | rag_chain.py | 历史沉渣净化中间件 |
| `_fix_and_close_sdk_code()` | rag_chain.py | 🔴 代码块自动闭合 + CDLL 补全（替代已删除的 `_ensure_code_blocks_closed`） |
| `_call_llm()` / `_stream_llm()` | rag_chain.py | LLM 调用 + 400 拦截 + Context 裁切 |

### 🔴 PDF 切片规则 (v18)

#### SDK 状态机边界触发条件 (`_SDK_BLOCK_BOUNDARY_RE`)

```
仅两路可验证边界:
  ① ^\d{1,2}[\.\、\s]\s*\S+       → "28. 机械臂电源上电" / "4. 机械臂上电"
  ② ^(?:函数名称|函数说明)\s*      → OpenC3/OpenR6 两种 API 表头格式
```

**严格禁止**匹配的模式：`^#{1,4}\s+`（Python 注释 `# 时间等待3秒` 与 Markdown 标题无法区分，已从边界正则中永久移除）。

#### Heading 代码注释拦截 (`_v4_extract_headings`)

所有 `#` 开头的候选标题需通过 **±120 字符上下文校验**：
```python
_CODE_KEYWORDS = ['restype', 'argtypes', 'CDLL', 'ctypes', 'robot.', 'c_int', 'c_float', 'import ']
```
任一关键词出现在上下文窗口 → 该 `#` 行被判定为代码注释 → 拒绝提权为 Heading。

#### 伪标题黑名单 (`_PSEUDO_SECTION_BLACKLIST`)

```python
frozenset({"时间等待", "命令发送", "示例代码", "代码示例", "调用示例",
           "参数说明", "返回值", "功能描述", "函数说明", "注意事项", "备注"})
```
标题清洗后长度 < 15 字符 且 包含黑名单关键词 → 返回 `""` → 调用方自动继承父级 H2 标题。

#### Golden Section 继承机制

当 `_sanitize_section_title()` 返回空字符串时：
- `_v4_build_child_docs_v2` c_sdk 路径：回退到 `breadcrumb` 路径信息
- `_emit_child` 非 SDK 路径：回退到 `_parent_title`（父级 H2 标题）
- **预留**: `_v4_extract_sdk_toc()` 已实现 Golden TOC 映射引擎，待接入空值回退链路

#### Micro-Chunk Auto-Merge 阈值

```python
_MIN_BLOCK_GAP = 20     # 边界邻近合并窗口（字符）
# 合并后文本 ≥ 60 字符 或 包含代码特征词 → 提交为独立 API 块
```

---

# 📋 当前生产配置

```python
# LLM
BASE_URL     = "http://localhost:8001/v1"
MODEL_NAME   = "Qwen/Qwen2.5-7B-Instruct-AWQ"
DEEPSEEK_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEEPSEEK_MODEL    = "glm-4.7-flash"
# max_tokens=1024 (v16: 代码+步骤完全充裕，从源头消解 vLLM 400)
LLM_INFERENCE_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=15.0, pool=5.0)

# 检索
CHUNK_SIZE=300 / CHUNK_OVERLAP=50 / RETRIEVAL_K=10 / SIMILARITY_THRESHOLD=0.68
_AUTOCUT_MIN_K=8 / _AUTOCUT_MAX_K=15  # SDK 检索时 MIN_K 动态提升至 10
_MIN_SUB_QUERY_LEN=2  # v22: 复合查询最小子句长度，两字动词不丢弃
CHUNK_MODE = "v4_dual"  # Parent(1000) + Child(400)
_MAX_CONTEXT_CHARS = 4000  # SDK 检索时动态提升至 8000

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
