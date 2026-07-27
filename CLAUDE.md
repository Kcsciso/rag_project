# 🔴 系统红线与开发规则（STRICT CONSTRAINTS）

## 1. 硬件与 GPU 管理（双 A100 智能自适应）

- **算力底座**: 2 × NVIDIA A100-PCIE-40GB（CUDA 12.4）。
- **GPU 自适应策略**（Dynamic GPU Detection）:
  - **禁止硬编码 `CUDA_VISIBLE_DEVICES`**。启动脚本 `start_services.sh` 和 `src/config.py` 已内置 `nvidia-smi` 空闲显存扫描。
  - 自动选择**剩余显存最大的 GPU**（过滤空闲 < 5 GB 的 GPU）。
  - 手动覆盖方式：`--gpu <id>` 参数 或 `VLLM_GPU_ID` 环境变量。
- **默认分配**（自动检测无结果时的回退）:
  - GPU 1（端口 **8001**）：本地 vLLM 推理服务，当前模型 **Qwen2.5-3B-Instruct**（已缓存 ~4.6GB，`--gpu-memory-utilization 0.20`）。升级目标：**Qwen2.5-7B-Instruct-AWQ**（4-bit 量化 ~8GB，待下载权重 ~15GB）。
  - GPU 0：向量检索引擎（ChromaDB / PyTorch 嵌入计算）。**注意**：GPU 0 为多人共享，高峰期空闲仅 ~16 MB。
- **GPU 升级路径**：当前默认 `Qwen2.5-3B-Instruct`（已缓存）。升级至 7B AWQ 需先下载权重：
  ```bash
  VLLM_USE_MODELSCOPE=true python -c "from vllm import LLM; LLM(model='Qwen/Qwen2.5-7B-Instruct-AWQ', quantization='awq')"
  ```
  下载完成后将 `src/config.py` 中 MODEL_NAME 改为 `Qwen/Qwen2.5-7B-Instruct-AWQ`。空闲 < 5GB 时自动降级至 1.5B。
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
  - **Layer 1（主模型）**: `Qwen/Qwen2.5-3B-Instruct`（已缓存），`http://localhost:8001/v1`，GPU 自适应，`--gpu-memory-utilization 0.20`，`--max-model-len 8192`，`--enforce-eager`。升级目标：`Qwen/Qwen2.5-7B-Instruct-AWQ`（4-bit 量化，用 `VLLM_USE_MODELSCOPE=true` 从 ModelScope 下载）。
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

## 5. 轻量化幻觉防御策略（ADR-9/ADR-10，2026-07-23）

针对 1.5B 小模型在工业文档问答中的幻觉倾向，采用**不硬拦截、靠上下文工程**的柔性方案：

### 5.1 父子切片上下文扩展（Parent-Child Chunk Expansion）
- **函数**: `_expand_parent_sections()` in `src/rag_chain.py`
- **逻辑**: 检索命中子切片后，按 `[章节: X.Y.Z]` ID 从 ChromaDB 捞取同章节兄弟切片（最多 `max_siblings=2`），确保 LLM 获得完整流程上下文。
- **适用场景**: TCP 四点法、关机步骤、安全区域设置等多步骤长流程。
- **调用点**: `rag_chat()` / `rag_chat_stream()` / `llm_generation_node()` / `run_graph_stream()` — 全部 4 个入口均已接入。

### 5.2 柔性 Grounding 提示（Flexible Grounding）
- **触发条件**: query 匹配 `(默认|初始|预设).{0,6}(密码|端口|参数)` 等数字关键词，但 Context 中无 ≥2 位数字。
- **行为**: 在 Context 末尾追加 `[提示：参考切片中未包含确切的数字参数...切勿猜测 admin、502 等通用默认值]`。
- **原则**: **绝不硬拦截真实业务提问**（如"JAKA 版本升级"）。只有查询含数字请求且 Context 无对应数字时才追加柔性提示。

### 5.3 多轮对话 Citation 前缀清洗
- **逻辑**: `_build_messages()` 处理历史消息时，剥离 assistant 回复中的 `根据《X》第 Y.Z 节【...】` 长前缀。
- **目的**: 防止第二轮模型将上一轮的章节溯源复读为上下文幻觉。
- **正则**: `r'^(?:根据|参考|依据)《[^》]+》第\s*[\d.]+\s*节【[^】]*】[，,，]\s*'`

### 5.4 设计原则
- ✅ **不硬拦截**：取消 `_is_impossible_query()` 中的升级/固件正则（真实业务提问必须走正常检索）
- ✅ **相信检索**：章节标题自动注入 (`[章节: X.Y.Z 标题]`) + BM25 自动术语提取 → 让检索本身召回正确答案
- ✅ **Context 物理标注**：每个切片带 `【出处: 《文档名》】` + `[章节: X.Y.Z]` 双重标签
- ✅ **Token 预算硬控制**：3 切片 × 200 字符 + `max_tokens=384` → 确保 input+output < 4096
- ⚠️ **已知限制**：截图中的数字（如 JAKA 端口号 6502）不可提取（需 OCR 引擎，待后续升级）

## 5.5 LangGraph v2 后处理控制层（ADR-11，2026-07-24）

针对 1.5B 小模型在工业文档问答中的属性词颠倒/篡改问题，在 LangGraph 状态图中新增 2 个后处理节点 + 2 条条件边，构建零特定数字补丁的通用对齐与自纠错环路。

### 5.5.1 通用属性对齐节点（ExtractAlignNode）
- **节点**: `extract_align_node` in `src/graph_rag.py`
- **逻辑**: 从 Context 中通过通用属性词库（50+ 物理属性词）+ 正则扫描提取 KV 映射（`{"端口号": "6502"}`），在生成后逐数字校验模型输出中的属性词是否与 Context 原文一致。若模型将属性词颠倒（如把"端口号 6502"误写为"从站地址 6502"），用 Context 中的正确属性词硬改写。
- **原则**: 绝不硬编码 6502、9600 等特定数值 — 通过通用词库 + 正则实现通用免疫。

### 5.5.2 SDK 代码自纠错条件环路（SDK_VerifyNode）
- **节点**: `sdk_verify_node` in `src/graph_rag.py`
- **条件边**: `_route_after_llm` → `sdk_verify` | `extract_align`；`_route_after_sdk_verify` → `llm_generation`（回环）| `extract_align`
- **逻辑**: 检测生成代码是否缺失 `set_` 前缀、`ctypes.CDLL` 加载或 `argtypes` 声明。若发现问题 → 写入 `State["feedback"]` + 递增 `State["retry_count"]` → 条件边路由回 `llm_generation_node`（上限 2 次）。
- **智能豁免**: CDLL 已存在时不报 CDLL 缺失；argtypes 已存在时不报 argtypes 缺失；`(?<!set_)` 负向后顾排除已有 `set_` 前缀的正确写法。

### 5.5.3 System Prompt Few-Shot 强化
- 新增 Rule 12 在 `RAG_SYSTEM_PROMPT` 中，含 2 个 Few-Shot 示例：
  - 示例 1：端口属性精确归因（6502 → 端口号，不是"从站地址"或"设备标识符"）
  - 示例 2：步骤原文逐字复述（不添加"初始为红色"、"右上角"、"约 3 秒"等未记载细节）

### 5.5.4 AgentState v2 扩展
- `agent_state.py` → `RAGState` 新增 5 个后处理控制字段：
  - `extracted_entities: Dict[str, str]` — Context KV 映射
  - `feedback: str` — SDK 校验反馈文本
  - `retry_count: int` — 自纠错重试计数（0-2）
  - `context_text: str` — Context 原始文本拼接
  - `raw_llm_answer: str` — 未修改的 LLM 原始输出

### 5.5.5 新图结构
```
START → query_fusion → product_routing → hybrid_retrieval → llm_generation
                                                                  │
                                          ┌───────────────────────┼───────────────────────┐
                                          │                                               │
                                     sdk_verify                                     extract_align
                                          │                                               │
                                  ┌───────┴───────┐                                       ▼
                                  │               │                                      END
                            llm_generation   extract_align
                            (retry ≤ 2)         │
                                                ▼
                                               END
```

### 5.5.6 设计原则
- ✅ **零特定数字补丁**：通用属性词库 + 正则扫描，不硬编码任何具体数值
- ✅ **通用免疫**：对齐逻辑基于 Context KV 对比，而非特定产品/数字的 if-else
- ✅ **自纠错不中断**：SDK 重试上限 2 次，超限后放弃修复进入对齐，绝不卡死
- ✅ **所有既有 API 兼容**：`run_graph()` / `run_graph_stream()` 签名不变，`app.py` 零改动

## 5.6 Extract-Render 两层分离架构（ADR-12，2026-07-24）

针对 1.5B 小模型的根本局限——token 概率分布中预训练模式始终压倒 Context 信号——将生成管线从"自由文本生成"改为"提取 + 确定性渲染"两层分离。

- **System Prompt Extract Mode**: 要求模型输出 `【提取】{JSON}【提取结束】` 块，只做实体提取不写完整答案。
- **`render_node`** (`src/graph_rag.py`): 解析 JSON → 确定性渲染 ctypes 代码/编号步骤/强制引用 `根据《{doc}》【{section}】的记载：`。
- **降级容错**: JSON 解析失败或不存在 → 透传原始回答，不中断服务。
- **已知限制**: 1.5B 模型对 JSON 格式指令遵循不稳定；升级至 7B+ 模型预期可显著提升提取模式触发率。

## 5.7 Plan-Execute-Synthesize 三层架构（ADR-14，2026-07-25）

针对 7B 模型在工业文档 RAG 中的 4 类架构级缺陷——静态 KV 无泛化、单库路由拦截跨产品对比、代码实体 Embedding 湮灭、product_id=None 空响应——将 LangGraph 管线从线性链重构为"规划→并行执行→融合"三层架构。

### 5.7.1 架构设计

```
SubGoalPlanner → [Parallel: QA | Attribute | CodeSearch] → Synthesize
     ↑ Fast Path: 单产品简单查询 100% 绕过 Planner
```

**核心原则**:
- ✅ **Fast Path 零延迟**: 有明确 product_id 的单产品查询绝不触发 Planner
- ✅ **防崩兜底**: Planner Markdown 解析失败 → 自动降级标准单路检索
- ✅ **product_id=None 不反问**: CrossProductRetrievalNode 全库 Top-K 检索 + 综合回答
- ✅ **零硬编码**: 属性意图由 LLM 动态提取，CodeEntityAnchor 正则做软加权非硬过滤

### 5.7.2 新增节点（graph_rag.py）

| 节点 | 功能 |
|------|------|
| `subgoal_planner_node` | LLM 轻量调用（max_tokens=256），拆分子目标；Fast Path 判断内置 |
| `cross_product_retrieval_node` | product_id=None 时，对所有已注册产品并行检索 |
| `synthesize_node` | 多路结果融合：透传/对比/反问/综合 |

### 5.7.3 新增模块

| 文件 | 功能 |
|------|------|
| `src/attribute_tool.py` | 动态属性意图工具 — LLM 提取属性关键词 → BM25 精准搜索 → 正则提取值。替代静态 KV 正则表。 |
| `src/kv_extractor.py` | 离线属性提取器 — 从 ChromaDB 文本 + 手动校准构建 KV 属性 JSON。phase-out 中，逐步被 attribute_tool 替代。 |

### 5.7.4 v3 AgentState 扩展

`agent_state.py` → `RAGState` 新增 7 个 Plan-Execute-Synthesize 字段：
- `sub_goals: List[Dict]` — SubGoalPlanner 拆分的子目标列表
- `sub_results: List[Dict]` — 各子目标并行执行的结果
- `cross_product_candidates: List[Dict]` — 跨产品检索候选
- `attribute_intent: Dict` — 动态属性意图提取结果
- `code_entities: List[str]` — CodeEntityAnchor 提取的代码实体名
- `plan_mode: str` — "single" | "multi" | "cross_product" | "attribute"
- `skip_planner: bool` — Fast Path 标志

### 5.7.5 v3 图结构

```
START → query_fusion → product_routing
                          │
          ┌───────────────┼───────────────┐
          │               │               │
      chitchat/       clarify/        generate/
      refuse          (→ planner)     (→ planner)
          │                               │
          ▼                       ┌───────┴───────┐
   build_direct_response          │               │
          │                   plan_mode=      plan_mode=
          ▼                    single      cross_product
         END                       │               │
                                   ▼               ▼
                            hybrid_retrieval  cross_product_retrieval
                                   │               │
                                   ▼               ▼
                            llm_generation   llm_generation
                                   │               │
                                   └─── synthesize ──┘
                                            │
                              ┌─────────────┼─────────────┐
                              │                           │
                         sdk_verify                 extract_align
                              │                           │
                         (retry ≤ 2)                    END
                              │
                         extract_align → END
```

### 5.7.6 设计原则与已知限制

- ✅ **Fast Path 零开销**: skip_planner=True 时 SubGoalPlanner 直接返回，不调 LLM
- ✅ **防崩兜底**: Planner 解析失败 → 降级标准单路检索，绝不死锁
- ✅ **全库检索不反问**: CrossProductRetrievalNode 输出候选+反问，不 Only 反问
- ⚠️ **已知限制**: SubGoalPlanner 对 1.5B 模型的指令遵循不稳定，需 7B+ 模型充分释放能力；CodeEntityAnchor 的正则初筛需与 BM25 tokenizer 集成才能达到最大效果

## 5.8 切片机制架构升级（ADR-15，2026-07-25）

针对固定字符切片（chunk_size=300）导致的 SDK 函数签名割裂、章节锚点错位、KV 参数语义稀释三大问题，重构为 API 原子化 + 标题感知 + 父子双层索引的 v4 切片策略。

### 5.8.1 核心策略
- **API-Level Atomic Chunking**: 用正则预分割器识别函数定义边界（C 签名/Python ctypes/Markdown 代码块），标记为 `api_atomic=True`，永不切分
- **Header-Aware Chunking**: 沿标题层级树切分，每个切片注入完整面包屑 `[路径: H1 > H2 > H3]`，兼容数字编号（3.1.5）和非 Markdown 格式
- **Parent-Child Dual Indexing**:
  - Parent Collection (`rag_v4_parent`): H2 章节级粗粒度切片，用于粗召回
  - Child Collection (`rag_v4_child`): H3/H4 函数级精粒度切片，带 `metadata.function_names`，API 原子块完整保留

### 5.8.2 关键函数（pdf_loader.py）
| 函数 | 功能 |
|------|------|
| `_v4_extract_headings()` | 多格式标题识别（数字编号/中文序号/Markdown/纯数字） |
| `_v4_build_breadcrumb()` | 层级面包屑生成（非 Markdown 兼容） |
| `_v4_extract_api_blocks()` | SDK 函数原子块识别与保护 |
| `_v4_build_parent_child_docs()` | Parent-Child 双层 Document 构建 |
| `load_pdfs_v4_dual()` | v4 主入口，返回 `(parents, children)` |

### 5.8.3 双 Collection 检索（vector_store.py）
| 函数 | 功能 |
|------|------|
| `create_dual_collections()` | 创建 Parent + Child ChromaDB Collection |
| `search_dual_index()` | Child 优先 + Parent 批量反查（高效单次查询模式） |

### 5.8.4 配置
```python
CHUNK_MODE = "v4_dual"    # "v4_dual" | "v3_legacy"
PARENT_CHUNK_SIZE = 1000   # H2 章节级父层
CHILD_CHUNK_SIZE = 400     # H3/H4 函数级子层（API 原子）
```

## 5.9 多模态增量更新与 GPU 批量加速（ADR-16，2026-07-25）

针对全量覆盖 O(N) 重建、OCR 文本盲区、CPU Embedding 串行慢三大痛点，构建增量引擎 + 多模态 OCR + GPU 批处理。

### 5.9.1 增量引擎（vector_store.py）
| 函数 | 功能 |
|------|------|
| `upsert_product_documents()` | 增量 Upsert 核心入口：MD5 去重 → 级联删除 → OCR 解析 → GPU 嵌入 → BM25 同步 |
| `delete_product_chunks()` | 按 product_id 级联清理 Parent + Child Collection |
| `_init_md5_store_from_chroma()` | 系统重启时从 ChromaDB metadata 自动恢复 MD5 记录 |
| `_persist_md5_store()` | 将 MD5 记录持久化到 Collection metadata |
| `bm25_upsert_product()` | BM25 增量同步 — 仅重建受影响产品索引 O(n) |
| `bm25_remove_product()` | BM25 级联删除 |

### 5.9.2 OCR 图文抽取（pdf_loader.py）
| 函数 | 功能 |
|------|------|
| `_v4_get_ocr_engine()` | RapidOCR ONNX 引擎懒加载（~15MB 模型，纯 CPU） |
| `_v4_inject_ocr_text()` | 遍历 PDF 内嵌图片 → OCR 识别 → `[OCR识别]` 标签注入切片正文 |

### 5.9.3 GPU 批量加速
- `EMBEDDING_BATCH_SIZE = 64`: SentenceTransformer 自动 GPU 批处理，A100 上 10-40× 加速
- `load_vector_store_from_name()`: 按 Collection 名称加载（支持 v4 Parent/Child 分离访问）

### 5.9.4 API 改造
- `POST /api/upload`: 从全量重建改为增量 Upsert（MD5 去重 + 级联清理 + OCR + BM25 同步）
- 返回新增字段: `parents`, `children`, `ocr_images`, `deleted_old`, `status`

## 5.10 检索幻觉修复与产品隔离强化（2026-07-25）

针对 v4 上线后暴露的 API 捏造（`set_robot_connect`/`robot_move_arc`）、跨产品函数混淆、检索盲区三大问题。

### 5.10.1 function_names 元数据检索增强
- **`_match_function_names()`** (`rag_chain.py`): 逗号分隔字符串 ↔ query 代码实体的模糊匹配（strip/lower/子串），解决 list→str 后的检索失效
- **`_extract_query_code_entities()`** (`rag_chain.py`): 从 query 中提取代码实体模式，用于 RRF boost 和 function_names 匹配
- **RRF 加权**: 匹配 function_names 的 Child doc 获得 0.08 RRF 加分（高于普通锚点 0.05）

### 5.10.2 检索过滤放宽 + 安全网
- **关键词分阈值**: 从 0.05 降至 0.03，含 `function_names` 元数据或代码关键词（robot_/set_/ctypes）的切片豁免过滤
- **安全网**: `kept_docs` 全空时自动恢复向量 Top-3 保底（防止 LLM 空上下文后激活幻觉代码生成）
- **候选池扩大**: 代码实体查询 `fetch_factor=8`（→ ~40 候选），boost 重排后截断至 top-k

### 5.10.3 防幻觉硬拦截
- **三层判断** (`_build_messages`): 检查 context 中是否有 (a) 真实函数签名 (b) 操作步骤 (c) metadata function_names
- **`_force_no_code`**: 三条件全空时在 system prompt 头部注入硬指令："禁止代码/禁止编造函数名/只允许拒答/限制 50 字"
- **条件约束**: 仅 metadata 有函数名但文本中无 → 弱约束提示

### 5.10.4 产品推断与隔离
- **`_infer_product_from_query()`** (`vector_store.py`): 从 query 关键词自动推断 product_id
- **`search_similar_with_threshold()`**: product_id 为空时自动推断，无法推断时按产品分组标注
- **`_is_sdk_code_query()`** (`rag_chain.py`): 本地 SDK 代码查询检测（避免跨模块循环导入）

- 🔴 **测试红线（v4.2）**: 每次修改 `rag_chain.py` 或 `graph_rag.py` 后，必须先运行 `python tests/run_eval.py --verbose` 并确保 **100% PASS**（含 4 项硬质量断言），才能交付。禁止跳过。
- 旧测试脚本: `test_robot_rag.py` 和 `test_multidoc_simulation.py` 已删除，用例已合并至 `tests/eval_cases.json`。`test_rag_eval.py` / `test_human_simulation.py` / `test_unified_suite.py` / `test_stability.py` 保留向后兼容。
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
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --served-model-name Qwen/Qwen2.5-7B-Instruct-AWQ \
    --max-model-len 8192 \
    --port 8001 \
    --gpu-memory-utilization 0.25 \
    --quantization awq \
    --trust-remote-code \
    --enforce-eager
```

| 参数 | 值 | 理由 |
|------|-----|------|
| `CUDA_VISIBLE_DEVICES` | `1`（或自动检测结果） | 隔离 GPU，避免与其他用户进程冲突 |
| `--port` | **8001** | FastAPI 占用 7860，vLLM 使用独立端口 |
| `--gpu-memory-utilization` | `0.25` | 7B AWQ 4-bit ~8GB，25% 硬锁定，不霸占共享显存 |
| `--quantization` | `awq` | 4-bit 量化，显存从 28GB → 8GB |
| `--max-model-len` | `8192` | 7B 模型支持更长上下文，适配多切片+系统 Prompt |
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
| `src/pdf_loader.py` | **PDF 加载** — pypdf 逐页提取 → RecursiveCharacterTextSplitter 13 级递归分块 → **Header Injection**（自动提取 C 函数名 `[Functions: xxx]` 注入切片头部） |
| `src/vector_store.py` | **向量知识库** — bge-small-zh-v1.5 (512维) + ONNX 回退双轨嵌入；**BM25 稀疏检索**（jieba 分词 + 正则标识符保护 + 自定义 SDK 函数词典）；`search_similar_with_threshold()` 阈值过滤；`cleanup_vector_store()` 资源释放 |
| `src/rag_chain.py` | **RAG 核心管线 (v4.1)** — 四层金字塔容灾；`_hybrid_retrieve` BM25+向量 RRF 融合 + `_match_function_names()` 元数据 boost + `_extract_query_code_entities()` 实体提取 + 放宽关键词过滤 + `kept_docs` 安全网；`_build_messages()` 含 `_force_no_code` 防幻觉硬拦截；`_is_sdk_code_query()` 本地 SDK 检测 |
| `src/agent_state.py` | **LangGraph 状态定义 (v3)** — `RAGState` TypedDict（21 字段），含 5 个 v2 后处理控制 + 7 个 v3 Plan-Execute-Synthesize 字段 |
| `src/graph_rag.py` | **LangGraph 状态图引擎 (v3)** — 9 节点 + 5 条件边 + 自纠错环路；v3 新增 `subgoal_planner_node`、`cross_product_retrieval_node`、`synthesize_node`；`_extract_code_entities()` CodeEntityAnchor；`run_graph()` / `run_graph_stream()` API 完全兼容 |
| `src/attribute_tool.py` | **动态属性意图工具 (v3)** — LLM 提取属性关键词 → BM25 精准搜索 → 正则提取值。替代静态 KV 正则表 |
| `src/kv_extractor.py` | **离线属性提取器** — 从 ChromaDB 文本切片 + 手动校准数据构建 KV JSON 存储。phase-out 中 |
| `src/pdf_loader.py` | **PDF 加载与分块 (v4)** — 保留 v3 `load_pdfs_from_directory()`；v4 新增 `load_pdfs_v4_dual()` API 原子切分 + `_v4_inject_ocr_text()` OCR 图文抽取 + 标题面包屑 + Parent-Child 双层索引 |
| `src/vector_store.py` | **向量知识库 (v4)** — 双 Collection 管理、`search_dual_index()` 高效检索、`upsert_product_documents()` 增量 Upsert、`delete_product_chunks()` 级联删除、`bm25_upsert_product()` BM25 动态同步；v4.1 新增 `_match_function_names()` 模糊匹配、`_infer_product_from_query()` 产品推断、`_sanitize_metadata()` ChromaDB 类型清洗 |
| `rebuild_v4.py` | **v4 向量库重建脚本** — 独立运行，用 v4 切分策略重新解析所有 PDF 并写入 Parent + Child Collection |
| `app.py` | **FastAPI 主入口 (7860)** — 5 条路由 + 安全中间件；SSE 防泄露；`shutdown` 事件；产品路由（`product_id` + `GET /api/products`）|
| `src/multimodal_loader.py` | **多模态解析** — PyMuPDF + pdfplumber 表格→Markdown、图片→Caption 注入、智能路由（纯文本→标准 pypdf） |
| `frontend_server.py` | **前端 UI 服务 (8501)** — Jinja2 模板渲染 + `/api/*` 反向代理到 7860 后端 |
| `check_status.py` | **健康检查** — vLLM + FastAPI + GPU 显存/温度/功率 + 四层容灾可用性 + vLLM 部署 GPU 识别 |
| `start_services.sh` | **一键启动** — GPU 智能选择（`detect_best_gpu` stdout/stderr 严格隔离）+ 端口检测 + vLLM 后台拉起 + FastAPI 启动 + 优雅退出 |
| `tunnel.py` | **ngrok 隧道** — 公网穿透，authtoken 认证 |
| `test_robot_rag.py` | **功能回归测试** — 4 题 × 流式/非流式双模式 |
| `test_stability.py` | **稳定性压力测试** — 多轮对话 + 并发 + 7 种异常降级场景 |
| `test_human_simulation.py` | **人类模拟测试** — 5 类 14 用例：口语噪音、错别字、多轮指代、长文本组合、边界攻击 |
| `test_multidoc_simulation.py` | **多文档产品隔离测试** — 多产品 PDF 同时入库：产品打标、物理隔离检索、混合查询主动反问 |
| `test_rag_eval.py` | **防过拟合评测** — 8 用例 (核心 5 + 泛化 3)，验证产品路由、物理隔离、SDK 函数精确匹配、防幻觉 |
| `streamlit_app.py` | **Streamlit 前端** — 端口 8501 的备用前端 UI |

## 产品级物理隔离架构（ADR-6）

```
用户提问 ──────────────────────────────────────────────────┐
  │                                                        │
  ▼                                                        │
┌──────────────────────────────────────┐                    │
│ 第 0 步：产品意图路由 (Product Router) │                   │
│                                      │                   │
│  product_id 已提供？（前端强指定）     │                   │
│    ├── 是 → 直接使用                  │                  │
│    └── 否 → _resolve_product_from_query()              │
│              ├── 命中 → 锁定 product_id                 │
│              └── 未命中 → 反问澄清                     │
│                    "请问您询问的是哪一款产品呢？"         │
└──────────────────────────────────────┘                   │
  │                                                        │
  ▼ (product_id 已确定)                                    │
┌──────────────────────────────────────┐                    │
│ ChromaDB 混合检索（product_id 物理隔离）│                  │
│   where={"product_id": "OpenR6"}      │  ← 100% 单库隔离 │
│   向量召回 4× → 关键词重排序 → Top-K   │                   │
└──────────────────────────────────────┘                   │
  │                                                        │
  ▼                                                        │
  四层容灾 (Layer 1→2→3→4)
```

### 产品映射规则（入库阶段 — 文件名 → product_id）

定义于 `src/config.py` → `PRODUCT_MAPPING_RULES`：

| product_id | filename_patterns | content_keywords |
|------------|-------------------|------------------|
| OpenR6 | `["OpenR6", "openr6", "R6", "windows系统"]` | `["py_dll", "Robot_.*", "windows"]` |
| OpenC3 | `["OpenC3", "openc3", "六轴机械臂", "collrob", "六轴"]` | `["六轴", "collrob", "OpenC3", "机械臂"]` |

**动态扩展**：新增产品只需追加一条规则，无需更改任何核心逻辑代码。

### 产品路由器（查询阶段 — query → product_id）

定义于 `src/config.py` → `PRODUCT_ROUTER_RULES`：

| product_id | 关键词 | priority |
|------------|--------|----------|
| OpenR6 | `"OpenR6"`, `"py_dll"`, `"R6"`, `"windows"`, `"windows系统"` | 10 |
| OpenC3 | `"OpenC3"`, `"collrob"`, `"六轴"`, `"六轴机械臂"` | 10 |

### 产品路由相关函数

| 函数 | 位置 | 用途 |
|------|------|------|
| `_resolve_product_from_query(query)` | `src/rag_chain.py` | 从用户查询中动态识别产品 |
| `_build_clarification_response()` | `src/rag_chain.py` | 非流式澄清反问 |
| `_build_clarification_response_stream()` | `src/rag_chain.py` | 流式澄清反问（15 字符/块打字机） |
| `_resolve_product_id_from_filename()` | `src/pdf_loader.py` | 文件名→product_id 打标 |
| `resolve_product_id(filename)` | `src/vector_store.py` | 公开的产品识别 API |
| `get_registered_products()` | `src/vector_store.py` | 查询已入库的产品列表 |
| `clear_vector_store()` | `src/vector_store.py` | 彻底清空 ChromaDB（Collection API + 物理删除双方案）|

```
用户提问
  │
  ▼
产品路由器 (Product Router) → product_id 物理隔离
  │
  ▼
BM25 + 向量混合检索（RRF 融合 + Autocut 动态截断）
  │  ├─ Dense: bge-small-zh-v1.5 向量召回 (fetch_factor=5)
  │  ├─ Sparse: BM25 jieba+正则 关键词召回
  │  ├─ RRF 融合排序 (K=60)
  │  └─ Autocut 动态截断 [2, 8]
  │
  ├── Layer 1: 本地 vLLM (_resolve_vllm_model 动态解析, :8001)
  │     • 超时: connect=5s / read=60s (推理) / read=5s (健康检查)
  │     • 采样: temperature=0.3, repetition_penalty=1.2, max_tokens=1024
  │     • 并发保护: threading.Lock + 预检 _check_vllm_health()
  │     └── 失败 → Layer 2
  │
  ├── Layer 2: 智谱 GLM-4.7-Flash (open.bigmodel.cn)
  │     • 主通道已是智谱时自动跳过（_FALLBACK_ENABLED）
  │     └── 失败 → Layer 3
  │
  ├── Layer 3: 纯向量检索智能直出 (CPU-only, 零显存/零API)
  │     • 行级归一化去重 → 结构化输出 → 模拟流式
  │     └── 失败 → Layer 4
  │
  └── Layer 4: 优雅中文错误提示
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
# 一键启动
./start_services.sh                         # 智能 GPU 检测 → vLLM → FastAPI
./start_services.sh --vllm-only             # 仅启动 vLLM
./start_services.sh --fastapi-only          # 仅启动 FastAPI

# 一键停止
pkill -f "app.py"; pkill -f "vllm.entrypoints"   # 或定义: alias stoprag='pkill -f app.py; pkill -f vllm'

# 健康检查
python check_status.py                      # 一次性完整报告
python check_status.py --watch 10           # 每 10 秒刷新

# 自动化测试
python test_human_simulation.py             # 全场景人类模拟（14 用例 × 5 类别）
conda run -n rag_agent python test_robot_rag.py      # RAG 功能回归（4 题 × 双模式）
conda run -n rag_agent python test_stability.py       # 稳定性压力测试（多轮 + 并发）
```

---

# 📋 当前生产配置摘要

```python
# Layer 1: 本地 vLLM
BASE_URL     = "http://localhost:8001/v1"
MODEL_NAME   = "Qwen/Qwen2.5-7B-Instruct-AWQ"

# Layer 2: 智谱 GLM-4.7-Flash（云端降级）
DEEPSEEK_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEEPSEEK_MODEL    = "glm-4.7-flash"

# LLM 超时（vLLM 预检 + 从容超时）
# _check_vllm_health()     → 预检 GET /v1/models（connect=3s/read=5s 独立短超时）
# _resolve_vllm_model()    → 动态获取 vLLM 实际模型名，缓存后用于所有 LLM 调用
LLM_TIMEOUT = httpx.Timeout(connect=5.0, read=15.0, write=15.0, pool=5.0)
LLM_INFERENCE_TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=15.0, pool=5.0)
_VLLM_HEALTH_TIMEOUT = httpx.Timeout(connect=3.0, read=5.0, write=5.0, pool=3.0)

# 检索参数（ADR-7 BM25 混合 + ADR-8 Autocut）
CHUNK_SIZE           = 300
CHUNK_OVERLAP        = 50
RETRIEVAL_K          = 8     # 最大保留数（Autocut 上限）
SIMILARITY_THRESHOLD = 0.68
DIRECT_RETRIEVAL_K   = 2
_AUTOCUT_MIN_K       = 2     # Autocut 下限（防止切太狠）
_AUTOCUT_MAX_K       = 8     # Autocut 上限（防止撑爆上下文）

# 🔴 混合检索 (ADR-7): BM25 + Vector RRF 融合
# _hybrid_retrieve() 内置：
#   ① Dense: bge-small-zh-v1.5 向量召回（fetch_factor=5）
#   ② Sparse: BM25 jieba+正则 关键词召回
#   ③ RRF 融合排序 (K=60)
#   ④ _autocut_knee() 断崖检测 → 动态截断 [2, 8]
#   ⑤ Header Injection: [Functions: xxx] 自动注入切片头部

# 嵌入模型
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"  # 512 维中文专优
FALLBACK_TO_ONNX     = True
HF_HUB_OFFLINE       = "1"  # 离线模式，模型已缓存

# 产品隔离（ADR-6）
# PRODUCT_MAPPING_RULES  → 入库时文件名打标 → product_id 写入 ChromaDB metadata
# PRODUCT_ROUTER_RULES   → 查询时意图识别 → product_id 物理隔离检索

# 部署端口映射
# FastAPI 后端: 7860 (内) → 50003 (外)
# Frontend UI:  8501 (内) → 50004 (外)
# vLLM 推理:    8001 (内)

# Web 服务
HOST = "0.0.0.0"
PORT = 8000

# API 路由
# GET  /                → NewsPage 主页面
# POST /api/chat        → RAG 对话（新增 product_id 表单参数，支持流式 SSE）
# POST /api/upload      → 上传 PDF 并重建向量库（上传前清空旧库，返回 product_distribution）
# GET  /api/status      → 向量库状态
# GET  /api/products    → 已注册产品列表 [新增]

# 嵌入模型
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FALLBACK_TO_ONNX     = True

# GPU 自适应
VLLM_GPU_ID = <auto-detected>  # 环境变量覆盖: export VLLM_GPU_ID=1
MIN_FREE_MEMORY_MIB = 5120     # 最低空闲显存门槛（5 GB）
```
