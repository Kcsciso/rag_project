# 🔴 系统红线与开发规则（STRICT CONSTRAINTS）

## 1. 硬件与 GPU 管理（双 A100 智能自适应）

- **算力底座**: 2 × NVIDIA A100-PCIE-40GB（CUDA 12.4）。
- **GPU 自适应**: 禁止硬编码 `CUDA_VISIBLE_DEVICES`。`start_services.sh` + `src/config.py` 内置 `nvidia-smi` 扫描，自动选择空闲显存最大的 GPU（过滤 <5GB）。手动覆盖：`--gpu <id>` / `VLLM_GPU_ID`。
- **默认**: GPU 1（:8001）→ vLLM Qwen2.5-7B-Instruct-AWQ (4-bit ~8GB)；GPU 0 → ChromaDB/嵌入。
- **降级**: 空闲<5GB → 自动降级 1.5B。

## 0. 🔴 四层架构排雷法（AI 辅助开发思想钢印 — v24 更新）

**任何代码修改前，必须先声明该修改属于哪一层，并自检是否会破坏该层的核心特性。**

### L1 — 数据摄入与切片层 (pdf_loader.py)
| 核心特性 | 严禁破坏 |
|----------|---------|
| `_v4_extract_headings()` 代码注释拦截 | 8 特征词 ±120 字符上下文校验 |
| 🔴 `_v4_extract_headings()` doc_type 动态双轨拦截 (v23) | gui_app 轨绝对禁止单数字编号(1. 2.)提权为标题，防操作步骤碎裂 |
| 🔴 `_V4_HEADING_PATTERNS` 多级数字编号 (v23) | `{1,5}` 支持最高 6 级深度标题 (3.1.5.2.1)，末尾带点兼容 |
| `_sanitize_section_title()` 伪标题黑名单 | frozenset 10 项 + 触发父级 H2 继承 |
| `_SDK_BLOCK_BOUNDARY_RE` 两类可验证边界 | 数字标题 + 函数名称/函数说明 |
| `_is_skeleton_chunk()` 骨架过滤 | 150 字符 + 代码特征词 + 实质参数三重判定 |
| `_clean_pdf_text()` 7 步清洗 | Step 6 SDK 代码换行修复不可省略；🔴 v25: JAKA/gui_app 数字保护特判（≥3 位参数保全，C-SDK 轨原逻辑不变） |
| 🔴 OCR 面积过滤 (v26) | gui_app 轨废除 `<100px` 硬过滤 → 放置矩形面积比（<1.5% 或边长 <18pt）；C-SDK 轨原逻辑不变 |
| 🔴 OCR 页尾追加 (v27) | **回退 v26 CTM Y 归位**（PDF 坐标系不一致污染切片）；OCR 块 `[本页图片解析参数补充]` + last_header 继承安全追加页尾；低密度页 OCR 文本同步更新标题追踪器 |
| 🔴 区域状态机标题提取 (v28) | `_v4_extract_headings` 必须 consult `_v4_find_protected_ranges`（代码块/表格/OCR 补充块内匹配跳过）；OCR 补充块锚定 `\n\n` 页分隔入保护区 |
| 🔴 line 级表格重建 (v28) | gui_app 轨必须用 `get_text("dict")` line bbox 按 y 聚类（`round(y/12)`）+ x 排序；仅 ≥2 项且单元格 ≤40 字符的带包装为 Markdown 表格行；C-SDK 轨 block 级逻辑严禁改动 |
| 🔴 last_header 层级栈 (v28) | last_header 仅接受数字编号/章节编号标题（裸字不入栈）；弹栈 = 层级不降或编号前缀不匹配；数字编号标题形态负向校验（首字符 `\|` 或仅数字/点/竖线 → 拒绝） |
| 🔴 OCR 键值法 (v29) | gui_app OCR 输出必须键值归一（`端口：\| 6502` → `端口：6502`）+ 跨行配对（`从站节点号：` + 纯数值）+ 按图子块化（`[图表内容包含：]` 前缀）；`_PROTECTED_BLOCK_RE` 第三分支与标记同步 |
| 🔴 图片过滤重构 (v29) | 过滤下限 = 面积 ≥0.5% 且边长 ≥40px（禁止恢复 1.5% 面积比）；必须 xref 全局去重 + 放置 >20 页跳过（页眉 logo） |
| 🔴 跨页表头继承 (v30) | gui_app 轨暂存 `_table_header`（每页首个 `\|` 行）→ 下页首行若为 `\|` 则强制注入；C-SDK 轨零触碰 |
| 🔴 OCR 标签化 (v30) | gui_app OCR 输出必须 `<OCR_BLOCK>...</OCR_BLOCK>` 包裹；`_PROTECTED_BLOCK_RE` 第二分支匹配；`_v4_find_protected_ranges` `group(2)` → `type="ocr"` |
| 🔴 软装箱算法 (v30) | Parent 截断前按 `_PROTECTED_BLOCK_RE` 解析文本为 普通/受保护 交替序列；受保护块整体装入允许超标；普通文本 `\n\n` 安全切断；受保护块**禁止物理切断** |
| 🔴 OCR 子块隔离 (v30) | gui_app child 切片中 `<OCR_BLOCK>` 强制 `_emit_ocr_child()` → `chunk_type="ocr_child"`；禁止混入普通 child |
| 🔴 孤儿行合并 (v30.final) | `_row_texts` 构建：`len(_cells)==1` 且 `<10ch` → 追加到上一行表格末单元格；修复单元格内换行碎裂 |
| 🔴 全景快照 OCR (v30.final) | gui_app 轨废弃 `get_images()` 逐图抠取 → `get_pixmap(Matrix(2,2))` 整页截图；矢量图盲区归零；逐图循环 `if doc_type=="gui_app": continue` 跳过 |
| 🔴 OCR 智能去重 (v30.final) | OCR 文本去空格后 vs `page_text` 交叉比对 → 已存在则丢弃；仅保留 PyMuPDF 未提取的增量参数 |
| 🔴 Y 轻量分组 (v30.final) | OCR 行 ≤12 行/组（`_GROUP_SIZE=12`），每组独立 `<OCR_BLOCK>`；防 giant block 撑爆向量窗口 |
| 🔴 OCR 前置 (v30.final) | gui_app OCR 块插入 `page_text` 最前方（非页尾追加）；C-SDK 轨 OCR 追加逻辑严禁触碰 |
| Micro-Chunk Auto-Merge API 排他锁 | `_extract_primary_api_name()` 提取后不同 API 不合并 |
| 4 级 Title Fallback 链 | L1 状态机标题 → L2 面包屑 → L3 父级 H2 → L4 硬兜底 |
| 受保护区域 | 代码块 (```) + Markdown 表格 绝不拦腰切断 |
| 🔴 动态切片容量分配 (v23) | GUI/JAKA: Child=1500, Parent=2000; SDK: Child=400, Parent=1000 |
| 🔴 父级标题跨级扫描 (v23) | `lv <= parent_level` 防止章节级标题(H1)被丢弃；前导文字自动保护 |
| 🔴 跨级大纲扫描终点 (v23) | TOC 扫描至下一个同级/更高级标题，H1 章节完整囊括子章节 |
| 🔴 微缩大纲上限 (v23) | Child TOC 上限 **5 条** (v22: 15)，超出显示 "... (更多章节略)" |
| 🔴 大纲标签统一 (v23) | 全部使用 `[章节大纲参考]:`，禁止旧 `【子章节】` / `[本章/本节包含以下子内容大纲]` |

### L2 — 检索与重排层 (vector_store.py + rag_chain.py `_hybrid_retrieve`)
| 核心特性 | 严禁破坏 |
|----------|---------|
| RRF 六大提权引擎 | Entity Anchor (+5.0) / Function Names (+0.08) / Text Rebalance (+0.03) / CODE BM25 三倍写入 / Title Exact Match (+5.0) / Chapter Isolation (+20.0/-10.0) |
| 三层保底召回 | 阈值空 → 原始 Top-3 → kept_docs 恢复 |
| Autocut 断崖检测 | `_autocut_knee()` — RRF 分数相邻差值 Knee Point; `_AUTOCUT_MIN_K=8` (SDK=10), `_AUTOCUT_MAX_K=15` |
| 复合查询拆解 | `_decompose_compound_query()` + `_MIN_SUB_QUERY_LEN=2` — 两字核心动词不丢弃 |
| LLM 意图重写 (ADR-19) | `_rewrite_query_with_llm()` 代词消解+产品名补全 — 禁止回退正则缝合；🔴 v26: **always-on**（无历史也执行）+ 规则 5/6/7（同音纠错/名词补全/注入旁路）+ 3 组泛化 Few-Shot + max_tokens=128；🔴 v27: 规则 9 产品名缺失保持缺失 + 路由责任切分（原始 query 优先 + 单轮澄清守卫 + `_resolve_product_from_history` 历史扫描）；🔴 v28: 规则 3 实体指代泛化（产品/函数/动作类型/参数）+ 补全实体逐字来自历史限定；🔴 v29: 规则 2 协议主题中立性限制 + `_PROTOCOL_TERMS_RE` 确定性兜底（重写后剥产品名） |
| BM25 标识符保护 | `_IDENTIFIER_RE` 正则预提取 → jieba 不拆蛇形函数名 |
| 🔴 复合词原子化 (v26) | `_COMPOUND_RE` 斜杠/连字符专有名词（Ethernet/IP、Modbus-RTU）整体 token **追加**（只增不删）；**排除 `.`** 防吞 `robot.set_move_line`；`_SPACE_SEP_RE` 空格归一化双侧对称 |
| 🔴 动态 BM25 权重 (v27) | `_BM25_WEIGHT = 3.0 if (len(query) <= 8 or _COMPOUND_RE.search(query)) else 1.2` — 短文本/复合词查询 Dense 漂移风险高，BM25 字面信号更可靠 |
| HyDE 防毒化 | SDK 轨 (OpenC3/OpenR6) + 🔴 JAKA (v23) + 短 Query (<6ch) + 精确 API 签名 → 全部禁用 |
| 🔴 GUI 噪声过滤豁免 (v23) | `_is_gui` (product_id==JAKA 或 doc_type==gui_app) → 完全豁免 kw_score<0.03 拦截 |
| 🔴 宏观提权 v2 (v23) | 广谱关键词扩展 ("内容/总结/介绍/大意/结构") + 双重判定 (chunk_type=="parent" 或正文含 `[章节大纲参考]`) |
| `[CODE:xxx]` 标签 | BM25 tokenizer 三倍写入实现 Boost=3.0 |
| 跨产品检索阈值一致性 | 禁止在 `cross_product_retrieval_node` 中硬编码不同于全局 `SIMILARITY_THRESHOLD` 的值 |

### L3 — 上下文组装与指令层 (rag_chain.py `_build_messages` + `RAG_SYSTEM_PROMPT`)

🔴 **v24 核心架构变更: Markdown 模板强约束 (Template Masking)**

| 核心特性 | 严禁破坏 |
|----------|---------|
| 🔴 **System Prompt 极简原则 (v24)** | `RAG_SYSTEM_PROMPT` 严格控制在 **~250 tokens** 以内。禁止向 System Prompt 追加新规则——所有格式约束走 `_dual_track_prefix` 模板。 |
| 🔴 **模板底端锚定 (v24)** | `_dual_track_prefix` 必须置于 **User Message 的末尾**（紧邻模型输出的前一个 token），利用 Recency Bias 实现注意力锚定。禁止将模板移至 System Prompt 或 User Message 中部。 |
| 🔴 **槽位填充模式 (v24)** | gui_app: 首句出处声明 + `[填写步骤]` 槽位 / c_sdk: 首句出处声明 + `[准确函数名]([参数])` 槽位。模板中的 `[填写xxx]` 标记是给小模型的认知提示——禁止删除或改为自由文本。 |
| 🔴 **双轨制模板 (v24 增强)** | gui_app: 六条铁律 (宏观总结/结构清晰/历史隔离/视觉屏蔽/禁止脑补/禁止代码) / c_sdk: SDK 两段式排版铁律 (首句出处+唯一代码块)。两条轨道有独立的输出格式模板。 |
| 🔴 **逃生舱条款 (v25)** | 双轨模板末尾追加：上下文无对应函数/硬件模块/参数/视觉等超纲内容，或触发【🚫 跨产品 API 隔离】警告 → LLM 必须彻底无视排版模板，仅输出 `_ESCAPE_REFUSAL` 一句拒答。**禁止**用业务正则做拒答判定——拒答决策完全交给 LLM 阅读条款 |
| 🔴 **逃生舱视觉加固 (v26)** | 逃生条款用 `> [!WARNING] ⛔🔴 绝密拦截` GitHub Alert 引用块语法（Qwen 训练语料识别度高）；**尾部对冲行已删除**（"请基于以上参考资料…请明确说明"）——模板即消息尾部，逃生指令获得极致 Recency Bias；铁律 3 与 `_ESCAPE_REFUSAL` 逐字一致 |
| 🔴 **模板选择守卫 (v27)** | L3 层三条件（A: query 点名函数不在 Context 函数集合（先 BM25 第二机会）/ B: 非 SDK 产品+SDK 问法 / C: coverage 句式+跨领域技术强词零命中）→ 双轨模板整体替换为拒答模板 + 回删 SDK Header。**非 L4 拦截，纯模板选择**；禁止添加第四条件（业务词表） |
| 🔴 **守卫 context 脱敏 (v28)** | 守卫命中路径必须应用 `_strip_code_from_context()`（``` 代码块 → `[代码内容省略]`、DLL 加载行 → `[DLL加载代码省略]`）——模型无代码可抄；正常路径零触碰 |
| 🔴 **Fast-Path 确定性拒答 (v29)** | `_build_messages` 必须返回 `(messages, refusal_flag)` 侧信道（**禁止**用模块级标志——FastAPI 线程池并发竞态）；守卫命中 → 四调用方在生成金字塔**之前**短路直出 `_HARD_REFUSAL`（含 Layer 3 泄漏封堵）；每次重建 messages 后重读 flag |
| 🔴 **数字守卫豁免 (v29)** | 守卫入参必须 `_SPACE_SEP_RE` 归一化 + `_COMPOUND_RE` 剥离后再跑 `_NUMERIC_QUERY_RE` 与关键词判定（剥离串同时用于两处）；剥离绝不污染真实 query |
| 🔴 **Top-1 来源锚定 (v24)** | `_doc_section_str` 仅取排名第一的章节 (`_sections[0]`)。禁止拼接多个章节名为大杂烩来源声明——单一锚点降低小模型认知负担。 |
| `_term_alignment_prefix` 动态术语对齐 | 仅在命中特定产品+同义词对时按需注入 (如 OpenR6 "使能"→`set_robot_arm_init`)，零全局 Token 损耗 |
| `_anti_bleed_prefix` 反跨产品泄露 | metadata function_names + 正文双重确认 → 仅目标缺失 + 非目标有 API 时注入 |
| Context Cap 整块剔除 | 从末尾 Parent 优先丢弃，不切割任何 Chunk 内部正文 |
| 历史沉渣净化 | `sanitize_chat_history()` + Citation 前缀清洗 + 代码块替换 + 尾部拒答剥离 |
| 柔性 Grounding 提示 | `_NUMERIC_QUERY_RE` 动态检测 → Context 无数值时追加诚实提示 |
| `_last_numeric_context_missing` 线程安全 | 禁止在非请求作用域外读写此变量（已知并发 unsafe，待修复为 State 字段） |

### L4 — 生成控制与后处理层 (graph_rag.py 后处理节点 + rag_chain.py LLM 调用)

🔴 **v24 核心架构变更: L4 从"擦屁股"简化为"兜底校验"**

| 核心特性 | 严禁破坏 |
|----------|---------|
| 🔴 **render_node 极简透传 (v24)** | `render_node` 退化为纯文本透传——**禁止**向其中添加 JSON 解析或正则清洗逻辑。格式正确性由 L3 的模板约束保证，render_node 只负责传递。 |
| 🔴 **流式极速穿透 (v24)** | `_stream_guardrail` 直接逐 chunk 透传，**绝对禁止**全量缓冲后再重新分块。TTFB 必须 <2s。任何需要在完整输出后才能执行的后处理逻辑必须移至流结束后的非关键路径。 |
| 🔴 **围栏闭合状态机 (v25)** | `_stream_guardrail` 透传中仅用 2 字符 carry 统计 ``` 奇偶性（零缓冲代价），流结束奇数时自动补发闭合行。`run_graph_stream` Layer 1/2 必须包裹守卫 |
| 🔴 **L4 正则最小化原则 (v24)** | `extract_align_node` 中禁止新增"屠魔版"正则清洗规则。L4 的职责是 KV 实体对齐校验 + SemanticDedup + 静默斩尾——不再试图纠正本应由模板约束预防的格式错误。 |
| SDK 两段式排版铁律 (ADR-22) | `_dual_track_prefix` 强制 "首句出处说明 + 唯一整合代码块"，`_dll_name` 基于 product_id 精确推断 |
| 静默斩尾 `_strip_hedging_tail()` | 8 模式 regex — "上述代码假设存在"/"参考文档未包含详细步骤" 等 |
| `_fix_and_close_sdk_code()` 过渡期兜底 | v24 标注为"过渡期兜底"，函数名修正表不再膨胀。模板约束生效后逐步缩减修正规则；🔴 v25: 闭合兜底接入 `extract_align_node` 覆盖 Graph 全路径（此前仅 legacy rag_chat） |
| SemanticDedup (v25 重构) | 无条件精确段落去重（连续相同 ≥80 字符段落移除，与 eval ② 定义一致）；含代码块的回答跳过模糊 trigram 去重。修复 kv_entities 为空时去重完全跳过的 Bug。注: v23 的 JAKA 豁免已由 v24 移除，v25 维持取消豁免 |
| SDK 自纠错硬熔断 | `retry_count >= 2 → skip`（入口检测 + 循环检测 双保险）。v24 模板约束下触发频率预期大幅下降 |
| NEVER-EMPTY 保证 | 所有 4 层 + 流式/非流式双路径均覆盖终极兜底 |
| Temperature 策略 | 非流式 t=0.2 / 流式 t=0.01（代码近确定性输出） |

### 跨层数据流约束
- **LangGraph 管线优先**: `app.py` → `run_graph`/`run_graph_stream`，`rag_chat`/`rag_chat_stream` 为废弃内部 fallback（v25 目标正式移除）
- **并发安全**: 模块级可变全局变量 (`_last_numeric_context_missing`, `_HYDE_CACHE`) 不保证线程安全，新逻辑优先使用 State 字段或请求作用域局部变量
- **Vector Store 注入**: 通过 `set_graph_vector_store()` 统一注入，禁止节点内直接 import ChromaDB 客户端绕过
- 🔴 **模板与后处理的边界 (v24)**: L3 模板负责**预防**格式错误，L4 后处理负责**兜底校验**。禁止在 L4 中为 L3 模板应预防的错误打补丁——发现格式错误应追溯到 L3 模板设计缺陷

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

## 5. 架构演进摘要（ADR-6~ADR-24）

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
| v23 | v23 | 🔴 GUI 轨专项攻坚: 双轨标题拦截 / 动态切片扩容 / 大纲降噪 / 章节隔离+标题强匹配提权 / GUI Prompt 六铁律 / L4 物理清洗引擎 / SemanticDedup 豁免 / HyDE JAKA 封杀 / GUI 噪声豁免 | `_v4_extract_headings()` +doc_type, `CHILD_CHUNK_SIZE=1500`(GUI), Chapter Isolation +20.0/-10.0, Title Exact Match +5.0, GUI Prompt 六条铁律, L4 5 道物理清洗正则 |
| 🔴 **v24** | **v24** | **Markdown 模板强约束 (Template Masking) + 极速流式穿透: System Prompt 极简瘦身 (210→15行) / 模板底端锚定 (Recency Bias) / `_doc_section_str` Top-1 来源 / `_stream_guardrail` 零缓冲透传 / `render_node` 退化为文本透传 (废弃JSON解析) / L4 正则最小化 / `run_graph_stream` 双重输出Bug修复** | `RAG_SYSTEM_PROMPT` 重写, `_dual_track_prefix` 模板, `_stream_guardrail` 透传, `render_node` 退化, `extract_align_node` 简化 |
| 🔴 **v25** | **v25** | **回归攻坚 (Regression Hardening): 围栏闭合状态机 (透传+```奇偶计数自动补发) / 双轨模板逃生舱条款 (拒答交 LLM 自主判定，零业务正则) / JAKA-gui_app 数字保护特判 (≥3 位参数保全) / OCR Y 坐标行对齐 (GUI 轨) / KV 属性注入放宽 (数字意图即触发, `_NUMERIC_QUERY_RE` 模块级) / SemanticDedup 无条件精确段落去重 (E04) / 闭合兜底接入 Graph 全路径 (E10/E17/E18/E23)** | `_stream_guardrail` 状态机, `_ESCAPE_REFUSAL` + `_dual_track_prefix` 逃生舱, `_clean_pdf_text` JAKA 特判, `lookup_attribute` tie-break, `extract_align_node` 精确去重 |
| 🔴 **v26** | **v26** | **最后一公里根治: OCR 放置面积比过滤 + CTM 矩阵 Y 归位 (截图参数插回对应章节，切片边界不变) / BM25 复合词原子化与空格归一化 (`Ethernet/IP` 整体 token，排除 `.` 防吞 snake_case，双侧对称) / 重写器 always-on + 同音纠错/名词补全规则与 Few-Shot (E18/E28) / 逃生舱 `> [!WARNING]` 视觉加固 + 删除尾部对冲行 (纯 Prompt 零 L4)** | `_px_to_page_point()`, `_COMPOUND_RE`/`_SPACE_SEP_RE`, `REWRITE_SYSTEM_PROMPT` 规则 5/6/7, `_dual_track_prefix` 视觉块 |
| 🔴 **v27** | **v27** | **回归反转: 产品路由责任切分 (原始 query 优先 + 单轮澄清守卫 + `raw_query` State + 历史扫描第三兜底 + coverage 例外, E01) / OCR 回退页尾追加 (`[本页图片解析参数补充]` 标识, 回退 CTM Y 归位防切片污染) / SDK 模板物理隔离 (删除 `import ctypes` 字面样例, `_extract_sdk_header` 修复裸 CDLL 提取) / L3 模板选择守卫 (函数级/产品级/超纲级三条件 → 拒答模板, E09/E21/E25) / 去重规范化 (忽略空白与尾标点) / 动态 BM25 权重 (短文本/复合词 3.0, E28/E29)** | `_refusal_override` 守卫, `_COVERAGE_QUERY_RE`/`_TECH_STRONG_TERMS_RE`, `_resolve_product_from_history`, `_extract_sdk_header` 修复, `_BM25_WEIGHT` 动态化 |
| 🔴 **v28** | **v28** | **切片状态机化: 标题提取区域状态机 (`_v4_extract_headings` consult 受保护区域, OCR 补充块锚定 `\n\n` 入保护区, 根治 309 污染路径) / gui_app line 级几何表格重建 (`get_text("dict")` y 聚类, ≥2 项短单元格带包装 Markdown 表格行) / last_header 数字编号层级栈 (前缀祖先校验修跨章叠加) / 数字编号形态负向校验 (`0.000 \| 0.000` 拒绝) / `第N 章` 空格兼容 / TOC 点线目录特征过滤 (E29) / 守卫命中 context 代码脱敏 (`_strip_code_from_context`) / 重写规则 3 实体指代泛化 (E17 动作指代)** | `_PROTECTED_BLOCK_RE` 第三分支, `_try_update_header` 层级栈, `_strip_code_from_context`, REWRITE 规则 3 |
| 🔴 **v29** | **v29** | **数据语义化 + 确定性拒答: OCR 键值法 (`端口：\|6502`, 跨行配对, 按图子块化) / 图片过滤重构 (0.5%/40px 下限 + xref 去重 + 放置次数) / 数字守卫复合词豁免 (Ethernet/IP 的 IP 不误杀) / Fast-Path 确定性拒答 (`_build_messages` 返回侧信道, 守卫命中跳过 LLM 直出 `_HARD_REFUSAL`, 封堵 Layer 3 泄漏) / 重写协议主题中立性 (规则 2 限制 + Few-Shot + `_PROTOCOL_TERMS_RE` 确定性兜底)** | `_ocr_kv_normalize_row`/`_ocr_merge_cross_line`, `_guard_query` 剥离, `(messages, refusal_flag)` 侧信道, `_PROTOCOL_TERMS_RE` |
| 🔴 **v30** | **v30** | **AST-Lite 软装箱: 跨页表格表头向下继承 (暂存 `_table_header` → 下页注入) / OCR 标签化防稀释 (`<OCR_BLOCK>` 包裹 + `_PROTECTED_BLOCK_RE` 第二分支 + `_v4_find_protected_ranges` `type="ocr"`) / Parent 软装箱算法 (受保护块不可分割 → 整体装入允许超标, 普通文本 `\n\n` 安全切断) / OCR 子块隔离 (`_emit_ocr_child` + `chunk_type="ocr_child"` 独立切片) / 过早封箱 Bug 修复 (`_packed_len >= parent_chunk_size` 条件封箱)** | `_table_header`, `<OCR_BLOCK>`, `_PROTECTED_BLOCK_RE` 第二分支, `_emit_ocr_child()`, `chunk_type="ocr_child"` |
| 🔴 **v30.final** | **v30.final** | **全景快照 OCR + 孤儿行合并: 孤儿行合并 (短文本 <10ch 追加到上一行表格末单元格, 修复 `Windows7及以上\n上` 碎裂) / 全景快照 OCR (`get_pixmap(Matrix(2,2))` 整页截图 → 废弃 `get_images()` 逐图抠取, 矢量图盲区归零) / 智能去重 (OCR 文本去空格后 vs `page_text` 交叉比对, 仅保留增量) / Y 轻量分组 (≤12 行/组, 防 giant block) / OCR 前置 (`page_text` 最前方插入, 旧逻辑页尾追加撕裂跨页上下文) / C-SDK 逐图跳过 (`if doc_type=="gui_app": continue`)** | `_pix = page.get_pixmap()`, `_page_text_norm` 去重, `_GROUP_SIZE=12`, OCR prepend, 孤儿行合并 `elif len(_cells)==1` |

### 当前关键配置

| 参数 | 值 | 说明 |
|------|-----|------|
| max_tokens | 1024 | v17: 代码+步骤完全充裕，从源头消解 vLLM 400 |
| _AUTOCUT_MIN_K | 8 | v22: 硬下限8，SDK 检索动态提升至 10 |
| _AUTOCUT_MAX_K | 15 | v22: 上限15，承载多参数/多步骤 SDK 切片 |
| _MIN_SUB_QUERY_LEN | 2 | v22: 复合查询最小子句长度，两字动词不丢弃 |
| _MAX_CONTEXT_CHARS | 4000 / 8000(SDK) | v20: 非SDK 4000 / SDK 8000，配合 Autocut 满载 |
| CHILD_CHUNK_SIZE | 400 / 1500(GUI) | 🔴 v23: H3/H4 函数级子层; GUI 轨扩容至 1500 防止长步骤断裂 |
| PARENT_CHUNK_SIZE | 1000 / 2000(GUI) | 🔴 v23: H2 章节级父层; GUI 轨同步扩容 |
| CHUNK_MODE | v4_dual | Parent+Child 双层索引 |
| SIMILARITY_THRESHOLD | 0.68 | 向量检索阈值 |
| RETRIEVAL_K | 10 | 单次检索召回数 |
| MAX_HISTORY_TURNS | 2 | v16: 滑动窗口 2 轮=4 条消息 |
| LLM_INFERENCE_TIMEOUT | connect=10.0, read=120.0, write=15.0, pool=5.0 | v20: 匹配 7B AWQ 多切片推理 |
| _VLLM_LOCK_TIMEOUT | 120.0s | v20: 对齐 inference read timeout |
| _temperature (stream) | 0.01 | v20: 极紧温度，代码近确定性输出 |
| _temperature (non-stream) | 0.2 | 非流式保持低随机性 |

### 🔴 v24 新增关键约束

| 约束 | 说明 |
|------|------|
| **System Prompt Token 预算** | `RAG_SYSTEM_PROMPT` 严格 ≤ **250 tokens**。所有格式约束走模板，不走 System Prompt |
| **模板底端锚定** | `_dual_track_prefix` 必须在 User Message **末尾**（模型输出的前一个 token 位置） |
| **Top-1 来源** | `_doc_section_str` 仅取 `_sections[0]`，禁止多章节拼接 |
| **流式零缓冲** | `_stream_guardrail` 禁止全量缓冲，必须逐 chunk 透传 |
| **render_node 纯透传** | 禁止向 render_node 添加 JSON 解析或正则清洗 |
| **L4 正则最小化** | `extract_align_node` 禁止新增正则清洗规则——格式问题应追溯到 L3 模板设计 |

### 🔴 v25 新增关键约束

| 约束 | 说明 |
|------|------|
| **围栏闭合状态机** | `_stream_guardrail` 逐 chunk 透传中统计 ``` 奇偶性，流结束奇数时自动补发闭合行；禁止回退到全量缓冲 |
| **逃生舱条款** | `_dual_track_prefix` 模板末尾必须保留【逃生舱条款】；**禁止**用业务正则做拒答判定（不写 `_USAGE_QUERY_RE`/`_COVERAGE_QUERY_RE` 类规则），拒答决策完全交给 LLM 阅读条款 |
| **JAKA 数字保护特判** | `_clean_pdf_text` 孤立数字删除仅限 1-2 位（页码），且仅对 `gui_app`/JAKA 生效；C-SDK 轨 `^\s*\d+\s*$` 原逻辑严禁改动 |
| **OCR 行对齐范围** | Y 聚类 + X 排序仅限 `gui_app` 轨；C-SDK 轨 OCR 处理严禁触碰 |
| **KV 注入放宽** | `_NUMERIC_QUERY_RE` 为模块级常量（graph_rag 与 rag_chain 共用）；数字意图查询即尝试 KV 属性注入，不依赖 Context 缺失守卫 |
| **精确段落去重** | `extract_align_node` 无条件执行连续相同段落（≥80 字符）去重；含代码块回答跳过模糊 trigram 去重 |
| **闭合兜底全路径** | `_fix_and_close_sdk_code` 已在 `extract_align_node` 入口接入——Graph 非流式路径必须保有代码闭合能力 |

### 🔴 v26 新增关键约束

| 约束 | 说明 |
|------|------|
| **OCR 面积过滤** | gui_app 轨图片过滤必须用放置矩形面积比（<1.5% 或边长 <18pt），**禁止**恢复 `<100px` 硬过滤；C-SDK 轨过滤逻辑严禁改动 |
| **CTM Y 归位** | OCR 行归位必须经 `_px_to_page_point()`（`get_image_info` transform 优先）；禁止把 OCR 文本一股脑追加页尾（gui_app 轨）；归位只改页面内容流，**禁止**触碰标题树/切片边界 |
| **复合词原子化** | `_COMPOUND_RE` 只允许 `/` 与 `-` 分隔符，**禁止**加入 `.`（防吞 `robot.set_move_line`）；复合 token 只能追加（只增不删），子段 token 必须保留 |
| **重写器 always-on** | `_rewrite_query_with_llm` 禁止恢复"无历史跳过"短路；`REWRITE_SYSTEM_PROMPT` 必须保留纠错/补全/注入旁路三条规则及其 Few-Shot |
| **逃生舱纯 Prompt** | 逃生条款必须用 `> [!WARNING]` 引用块语法且位于 User Message 最底部；**禁止**在消息尾部追加任何对冲性指令（如"如果不足以回答请明确说明"）；**禁止**任何 L4 if/else 拒答拦截 |

### 🔴 v27 新增关键约束

| 约束 | 说明 |
|------|------|
| **路由责任切分** | 产品路由必须主判原始 query（`raw_query` State）、辅判重写 query；单轮 + 原始无产品名 + 非 coverage 提问 → 直接澄清；**禁止**重写器越权补产品名（REWRITE 规则 9） |
| **coverage 例外** | `_COVERAGE_QUERY_RE` 命中（有没有/是否提到）→ 不得澄清，进 generation 由 L3 模板守卫拒答——E01 澄清与 E21 拒答的相反需求靠此区分 |
| **OCR 页尾追加** | gui_app 轨 OCR 块必须用 `[本页图片解析参数补充]` 标识页尾追加；**禁止**恢复 CTM Y 归位（v26 已实证污染切片）；面积比过滤保留 |
| **SDK 模板隔离** | SDK 轨模板**禁止**再出现 `import ctypes`/`ctypes.CDLL` 字面样例（诱导源）；CDLL 加载行唯一来源 = `_sdk_header_injected`（`_extract_sdk_header` 已修复裸 `CDLL(r...)` 提取） |
| **模板选择守卫** | 守卫三条件（函数级/产品级/超纲级）禁止增加第四条件（业务词表）；命中 → 拒答模板 + 回删 SDK Header；条件 A 必须保留 BM25 第二机会防漏召回误拒 |
| **动态 BM25 权重** | `_BM25_WEIGHT` 动态化仅限 `len(query) <= 8 or _COMPOUND_RE.search(query)` → 3.0，其余 1.2 不变 |

### 🔴 v28 新增关键约束

| 约束 | 说明 |
|------|------|
| **区域状态机标题提取** | `_v4_extract_headings` 必须跳过受保护区域（代码块/表格/OCR 补充块）内的匹配；OCR 补充块区域**必须锚定 `\n\n` 页分隔**（禁止"到下一个标记"——吞下页正文）；禁止用"包含 \| 就不算标题"式单行启发式 |
| **line 级表格重建** | gui_app 轨表格重建必须用 `get_text("dict")` line bbox（block 级无效——整行单元格在同一 block）；仅 ≥2 项且单元格 ≤40 字符的带包装；单 item 带原样输出 |
| **last_header 层级栈** | last_header 仅接受数字编号/章节编号标题；弹栈 = 层级不降或编号前缀不匹配；禁止恢复 `_prev_path[:num_dots]` 截取式路径拼接 |
| **形态负向校验** | 数字编号标题文字首字符 `\|` 或整段仅数字/点/竖线/空白 → 拒绝（`0.000 \| 0.000` 类 OCR 假标题） |
| **守卫脱敏** | 模板守卫命中路径必须应用 `_strip_code_from_context()`；正常路径禁止调用 |
| **重写指代限定** | 规则 3 实体补全必须逐字来自历史——禁止生成历史外函数名/标识符 |

### 🔴 v29 新增关键约束

| 约束 | 说明 |
|------|------|
| **OCR 键值法** | gui_app OCR 输出必须经 `_ocr_kv_normalize_row`（行内键值归一）+ `_ocr_merge_cross_line`（跨行配对，防页码误伤）+ 按图子块化（`[图表内容包含：]` 前缀）；**禁止**恢复 `\|` 离散分隔输出 |
| **图片过滤** | 过滤下限 = 面积 ≥0.5% 且边长 ≥40px；必须 xref 全局去重（同一 xref 只 OCR 一次）+ 放置 >20 页跳过；禁止恢复 1.5% 面积比 |
| **数字守卫豁免** | 守卫入参必须 `_SPACE_SEP_RE` 归一化 + `_COMPOUND_RE` 剥离后再跑 `_NUMERIC_QUERY_RE` 与关键词成员判定（剥离串同时用于两处）；剥离绝不污染真实 query |
| **Fast-Path 短路** | `_build_messages` 必须返回 `(messages, refusal_flag)`（禁止模块级标志——并发竞态）；守卫命中 → 四调用方在生成金字塔之前短路；每次重建 messages 后重读 flag |
| **重写中立性** | 规则 2 必须含协议主题中立性限制（Ethernet/IP/TCP/IP/Modbus 等不拼接产品名）；`_PROTOCOL_TERMS_RE` 确定性兜底必须保留；"泛泛步骤问法"措辞禁止加入（与 E18 few-shot 矛盾） |

### 🔴 v30 / v30.final 新增关键约束

| 约束 | 说明 |
|------|------|
| **跨页表头继承 (v30)** | gui_app 轨每页结束后暂存第一个 `\|` 行至 `_table_header`；下页首行若为 `\|` 表续行 → `_row_texts.insert(0, _table_header)` 强制注入；C-SDK 轨逻辑严禁触碰 |
| **OCR 标签化 (v30)** | gui_app OCR 输出必须用 `<OCR_BLOCK>...</OCR_BLOCK>` 包裹；`_PROTECTED_BLOCK_RE` 第二分支匹配 `<OCR_BLOCK>[\s\S]*?</OCR_BLOCK>`；`_v4_find_protected_ranges` `group(2)` → `type="ocr"` |
| **软装箱 (v30)** | Parent 截断前必须按 `_PROTECTED_BLOCK_RE` 解析文本为 普通/受保护 交替序列；受保护块**整体装入允许超标**，普通文本 `\n\n` 安全切断；**禁止**任何受保护块从中间物理切断 |
| **OCR 子块隔离 (v30)** | `_split_text_into_children` gui_app 路径必须扫描 `<OCR_BLOCK>` 标签 → 独立 `_emit_ocr_child()` → `chunk_type="ocr_child"`；OCR 块**禁止**混入普通 child 切片 |
| **封箱条件 (v30.final)** | 受保护块装入后**仅当 `_packed_len >= parent_chunk_size` 才 `_sealed = True`**；未超标则箱子继续装后续段——禁止过早封箱致后续内容丢失 |
| **孤儿行合并 (v30.final)** | `_row_texts` 构建：`len(_cells)==1` 且 `<10` 字符且前一行存在 → 追加到上一行表格末单元格；禁止将单元格内换行误判为独立行 |
| **全景快照 OCR (v30.final)** | gui_app 轨**废弃 `page.get_images()` + `doc.extract_image()`** 逐图抠取 → `page.get_pixmap(matrix=fitz.Matrix(2, 2))` 整页 2× 高清截图；矢量图设置框零盲区 |
| **智能去重 (v30.final)** | OCR 文本行去空格后若已存在于 `page_text`（PyMuPDF 提取）→ 直接丢弃；只保留 PyMuPDF 未提取的增量幽灵参数 |
| **Y 轻量分组 (v30.final)** | OCR 行按 Y 坐标 ≤12 行合并为一组（`_GROUP_SIZE=12`），每组独立 `<OCR_BLOCK>`——防整页 giant block 撑爆 512 维向量窗口 |
| **OCR 前置 (v30.final)** | gui_app OCR 块必须插入 `page_text`**最前方**（非页尾追加）；C-SDK 轨 OCR 追加逻辑严禁触碰；逐图循环入口 `if doc_type=="gui_app": continue` 跳过 |

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
| `src/rag_chain.py` | 🔴 RAG 核心管线 (~3,242 行, v25 计划拆分为 6 子模块) — 四层容灾、混合检索、HyDE、🔴 v24: Markdown 模板约束 + 极速流式穿透 |
| `src/graph_rag.py` | 🔴 LangGraph 状态图引擎 (~1,926 行, v25 计划拆分为 4 子模块) — 9 节点 + 条件边 + SDK 自纠错 + 硬熔断 + 🔴 v24: render_node 退化 + 流式双重输出 Bug 修复 |
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
| `_v4_extract_headings()` | pdf_loader.py | 标题提取 + 🔴 代码注释拦截（8 特征词上下文校验）+ 🔴 v23: doc_type 动态双轨拦截（GUI 禁止单数字编号） |
| `_sanitize_section_title()` | pdf_loader.py | 标题清洗器 + 🔴 伪标题黑名单（10 项 frozenset） |
| `_v4_extract_sdk_toc()` | pdf_loader.py | 🔴 Golden TOC 目录树预解析（预留回退基础设施） |
| `_is_skeleton_chunk()` | pdf_loader.py | 离线骨架过滤 |
| `_clean_pdf_text()` | pdf_loader.py | 7 步通用文本清洗 + 🔴 Step 6 SDK 代码换行修复 + 🔴 v25: JAKA/gui_app 数字保护特判（≥3 位参数保全） |
| `_v4_build_parent_child_docs()` | pdf_loader.py | 🔴 v23: 父级跨级扫描 + 动态切片容量分配 (GUI=1500/2000) + 前导文字保护 + 跨级大纲扫描 |
| `_v4_build_child_docs_v2()` | pdf_loader.py | 🔴 v23: 微缩大纲上限 5 条 + 标签统一 `[章节大纲参考]` |
| `_hybrid_retrieve()` | rag_chain.py | BM25+向量 RRF 混合检索 + 🔴 v23: 六大提权引擎 |
| `_decompose_compound_query()` | rag_chain.py | 复合查询拆解 (顺序连接词) |
| `_rewrite_query_with_llm()` | rag_chain.py | 🔴 ADR-19: LLM 意图重写引擎 (代词消解+产品补全) + 🔴 v26: always-on + 纠错/补全规则与 Few-Shot (max_tokens=128) |
| `_build_messages()` | rag_chain.py | 🔴 v24: Prompt 组装 + 双轨模板底端锚定 + Top-1 来源 + 反泄露门控 + 🔴 v25: 逃生舱条款（模板末尾，拒答交由 LLM 判定，零业务正则）+ 🔴 v26: `> [!WARNING]` 视觉加固 + 删除尾部对冲行 |
| `sanitize_chat_history()` | rag_chain.py | 历史沉渣净化中间件 |
| `_stream_guardrail()` | rag_chain.py | 🔴 v24: 极速流式透传（零缓冲，逐 chunk 直接 yield）+ 🔴 v25: 围栏闭合状态机（2 字符 carry 计数，奇数自动补发 ``` ） |
| `_fix_and_close_sdk_code()` | rag_chain.py | 🔴 v24: 过渡期兜底 — 代码块自动闭合 + CDLL 补全 + 函数名修正表（不再膨胀）+ 🔴 v25: 闭合兜底接入 extract_align_node 覆盖 Graph 全路径 |
| `_call_llm()` / `_stream_llm()` | rag_chain.py | LLM 调用 + 400 拦截 + Context 裁切 |
| `render_node()` | graph_rag.py | 🔴 v24: 退化为极简文本透传（废弃 JSON 解析） |
| `extract_align_node()` | graph_rag.py | 🔴 v24: 简化版属性对齐校验 + SemanticDedup + 静默斩尾（移除屠魔版正则）+ 🔴 v25: 无条件精确段落去重 + 代码块跳过模糊去重 + `_fix_and_close_sdk_code` 入口接入 |
| `run_graph_stream()` | graph_rag.py | 🔴 v24: 流式图执行 + SDK 自纠错回路 + 双重输出 Bug 修复 |
| `_tokenize_for_bm25()` | vector_store.py | BM25 分词器 — jieba + 标识符保护 + CODE 标签三倍写入 + 🔴 v26: 复合词原子化（`_COMPOUND_RE` 排除 `.`）与空格归一化（`_SPACE_SEP_RE`） |
| `_ocr_kv_normalize_row()` | pdf_loader.py | 🔴 v29: OCR 行内键值归一化 — `端口：\| 6502` → `端口：6502`（`\|` 离散分隔转 Dense 友好键值语义） |
| `_ocr_merge_cross_line()` | pdf_loader.py | 🔴 v29: OCR 跨行键值配对 — `从站节点号：` + 纯数值 → `从站节点号：1`（防页码 ±1 误伤） |
| `_emit_ocr_child()` | pdf_loader.py | 🔴 v30: OCR 子块独立 emit — `_split_text_into_children` gui_app 路径内定义，`chunk_type="ocr_child"`，不提取 function_names |
| `_v4_extract_text_universal()` 全景 OCR | pdf_loader.py | 🔴 v30.final: gui_app 轨 `get_pixmap(Matrix(2,2))` 整页截图 + 智能去重（vs `page_text`）+ Y 轻量分组（`_GROUP_SIZE=12`）+ OCR 前置插入 |
| `_v4_build_parent_child_docs()` 软装箱 | pdf_loader.py | 🔴 v30: AST-Lite 软装箱替代暴力腰斩 — `_PROTECTED_BLOCK_RE` 交替序列 + 受保护块整体装入 (允许超标) + 普通文本 `\n\n` 安全切断 + 🔴 v30.final: 条件封箱 (`_packed_len >= parent_chunk_size`) |
| `_strip_code_from_context()` | rag_chain.py | 🔴 v28: 通用代码脱敏 — ``` 代码块 → `[代码内容省略]`、DLL 加载行 → `[DLL加载代码省略]`（仅守卫命中路径） |
| `_resolve_product_from_history()` | rag_chain.py | 🔴 v27: 多轮产品解析第三兜底 — PRODUCT_ROUTER_RULES 扫最近 6 条历史锁定产品 |
| `_extract_sdk_header()` | pdf_loader.py | SDK 全局代码头提取 — 🔴 v27: 兼容裸 `CDLL(r"...")`（`from ctypes import *` 前缀省略）与 raw 前缀 |

### 🔴 PDF 切片规则 (v23)

#### SDK 状态机边界触发条件 (`_SDK_BLOCK_BOUNDARY_RE`)

```
仅两路可验证边界:
  ① ^\d{1,2}[\.\、\s]\s*\S+       → "28. 机械臂电源上电" / "4. 机械臂上电"
  ② ^(?:函数名称|函数说明)\s*      → OpenC3/OpenR6 两种 API 表头格式
```

**严格禁止**匹配的模式：`^#{1,4}\s+`（Python 注释 `# 时间等待3秒` 与 Markdown 标题无法区分，已从边界正则中永久移除）。

#### 🔴 v23: 标题正则深度扩展

多级数字编号 `{1,5}` 支持最高 6 级深度标题（如 `3.1.5.2.1`），兼容末尾带点和数字汉字粘连的极端排版。

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
CHUNK_MODE = "v4_dual"  # Parent(1000) + Child(400), GUI: Parent(2000) + Child(1500)
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
