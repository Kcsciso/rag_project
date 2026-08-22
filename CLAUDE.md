# 🔴 系统红线与开发规则（STRICT CONSTRAINTS）

## 1. 硬件与 GPU 管理（双 A100 智能自适应）

- **算力底座**: 2 × NVIDIA A100-PCIE-40GB（CUDA 12.4）。
- **GPU 自适应**: 禁止硬编码 `CUDA_VISIBLE_DEVICES`。`start_services.sh` + `src/config.py` 内置 `nvidia-smi` 扫描，自动选择空闲显存最大的 GPU（过滤 <5GB）。手动覆盖：**仅** `VLLM_GPU_ID` 环境变量（🔴 Stage 2-4 审查实测：`start_services.sh` 无 `--gpu`/`--vllm-only`/`--fastapi-only` CLI 参数，早期文档记载有误，禁止再写入）。
- **默认**: GPU 1（:8001）→ vLLM Qwen2.5-7B-Instruct-AWQ (4-bit ~8GB)；GPU 0 → ChromaDB/嵌入。
- **降级**: 空闲<5GB → 自动降级 1.5B。

## 0. 🔴 四层架构排雷法（AI 辅助开发思想钢印 — v24 更新）

**任何代码修改前，必须先声明该修改属于哪一层，并自检是否会破坏该层的核心特性。**

### L1 — 数据摄入与切片层 (src/pdf_loader.py — 🔴 Stage 1 (v32 收口) 双轨统一模块)

> **Stage 1 重构**: pdf_loader.py 由 ~2,900 行收敛为 ~660 行统一模块——SDK 专轨 PyMuPDF 状态机 + JAKA 专轨 MinerU/Qwen2-VL 多模态提纯 + KV 属性库 + 统一入口 `load_all_documents_v4_dual()`。以下红线表为准；原 v23–v30.final 自研 L1 管线整体归档（见下方归档表，恢复 GUI 自研轨需参照归档行）。

| 🔴 Stage 1 现行红线 | 严禁破坏 |
|----------|---------|
| SDK 专轨引擎 | 必须 `fitz` `get_text("text", sort=True)` 物理坐标流排序（`_extract_text_with_fitz`）——**禁止**回退 pypdf（表格/代码块断层、跨节漂移） |
| SDK 章节严格正则 | `_SDK_CHAPTER_BOUNDARY_RE = r'(?:^|\n)(?=[ \t]*\d{1,2}\s*\.\s*[一-龥a-zA-Z])'`；禁止恢复 `^#{1,4}\s+`（Python 注释无法区分） |
| SDK 原子闭环 | 单切片完整容纳「章节标题+函数签名+参数说明+返回值+示例代码」；`api_atomic`/`function_names` 元数据必填 |
| Ctypes 类型名黑名单 | `_CTYPES_BLACKLIST` frozenset（c_float/c_int/c_char_p…+restype/argtypes/byref 等）——函数名提取零 CTypes 污染（实测 29/31 函数 0 污染） |
| SDK Header 注入 | `_extract_sdk_header()` 提取 CDLL 加载行 + POSE/Joint 结构体，注入 API 切片 |
| JAKA 专轨 | MinerU 离线解析（`src/parse_jaka_mineru.py`）→ `load_jaka_mineru_dual()`；禁止将 JAKA 手册回切自研管线 |
| VLM 提纯注入 | :8005 Qwen2-VL-7B 提取图表参数；Prompt 严禁"未提供"类模板废话占位符；纯示意图统一标注 `仅为UI示意图` |
| 三重图片防线 | 1. 几何过滤（边长<80px 或长宽比>8）；2. 上下文图注强校验（±100 字）；3. VLM 纯参数提纯注入 |
| HTML 表格规整 | `clean_html_tables()` → 标准 GitHub Markdown 表格 + html/body/div 外层标签剥离 + 缺损单元格补齐（实测 34 表格→0 残留/444 行） |
| Markdown 面包屑栈 | 仅编号章节（`# 1.1`、`# 第一章`）维护 4 槽路径；`# 注意：` 类强调行禁止重置路径 |
| JAKA 软装箱 | 段落边界封箱（child_chunk_size=1500），表格保护块整体装入允许超标；TOC 点线特征行过滤 |
| 多模态缓存优先 | 优先读 `data/jaka_manual_chunks.json`（含 ≥50 个 VLM 实体切片才采信），无缓存才全量重解析 |
| KV 属性库 | `export_kv_attributes()` 摄入后自动生成 `kv_db/attribute_kv.json`；`_MANUAL_CALIBRATION`（6502/9600 等）强制覆盖 |
| 统一入口 | `load_all_documents_v4_dual()` = JAKA 轨 + SDK 轨 + KV 导出；`src/rebuild_v4.py`/`app.py` 必须走此入口 |
| 切片容量分配 | SDK: `SDK_CHILD_CHUNK_SIZE=400`/`SDK_PARENT_CHUNK_SIZE=1000`；GUI/JAKA: `GUI_CHILD_CHUNK_SIZE=1500`/`GUI_PARENT_CHUNK_SIZE=2000`；`CHILD_CHUNK_SIZE`/`PARENT_CHUNK_SIZE` 向下兼容别名 |
| SDK 切片规格 | 章节原子切片实测 270~890 字符（均值 OpenC3 414 / OpenR6 540）；JAKA Child 均值 ~480、p90 ~1090、保护块超标至 ~2600 |

### 🕰️ 历史归档 (v23–v30.final 自研 L1 管线 — Stage 1 已移除，恢复需参照)

| 归档特性 | 原约束 |
|----------|--------|
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
| 🔴 数据摄入双轨制 (v31) | JAKA 手册 → MinerU 离线解析（`src/parse_jaka_mineru.py`）；SDK 文档 → 原 L1 状态机管线。两轨互不侵入；SDK 轨零触碰；gui_app 轨 L1 逻辑保留不删除 |
| 🔴 **多模态 VLM 提纯注入 (v32)** | MinerU Markdown 切片时调用本地 :8005 Qwen2-VL-7B 提取图表参数；Prompt 严禁模板废话占位符（"未提供"等），纯示意图统一标注 `仅为UI示意图` |
| 🔴 **多模态三重图片防线 (v32)** | 1. 物理几何过滤（边长<80px或长宽比>8）；2. 上下文图注强校验（前后100字命中图/表/参数/设置等）；3. VLM 纯参数提纯注入 |
| 🔴 **HTML 表格规整转 Markdown (v32)** | 彻底清洗独立 `<table>` 及嵌套 HTML 表格为标准 GitHub Markdown 表格，并自动补齐缺损单元格 |
| 🔴 **Markdown 层级面包屑栈 (v32)** | 仅带编号的章节（`# 1.1`, `# 第一章`）维护层级路径；普通强调行（如 `# 注意：`）禁止重置路径与过度打碎切片 |

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

### 🔴 Stage 2/3/4 收口红线（2026-08-22 全量架构审查落盘）

| 收口项 | 红线 |
|--------|------|
| **单例数据库连接规范** | ChromaDB 读写**统一走 LangChain `Chroma` 包装器**（`persist_directory=CHROMA_PERSIST_DIR`，禁止传自建 `client=`）。嵌入函数唯一入口 `get_embedding_function()`（模块级懒加载单例，HF bge-small-zh → ONNX 回退）。**仅有的两个写入口**: `create_dual_collections()`（全量重建）/ `upsert_product_documents()`（增量摄入） |
| 🔴 **禁止原生 PersistentClient 混用** | 业务代码**禁止**直接 `chromadb.PersistentClient` 打开 `CHROMA_PERSIST_DIR`——原生客户端（自定义 `Settings`）与 LangChain 包装器（默认 Settings）并发打开同一目录 → ChromaDB Settings 冲突 / SQLite 文件锁冲突（实测建库写入失败）。**仅允许的只读例外**: `check_status.py` 健康检查、`app.py` `/api/debug/inspect_chunks` 调试接口、`vector_store.py` MD5 持久化（`_persist_md5_store`）。新增代码必须走 LangChain 包装器 |
| 🔴 **空 ID 拦截** | 文档 ID 生成必须用 `or` 链兜底（`doc.metadata.get("chunk_id") or f"c_{product_id}_{i}"`）——`get(key, default)` 对空字符串 `""` 不触发默认值，空 ID 导致 ChromaDB `add` 写入异常。禁止用 Python 内置 `hash()` 生成 ID（进程重启盐值漂移，ID 不稳定） |
| **BM25Okapi 增量索引** | 内存索引 `_bm25_indexes: {product_id: BM25Okapi}`；全量 `build_bm25_index()` / 增量 `bm25_upsert_product()`（corpus 扩展后整体重建重算 IDF）；进程重启由 `app.py` startup 调 `build_bm25_from_chromadb()` 恢复。⚠️ 已知债: build 路径 corpus 存 Document、upsert 路径存 str，`bm25_search` 返回类型随路径而异——禁止再引入第三种形态 |
| 🔴 **KV 确定性属性侧信道** | `src/kv_extractor.py` `lookup_attribute(query, product_id)` 读取 `kv_db/attribute_kv.json`（两级嵌套 {产品: {键: 值}}）→ **4 个注入点**（`rag_chat`/`rag_chat_stream`/`llm_generation_node`/`run_graph_stream`，均在 `_build_messages` 之后、LLM 调用之前前置注入），触发条件 `_last_numeric_context_missing or _NUMERIC_QUERY_RE`（模块级常量 rag_chain.py:2280）。未命中 → BM25 第二机会 → 硬拒答。`attribute_tool.py` 为**死代码**（0 导入者），禁止接线 |
| 🔴 **OpenR6 目录噪声剔除** | `_strip_openr6_toc()` 在 SDK 轨清洗后、状态机切分前执行（仅 `product_id=="OpenR6"` 或文件名含 openr6）。**分隔符正则必须为 `[☆★\*]{5,}`**——真实文档目录分隔线是 ☆ (U+2606) 非 ASCII `*`，纯 `[\*]` 实测剥离 0 字符。修复后实测: 剥离 465 字符、状态机块 119→61、切片 32→31（30 真实章节 + SDK 基础配置块） |
| **upsert 双轨路由** | `upsert_product_documents(file_path, product_id)` 按扩展名路由: `.md` → `load_jaka_mineru_dual` / `.pdf` → `load_single_sdk_pdf(file_path, product_id=...)`。禁止恢复旧 `_v4_extract_text_universal` import（已删除，曾致 /api/upload 500）。⚠️ 已知债: 新 upsert 无旧切片级联删除（下标式 ID 收缩时尾部旧 ID 残留），产品切片数收缩后需全量重建 |
| **数据目录纪律** | `data/` 下仅放产品文档；非产品 PDF（如 ROS 书籍）会被统一入口摄入为 `General` 产品线噪声切片（当前 DB 实测 1P+1C），入库前必须移出或配置产品映射 |

### 🔴 模块调用边界（Stage 2-4 收口实测）

| 模块 | 允许被调用方 | 边界 |
|------|------------|------|
| `pdf_loader.py` | `rebuild_v4.py` / `vector_store.upsert_product_documents` / `tests/` | 统一入口 `load_all_documents_v4_dual()`；单文件摄入用 `load_single_sdk_pdf(file_path, product_id=...)` / `load_jaka_mineru_dual()` |
| `vector_store.py` | `app.py` / `rebuild_v4.py` / `rag_chain.py`（检索函数）/ `graph_rag.py`（`search_similar_with_threshold`）/ `tests/` | 写入口仅 `create_dual_collections` + `upsert_product_documents`；读入口 `load_vector_store` / `search_similar_with_threshold` / `bm25_search` / `get_registered_products` |
| `kv_extractor.py` | `rag_chain.py`（2 处）/ `graph_rag.py`（2 处） | 唯一入口 `lookup_attribute`；`attribute_tool.py` 禁止接线（死代码） |
| `rag_chain.py` | `app.py`（经 `run_graph`/`run_graph_stream`）/ `tests/` | `rag_chat`/`rag_chat_stream` 为废弃 fallback |
| `graph_rag.py` | `app.py` / `tests/` | `set_graph_vector_store()` 统一注入，禁止节点内绕过 |

---

## 2. 核心依赖红线（严禁升级）

Conda `rag_agent` (Python 3.10)。**严禁 `pip install --upgrade`**：
`torch==2.6.0+cu124` `vllm==0.16.0` `sentence-transformers==2.7.0`
允许新增：`pypdf` `langchain-chroma` `rank-bm25`

## 3. RAG 架构与 AI 生态

- **框架**: LangChain、LangGraph、ChromaDB。**LLM**: vLLM Qwen2.5-7B-Instruct-AWQ @ :8001；云端降级 glm-4.7-flash。
- **嵌入**: BAAI/bge-small-zh-v1.5 (512维) → ONNX 自动回退。
- **UI 命名**: **比邻星 (ProximaRAG)**。
- **测试红线**: 修改 `rag_chain.py`/`graph_rag.py` 后必须运行 `python tests/run_eval.py --verbose`，8 项硬断言（①JSON泄露 ②段落重复 ③界面套话 ④函数签名错误 ⑤提示词泄露 ⑥API幻觉 ⑦零脑补 ⑧代码截断）全部通过方为合格。
- **Stage 1 摄入验收**: 修改 `pdf_loader.py` 后必须运行 `python tests/test_stage1.py`（18 断言 / 4 测试组：SDK 专轨 9 + JAKA 专轨 6 + KV 校准 1 + 统一入口 2），退出码 0 为合格。
- **向量库白盒质检**: 摄入/建库后运行 `python tests/audit_ingestion.py`（4 规则：零切片 / 垃圾切片 / 高压实体存活 / 架构标记注入）。⚠️ 当前规则 3b/4a/4b 断言口径与 Stage 1 脱节（6502 走 KV 侧信道而非切片、`[OCR补漏:]` 标记被 VLM 注入替代、Parent `[章节大纲参考]` 注入 0/12）——该三项待口径适配，判定以 test_stage1.py + run_eval.py 为准。

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
| 🔴 **v31** | **v31** | **数据摄入双轨制: JAKA 手册 → MinerU 离线解析 (magic-pdf 1.3.12, doclayout_yolo 版面 + rapid_table 表格 + unimernet `cache_position` 文件级补丁) 产出 `data/jaka_markdown/JAKA_Manual/auto/JAKA_Manual.md`；SDK 文档保留原 L1 状态机管线零改动。连环排雷: layout-config 缺省回退 detectron2 NoneType 崩溃 / `table-master` 非法表名 / transformers 4.49 毒药参数 (subprocess 内 monkeypatch 无效 → 文件级补丁) / modelscope 模型快照漂移 (OCR v3 det 缺失) / GPU 争抢自适应** | `src/parse_jaka_mineru.py`, `auto_patch_mbart.py`, `patch_unimernet.py` |
| 🔴 **v32** | **v32** | **多模态 Markdown 提纯与双模型微服务架构**: 本地部署 Qwen2-VL-7B-Instruct (:8005) + 双重图片几何/语义过滤 + HTML 表格转 Markdown + 防模板化 VLM Prompt + 切片 JSON 持久化 | `src/pdf_loader.py` (v32 逻辑统一收口), `data/jaka_manual_chunks.json` (提纯缓存) |
| 🔴 **Stage 1** | **Stage 1 (v32 收口)** | **数据摄入统一收口: SDK 专轨 PyMuPDF (fitz) 状态机重构 (pypdf 废弃, `_SDK_CHAPTER_BOUNDARY_RE` 严格章节正则, Ctypes 黑名单零污染) / JAKA 专轨 VLM 提纯 + KV 属性库自动生成 / 统一入口 `load_all_documents_v4_dual()` + `src/rebuild_v4.py` 双轨建库 / check_status vLLM 1-token 微探针 (防 CUDA 假死) / 验收: OpenC3 27 · OpenR6 30 · JAKA 225+9 · 189 截图提纯** | `load_single_sdk_pdf()`, `load_jaka_mineru_dual()`, `export_kv_attributes()`, `load_all_documents_v4_dual()`, `tests/test_stage1.py` |


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

### 🔴 v31 新增关键约束

| 约束 | 说明 |
|------|------|
| **数据摄入双轨制** | JAKA 手册 → MinerU 离线解析（`src/parse_jaka_mineru.py`，magic-pdf 1.3.12）；SDK 文档 → 原 L1 状态机管线。两轨互不侵入；禁止将 JAKA 手册回切自研管线（除非 MinerU 失效且 gui_app 轨仍可用） |
| **MinerU 配置三键** | `~/magic-pdf.json` 必须含：`layout-config: {"model": "doclayout_yolo"}`（**缺省回退 layoutlmv3 → detectron2/fvcore 抛 NoneType**）、`table-config` 合法名（tablemaster/rapid_table/struct_eqtable，带连字符非法）、`models-dir` 绝对路径 |
| **unimernet 补丁文件级** | transformers ≥4.49 下必须执行 `auto_patch_mbart.py`/`patch_unimernet.py`（site-packages 文件级补丁）；**进程内 monkeypatch 对 subprocess 调用的 magic-pdf 无效** |
| **MinerU 依赖互斥** | magic-pdf 1.3.12 要求 transformers≥4.49，vLLM 0.5.4 要求 <4.46——升级/安装任一栈前必须评估对方影响；禁止将 MinerU 依赖写入核心依赖锁定清单 |
| **SDK 轨零触碰** | MinerU 双轨制下 `pdf_loader.py` C-SDK 轨逻辑保持原样；gui_app 轨 L1 逻辑保留（其他 GUI 文档仍依赖），禁止删除 |

# 🚀 本地服务启动

```bash
./start_services.sh                    # 一键启动 (GPU 智能检测 → vLLM :8001 → FastAPI :8000)
VLLM_GPU_ID=0 ./start_services.sh      # 手动指定 GPU (环境变量覆盖)
# 🔴 Stage 2-4 审查实测: start_services.sh 无 --vllm-only/--fastapi-only/--gpu CLI 参数
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
| `src/pdf_loader.py` | 🔴 Stage 1: 统一数据摄入与切片 (~660 行) — SDK 专轨 fitz 状态机原子切片 + JAKA 专轨 MinerU/Qwen2-VL 多模态提纯 + KV 属性库 + 统一入口 `load_all_documents_v4_dual()` |
| `src/vector_store.py` | 🔴 Stage 2: 向量知识库 — LangChain Chroma 单例规范 (Parent-Child 双集合) + BM25Okapi 增量分词索引 + bge-small-zh-v1.5 + ONNX 回退 |
| `src/kv_extractor.py` | 🔴 Stage 3: 轻量 KV 属性事实检索 — `lookup_attribute()` 确定性侧信道注入 (替代已失效的 attribute_tool 接线) |
| `src/rag_chain.py` | 🔴 RAG 核心管线 (~3,242 行, v25 计划拆分为 6 子模块) — 四层容灾、混合检索、HyDE、🔴 v24: Markdown 模板约束 + 极速流式穿透 |
| `src/graph_rag.py` | 🔴 LangGraph 状态图引擎 (~1,926 行, v25 计划拆分为 4 子模块) — 11 节点 + 条件边 + SDK 自纠错 + 硬熔断 + 🔴 v24: render_node 退化 + 流式双重输出 Bug 修复 |
| `src/agent_state.py` | RAGState TypedDict (21 字段) |
| `src/attribute_tool.py` | ⚠️ 死代码 (0 导入者, Stage 3 审查确认) — 动态属性意图 LLM 提取→BM25→正则 KV，已被 `kv_extractor.lookup_attribute` 替代，禁止重新接线 |
| `app.py` | FastAPI (:8000) — /api/chat, /api/upload, /api/status, /api/products；startup 加载向量库 + `build_bm25_from_chromadb` 恢复 BM25 + `set_graph_vector_store` 注入 |
| `frontend_server.py` | 前端 UI (:8501) — Jinja2 + /api/* 反向代理；⚠️ Stage 2-4 审查发现: 代理目标 `API_BACKEND=localhost:7860` 与 FastAPI 实际端口 8000 失配，前端 API 链路待修 |
| `tunnel.py` | 🔴 ngrok 内网穿透 — pyngrok，`python tunnel.py --token X --port 8000`（或 `NGROK_AUTHTOKEN` 环境变量，默认端口 8000） |
| `src/rebuild_v4.py` | 🔴 Stage 1: 双轨建库脚本 — 物理清库 → `load_all_documents_v4_dual` → ChromaDB 双 Collection → 质检统计；运行 `python src/rebuild_v4.py` 或 `python -m src.rebuild_v4` |
| `check_status.py` | 健康检查 — vLLM + FastAPI + GPU |
| `start_services.sh` | 一键启动 — GPU 智能选择 + 就绪轮询 + 优雅退出 |
| `audit_chunks.py` | ⚠️ 切片健康度审计 (Health Score) — 依赖旧 `load_pdfs_v4_dual` 入口，Stage 1 待适配 |
| `src/parse_jaka_mineru.py` | 🔴 v31: MinerU 离线解析入口 (JAKA 手册) — modelscope 权重检查 + `~/magic-pdf.json` 生成 (doclayout_yolo + rapid_table) + GPU 自适应 + magic-pdf CLI 调用 |
| `auto_patch_mbart.py` / `patch_unimernet.py` | 🔴 v31: unimernet × transformers 4.49 `cache_position` 文件级补丁 (site-packages 注入) |

### 关键函数索引

| 函数 | 位置 | 用途 |
|------|------|------|
| `_v4_parse_sdk_state_machine()` | pdf_loader.py | 🔴 Stage 1: SDK 轨状态机 API 块解析器 — `_SDK_CHAPTER_BOUNDARY_RE` 严格章节正则切分（仅行首 1-2 位编号+中英文跟随），preamble → "SDK 基础配置" |
| `_extract_text_with_fitz()` | pdf_loader.py | 🔴 Stage 1: PyMuPDF `get_text("text", sort=True)` 物理坐标流排序提取 — 废弃 pypdf，根除表格/代码块断层与跨节漂移 |
| `_clean_sdk_pdf_text()` | pdf_loader.py | 🔴 Stage 1: SDK 文本清洗 — 表格纵向断字修复（`函数名\n称`→`函数名称`）+ 下划线/API 断行拼接 + I/O 归一化 |
| `_strip_openr6_toc()` | pdf_loader.py | 🔴 Stage 2-4: OpenR6 目录噪声剔除 — 剥离文档开头 1~29 项目录块；分隔符正则 `[☆★\*]{5,}`（☆ U+2606 为真实分隔线，纯 `[\*]` 实测剥离 0 字符）；修复后实测剥离 465 字符 → 32→31 切片（30 真实章节） |
| `_v4_extract_function_names()` | pdf_loader.py | 🔴 Stage 1: 三模式函数名提取 + `_CTYPES_BLACKLIST` 黑名单（c_float/c_int/restype/byref…零 CTypes 污染） |
| （v23–v30.final 归档） | pdf_loader.py | `_v4_extract_headings`/`_sanitize_section_title`/`_v4_extract_sdk_toc`/`_is_skeleton_chunk`/`_clean_pdf_text`/`_v4_build_parent_child_docs`/`_v4_build_child_docs_v2` 已随 Stage 1 重构移除（见 L1 历史归档表） |
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
| `create_dual_collections()` | vector_store.py | 🔴 Stage 2: Parent-Child 双集合建/重建 — 统一 LangChain `Chroma` 包装器（`persist_directory`，禁止原生 `PersistentClient` 混用）+ 空 ID `or` 链拦截 + 写入后同步 `build_bm25_index` |
| `upsert_product_documents()` | vector_store.py | 🔴 Stage 2: 增量摄入 — `file_path`+`product_id` 显式参数 → 按扩展名双轨路由（`.md`→JAKA 轨 / `.pdf`→SDK 轨）→ metadata 清洗 → 双集合增量写入 + BM25 增量同步 |
| `build_bm25_index()` / `build_bm25_from_chromadb()` | vector_store.py | 🔴 Stage 2: BM25Okapi 内存索引全量构建（按 product_id 分组）/ 进程重启恢复（`app.py` startup 调用，从 ChromaDB 全量拉取重建） |
| （v29–v30.final OCR 归档） | pdf_loader.py | `_ocr_kv_normalize_row`/`_ocr_merge_cross_line`/`_emit_ocr_child`/`_v4_extract_text_universal`/AST-Lite 软装箱 已随 Stage 1 移除 — JAKA 参数提纯由 Qwen2-VL 多模态管线替代 |
| `_strip_code_from_context()` | rag_chain.py | 🔴 v28: 通用代码脱敏 — ``` 代码块 → `[代码内容省略]`、DLL 加载行 → `[DLL加载代码省略]`（仅守卫命中路径） |
| `_resolve_product_from_history()` | rag_chain.py | 🔴 v27: 多轮产品解析第三兜底 — PRODUCT_ROUTER_RULES 扫最近 6 条历史锁定产品 |
| `_extract_sdk_header()` | pdf_loader.py | SDK 全局代码头提取 — 🔴 v27: 兼容裸 `CDLL(r"...")`（`from ctypes import *` 前缀省略）与 raw 前缀 |
| `clean_html_tables()` | pdf_loader.py | 🔴 Stage 1/v32: HTML 表格无损转 Markdown 表格 + 外层标签剥离 + 缺损单元格补齐（实测 34 表格→0 残留/444 行） |
| `_preprocess_all_images()` / `_call_vlm_worker()` | pdf_loader.py | 🔴 Stage 1/v32: 三重图片防线（几何/图注/VLM）6 线程并发提纯 — 对接 :8005 Qwen2-VL，Prompt 禁模板废话占位符，空结果标 `仅为UI示意图` |
| `load_jaka_mineru_dual()` | pdf_loader.py | 🔴 Stage 1/v32: JAKA 专轨切片器 — 提纯缓存优先 (`data/jaka_manual_chunks.json`) + 章节面包屑栈 + 1500ch 软装箱 + TOC 点线过滤 |
| `load_single_sdk_pdf()` | pdf_loader.py | 🔴 Stage 1: SDK 专轨入口 — fitz sort=True 提取 → 清洗 → 章节原子切片 + SDK Header 注入 |
| `export_kv_attributes()` / `load_all_documents_v4_dual()` | pdf_loader.py | 🔴 Stage 1: KV 属性库自动生成 (`kv_db/attribute_kv.json` + 人工校准覆盖) / 统一摄入入口（JAKA 轨 + SDK 轨 + KV 导出） |
| `lookup_attribute()` | kv_extractor.py | 🔴 Stage 3: KV 确定性属性侧信道 — 读取 `kv_db/attribute_kv.json`（两级嵌套，缓存单例）→ 4 注入点（rag_chat/rag_chat_stream/llm_generation_node/run_graph_stream，均 `_build_messages` 后、LLM 前前置注入）；触发 `_last_numeric_context_missing or _NUMERIC_QUERY_RE` |

### 🔴 PDF 切片规则 (Stage 1 / v32 收口)

#### SDK 专轨章节边界 (`_SDK_CHAPTER_BOUNDARY_RE`)

```
r'(?:^|\n)(?=[ \t]*\d{1,2}\s*\.\s*[一-龥a-zA-Z])'   # re.MULTILINE
```
- 仅匹配行首 1-2 位编号 + 可选空格 + `.` + 空格 + 中文/英文字符跟随（如 `28. 机械臂电源上电`）→ 整章原子闭环切片（OpenC3 27 / OpenR6 30 章节）。
- `\d{1,2}` 上限 2 位 — 浮点数（`3.14`）与长数字不误切；<20 字符碎片块由加载器丢弃兜底。
- **严格禁止**匹配的模式：`^#{1,4}\s+`（Python 注释 `# 时间等待3秒` 与 Markdown 标题无法区分）。
- 历史 `_SDK_BLOCK_BOUNDARY_RE` 双边界（数字标题 + 函数名称/函数说明）与 v23 多级编号 `{1,5}` 深度扩展随 Stage 1 归档（见 L1 历史归档表）。

---

# 📋 当前生产配置

```python
# LLM
BASE_URL     = "http://localhost:8001/v1"
MODEL_NAME   = env LLM_MODEL_NAME, 默认本地快照 /home/kasm-user/LLM/mo/models/Qwen--Qwen2.5-7B-Instruct-AWQ/snapshots/master
DEEPSEEK_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEEPSEEK_MODEL    = "glm-4.7-flash"
# VLM (Stage 1/v32): VLM_BASE_URL="http://localhost:8005/v1" / VLM_MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# max_tokens=1024 (v16: 代码+步骤完全充裕，从源头消解 vLLM 400)
LLM_INFERENCE_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=15.0, pool=5.0)

# 检索
CHUNK_SIZE=300 / CHUNK_OVERLAP=50 / RETRIEVAL_K=10 / SIMILARITY_THRESHOLD=0.68
_AUTOCUT_MIN_K=8 / _AUTOCUT_MAX_K=15  # SDK 检索时 MIN_K 动态提升至 10
_MIN_SUB_QUERY_LEN=2  # v22: 复合查询最小子句长度，两字动词不丢弃
CHUNK_MODE = "v4_dual"  # Stage 1: SDK_PARENT=1000/SDK_CHILD=400, GUI_PARENT=2000/GUI_CHILD=1500 (兼容别名 PARENT_CHUNK_SIZE/CHILD_CHUNK_SIZE)
_MAX_CONTEXT_CHARS = 4000  # SDK 检索时动态提升至 8000

# 嵌入
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"  # 512维, HF→ONNX 回退

# 端口
# FastAPI :8000 | vLLM :8001 | Frontend :8501
```

### 运维命令

```bash
./start_services.sh              # 一键启动 (GPU 智能检测 → vLLM → FastAPI)
pkill -f "app.py"; pkill -f "vllm"  # 一键停止
python check_status.py           # 健康检查 (vLLM 1-token 微探针)
python tests/run_eval.py --verbose  # 回归评测 (35 用例 / 8 硬断言)
python tests/test_stage1.py      # 🔴 Stage 1: 数据摄入/双轨切片离线冒烟测试 (18 断言)
python tests/audit_ingestion.py  # 🔴 Stage 2: 向量库白盒质检 (4 规则; 规则 3b/4a/4b 口径待适配)
python src/rebuild_v4.py         # 🔴 Stage 1: 双轨全量建库 (或 python -m src.rebuild_v4)
python src/parse_jaka_mineru.py  # v31: JAKA 手册 MinerU 离线解析
python tunnel.py                 # ngrok 内网穿透 (--token X / NGROK_AUTHTOKEN, 默认 :8000)
# ⚠️ audit_chunks.py 依赖旧 pdf_loader 入口，Stage 1 待适配 (以 tests/audit_ingestion.py 为准)
```
