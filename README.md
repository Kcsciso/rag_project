# 📰 比邻星 (ProximaRAG) — 湖南比邻星科技文档智能问答系统

基于 **RAG（Retrieval-Augmented Generation）** 架构的官方技术文档与使用手册智能问答系统。专为**湖南比邻星科技有限公司**的开发者与用户打造，采用双 A100 GPU 算力底座，底层搭载 **vLLM + Qwen2.5-7B-Instruct-AWQ** 实现完全私有化、低延迟的本地推理。

> **🔴 Stage 1 结项 — 数据摄入统一收口 (2026-08-21)**: **SDK 专轨 PyMuPDF 状态机重构**（废弃 pypdf，`sort=True` 版面排序，OpenC3 27 / OpenR6 30 章节原子切片，Ctypes 零污染）+ **JAKA 专轨 MinerU × Qwen2-VL (:8005) 多模态提纯**（三重图片防线筛出 189 张截图，IP / Modbus 6502 / 波特率 9600 参数文本化，226+ 切片，KV 属性库自动生成）+ 统一建库入口 `python src/rebuild_v4.py` 与离线验收 `python tests/test_stage1.py`。

> **🔴 Stage 2-4 收口 (2026-08-22)**: **ChromaDB LangChain 单例规范**（Parent-Child 双集合写入统一走 LangChain `Chroma` 包装器，根除原生 `PersistentClient` 混用引发的 Settings/文件锁冲突）+ **BM25Okapi 增量分词索引**（建库全量构建 + upsert 增量重建 IDF + 启动时从 ChromaDB 恢复）+ **KV 确定性属性侧信道**（`kv_extractor.lookup_attribute()` 读取 `kv_db/attribute_kv.json`，4 注入点前置注入，6502/9600 端到端实测命中）+ **OpenR6 目录噪声剔除**（`_strip_openr6_toc()`，分隔线 ☆ 正则修复，切片 32→31）+ 空 ID 拦截与显式参数修复。详见 [ARCHITECTURE_AUDIT.md](./ARCHITECTURE_AUDIT.md)。

> **🔴 v24 架构升级 (2026-08-04)**: 全面转向 **Markdown 模板强约束 (Template Masking) + 极速流式穿透** 架构。废弃了此前的 JSON 提取+正则清洗后处理管线，System Prompt 从 210 行压缩至 ~15 行（Token 节省 83%），TTFB 从 60-90s 降至 <2s。

> **🔴 v25 回归攻坚 (2026-08-05)**: 代码围栏闭合状态机（`_stream_guardrail` 透传+奇偶计数自动补 `` ``` ``）+ 双轨模板【逃生舱条款】（拒答交由 LLM 自主判定，零业务正则）+ JAKA/gui_app 数字保护特判（≥3 位参数保全）+ KV 属性注入放宽（E05/E07 确定性数值）+ SemanticDedup 无条件精确段落去重。

> **🔴 v26 最后一公里 (2026-08-05)**: OCR 面积比过滤 + CTM 矩阵 Y 归位（截图参数插回对应章节，切片结构不变）+ BM25 复合词原子化与空格归一化（`Ethernet/IP` 整体 token，排除 `.` 防吞 snake_case）+ 重写器 always-on 与纠错/补全 Few-Shot（E18 错别字、E28 纯名词）+ 逃生舱 `> [!WARNING]` 视觉加固与尾部对冲行删除（纯 Prompt，零 L4 拦截）。

> **🔴 v27 回归反转 (2026-08-05)**: 产品路由责任切分（原始 query 优先 + 单轮澄清守卫 + 历史扫描，E01 确定性澄清）+ OCR 回退页尾追加（`[本页图片解析参数补充]` 标识，回退 CTM Y 归位防切片污染）+ SDK 模板物理隔离（删除 `import ctypes` 字面样例，`_extract_sdk_header` 修复裸 `CDLL` 提取）+ L3 模板选择守卫（函数级/产品级/超纲级三条件 → 拒答模板，E09/E21/E25）+ 去重规范化 + 动态 BM25 权重（短文本/复合词 3.0，E28/E29）。

> **🔴 v28 切片状态机化 (2026-08-05)**: 标题提取区域状态机（`_v4_extract_headings` 感知受保护区域 + OCR 补充块入保护区，根治 309 个污染路径）+ gui_app 轨 line 级几何表格重建（表 1-1 变 Markdown 表格行，单元格不裸行不被 H3 提权）+ last_header 数字编号层级栈（前缀祖先校验修复跨章叠加）+ 数字编号形态负向校验（`0.000 | 0.000` 类 OCR 假标题拒绝）+ TOC 点线目录特征过滤（E29 误拒答根因之一）+ 守卫命中 context 代码脱敏（无代码可抄）+ 重写规则 3 实体指代泛化（E17 动作指代）。

> **🔴 v29 数据语义化 + 确定性拒答 (2026-08-05)**: OCR 键值法语义（`端口： | 6502` → `端口：6502`，`|` 离散转 Dense 友好键值；跨行键值配对；按图子块化）+ 图片过滤重构（0.5%/40px 数据支撑下限 + xref 全局去重 + 放置次数启发式，230 个小参数图入库）+ 数字守卫复合词豁免（Ethernet/IP 的 IP 不再误杀）+ **Fast-Path 确定性拒答**（`_build_messages` 返回侧信道，守卫命中跳过 LLM 直出固定话术——物理根除拒答记忆中毒）+ 重写引擎协议主题中立性（规则 2 限制 + Few-Shot + 确定性兜底，单发 "Ethernet/IP" 不再被强加 OpenC3）。

> **🔴 v30 AST-Lite 软装箱 (2026-08-07)**: 跨页表格表头向下继承（暂存 `_table_header` → 下页强制注入，字段语义不丢失）+ OCR 标签化防稀释（`<OCR_BLOCK>` 包裹 + `_PROTECTED_BLOCK_RE` 同步 + `type="ocr"` 保护区识别）+ **AST-Lite 软装箱算法**（Parent 截断替代暴力腰斩：文本解析为 普通/受保护 交替序列 → 受保护块整体装入允许超标、普通文本 `\n\n` 安全切断 → 表格/代码/OCR 零物理切断）+ OCR 子块隔离（`_emit_ocr_child` + `chunk_type="ocr_child"` 独立切片，OCR 参数不稀释 Dense 语义）+ **过早封箱 Bug 修复**（受保护块装入后仅超标才封箱，未满则继续装）。

> **🔴 v30.final 全景快照 OCR + 孤儿行合并 (2026-08-07)**: 孤儿行合并（单元格内换行 `<10ch` → 追加到上一行表格末单元格，修复 `Windows7及以上\n上` Markdown 碎裂）+ **全景快照 OCR**（废弃 `get_images()` 逐图抠取 → `get_pixmap(Matrix(2,2))` 整页 2× 高清截图，矢量图设置框零盲区）+ 智能去重（OCR 文本 vs `page_text` 交叉比对，仅保留增量幽灵参数）+ Y 轻量分组（≤12 行/组，防 giant block）+ **OCR 前置**（插入 `page_text` 最前方，旧逻辑页尾追加撕裂跨页上下文已废弃）+ C-SDK 逐图跳过（`if doc_type=="gui_app": continue`，零触碰隔离）。

> **🔴 v31 数据摄入双轨制 (2026-08-19)**: **JAKA 手册 → MinerU (magic-pdf 1.3.12) 离线解析**（doclayout_yolo 版面 + rapid_table 表格 + unimernet `cache_position` 文件级补丁），产出 `data/jaka_markdown/JAKA_Manual/auto/JAKA_Manual.md`（2659 行，含 Modbus 寄存器大表）；**SDK 文档保留原 L1 状态机管线零改动**。连环排雷记录：layout-config 缺省回退 detectron2 NoneType 崩溃 / `table-master` 非法表名 / transformers 4.49 毒药参数 / modelscope 模型快照漂移 / GPU 争抢自适应（详见 dev_log）。

---

## 🏛️ RAG 四层系统架构

比邻星 (ProximaRAG) 遵循工业级 RAG 四层架构设计：

```
                          ┌──────────────────────────────────────────────┐
                          │         FastAPI Gateway (:8000)               │
                          │   /api/chat (SSE) · /api/upload · /api/status │
                          │     LangGraph: run_graph / run_graph_stream   │
                          └───────────────┬────────────────┬──────────────┘
                                          │                │
                         在线查询路径 ────►│                │◄──── 离线摄入路径 (Stage 1)
                                          ▼                ▼
      ┌───────────────────────────────┐      ┌────────────────────────────────┐
      │  L2 检索与重排层               │      │  L1 数据摄入与切片层            │
      │  (rag_chain + vector_store)   │      │  (pdf_loader.py · 双轨统一)     │
      │  • LLM 意图重写 (always-on)   │      │  • SDK 专轨: fitz 状态机       │
      │  • RRF 混合检索 向量+BM25     │      │  • JAKA 专轨: MinerU×Qwen2-VL  │
      │  • 六大提权引擎 + Autocut     │      │  • 三重图片防线 (:8005 VLM)    │
      │  • 复合查询拆解 + 保底召回    │      │  • OpenR6 目录噪声剔除          │
      │  • 产品物理隔离 (where 过滤)  │      │  • HTML表格→MD + KV 属性库导出 │
      │  • HyDE SDK/JAKA 全线禁用     │      │  • 统一入口 load_all_documents  │
      └──────────────┬────────────────┘      │    _v4_dual()                  │
                     │ 读取                  └──────────────┬─────────────────┘
                     ▼                                     │ 写入
      ┌─────────────────────────────────────────────────────┴────────────────┐
      │   Stage 2 向量知识库 (vector_store.py — LangChain Chroma 单例规范)     │
      │   rag_v4_parent (父层·章节级) │ rag_v4_child (子层·函数/步骤级)       │
      │   BM25Okapi 增量分词索引 (内存按产品分组 · 启动自 ChromaDB 恢复)       │
      │   kv_db/attribute_kv.json (KV 确定性属性库 · 人工校准 6502/9600)      │
      └──────────────────────────────┬───────────────────────────────────────┘
                                     │ 检索命中 Context
                                     ▼
      ┌─────────────────────────────────────────────────────────────────────┐
      │   L3 上下文组装与指令层 (rag_chain._build_messages)                  │
      │   System Prompt 极简 (≤250 tok) │ 双轨模板底端锚定 (gui_app/c_sdk)   │
      │   模板守卫三条件 → 拒答模板 + 代码脱敏 (_strip_code_from_context)     │
      │   🔴 KV 确定性侧信道 (kv_extractor.lookup_attribute · 4 注入点)      │
      │   历史净化 sanitize_chat_history │ Context Cap 4000/8000             │
      └──────────────────────────────┬──────────────────────────────────────┘
                                     │ messages
                                     ▼
      ┌─────────────────────────────────────────────────────────────────────┐
      │   L4 生成控制与后处理层 (graph_rag 后处理 + LLM 调用)                 │
      │   四层容灾金字塔: vLLM(:8001) → 智谱 GLM → 纯检索直出 → 硬拒答       │
      │   极速流式穿透 (_stream_guardrail · TTFB<2s · 围栏闭合状态机)         │
      │   SDK 自纠错回路 (硬熔断 retry≤2) │ render_node 纯文本透传            │
      │   extract_align (SemanticDedup + 静默斩尾 + 代码闭合兜底)             │
      └──────────────────────────────┬──────────────────────────────────────┘
                                     │ SSE 流式
                                     ▼
                      ┌─────────────────────────────────┐
                      │         前端 UI (:8501)          │
                      └─────────────────────────────────┘
```

> **两条数据流**: ① **离线摄入路径**（Stage 1 双轨解析 → Stage 2 双集合写入 + BM25 构建，经 `src/rebuild_v4.py` 全量重建或 `/api/upload` 增量触发，L1 不参与在线请求）；② **在线查询路径**（Gateway → L2 混合检索 → L3 上下文组装 → L4 生成，由 LangGraph `run_graph`/`run_graph_stream` 编排，SSE 流式返回前端）。

### L1 — 数据摄入与切片层 (src/pdf_loader.py, ~660 行 — 🔴 Stage 1 双轨统一模块)

**处理流程 (SDK 专轨)**: `PDF` → `_extract_text_with_fitz()` (PyMuPDF `sort=True` 版面排序) → `_clean_sdk_pdf_text()` → `_strip_openr6_toc()` (OpenR6 目录噪声剔除) → `_extract_sdk_header()` + `_v4_parse_sdk_state_machine()` (`_SDK_CHAPTER_BOUNDARY_RE` 章节原子切分) → `load_single_sdk_pdf()`

**处理流程 (JAKA 专轨)**: `MinerU Markdown` → `_preprocess_all_images()` (三重防线 + Qwen2-VL 提纯) → `clean_html_tables()` (HTML→Markdown) → `load_jaka_mineru_dual()` (面包屑栈 + 软装箱)

| 能力                   | 说明                                                                                                                                                                                            |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🔴 SDK 引擎            | PyMuPDF`fitz` + `get_text("text", sort=True)` 物理坐标流排序 — 废弃 pypdf，根除表格/代码块断层与跨节漂移                                                                                   |
| 🔴 SDK 章节正则        | `_SDK_CHAPTER_BOUNDARY_RE` 仅匹配行首 1-2 位编号+中英文字符 — OpenC3 27 章 / OpenR6 30 章原子闭环切片                                                                                        |
| 🔴 原子闭环            | 单切片 = 章节标题+函数签名+参数说明+返回值+示例代码；`api_atomic`/`function_names` 元数据 100% 覆盖                                                                                         |
| 🔴 Ctypes 黑名单       | `_CTYPES_BLACKLIST` frozenset (c_float/c_int/restype/byref…) — 函数名提取零类型名污染                                                                                                       |
| 🔴 SDK Header          | `_extract_sdk_header()` 提取 CDLL 加载行 + POSE/Joint 结构体注入 API 切片                                                                                                                     |
| 🔴 JAKA 表格规整       | `clean_html_tables()` — 34 处 HTML `<table>` 转 GitHub Markdown + 外层标签剥离 + 缺损单元格补齐（0 残留）                                                                                  |
| 🔴 三重图片防线        | 几何过滤 (<80px / 长宽比>8) → 图注校验 (±100 字) → Qwen2-VL 提纯 — 262 张图片筛出 189 张截图并发提取                                                                                        |
| 🔴 VLM 防废话          | Prompt 禁"未提供"类模板占位符；纯示意图统一标`仅为UI示意图`；结果缓存 `data/jaka_manual_chunks.json`                                                                                        |
| 🔴 JAKA 软装箱         | 段落边界封箱 (1500ch) + 章节面包屑栈 (`第N章`/`N.N`) + TOC 点线特征过滤 + 表格保护块允许超标                                                                                                |
| 🔴 KV 属性库           | `export_kv_attributes()` 摄入后自动生成 `kv_db/attribute_kv.json` + 人工校准强制覆盖 (6502/9600)                                                                                            |
| 🔴 统一入口            | `load_all_documents_v4_dual()` = JAKA 轨 + SDK 轨 + KV 导出；`src/rebuild_v4.py` 全量重建对接点                                                                                             |
| 切片规格               | SDK 章节原子 270~890 字符 (均值 414/540)；JAKA Child 均值 ~480、p90 ~1090、保护块超标至 ~2600                                                                                                   |
| 🔴 OpenR6 目录噪声剔除 | `_strip_openr6_toc()` 剥离文档开头 1~29 项目录文本块 + 星号分隔线（仅 OpenR6 轨）；分隔线为 ☆ (U+2606)——修复前纯 `[\*]` 正则剥离 0 字符，修复后剥离 465 字符、切片 32→31（30 真实章节） |
| 历史归档               | v23–v30.final 自研 L1 管线（OCR 归位/软装箱/标题状态机等）随 Stage 1 归档移除，详见 CLAUDE.md 归档表                                                                                           |

### L2 — 检索与重排层 (rag_chain.py + vector_store.py)

**处理流程**: `Query` → `_rewrite_query_with_llm()` (LLM 意图重写) → `_preprocess_query()` (口语剥离) → `_generate_hyde_doc()` (SDK/GUI 全线禁用) → `_hybrid_retrieve()` (向量 + BM25 + RRF 六大引擎 + Autocut)

| 能力                  | 说明                                                                                                                                                                                                                                                                                                                                                                                                                |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 向量检索              | ChromaDB cosine (bge-small-zh-v1.5, 512维) — 🔴 Stage 2: LangChain`Chroma` 包装器单例规范（禁止原生 `PersistentClient` 混用同一 persist 目录）+ Parent-Child 双集合；候选池放大 fetch_factor=5×, SDK 查询 8×                                                                                                                                                                                                 |
| 🔴 BM25Okapi 增量索引 | 内存索引按 product_id 分组 (`_bm25_indexes`)；建库全量构建 + `bm25_upsert_product` 增量重建 IDF + FastAPI startup `build_bm25_from_chromadb` 恢复                                                                                                                                                                                                                                                             |
| BM25 检索             | jieba + 标识符保护 — snake_case 函数名不被切碎; v26: 复合词原子化（`Ethernet/IP`→整体 token，排除 `.`）+ 分隔符空格归一化（双侧对称）; v27: 短文本(≤8字)/复合词查询 BM25 权重动态 3.0                                                                                                                                                                                                                        |
| RRF 六大提权引擎      | Entity Anchor (+5.0) / Function Names (+0.08) / Text Rebalance (+0.03) / CODE BM25 三倍写入 / Title Exact Match (+5.0) / Chapter Isolation (+20.0/-10.0)                                                                                                                                                                                                                                                            |
| Autocut 动态截断      | `_autocut_knee()` 断崖检测 — 找 RRF 分数相邻差值最大点; SDK 场景 min_k=10                                                                                                                                                                                                                                                                                                                                        |
| 复合查询拆解          | `_decompose_compound_query()` 顺序连接词 — `_MIN_SUB_QUERY_LEN=2` 保留两字核心动词                                                                                                                                                                                                                                                                                                                             |
| 保底召回              | 三层防护: 阈值 0→原始 Top-3 / 噪声全杀→kept_docs 恢复 / 最终空→BM25 第二机会                                                                                                                                                                                                                                                                                                                                     |
| 产品隔离              | ChromaDB`where={"product_id":"xxx"}` + 未指定时 Search-First 软路由                                                                                                                                                                                                                                                                                                                                               |
| HyDE 防毒化           | SDK 轨 + JAKA 全线封杀; 短 Query/非技术符号/精确 API 签名 → 禁用                                                                                                                                                                                                                                                                                                                                                   |
| LLM 意图重写          | `_rewrite_query_with_llm()` ADR-19 — 代词消解 + 产品名补全 (t=0.0, max_tokens=50)；v26: always-on 执行 + 同音纠错/名词补全规则与 Few-Shot + max_tokens=128；v27: 规则 9 产品名缺失保持缺失（严禁脑补）+ 路由责任切分（原始 query 优先判定）；v28: 规则 3 实体指代泛化（产品/函数/动作类型）+ 逐字来源限定；v29: 规则 2 协议主题中立性限制 + `_PROTOCOL_TERMS_RE` 确定性兜底（单发 "Ethernet/IP" 不拼接产品名） |
| GUI 噪声豁免 (v23)    | `_is_gui` 判定 → 跳过 kw_score 拦截                                                                                                                                                                                                                                                                                                                                                                              |
| 宏观提权 v2 (v23)     | 多关键词广谱判定 + chunk_type 双重检测 → +5.0 登顶                                                                                                                                                                                                                                                                                                                                                                 |

### L3 — 上下文组装与指令层 (rag_chain.py `_build_messages` + `RAG_SYSTEM_PROMPT`)

🔴 **v24 核心变更: Markdown 模板强约束 (Template Masking)**

| 能力                         | 说明                                                                                                                                                                                                                                                                                                                                      |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🔴 System Prompt 极简        | 从 210 行压缩至 ~15 行 (~250 tokens, v23 的 1/6) — 所有格式约束走模板                                                                                                                                                                                                                                                                    |
| 🔴 Markdown 填空模板         | gui_app: 首句出处 +`[填写操作步骤]` 槽位 / c_sdk: 首句出处 + `[准确函数名]([参数])` 槽位                                                                                                                                                                                                                                              |
| 🔴 模板底端锚定              | 模板置于 User Message 末尾，利用 Recency Bias 实现注意力锚定                                                                                                                                                                                                                                                                              |
| 🔴 Top-1 来源                | `_doc_section_str` 仅取排名第一的章节 — 单一锚点降低小模型认知负担                                                                                                                                                                                                                                                                     |
| 双轨制 Prompt                | c_sdk (两段式铁律) / gui_app (六条铁律: 宏观总结/结构清晰/历史隔离/视觉屏蔽/禁止脑补/禁止代码)                                                                                                                                                                                                                                            |
| 🔴 v25: 逃生舱条款           | 双轨模板末尾追加 — 上下文无对应函数/硬件/超纲内容或触发隔离警告时，LLM 彻底无视排版模板，仅输出一句拒答 (零业务正则)                                                                                                                                                                                                                     |
| 🔴 v26: 逃生舱视觉加固       | `> [!WARNING] ⛔🔴 绝密拦截` GitHub Alert 引用块语法 + 删除尾部对冲行（"请明确说明"软出口）— 模板即消息尾部，极致 Recency Bias                                                                                                                                                                                                         |
| 🔴 v27: 模板选择守卫         | L3 层三条件（query 点名函数不在 Context / 非 SDK 产品+SDK 问法 / coverage 提问+技术强词零命中）→ 双轨模板整体替换为拒答模板 + 回删 SDK Header；Plan 代理逐例评审 35 用例误伤面为零                                                                                                                                                       |
| 🔴 v28: 守卫 context 脱敏    | 守卫命中 →`_strip_code_from_context()` 剥离 ``` 代码块与 DLL 加载行（`[代码内容省略]`）——模型无代码可抄；误伤面为零                                                                                                                                                                                                                |
| 🔴 v29: Fast-Path 确定性拒答 | `_build_messages` 返回 `(messages, refusal_flag)` 侧信道；守卫命中 → 四调用方跳过 LLM 直出 `_HARD_REFUSAL`（物理根除拒答记忆中毒；检查点在生成金字塔之前封堵 Layer 3 泄漏）                                                                                                                                                        |
| 🔴 v29: 数字守卫豁免         | `_SPACE_SEP_RE` 归一化 + `_COMPOUND_RE` 剥离（Ethernet/IP、TCP-IP 整体）后跑 `_NUMERIC_QUERY_RE` 与关键词判定——协议名词不再被裸 "IP" 误杀                                                                                                                                                                                         |
| 动态术语对齐                 | `_term_alignment_prefix` 按需注入 — OpenR6 "使能"→`set_robot_arm_init`，零全局 Token 损耗                                                                                                                                                                                                                                           |
| 反跨产品泄露                 | `_anti_bleed_prefix` metadata + 正文双重确认                                                                                                                                                                                                                                                                                            |
| Context Cap                  | 非SDK 4000 / SDK 8000 字符整块剔除 — Parent 背景优先丢弃                                                                                                                                                                                                                                                                                 |
| 历史净化                     | `sanitize_chat_history()` 5 步清洗 — Citation 剥离/代码块替换/拒答过滤/尾部套话擦除                                                                                                                                                                                                                                                    |
| 柔性 Grounding               | `_NUMERIC_QUERY_RE` 动态检测 — Context 无数值 → 追加诚实提示                                                                                                                                                                                                                                                                          |
| 🔴 Stage 3: KV 确定性侧信道  | `kv_extractor.lookup_attribute(query, product_id)` 读 `kv_db/attribute_kv.json`（两级嵌套 + 人工校准 6502/9600）→ 4 注入点（run_graph / run_graph_stream / rag_chat / rag_chat_stream，均在 `_build_messages` 后、LLM 前）前置注入；触发 `_last_numeric_context_missing or _NUMERIC_QUERY_RE`；未命中 → BM25 第二机会 → 硬拒答 |

### L4 — 生成控制与后处理层 (graph_rag.py 后处理节点 + rag_chain.py LLM 调用)

🔴 **v24 核心变更: L4 从"擦屁股"简化为"兜底校验"**

| 能力                   | 说明                                                                                                                                         |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 🔴 极速流式穿透        | `_stream_guardrail` 零缓冲逐 chunk 透传 — TTFB <2s (v23: 60-90s)                                                                          |
| 🔴 v25: 围栏闭合状态机 | `_stream_guardrail` 透传中统计 `` ``` `` 奇偶性，流结束奇数时自动补发闭合行 — 零缓冲代价                                                  |
| 🔴 render_node 退化    | 从 JSON 解析+结构化渲染退化为纯文本透传 — 格式正确性由 L3 模板保证                                                                          |
| 四层容灾金字塔         | L1 本地 vLLM → L2 智谱 API → L3 纯检索直出 → L4 硬拒答 — NEVER-EMPTY 保证                                                                |
| 静默斩尾               | `_strip_hedging_tail()` 8 模式 — "上述代码假设存在"/"参考文档未包含详细步骤"等                                                            |
| 属性词硬改写           | `extract_align_node` 50+ 领域词库 — 数值前后 12+8 字符窗口 + Context 原词强制覆盖                                                         |
| SDK 自纠错             | `sdk_verify_node` → `llm_generation` 回环 — set_前缀/CDLL/argtypes 检测 + 硬熔断 retry≤2                                              |
| 代码块闭合             | `_fix_and_close_sdk_code()` 过渡期兜底 — Markdown ``` 自动闭合 + CDLL 补全 + 函数名修正表; v25: 接入 extract_align_node 覆盖 Graph 全路径 |
| SemanticDedup          | trigram overlap > 0.55 截断 — v25: 无条件精确段落去重 (连续相同 ≥80ch) + 代码块跳过模糊去重                                                |
| Temperature            | 非流式 0.2 / 流式 0.01 — 代码生成近确定性输出                                                                                               |

---

## 📦 支持产品线

| 产品线                      | 文档源                                                                | 解析专轨                                                                   | 当前库量 (2026-08-22 实测) |
| --------------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------------- | -------------------------- |
| **JAKA 六轴机器人**   | `data/JAKA_Manual.pdf` → MinerU Markdown (`data/jaka_markdown/`) | 🔴 JAKA 专轨: MinerU + Qwen2-VL 多模态提纯 + HTML 表格规整 + 1500ch 软装箱 | Parent 9 / Child 225       |
| **OpenC3 六轴机械臂** | `data/OpenC3六轴机械臂SDK说明文档_win.pdf`                          | 🔴 SDK 专轨: fitz`sort=True` 物理坐标流 + 章节状态机原子切片             | Parent 1 / Child 27        |
| **OpenR6 六轴机械臂** | `data/windows系统OpenR6_sdk使用文档.pdf`                            | 🔴 SDK 专轨 + OpenR6 目录噪声剔除 (`_strip_openr6_toc`)                  | Parent 1 / Child 32*       |
| ⚠️ General（噪声）        | `data/ROS机器实践应用.pdf`（ROS 书籍，非产品文档）                  | 统一入口扫描误摄入                                                         | Parent 1 / Child 1         |

\* OpenR6 当前库中 32 切片含 2 个目录页 TOC 噪声块；`_strip_openr6_toc` 修复后重解析实测 31 切片（30 真实章节 + SDK 基础配置前置块），**需全量重建后入库生效**。ROS 书籍 PDF 属数据目录噪声，建议移出 `data/` 或配置产品映射。

---

## 📄 文档解析双轨制（v31）

不同文档类型走不同的解析管线，互不侵入：

| 文档类型                           | 解析管线                                                                                                                        | 入口                         |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| **JAKA 手册 / GUI 文档**     | **MinerU (magic-pdf 1.3.12)** — 深度学习版面分析 + OCR + 表格识别 → Markdown                                            | `src/parse_jaka_mineru.py` |
| **SDK 文档 (OpenC3/OpenR6)** | **Stage 1 重构**: PyMuPDF (fitz) 状态机管线 `_v4_parse_sdk_state_machine()` + `_SDK_CHAPTER_BOUNDARY_RE` 章节原子切片 | `src/rebuild_v4.py`        |
| **JAKA 截图参数**            | Qwen2-VL-7B (:8005) 多模态提纯 — 三重图片防线 + 防废话 Prompt + 提纯缓存优先 (`data/jaka_manual_chunks.json`)                | `load_jaka_mineru_dual()`  |

### MinerU 离线解析使用

```bash
conda activate rag_agent
python src/parse_jaka_mineru.py     # 权重检查 → magic-pdf.json 生成 → GPU 自适应 → 解析
```

- **模型权重**: `~/LLM/MinerU_Models`（modelscope `OpenDataLab/PDF-Extract-Kit-1.0`）
- **输出**: `data/jaka_markdown/JAKA_Manual/auto/JAKA_Manual.md`
- **配置要点**（`~/magic-pdf.json` 由脚本自动生成）:
  - `layout-config: {"model": "doclayout_yolo"}` — **必须显式指定**，缺省会回退 layoutlmv3（detectron2 崩溃）
  - `table-config: {"model": "rapid_table"}` — 合法名 `tablemaster` / `rapid_table` / `struct_eqtable`
- **环境补丁**: transformers ≥4.49 下需先执行 `auto_patch_mbart.py` / `patch_unimernet.py`（unimernet `cache_position` 文件级补丁）
- **已知事项**: 大表输出为 HTML `<table>` 混合 Markdown（已由 `clean_html_tables()` 无损规整为 GitHub Markdown）；MinerU 依赖（transformers≥4.49）与 vLLM 0.5.4（<4.46）版本互斥，升级任一方前需评估

---

## ⚙️ 系统环境与配置

| 项目                                   | 值                                                                                                                                                                                                                                                      |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **硬件底座**                     | 2 × NVIDIA A100-PCIE-40GB（CUDA 12.4）                                                                                                                                                                                                                 |
| **环境管理器**                   | Conda（`rag_agent`，Python 3.10）                                                                                                                                                                                                                     |
| **推理引擎**                     | vLLM 0.16.0 @ 端口**8001**                                                                                                                                                                                                                        |
| **默认模型**                     | `Qwen/Qwen2.5-7B-Instruct-AWQ` (4-bit ~8 GB)                                                                                                                                                                                                          |
| **云端降级**                     | 智谱 GLM-4.7-Flash (`open.bigmodel.cn`)                                                                                                                                                                                                               |
| **嵌入模型**                     | `BAAI/bge-small-zh-v1.5` (512 维，中文专优) → ONNX 自动回退                                                                                                                                                                                          |
| **Web 框架**                     | FastAPI (`8000`) + Jinja2 + 前端 UI (`8501`)                                                                                                                                                                                                        |
| **当前向量库 (2026-08-22 实测)** | 12 Parent + 285 Child（JAKA 9/225 · OpenC3 1/27 · OpenR6 1/32* · General 1/1）——\*OpenR6 含 2 个目录页 TOC 噪声块，`_strip_openr6_toc` 修复后全量重建预期 12P+284C（OpenR6 31）；General 来自 `ROS机器实践应用.pdf` 误摄入，建议移出 `data/` |

### 关键配置参数

| 参数                                                 | 值                           | 说明                                                                                      |
| ---------------------------------------------------- | ---------------------------- | ----------------------------------------------------------------------------------------- |
| `RETRIEVAL_K`                                      | 10                           | 单次检索召回数                                                                            |
| `SIMILARITY_THRESHOLD`                             | 0.68                         | Cosine 距离阈值                                                                           |
| `_AUTOCUT_MIN_K`                                   | 8 (SDK: 10)                  | Autocut 硬下限                                                                            |
| `_AUTOCUT_MAX_K`                                   | 15                           | Autocut 硬上限                                                                            |
| `_MIN_SUB_QUERY_LEN`                               | 2                            | 复合查询拆解最小子句长度                                                                  |
| `_MAX_CONTEXT_CHARS`                               | 4000 / 8000 (SDK)            | Context 字符 Cap                                                                          |
| `max_tokens`                                       | 1024                         | LLM 最大输出                                                                              |
| `MAX_HISTORY_TURNS`                                | 2                            | 滑动窗口轮数                                                                              |
| `SDK_CHILD_CHUNK_SIZE` / `SDK_PARENT_CHUNK_SIZE` | 400 / 1000                   | Stage 1: SDK 轨切片容量                                                                   |
| `GUI_CHILD_CHUNK_SIZE` / `GUI_PARENT_CHUNK_SIZE` | 1500 / 2000                  | Stage 1: JAKA/GUI 轨切片容量（`CHILD_CHUNK_SIZE`/`PARENT_CHUNK_SIZE` 为向下兼容别名） |
| 🔴**System Prompt tokens**                     | **~250** (v23: ~1,500) | v24: 极简瘦身, Token 节省 83%                                                             |
| 🔴**TTFB (流式)**                              | **<2s** (v23: 60-90s)  | v24: 零缓冲透传                                                                           |
| `_temperature` (stream)                            | 0.01                         | 代码近确定性输出                                                                          |

**🔴 核心锁定依赖（严禁升级）**：

- `torch==2.6.0+cu124` / `torchvision==0.21.0+cu124` / `torchaudio==2.6.0+cu124`
- `vllm==0.16.0`（`--no-deps` 隔离安装）
- `sentence-transformers==2.7.0`

---

## 🚀 部署与启动

### 1. 准备文档

将 PDF 文档放入 **`data/`** 目录。

### 2. 一键部署与 Web 启动（start_services.sh / app.py）

```bash
chmod +x start_services.sh

./start_services.sh                    # 一键部署 (GPU 智能检测 → vLLM :8001 → FastAPI :8000)
VLLM_GPU_ID=0 ./start_services.sh      # 手动指定 GPU (环境变量覆盖)

python app.py                          # 仅启动 Web 后端 FastAPI (:8000)
python frontend_server.py              # 前端 UI (:8501)

# Stage 1 双轨建库与验收 (v32 收口)
python src/rebuild_v4.py           # 双轨切片 + VLM 提纯 + ChromaDB 全量重建
python tests/test_stage1.py        # Stage 1 离线验收测试 (切片/表格/KV 断言)
```

> 🔴 注 (2026-08-22 审查实测): `start_services.sh` **无** `--vllm-only`/`--fastapi-only`/`--gpu` CLI 参数（早期文档记载有误），GPU 覆盖仅支持 `VLLM_GPU_ID` 环境变量；项目根目录**无 run.sh**，唯一一键脚本为 `start_services.sh`。

### 3. 手动启动

**终端 A — vLLM 推理服务**：

```bash
conda activate rag_agent
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --port 8001 --gpu-memory-utilization 0.25 \
    --max-model-len 8192 --enforce-eager --quantization awq
```

**终端 B — 比邻星 FastAPI 后端**：

```bash
conda activate rag_agent
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
python app.py
```

**终端 C — 前端 UI**：

```bash
conda activate rag_agent
python frontend_server.py
```

访问：**`http://localhost:8501`** | API 文档：`http://localhost:8000/docs`

### 4. v4 向量库重建

```bash
# 全量重建 (物理清库 → 双轨切片 → VLM 提纯 → ChromaDB 双 Collection)
conda run -n rag_agent python src/rebuild_v4.py
# 或: python -m src.rebuild_v4

# 增量摄入 (按扩展名双轨路由: .pdf → SDK 专轨 / .md → JAKA 专轨)
curl -X POST -F "file=@your_document.pdf" http://localhost:8000/api/upload
```

### 5. 一键停止

```bash
fuser -k 8000/tcp 2>/dev/null    # FastAPI
fuser -k 8501/tcp 2>/dev/null    # 前端
pkill -f "vllm.entrypoints"      # vLLM
```

### 6. 系统健康检查

```bash
python check_status.py                # 一次性完整报告
python check_status.py --watch 10     # 每 10 秒刷新
python tests/audit_ingestion.py       # 向量库白盒质检 (4 规则)
```

### 7. ngrok 内网穿透（tunnel.py）

基于 pyngrok 的公网映射脚本，将本地 FastAPI 后端暴露到外网：

```bash
python tunnel.py                          # 匿名模式 (有速率限制), 默认映射 http://localhost:8000
python tunnel.py --token <NGROK_TOKEN>    # 官方 token (推荐)
export NGROK_AUTHTOKEN=<NGROK_TOKEN> && python tunnel.py   # 环境变量方式
python tunnel.py --port 8501              # 映射其它端口 (如前端 UI)
```

- 默认目标: `http://localhost:8000`（FastAPI 后端）；前端 UI 需 `--port 8501`。
- 隧道建立后脚本保持运行，`Ctrl+C` 断开；免费版 URL 每次重启变化，需重新获取。
- 传统端口映射对照（内网 → 外网）：FastAPI `8000` → `50003`；前端 UI `8501` → `50004`；vLLM `8001` 仅内网。

---

## 🧪 自动化测试

**统一评测入口**: `python tests/run_eval.py --verbose`（35 用例 × 8 硬断言）

| 脚本                         | 覆盖范围                                                                                                                                                                           | 命令                                   |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `tests/run_eval.py`        | 35 用例 (GT-1~6 + E01~E29) × 8 硬断言 (①JSON泄露 ②段落重复 ③界面套话 ④函数签名 ⑤提示词泄露 ⑥API幻觉 ⑦零脑补 ⑧代码截断)；`--quick` 仅检索不调 LLM、`-f <用例ID>` 过滤 | `python tests/run_eval.py --verbose` |
| `tests/test_stage1.py`     | 🔴 Stage 1 离线验收 — 18 断言 / 4 组 (SDK 章节数+Ctypes 零污染 / JAKA 表格零残留+附录四 / KV 校准 6502·9600 / 统一入口三产品线)                                                  | `python tests/test_stage1.py`        |
| `tests/audit_ingestion.py` | 🔴 Stage 2 向量库白盒质检 — 4 规则 (零切片 / 垃圾切片 / 高压实体存活 / 架构标记注入)；⚠️ 规则 3b/4a/4b 口径待适配 Stage 1 (6502 走 KV 侧信道等)                                 | `python tests/audit_ingestion.py`    |
| `tests/test_stability.py`  | ⚠️ 多轮对话 + 并发保护 (依赖旧 rag_chat 入口, 待适配)                                                                                                                            | —                                     |
| `audit_chunks.py`          | ⚠️ 切片 8 维健康度审计 (依赖旧 pdf_loader 入口, 已被`tests/audit_ingestion.py` 取代)                                                                                           | —                                     |

---

## 📡 API 接口

| 方法     | 路径                          | 说明                                                                                                       |
| -------- | ----------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `GET`  | `/`                         | 渲染 比邻星 (ProximaRAG) 主页面                                                                            |
| `POST` | `/api/chat`                 | RAG 对话 (SSE 流式)。`query`(必填) / `history`(JSON可选) / `stream`(默认true) / `product_id`(可选) |
| `POST` | `/api/upload`               | 上传 PDF/MD 并增量更新向量库 (按扩展名双轨路由至 Stage 1 解析引擎)                                         |
| `GET`  | `/api/status`               | 向量库就绪状态与文档片段数                                                                                 |
| `GET`  | `/api/products`             | 已入库产品 ID 列表                                                                                         |
| `GET`  | `/api/debug/inspect_chunks` | 切片检查器 (按产品/关键词过滤)                                                                             |
| `POST` | `/api/debug/retrieve`       | 检索沙盒 (不调 LLM，仅输出管线中间结果)                                                                    |

### 外网端口映射

| 服务         | 内部端口 | 外部端口 |
| ------------ | -------- | -------- |
| FastAPI 后端 | 8000     | 50003    |
| 前端 UI      | 8501     | 50004    |
| vLLM 推理    | 8001     | 仅内网   |

---

## 📝 开发日志与架构审计

- **[dev_log.md](./dev_log.md)**: 从 2026-07-20 至今完整开发记录与架构决策（最新: Stage 2-4 收口 — 向量库单例化 + BM25 增量 + KV 侧信道 + OpenR6 目录噪声剔除 + 端到端验证）
- **[ARCHITECTURE_AUDIT.md](./ARCHITECTURE_AUDIT.md)**: 🔴 Stage 1→4 全量架构审查报告 (2026-08-22)（四层评分 / 六项 Bug 根因 / 13 项新识别架构债；历史 v24/v30 审计归档于附录）
- **[CLAUDE.md](./CLAUDE.md)**: AI 协同开发规范（含 v24 四层架构排雷法思想钢印：System Prompt 极简/模板底端锚定/流式零缓冲/render_node 纯透传/L4 正则最小化；v25: 逃生舱条款/围栏闭合状态机/JAKA 数字保护特判；v26: OCR Y 归位/复合词原子化/重写器 always-on；v27: 路由责任切分/模板选择守卫/OCR 回退；v28: 区域状态机标题提取/line 级表格重建/last_header 层级栈；v29: OCR 键值法/Fast-Path 确定性拒答/数字守卫豁免/重写中立性；v30/v30.final: AST-Lite 软装箱/全景快照 OCR/孤儿行合并；v31: 数据摄入双轨制 — JAKA→MinerU / SDK→原管线；Stage 1: SDK 专轨 fitz 状态机 + JAKA VLM 提纯 + KV 属性库 + 统一入口）
- **[tests/TEST_REPORT.md](./tests/TEST_REPORT.md)**: 评测报告归档
