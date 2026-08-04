# 📰 比邻星 (ProximaRAG) — 湖南比邻星科技文档智能问答系统

基于 **RAG（Retrieval-Augmented Generation）** 架构的官方技术文档与使用手册智能问答系统。专为**湖南比邻星科技有限公司**的开发者与用户打造，采用双 A100 GPU 算力底座，底层搭载 **vLLM + Qwen2.5-7B-Instruct-AWQ** 实现完全私有化、低延迟的本地推理。

> **🔴 v24 架构升级 (2026-08-04)**: 全面转向 **Markdown 模板强约束 (Template Masking) + 极速流式穿透** 架构。废弃了此前的 JSON 提取+正则清洗后处理管线，System Prompt 从 210 行压缩至 ~15 行（Token 节省 83%），TTFB 从 60-90s 降至 <2s。

---

## 🏛️ RAG 四层系统架构

比邻星 (ProximaRAG) 遵循工业级 RAG 四层架构设计：

```
                            ┌──────────────────────────────┐
                            │   FastAPI Gateway (:8000)     │
                            │ (输入清洗 / 路径防御 / SSE)   │
                            └─────────────┬────────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
              ▼                           ▼                           ▼
┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
│  L1: 数据摄入与切片层    │ │  L2: 检索与重排层        │ │  L3: 上下文组装与指令层   │
│                         │ │                         │ │ 🔴 v24: 模板约束架构     │
│ • PDF 通用文本提取      │ │ • Dense 向量 (bge-zh)   │ │ • System Prompt ~250t   │
│ • 7 步 PDF 文本清洗     │ │ • Sparse BM25 (jieba)   │ │ • 🔴 Markdown 填空模板  │
│ • 5 类标题模式识别      │ │ • RRF 六大提权引擎      │ │ • 🔴 模板底端锚定       │
│ • 代码注释拦截 (8特征词) │ │ • Autocut 断崖动态截断   │ │ • 反跨产品泄露门控       │
│ • 伪标题黑名单 (10项)   │ │ • HyDE 防毒化 (全线禁用) │ │ • 🔴 Top-1 来源锚定     │
│ • Parent-Child 双层索引 │ │ • 三层保底召回机制       │ │ • 动态术语对齐 (零Token) │
│ • SDK 状态机 API 原子块 │ │ • 代码实体 BM25 3倍写入  │ │ • Context Cap 整块剔除   │
│ • 4 级 Title Fallback   │ │ • 产品级物理隔离         │ │ • 历史沉渣净化 + 滑动窗口 │
│ • GUI 动态切片 1500ch   │ │ • GUI 噪声过滤豁免       │ │ • 父子结构化组装          │
│ • 微缩大纲降噪 (上限5条) │ │ • LLM 意图重写 (ADR-19) │ │ • SDK Header 单次注入     │
└────────────┬────────────┘ └────────────┬────────────┘ └────────────┬────────────┘
             │                           │                           │
             └───────────────────────────┼───────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  L4: 生成控制与后处理层 🔴 v24: 从"擦屁股"简化为"兜底校验"                      │
│                                                                                │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐  ┌───────────┐ │
│  │ 四层容灾金字塔   │  │ 🔴 极速流式穿透  │  │ 静默斩尾         │  │ 属性词    │ │
│  │ L1 vLLM → L2 智谱│  │ _stream_guardrail│  │ _strip_hedging_  │  │ 硬改写    │ │
│  │ → L3 直出 → L4   │  │ 零缓冲逐chunk    │  │ tail()           │  │ Extract   │ │
│  │ 硬拒答           │  │ 透传 TTFB <2s    │  │ 8 模式正则       │  │ Align     │ │
│  └─────────────────┘  └──────────────────┘  └─────────────────┘  └───────────┘ │
│  ┌─────────────────┐  ┌──────────────────┐                                      │
│  │ SDK 自纠错回路   │  │ 🔴 render_node   │                                      │
│  │ (set_前缀/CDLL/  │  │ 文本透传 (废弃   │                                      │
│  │  argtypes)       │  │ JSON 解析)       │                                      │
│  │ 硬熔断 retry≤2   │  │                  │                                      │
│  └─────────────────┘  └──────────────────┘                                      │
└────────────────────────────────────────────────────────────────────────────────┘
```

### L1 — 数据摄入与切片层 (pdf_loader.py, ~1938 行)

**处理流程**: `PDF 文件` → `_v4_extract_text_universal()` (PyMuPDF + OCR 归位) → `_clean_pdf_text()` (7 步清洗) → `_v4_build_parent_child_docs()` (标题树切分 + 双层索引)

| 能力 | 说明 |
|------|------|
| 标题识别 | `_v4_extract_headings()` 5 类模式 + `{1,5}` 支持 6 级深度; v23: `doc_type` 动态双轨拦截, GUI 禁止单数字编号提权 |
| 代码注释拦截 | ±120 字符窗口 + 8 特征词校验 — 防止 `# 时间等待3秒` 提权为 Heading |
| 伪标题黑名单 | `_PSEUDO_SECTION_BLACKLIST` frozenset 10 项 — 触发后自动继承父级 H2 标题 |
| API 原子块 | `_v4_parse_sdk_state_machine()` 状态机 — 仅 `数字标题` + `函数名称/函数说明` 两类可验证边界 |
| 微碎片缝合 | Micro-Chunk Auto-Merge (≥60ch 或含代码 → 提交) — API 排他锁：不同函数名不合并 |
| 骨架过滤 | `_is_skeleton_chunk()` — <150ch + 无代码 + 无实质参数 → 丢弃 |
| Parent-Child 双层 | H2 章节级 Parent(1000ch) + H3/H4 函数级 Child(400ch); v23: GUI 轨动态扩容 Child=1500/Parent=2000 |
| Title Fallback | 4 级链: L1 状态机标题 → L2 面包屑 → L3 父级 H2 → L4 硬兜底 |
| PDF 清洗 | 7 步 `_clean_pdf_text()` — Unicode 连字/括号空格/下划线归一化/I/O 修复/边界错位/表格竖线/JAKA 版式 |
| v23: 跨级大纲扫描 | Parent TOC 延伸到下一个同级/更高级标题 — H1 章节完整囊括子章节 |
| v23: 微缩大纲降噪 | Child TOC 上限 5 条 + `[章节大纲参考]:` 标签统一 |

### L2 — 检索与重排层 (rag_chain.py + vector_store.py)

**处理流程**: `Query` → `_rewrite_query_with_llm()` (LLM 意图重写) → `_preprocess_query()` (口语剥离) → `_generate_hyde_doc()` (SDK/GUI 全线禁用) → `_hybrid_retrieve()` (向量 + BM25 + RRF 六大引擎 + Autocut)

| 能力 | 说明 |
|------|------|
| 向量检索 | ChromaDB cosine (bge-small-zh-v1.5, 512维) — 候选池放大 fetch_factor=5×, SDK 查询 8× |
| BM25 检索 | jieba + 标识符保护 — snake_case 函数名不被切碎 |
| RRF 六大提权引擎 | Entity Anchor (+5.0) / Function Names (+0.08) / Text Rebalance (+0.03) / CODE BM25 三倍写入 / Title Exact Match (+5.0) / Chapter Isolation (+20.0/-10.0) |
| Autocut 动态截断 | `_autocut_knee()` 断崖检测 — 找 RRF 分数相邻差值最大点; SDK 场景 min_k=10 |
| 复合查询拆解 | `_decompose_compound_query()` 顺序连接词 — `_MIN_SUB_QUERY_LEN=2` 保留两字核心动词 |
| 保底召回 | 三层防护: 阈值 0→原始 Top-3 / 噪声全杀→kept_docs 恢复 / 最终空→BM25 第二机会 |
| 产品隔离 | ChromaDB `where={"product_id":"xxx"}` + 未指定时 Search-First 软路由 |
| HyDE 防毒化 | SDK 轨 + JAKA 全线封杀; 短 Query/非技术符号/精确 API 签名 → 禁用 |
| LLM 意图重写 | `_rewrite_query_with_llm()` ADR-19 — 代词消解 + 产品名补全 (t=0.0, max_tokens=50) |
| GUI 噪声豁免 (v23) | `_is_gui` 判定 → 跳过 kw_score 拦截 |
| 宏观提权 v2 (v23) | 多关键词广谱判定 + chunk_type 双重检测 → +5.0 登顶 |

### L3 — 上下文组装与指令层 (rag_chain.py `_build_messages` + `RAG_SYSTEM_PROMPT`)

🔴 **v24 核心变更: Markdown 模板强约束 (Template Masking)**

| 能力 | 说明 |
|------|------|
| 🔴 System Prompt 极简 | 从 210 行压缩至 ~15 行 (~250 tokens, v23 的 1/6) — 所有格式约束走模板 |
| 🔴 Markdown 填空模板 | gui_app: 首句出处 + `[填写操作步骤]` 槽位 / c_sdk: 首句出处 + `[准确函数名]([参数])` 槽位 |
| 🔴 模板底端锚定 | 模板置于 User Message 末尾，利用 Recency Bias 实现注意力锚定 |
| 🔴 Top-1 来源 | `_doc_section_str` 仅取排名第一的章节 — 单一锚点降低小模型认知负担 |
| 双轨制 Prompt | c_sdk (两段式铁律) / gui_app (六条铁律: 宏观总结/结构清晰/历史隔离/视觉屏蔽/禁止脑补/禁止代码) |
| 动态术语对齐 | `_term_alignment_prefix` 按需注入 — OpenR6 "使能"→`set_robot_arm_init`，零全局 Token 损耗 |
| 反跨产品泄露 | `_anti_bleed_prefix` metadata + 正文双重确认 |
| Context Cap | 非SDK 4000 / SDK 8000 字符整块剔除 — Parent 背景优先丢弃 |
| 历史净化 | `sanitize_chat_history()` 5 步清洗 — Citation 剥离/代码块替换/拒答过滤/尾部套话擦除 |
| 柔性 Grounding | `_NUMERIC_QUERY_RE` 动态检测 — Context 无数值 → 追加诚实提示 |

### L4 — 生成控制与后处理层 (graph_rag.py 后处理节点 + rag_chain.py LLM 调用)

🔴 **v24 核心变更: L4 从"擦屁股"简化为"兜底校验"**

| 能力 | 说明 |
|------|------|
| 🔴 极速流式穿透 | `_stream_guardrail` 零缓冲逐 chunk 透传 — TTFB <2s (v23: 60-90s) |
| 🔴 render_node 退化 | 从 JSON 解析+结构化渲染退化为纯文本透传 — 格式正确性由 L3 模板保证 |
| 四层容灾金字塔 | L1 本地 vLLM → L2 智谱 API → L3 纯检索直出 → L4 硬拒答 — NEVER-EMPTY 保证 |
| 静默斩尾 | `_strip_hedging_tail()` 8 模式 — "上述代码假设存在"/"参考文档未包含详细步骤"等 |
| 属性词硬改写 | `extract_align_node` 50+ 领域词库 — 数值前后 12+8 字符窗口 + Context 原词强制覆盖 |
| SDK 自纠错 | `sdk_verify_node` → `llm_generation` 回环 — set_前缀/CDLL/argtypes 检测 + 硬熔断 retry≤2 |
| 代码块闭合 | `_fix_and_close_sdk_code()` 过渡期兜底 — Markdown ``` 自动闭合 + CDLL 补全 + 函数名修正表 |
| SemanticDedup | trigram overlap > 0.55 截断 — v23: JAKA GUI 轨完整保留重复句 |
| Temperature | 非流式 0.2 / 流式 0.01 — 代码生成近确定性输出 |

---

## ⚙️ 系统环境与配置

| 项目 | 值 |
|------|-----|
| **硬件底座** | 2 × NVIDIA A100-PCIE-40GB（CUDA 12.4） |
| **环境管理器** | Conda（`rag_agent`，Python 3.10） |
| **推理引擎** | vLLM 0.16.0 @ 端口 **8001** |
| **默认模型** | `Qwen/Qwen2.5-7B-Instruct-AWQ` (4-bit ~8 GB) |
| **云端降级** | 智谱 GLM-4.7-Flash (`open.bigmodel.cn`) |
| **嵌入模型** | `BAAI/bge-small-zh-v1.5` (512 维，中文专优) → ONNX 自动回退 |
| **Web 框架** | FastAPI (`8000`) + Jinja2 + 前端 UI (`8501`) |
| **当前向量库** | 120 Parent + 376 Child = **496 chunks** (v4 dual index) |

### 关键配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `RETRIEVAL_K` | 10 | 单次检索召回数 |
| `SIMILARITY_THRESHOLD` | 0.68 | Cosine 距离阈值 |
| `_AUTOCUT_MIN_K` | 8 (SDK: 10) | Autocut 硬下限 |
| `_AUTOCUT_MAX_K` | 15 | Autocut 硬上限 |
| `_MIN_SUB_QUERY_LEN` | 2 | 复合查询拆解最小子句长度 |
| `_MAX_CONTEXT_CHARS` | 4000 / 8000 (SDK) | Context 字符 Cap |
| `max_tokens` | 1024 | LLM 最大输出 |
| `MAX_HISTORY_TURNS` | 2 | 滑动窗口轮数 |
| `CHILD_CHUNK_SIZE` | 400 / 1500 (GUI) | v23: GUI 轨扩容防止长步骤断裂 |
| `PARENT_CHUNK_SIZE` | 1000 / 2000 (GUI) | v23: GUI 轨同步扩容 |
| 🔴 **System Prompt tokens** | **~250** (v23: ~1,500) | v24: 极简瘦身, Token 节省 83% |
| 🔴 **TTFB (流式)** | **<2s** (v23: 60-90s) | v24: 零缓冲透传 |
| `_temperature` (stream) | 0.01 | 代码近确定性输出 |

**🔴 核心锁定依赖（严禁升级）**：
- `torch==2.6.0+cu124` / `torchvision==0.21.0+cu124` / `torchaudio==2.6.0+cu124`
- `vllm==0.16.0`（`--no-deps` 隔离安装）
- `sentence-transformers==2.7.0`

---

## 🚀 部署与启动

### 1. 准备文档

将 PDF 文档放入 **`data/`** 目录。

### 2. 一键启动（推荐）

```bash
chmod +x start_services.sh

./start_services.sh                    # 完整启动 (GPU 智能检测 → vLLM → FastAPI)
./start_services.sh --vllm-only        # 仅 vLLM
./start_services.sh --fastapi-only     # 仅 FastAPI
./start_services.sh --gpu 0            # 手动指定 GPU
```

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
# 全量重建
conda run -n rag_agent python rebuild_v4.py

# 增量上传 (MD5 去重 + 级联清理)
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
python audit_chunks.py                # 切片健康度审计
```

---

## 🧪 自动化测试

**统一评测入口**: `python tests/run_eval.py --verbose`（33 用例，8 硬断言）

| 脚本 | 覆盖范围 | 命令 |
|------|---------|------|
| `tests/run_eval.py` | 33 用例 (GT + SDK 函数 + 安全注入 + 多轮指代 + 错别字容错 + v23: 微观防泛化/短文本召回/特殊符号) | `python tests/run_eval.py --verbose` |
| `test_stability.py` | 多轮对话 + 并发保护 + 7 种异常降级 | `python test_stability.py` |
| `audit_chunks.py` | 切片 8 维健康度审计 | `python audit_chunks.py` |

---

## 📡 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 渲染 比邻星 (ProximaRAG) 主页面 |
| `POST` | `/api/chat` | RAG 对话 (SSE 流式)。`query`(必填) / `history`(JSON可选) / `stream`(默认true) / `product_id`(可选) |
| `POST` | `/api/upload` | 上传 PDF 并增量更新向量库 (MD5 去重 + 级联清理) |
| `GET` | `/api/status` | 向量库就绪状态与文档片段数 |
| `GET` | `/api/products` | 已入库产品 ID 列表 |
| `GET` | `/api/debug/inspect_chunks` | 切片检查器 (按产品/关键词过滤) |
| `POST` | `/api/debug/retrieve` | 检索沙盒 (不调 LLM，仅输出管线中间结果) |

### 外网端口映射

| 服务 | 内部端口 | 外部端口 |
|------|---------|---------|
| FastAPI 后端 | 8000 | 50003 |
| 前端 UI | 8501 | 50004 |
| vLLM 推理 | 8001 | 仅内网 |

---

## 📝 开发日志与架构审计

- **[dev_log.md](./dev_log.md)**: 从 2026-07-20 至今共 28 章完整开发记录与架构决策（最新: v24 模板约束+流式穿透重构）
- **[ARCHITECTURE_AUDIT.md](./ARCHITECTURE_AUDIT.md)**: v24 全盘四层架构审计报告（含模板约束理论分析/代码结构体检/拆分方案/未来升级推演）
- **[CLAUDE.md](./CLAUDE.md)**: AI 协同开发规范（含 v24 四层架构排雷法思想钢印：System Prompt 极简/模板底端锚定/流式零缓冲/render_node 纯透传/L4 正则最小化）
- **[tests/TEST_REPORT.md](./tests/TEST_REPORT.md)**: 评测报告归档
