# 📰 比邻星 (ProximaRAG) — 湖南比邻星科技文档智能问答系统

基于 **RAG（Retrieval-Augmented Generation）** 架构的官方技术文档与使用手册智能问答系统。专为**湖南比邻星科技有限公司**的开发者与用户打造，采用双 A100 GPU 算力底座，底层搭载 **vLLM + Qwen2.5-7B-Instruct-AWQ** 实现完全私有化、低延迟的本地推理。

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
│                         │ │                         │ │                         │
│ • PDF 通用文本提取      │ │ • Dense 向量 (bge-zh)   │ │ • System Prompt 210行   │
│ • 7 步 PDF 文本清洗     │ │ • Sparse BM25 (jieba)   │ │ • 双轨制 Prompt 前缀     │
│ • 5 类标题模式识别      │ │ • RRF 四大提权引擎      │ │ • 反跨产品泄露门控       │
│ • 代码注释拦截 (8特征词) │ │ • Autocut 断崖动态截断   │ │ • Context Cap 整块剔除   │
│ • 伪标题黑名单 (10项)   │ │ • HyDE 防毒化 (SDK禁用) │ │ • 历史沉渣净化 + 滑动窗口 │
│ • Parent-Child 双层索引 │ │ • 三层保底召回机制       │ │ • 柔性 Grounding 提示     │
│ • SDK 状态机 API 原子块 │ │ • 代码实体 BM25 3倍写入  │ │ • 父子结构化组装          │
│ • 4 级 Title Fallback   │ │ • 产品级物理隔离 (where) │ │ • SDK Header 单次注入     │
│ • OCR 图片文本归位      │ │                         │ │                         │
└────────────┬────────────┘ └────────────┬────────────┘ └────────────┬────────────┘
             │                           │                           │
             └───────────────────────────┼───────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  L4: 生成控制与后处理层                                                         │
│                                                                                │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐  ┌───────────┐ │
│  │ 四层容灾金字塔   │  │ SDK 自纠错回路   │  │ 静默斩尾         │  │ 属性词    │ │
│  │ L1 vLLM → L2 智谱│  │ (set_前缀/CDLL/  │  │ _strip_hedging_  │  │ 硬改写    │ │
│  │ → L3 直出 → L4   │  │  argtypes)       │  │ tail()           │  │ Extract   │ │
│  │ 硬拒答           │  │ 硬熔断 retry≤2   │  │ 8 模式正则       │  │ Align     │ │
│  └─────────────────┘  └──────────────────┘  └─────────────────┘  └───────────┘ │
└────────────────────────────────────────────────────────────────────────────────┘
```

### L1 — 数据摄入与切片层 (pdf_loader.py, ~1938 行)

**处理流程**: `PDF 文件` → `_v4_extract_text_universal()` (PyMuPDF + OCR 归位) → `_clean_pdf_text()` (7 步清洗) → `_v4_build_parent_child_docs()` (标题树切分 + 双层索引)

| 能力 | 实现 | 说明 |
|------|------|------|
| 标题识别 | `_v4_extract_headings()` 5 类模式 | 数字编号/中文章节/Markdown #/中文序号/装饰符 |
| 代码注释拦截 | ±120 字符窗口 + 8 特征词校验 | 防止 `# 时间等待3秒` 提权为 Heading |
| 伪标题黑名单 | `_PSEUDO_SECTION_BLACKLIST` frozenset 10 项 | 触发后自动继承父级 H2 标题 |
| API 原子块 | `_v4_parse_sdk_state_machine()` 状态机 | 仅 `数字标题` + `函数名称/函数说明` 两类可验证边界 |
| 微碎片缝合 | Micro-Chunk Auto-Merge (≥60ch 或含代码 → 提交) | API 排他锁：不同函数名不合并 |
| 骨架过滤 | `_is_skeleton_chunk()` | <150ch + 无代码 + 无实质参数 → 丢弃 |
| Parent-Child | H2 章节级 Parent(1000ch) + H3/H4 函数级 Child(400ch) | 标题树驱动，零厂商硬正则 |
| Title Fallback | 4 级链 | L1 状态机标题 → L2 面包屑 → L3 父级 H2 → L4 硬兜底 |
| PDF 清洗 | 7 步 `_clean_pdf_text()` | Unicode 连字/括号空格/下划线归一化/I/O 修复/边界错位/表格竖线/JAKA 版式 |

### L2 — 检索与重排层 (rag_chain.py + vector_store.py)

**处理流程**: `Query` → `_preprocess_query()` (口语剥离) → `_normalize_punctuation()` → `_generate_hyde_doc()` (SDK 轨禁用) → `_hybrid_retrieve()` (向量 + BM25 + RRF + Autocut)

| 能力 | 实现 | 说明 |
|------|------|------|
| 向量检索 | ChromaDB cosine (bge-small-zh-v1.5, 512维) | 候选池放大 fetch_factor=5×, SDK 查询 8× |
| BM25 检索 | jieba + 标识符保护 + jieba 自定义词典 | snake_case 函数名不被切碎 |
| RRF 融合 | K=60, BM25 weight=1.2× | 四大提权引擎同时生效 |
| 实体锚点提权 | Entity Anchor Boost (+0.05) | Query 中的数字/协议名/动作词精确匹配 |
| 函数名提权 | Function Names Boost (+0.08) | metadata function_names 与 query 代码实体模糊匹配 |
| 文本平衡 | Text-Chunk Rebalance (+0.03) | 纯文本切片在 RRF 中不被代码切片完全压制 |
| 代码实体三倍写入 | `[CODE:xxx]` → BM25 tokens ×3 | 等效 Boost=3.0，对抗 Dense Vector 盲区 |
| Autocut | `_autocut_knee()` 断崖检测 | 找 RRF 分数相邻差值最大点 (Knee Point) |
| 保底召回 | 三层防护 | 阈值 0→原始 Top-3 / 噪声全杀→kept_docs 恢复 / 最终空→BM25 第二机会 |
| 产品隔离 | ChromaDB `where={"product_id":"xxx"}` | 入库打标 + 检索物理隔离 + 未指定时 Search-First 软路由 |
| 跨产品检索 | `cross_product_retrieval_node` 全库并行 | 多产品拆分 + 交错合并 |
| HyDE 防毒化 | 3 条 skip 条件 | 短 Query/非技术符号/精确 API 签名 → 禁用 |

### L3 — 上下文组装与指令层 (rag_chain.py `_build_messages` + `RAG_SYSTEM_PROMPT`)

| 能力 | 实现 | 说明 |
|------|------|------|
| 双轨制 Prompt | c_sdk (API 即答案) / gui_app (步骤列表,禁止代码) | `_resolve_doc_type()` 根据 product_id 自动分轨 |
| 首句 Python 锚定 | `_dual_track_prefix` f-string 提取真实 source+section | 消除 LLM 编造章节引用 |
| 反跨产品泄露 | `_anti_bleed_prefix` metadata + 正文双重确认 | 仅目标产品缺失 API 且非目标产品泄露时注入 |
| Context Cap | 非SDK 4000 / SDK 8000 字符整块剔除 | Parent 背景优先丢弃 |
| SDK Header 注入 | 单次挂载到 Context 顶部 | CDLL 加载 + POSE/Joint 结构体 |
| 滑动窗口 | `MAX_HISTORY_TURNS=2` (4 条消息) | 超限自动裁剪至最近 2 轮 |
| 历史净化 | `sanitize_chat_history()` 5 步清洗 | Citation 剥离/代码块替换/拒答过滤/尾部套话擦除/注入检测 |
| 柔性 Grounding | `_NUMERIC_QUERY_RE` 动态检测 | Context 无数值 → 追加诚实提示 |
| 父子结构化组装 | Child【精确定位小节】优先 + Parent【章节背景】附后 | 确保 LLM 先读精确定位再读章节背景 |

### L4 — 生成控制与后处理层 (graph_rag.py 后处理节点 + rag_chain.py LLM 调用)

| 能力 | 实现 | 说明 |
|------|------|------|
| 四层容灾金字塔 | L1 本地 vLLM → L2 智谱 API → L3 纯检索直出 → L4 硬拒答 | 每层独立 try/except + NEVER-EMPTY 保证 |
| 静默斩尾 | `_strip_hedging_tail()` 8 模式 | "上述代码假设存在"/"参考文档未包含详细步骤"等 |
| 属性词硬改写 | `extract_align_node` 50+ 领域词库 | 数值前后 12+8 字符窗口 + Context 原词强制覆盖 |
| SDK 自纠错 | `sdk_verify_node` → `llm_generation` 回环 | set_前缀/CDLL/argtypes 检测 + 硬熔断 retry≤2 |
| 代码块闭合 | `_fix_and_close_sdk_code()` | Markdown ``` 自动闭合 + CDLL 智能补全 |
| SemanticDedup | trigram overlap > 0.55 截断 | 消除 1.5B 小模型段落重复 |
| 流式输出 | SSE async/await + bounded queue(256) | 线程池隔离 + 客户端断开取消保护 |
| Temperature | 非流式 0.2 / 流式 0.01 | 代码生成近确定性输出 |

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
| `_AUTOCUT_MIN_K` | 4 (SDK: 6) | Autocut 硬下限 |
| `_AUTOCUT_MAX_K` | 10 | Autocut 硬上限 |
| `_MAX_CONTEXT_CHARS` | 4000 / 8000 (SDK) | Context 字符 Cap |
| `max_tokens` | 1024 | LLM 最大输出 |
| `MAX_HISTORY_TURNS` | 2 | 滑动窗口轮数 |
| `CHILD_CHUNK_SIZE` | 400 | 子层切片 |
| `PARENT_CHUNK_SIZE` | 1000 | 父层切片 |

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
# 全量重建 (物理清空旧库 → 重新解析所有 PDF → 创建双索引)
conda run -n rag_agent python rebuild_v4.py

# 增量上传 (MD5 去重 + 级联清理旧数据 → 仅处理新/更新的 PDF)
curl -X POST -F "file=@your_document.pdf" http://localhost:8000/api/upload
```

`rebuild_v4.py` 流程：
1. **物理隔离**: `shutil.rmtree(CHROMA_PERSIST_DIR)` 彻底清除旧 SQLite/Parquet
2. **文档解析**: `load_pdfs_v4_dual()` Parent-Child 双层构建
3. **手动嵌入**: SentenceTransformer 本地 batch=64，无 ONNX 下载
4. **原生写入**: ChromaDB `collection.add(embeddings=precomputed)`，网络免疫
5. **BM25 同步**: jieba 分词 + 标识符保护 + 术语自动注册

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

# 切片健康度审计
python audit_chunks.py
```

---

## 🧪 自动化测试

**统一评测入口**: `python tests/run_eval.py --verbose`（30 用例，8 硬断言）

| 脚本 | 覆盖范围 | 命令 |
|------|---------|------|
| `tests/run_eval.py` | 30 用例 (GT + SDK 函数 + 安全注入 + 多轮指代 + 错别字容错) | `python tests/run_eval.py --verbose` |
| `test_stability.py` | 多轮对话 + 并发保护 + 7 种异常降级 | `python test_stability.py` |
| `audit_chunks.py` | 切片 8 维健康度审计 (骨架/粘连/标题/OCR/碎片/面包屑/SDK/倒挂) | `python audit_chunks.py` |

📊 **最新评测** (v21, 7B-AWQ, 496 chunks): 切片健康度 **近满分** (8 项指标 7 项零缺陷, 仅 1 骨架块) · Multi-API Sticky 0 · Corrupted Title 0 · OCR Artifacts 0 · SDK 碎化 0 · AST Collapse 0

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

### 环境变量覆盖

```bash
export LLM_BASE_URL="http://localhost:8001/v1"
export LLM_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct-AWQ"
export VLLM_GPU_ID=0
export ZHIPU_API_KEY="your-key-here"
```

---

## 📝 开发日志与架构审计

- **[dev_log.md](./dev_log.md)**: 从 2026-07-20 至今共 25 章完整开发记录与架构决策
- **[ARCHITECTURE_AUDIT.md](./ARCHITECTURE_AUDIT.md)**: v21 全盘四层架构审计报告 (10 项隐患 + 三阶段修复路线图)
- **[CLAUDE.md](./CLAUDE.md)**: AI 协同开发规范 (含四层架构排雷法思想钢印)
- **[tests/TEST_REPORT.md](./tests/TEST_REPORT.md)**: 评测报告归档
