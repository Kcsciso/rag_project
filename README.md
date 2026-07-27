# 📰 NewsPage — 湖南比邻星科技文档智能问答系统

基于 **RAG（Retrieval-Augmented Generation）** 架构的官方技术文档与使用手册智能问答系统。专为**湖南比邻星科技有限公司**的开发者和用户打造，采用双 A100 GPU 算力底座，底层搭载 **vLLM + 开源大模型**实现完全私有化、低延迟的本地推理。

---

## 🚀 核心特性

### 🤖 智能推理引擎
- **四层金字塔容灾**：本地 vLLM → 智谱 GLM-4.7-Flash 云端 API → 智能结构化纯检索直出（行级归一化去重）→ 优雅错误提示，极端故障下仍可服务。
- **显卡智能自适应部署**：`start_services.sh` 通过 `nvidia-smi` 实时扫描所有 GPU 空闲显存，自动绑定剩余空间最大的 GPU，避免硬编码导致的 OOM 崩溃。`detect_best_gpu()` 函数的 stdout/stderr 已严格隔离，杜绝日志污染变量。
- **毫秒级流式秒回**：FastAPI SSE 异步非阻塞线程池隔离 + 前端 50ms 节流渲染，LLM 读取超时激进缩短至 12s。

### 🔍 智能检索优化（ADR-7/ADR-8）
- **中文专优嵌入**（`BAAI/bge-small-zh-v1.5`）：512 维中文语义向量，精准匹配中文技术查询与 C 函数名。
- **BM25 + Vector 混合检索**：Dense 向量召回 (bge 语义) + Sparse BM25 关键词召回 (jieba 分词 + 正则标识符保护) → RRF (Reciprocal Rank Fusion) 融合排序。
- **Autocut 动态截断**（`_autocut_knee`）：基于 RRF 分数断崖/跳变点 (Knee Point) 检测，自适应确定最佳截断位置 [2, 8]。
- **C 函数 Header Injection**（`[Functions: xxx]`）：切片头部自动注入所含函数名，极大增强 Dense/Sparse 检索敏感度。
- **Query 预处理**（`_preprocess_query`）：多层迭代剥离口语化噪音（25+ 模式）。
- **产品级物理隔离**（ADR-6）：入库自动打标 → 检索 `where={"product_id":"OpenR6"}` → 未指定时主动反问。

### 🧠 智能上下文扩展与防幻觉（ADR-9/ADR-10/ADR-11）
- **LangGraph v3 Plan-Execute-Synthesize 架构**（ADR-14）：SubGoalPlanner 任务分解 + CrossProductRetrieval 全库检索 + Synthesize 多路融合 + CodeEntityAnchor 代码实体锚定。
- **v4 切片机制升级**（ADR-15）：API 原子切分 + 标题感知 + Parent-Child 双层索引（70P + 488C，59 个 API 原子块）。
- **多模态增量更新 + GPU 加速**（ADR-16）：RapidOCR 图片文本抽取 → 切片注入 + MD5 去重 + 级联 Upsert + BM25 动态同步 + GPU 批量嵌入。
- **检索幻觉修复**（v4.1）：function_names 元数据模糊匹配 + 放宽关键词过滤 + kept_docs 安全网 + `_force_no_code` 防幻觉硬拦截 + 产品自动推断隔离。
- **Extract-Render 两层分离架构**（ADR-12）：**信息提取模式** — 1.5B 模型只从 Context 提取结构化 JSON 实体，代码/步骤/引用由 Python 确定性渲染器生成，从根源消除伪 API 编造、步骤泛化与引用缺失。
- **SemanticDedup 语义去重**: trigram overlap 检测段落级重复生成并自动截断。
- **Multi-Product Intent Classifier**: 多产品对比查询自动拆分检索 + 交错合并。
- **Entity Anchor 实体锚定重排**: 查询含具体数字/专有词时置顶物理包含切片的 RRF 权重。
- **ABSTAIN 硬弃答网关**: Context 中实体缺失时直接返回诚实拒答，零 LLM 调用（26ms）。
- **Contextual Prefixing**: 每个切片注入 `[文档: X | 章节: Y]` 前缀，从物理切片源头隔离参数概念。
- **父子切片上下文扩展**（Parent-Child Chunking）：检索命中子切片时，自动按章节 ID 捞取同章节兄弟切片，补充完整流程上下文，彻底解决 TCP 四点法、关机步骤等长流程因截断导致总结不完整的问题。
- **柔性 Grounding 提示**：动态检测 query 中含数字请求（密码/端口/IP），若 Context 中无具体数值则自动追加诚实提示，引导模型明确告知"文档未记载"而非猜测 `admin`、`502` 等通用默认值。
- **多轮对话 Citation 前缀清洗**：剥离 chat_history 中助理回复的章节溯源长前缀（`根据《X》第 Y.Z 节【...】`），防止后续轮次复读背景幻觉。
- **章节标题自动注入切片**（`[章节: 2.2.4.3 版本升级]`）：PDF 解析阶段自动识别 5 类标题模式（编号型/章型/中文序号型/Markdown # 型/装饰符型）并注入切片头部，大幅提升向量检索对章节关键词的召回率。
- **文档术语自动提取**（`_auto_extract_and_register_terms`）：BM25 构建时自动扫描章节标题、表格表头、英文缩写、SDK 函数名，批量注册到 jieba 分词词典（零人工维护）。
- **Context Token 预算控制**：3 切片 × 200 字符截断 + `max_tokens=384`，确保总请求 ≤ 3584 + 384 < 4096 vLLM 硬限制。

### 🔒 企业级安全与稳定性
- **全栈输入防御**：防路径遍历（`sanitize_filename`）、Null 字节与控制字符清洗（`sanitize_query`）、Prompt 注入过滤（`_contains_injection_pattern`）、历史消息角色白名单（`validate_chat_history`）。
- **滑动窗口记忆**：多轮对话最多保留 3 轮历史，防止上下文超出 4096 Token 限制。
- **全链路异常自动降级**：覆盖 9 种故障场景（含 SSE 客户端断开），OOM/超时/限流自动跌落至纯检索直出。
- **资源泄露防范**：`shutdown_clients()` 释放 LLM 连接池 + `cleanup_vector_store()` 释放嵌入模型显存，FastAPI `shutdown` 事件自动触发。

### 🎨 现代化 Web 体验
- **NewsPage** 科技蓝深色主题，双栏布局（对话 + 上传/状态面板）。
- SSE 流式打字机效果 + Markdown 实时渲染 + `highlight.js` 代码高亮。

---

## 📁 项目目录结构

```text
rag_project/
├── src/
│   ├── config.py              # 全局配置中心 + GPU 智能探测 API
│   ├── agent_state.py         # LangGraph v3 RAGState（21 字段，PES 架构）
│   ├── graph_rag.py           # LangGraph v3 状态图引擎（9 节点 + 5 条件边）
│   ├── attribute_tool.py      # 动态属性意图工具（v3，LLM 提取 + BM25 搜索）
│   ├── kv_extractor.py        # 离线 KV 属性提取器（phase-out 中）
│   ├── pdf_loader.py          # v4 PDF 加载器（API 原子切分 + OCR 注入 + Parent-Child）
│   ├── pdf_loader.py          # PDF 解析与递归字符级文本分块
│   ├── vector_store.py        # ChromaDB 向量库（HF→ONNX 双轨嵌入）
│   ├── multimodal_loader.py   # 多模态解析（PyMuPDF + pdfplumber 表格→Markdown）
│   ├── attribute_tool.py      # 动态属性意图工具（v3）
│   ├── kv_extractor.py        # 离线 KV 属性提取器
│   ├── rebuild_v4.py          # v4 向量库重建脚本
│   └── rag_chain.py           # RAG 四层容灾 + 混合检索 + 口语化预处理 + 安全防御
├── templates/
│   └── index.html             # NewsPage 聊天与文档交互主页面
├── static/
│   ├── style.css              # 科技蓝深色主题样式
│   └── app.js                 # SSE 流式通信 + 50ms 节流渲染
├── data/                      # 用户上传的 PDF 文档目录
├── vector_db/                 # ChromaDB 向量数据持久化目录
├── app.py                     # FastAPI 异步应用入口（含安全中间件）
├── tunnel.py                  # ngrok 公网穿透脚本
├── check_status.py            # 统一服务健康检查（GPU 实时监测）
├── start_services.sh          # 一键自适应启动脚本（GPU 智能选择）
├── test_robot_rag.py          # 核心 RAG 功能自动化回归测试
├── test_stability.py          # 多轮对话 + 并发 + 异常降级压力测试
├── test_human_simulation.py   # 全场景人类模拟测试（14 用例 × 5 类别）
├── dev_log.md                 # 完整开发与迭代演进日志（20 章）
├── CLAUDE.md                  # AI 协同开发规范与系统红线
└── README.md                  # 本文件
```

---

## ⚙️ 系统环境与约束

| 项目 | 说明 |
|------|------|
| **硬件底座** | 2 × NVIDIA A100-PCIE-40GB（CUDA 12.4） |
| **环境管理器** | Conda（`rag_agent`，Python 3.10） |
| **推理引擎** | vLLM 0.16.0（OpenAI 兼容 API，端口 **8001**） |
| **默认模型** | `Qwen/Qwen2.5-1.5B-Instruct`（~3.7 GB，GPU 自适应部署） |
| **云端降级** | 智谱 GLM-4.7-Flash（免费模型，`open.bigmodel.cn`） |
| **嵌入模型** | `BAAI/bge-small-zh-v1.5`（512 维，中文专优）→ ONNX 自动回退 |
| **相似度阈值** | 0.68（cosine distance，配合 BM25+RRF 混合检索） |
| **Web 框架** | FastAPI + Jinja2（API `7860`/前端 UI `8501`） |

**🔴 核心锁定依赖（严禁升级）**：
- `torch==2.6.0+cu124` / `torchvision==0.21.0+cu124` / `torchaudio==2.6.0+cu124`
- `vllm==0.16.0`（`--no-deps` 隔离安装）
- `sentence-transformers==2.7.0`

---

## 🚀 部署与启动指南

### 1. 准备文档

将湖南比邻星科技有限公司的开发文档、API 规范或产品使用手册（PDF 格式）放入 **`data/`** 目录。

### 2. 一键启动（推荐）

```bash
chmod +x start_services.sh

# 完整启动（智能 GPU 检测 → vLLM → FastAPI）
./start_services.sh

# 仅启动 vLLM 推理服务
./start_services.sh --vllm-only

# 仅启动 FastAPI 后端（vLLM 已运行）
./start_services.sh --fastapi-only

# 手动指定 GPU（覆盖自动检测）
./start_services.sh --gpu 0
```

### 3. 手动启动（终端 A + B）

**终端 A — vLLM 推理服务**：
```bash
conda activate rag_agent
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --served-model-name Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 4096 \
    --port 8001 \
    --gpu-memory-utilization 0.20 \
    --trust-remote-code \
    --enforce-eager
```

**终端 B — NewsPage FastAPI 后端 (端口 7860)**：
```bash
conda activate rag_agent
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
python app.py
```

**终端 C — 前端 UI (端口 8501)**：
```bash
conda activate rag_agent
python frontend_server.py
```

访问：**`http://localhost:8501`**（页面标题：**NewsPage**）｜API 文档：`http://localhost:7860/docs`

### 4. 外网端口映射

| 服务 | 内部端口 | 外部端口 | 外部访问 URL |
|------|---------|---------|-------------|
| FastAPI 后端 | 7860 | 50003 | `http://<服务器IP>:50003` |
| 前端 UI | 8501 | 50004 | `http://<服务器IP>:50004` |
| vLLM 推理 | 8001 | — | 仅内网 |

### 5. 一键停止所有服务

```bash
fuser -k 7860/tcp 2>/dev/null   # 停止 FastAPI
fuser -k 8501/tcp 2>/dev/null   # 停止前端
pkill -f "vllm.entrypoints" 2>/dev/null  # 停止 vLLM
```

### 6. 系统健康检查

```bash
python check_status.py                # 一次性完整报告
python check_status.py --watch 10     # 每 10 秒自动刷新
```

### 6. 环境变量覆盖

```bash
export LLM_BASE_URL="http://localhost:8001/v1"           # 本地 vLLM
export LLM_MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
export VLLM_GPU_ID=0                                      # 手动指定 GPU
```

---

## 🧪 自动化测试

| 脚本 | 覆盖范围 | 命令 |
|------|---------|------|
| `test_human_simulation.py` | 5 类 14 用例（口语噪音、错别字、多轮指代、长文本组合、边界攻击） | `python test_human_simulation.py` |
| `test_robot_rag.py` | 核心 RAG 功能回归（4 题 × 双模式） | `conda run -n rag_agent python test_robot_rag.py` |
| `test_stability.py` | 多轮对话 + 并发保护 + 7 种异常降级 | `conda run -n rag_agent python test_stability.py` |

---

## 📡 API 接口文档

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 渲染 **NewsPage** 主页面 |
| `POST` | `/api/chat` | RAG 对话（SSE 流式）。参数：`query`（必填）、`history`（可选 JSON）、`stream`（默认 true） |
| `POST` | `/api/upload` | 上传 PDF 并自动重建向量库 |
| `GET` | `/api/status` | 返回向量库就绪状态与已索引文档片段数 |

---

## 📝 开发与排错日志

有关环境排查、兼容补丁、四层容灾、GPU 自适应、安全加固、混合检索、人类模拟测试等 20 个章节的详细开发记录与架构决策（ADR），请参阅 [dev_log.md](./dev_log.md)。

经过从 2026-07-20 到 07-24 的多轮迭代，NewsPage RAG 系统已经从早期的“单向线性 RAG 管道”**完全演进为**基于 LangGraph 的“Plan-Execute-Synthesize + Extract-Render”确定性 Agent 状态图架构。

系统彻底废弃了针对特定数字/函数的硬编码补丁，形成了具备**高容灾、多产品隔离、语义精准对齐与确定性代码生成**的产品级 RAG 架构。

---

## 🏛️ 升级后系统整体架构拓扑

整体架构分为**接入防护层**、**LangGraph 智能调度控制层**、**双轨混合检索层**、**确定性渲染层**与**四层金字塔容灾底座**。

```
                              ┌──────────────────────────────────┐
                              │  前端 WebUI / FastAPI Gateway    │
                              │ (输入清洗 / 路径防御 / SSE 异步) │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │    第 0 步：Product Router       │
                              │ (产品意图识别 / 未指定主动反问)  │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        LangGraph 状态图引擎 (RAGState v3.0)                             │
│                                                                                        │
│  ┌──────────────────────┐    ┌───────────────────────┐    ┌──────────────────────────┐ │
│  │  Sub-Goal Planner    │ ──►│    ABSTAIN Gateway    │ ──►│   Hybrid Retrieval Node  │ │
│  │ (Map-Reduce 多产品)  │    │  (Context缺失硬弃答)   │    │  (Dense+BM25+Autocut)    │ │
│  └──────────────────────┘    └───────────────────────┘    └────────────┬─────────────┘ │
│                                                                        │               │
│  ┌──────────────────────┐    ┌───────────────────────┐                 │               │
│  │ Extract-Render Node  │◄───│  llm_generation_node  │◄────────────────┘               │
│  │(确定性代码/步骤渲染) │    │  (小模型结构化 JSON)  │                                 │
│  └──────────┬───────────┘    └───────────┬───────────┘                                 │
│             │                            │ (SDK代码校验失败)                            │
│             │                            ▼                                             │
│             │                ┌───────────────────────┐                                 │
│             │                │    SDK_VerifyNode     │ ──► (回环重试 max_retries=2)     │
│             │                │  (前缀/CDLL/参数自纠) │                                 │
│             │                └───────────┬───────────┘                                 │
│             │                            │                                             │
│             ▼                            ▼                                             │
│  ┌───────────────────────────────────────────────────┐                                 │
│  │                 ExtractAlignNode                  │                                 │
│  │      (通用物理属性词与数值对齐 / 防属性词颠倒)     │                                 │
│  └───────────────────────────┬───────────────────────┘                                 │
└──────────────────────────────┼─────────────────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                             四层金字塔容灾底座 (Failover)                               │
│  ┌───────────────────┐  ┌────────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │ Layer 1: 本地vLLM │─►│ Layer 2: 云端 API  │─►│ Layer 3: 结构化  │─►│ Layer 4: 503 │ │
│  │  (GPU 智能选择)   │  │ (GLM-4.7-Flash)    │  │ CPU 直出模式     │  │ 友好错误提示 │ │
│  └───────────────────┘  └────────────────────┘  └──────────────────┘  └──────────────┘ │
└────────────────────────────────────────────────────────────────────────────────────────┘

```

---

## 🧩 核心模块与架构升级细节

### 1. 核心控制层：LangGraph 状态图与 Agent 工作流

系统核心调度完全基于 `LangGraph` 状态图（`RAGState` 扩展至 14 个控制字段），实现逻辑剥离与决策可逆：

* **Product Router (第 0 步路由)**：
根据 `PRODUCT_ROUTER_RULES` 自动匹配用户 Query 意图。若未提及产品，不盲目全库搜索，而是触发 `_build_clarification_response()` 主动反问澄清。
* **Sub-Goal Planner (Map-Reduce 规划器)**：
当识别到跨产品对比（如 OpenC3 与 OpenR6 对比）时，自动将 Query 拆解为平行子目标，并行触发单库检索后再进行交错融合（Reduce）。
* **ABSTAIN Gateway (硬弃答网关)**：
当 Query 中包含的实体在 Context 中完全不存在时，直接触发硬弃答，零开销拦截幻觉，不调用 LLM。
* **Extract-Render 两层分离架构 (ADR-12)**：
* **抽取层**：System Prompt 约束 LLM 仅输出结构化 JSON 提取块（包含函数名、参数、步骤原文）。
* **确定性渲染层**：由 Python 渲染器通过代码模板渲染 Python ctypes 代码、编号步骤与出处引用。避免 1.5B 小模型在自由文本生成时语法错乱。


* **后处理自纠错环路 (ADR-11)**：
* **`SDK_VerifyNode`**：自动扫描生成代码中的 `set_` 前缀缺失、CDLL 加载缺失与 `.argtypes` 声明缺失。未通过时带反馈触发 `llm_generation` 重试（最多 2 次）。
* **`ExtractAlignNode`**：使用 50+ 领域物理属性词库，扫描数值与其前后 20 字符窗口，强制用 Context 原词修正 LLM 颠倒或篡改的属性词。



---

### 2. 数据解析与物理隔离机制

* **多模态增强解析 (`multimodal_loader.py`)**：
结合 `pdfplumber` 表格提取与 `PyMuPDF` 图片 Caption 注入；自动化提取 PDF 中的 C 函数名并执行 Header Injection（`[Functions: xxx]`），提升代码检索敏感度。
* **动态产品打标与 100% 物理隔离**：
入库时通过 `PRODUCT_MAPPING_RULES` 自动将切片打上 `product_id` 标签；检索时通过 ChromaDB 的 `where={"product_id": "..."}` 实现物理隔离，杜绝跨产品代码张冠李戴。

---

### 3. 双轨混合检索与上下文工程

* **Code-Aware BM25 分词器**：
自定义正则在 jieba 分词前预提取 `set_move_line`、`robot_brkopen` 等 C/Python 变量和函数，确保 SDK 接口名不被切碎。
* **Dense + Sparse RRF 混合检索**：
向量检索（ChromaDB Cosine）与 BM25 结合，放大 candidate 池后通过 RRF 融合。
* **Autocut 动态自适应截断**：
基于 RRF 得分断崖点（Knee Point）自动截断无用切片，将召回数自适应钳制在 2~8 片之间。
* **父子切片上下文扩展 (Parent-Child Expansion)**：
提取已命中切片的 `[章节: X.Y.Z]` 标识，自动捞取同章节兄弟切片，解决长步骤跨块被截断的问题。
* **保底召回机制 (Fallback Retrieval)**：
当阈值过滤或噪声拦截导致 0 结果时，自动强行保留原始向量 Top-3 切片，不直接硬拦截，确保后续 LLM 读写能力生效。

---

### 4. 四层金字塔容灾底座 (Failover Pyramid)

系统具备应对 GPU 卡死、网络断连、API 限流与向量库异常的全自动平滑降级能力：

| 容灾层级 | 运行环境 | 触发条件 | 输出形式 |
| --- | --- | --- | --- |
| **Layer 1: 本地 vLLM** | GPU (Qwen2.5-1.5B/7B) | 主通道健康且获得线程锁 | 智能对话 + 完整代码渲染 |
| **Layer 2: 云端 API** | Cloud (智谱 GLM-4.7-Flash) | 本地 vLLM 超时 (2s/12s)、OOM 或未启动 | 无缝无感切换云端 LLM 输出 |
| **Layer 3: 结构化直出** | CPU-Only (零显存/零 API 费) | vLLM 与云端 API 均不可用 | 智能提取函数/参数/示例，Markdown 结构化直出 |
| **Layer 4: 优雅错误** | API Gateway | 向量库损坏等极端故障 | 503 HTTP 响应 + 中文友好提示 JSON |

---

### 5. 基础设施、安全与运维工具链

* **智能 GPU 动态探测**：`start_services.sh` 与 `src/config.py` 实时扫描所有 GPU 空闲显存，自动绑定空闲显存最大的 GPU 节点（如自动识别绑定 GPU 0 或 GPU 1）。
* **纵深防御安全体系**：
* 路径遍历清洗（`sanitize_filename`）；
* Prompt 注入启发式检测（防 DAN 越狱、角色扮演、指令覆盖）；
* Null 字节与控制字符清洗（`sanitize_query`）。


* **异步非阻塞 SSE 与并发保护**：
* 使用 `asyncio.Queue` (maxsize=256) 隔离线程池与主事件循环；
* `_vllm_lock` 互斥锁防止 GPU 多线程并发 OOM；
* 客户端断开自动捕获 `CancelledError` 停止 GPU 算力浪费。


* **运维工具链**：
* `check_status.py`：实时服务健康检查与 GPU 显存/温度轮询（v4 支持双索引统计）；
* `start_services.sh`：一键自动探测 GPU、检查端口并拉起服务（含端口冲突自动修复）；
* `rebuild_v4.py`：v4 Parent-Child 双索引向量库离线重建脚本。

---

## 🏗️ 企业级架构审查

完整审查报告见 **[ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md)**（评分 B+/82）。

### 核心发现

| 等级 | 数量 | 关键问题 |
|------|------|----------|
| 🔴 P0 安全红线 | 5 | API Key 硬编码泄露、零认证鉴权、文件上传无魔数校验、全局变量并发不安全、SSE 线程泄漏 |
| 🟡 P1 可靠性 | 6 | 31 处裸 `except Exception`、ChromaDB 连接泄漏、嵌入 GPU 未启用、BM25 无持久化、日志无动态控制、LLM 调用无重试 |
| 🟢 P2 架构增强 | 10 | 多用户会话、Prometheus 监控、A/B 测试、用户反馈闭环、Docker 化、向量版本管理等 |

### 优先修复路线

| 阶段 | 内容 | 预估 |
|------|------|------|
| Week 1 | 删除硬编码 API Key + API 鉴权中间件 + 文件魔数校验 | 3d |
| Week 2 | 全局状态并发锁 + SSE 线程追踪 + ChromaDB 连接池 | 3d |
| Week 3 | 异常处理规范化 + 嵌入 GPU 修复 + 动态日志 + LLM 超时重试 | 3d |
| Week 4+ | P2 架构增强（按需选做） | N/A |
