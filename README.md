# 比邻星 (ProximaRAG) 工业级多模态技术文档问答系统

---

### 一、 复现系统前期准备

在复现与部署系统前，请确保完成以下硬件基线、环境隔离、依赖锁定、模型下载与配置补丁准备。

#### 1. 硬件与算力基线

* **计算卡**：推荐 2 × NVIDIA A100-PCIE-40GB（或单卡显存 ≥ 24GB，CUDA 12.4+）。
* **存储空间**：预留 ≥ 60GB 磁盘空间（用于模型权重、向量知识库与 OCR 缓存）。

#### 2. Python 环境与核心锁定依赖

创建独立的 Python 3.10 环境，并严格锁定底层核心组件版本（**严禁擅自升级以防 CUDA 算子与 vLLM 冲突**）：

```bash
conda create -n rag_agent python=3.10 -y
conda activate rag_agent

# 1. 严格锁定 PyTorch 与 CUDA 12.4 基础运行时
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

# 2. 隔离安装 vLLM 推理引擎
pip install vllm==0.16.0 --no-deps

# 3. 安装 RAG、解析与编排依赖
pip install sentence-transformers==2.7.0 \
    magic-pdf==1.3.12 \
    pymupdf \
    chromadb \
    rank_bm25 \
    jieba \
    langchain \
    langchain-community \
    langgraph \
    fastapi \
    uvicorn \
    jinja2 \
    pyngrok

```

#### 3. 模型权重准备

将所需模型下载至本地对应权重目录（支持 HuggingFace 或 ModelScope）：

* **LLM 推理**：`Qwen/Qwen2.5-7B-Instruct-AWQ`（4-bit 量化，显存占用约 8GB）。
* **VLM 视觉提取**：`Qwen/Qwen2-VL-7B-Instruct`（就地部署在 `:8005` 端口）。
* **Dense Embedding**：`BAAI/bge-small-zh-v1.5`（512 维向量空间，针对中文工业术语优化）。
* **MinerU 版面解析套件**：下载 `OpenDataLab/PDF-Extract-Kit-1.0` 权重至 `~/LLM/MinerU_Models`。

#### 4. 配置文件生成与环境补丁打桩

* **生成 MinerU 配置 (`~/magic-pdf.json`)**：
必须指定版面分析模型为 `doclayout_yolo`（防止缺失时自动回退至 detectron2 引起崩溃），表格模型使用 `rapid_table`。
* **Transformers 4.49+ 补丁**：
若 MinerU 依赖的 `transformers` 与 `unimernet` 发生 `cache_position` 参数冲突，执行 `python patch_unimernet.py` 进行文件级打桩。

#### 5. 文档源放入

将原始工业 PDF（`JAKA_Manual.pdf`、`OpenC3六轴机械臂SDK说明文档_win.pdf`、`windows系统OpenR6_sdk使用文档.pdf`）存入 `data/` 目录。

---

### 二、 系统整体架构拓扑

系统遵循工业级四层分层控制架构，实现了离线双轨解析摄入与在线毫秒级流式问答的解耦：

```
                              ┌──────────────────────────────────────────────┐
                              │            FastAPI Gateway (:8000)           │
                              │   /api/chat (SSE) · /api/upload · /api/status│
                              │     LangGraph: run_graph / run_graph_stream  │
                              └───────────────┬────────────────┬─────────────┘
                                              │                │
                             在线查询路径 ────►│                │◄──── 离线摄入路径 (Stage 1)
                                              ▼                ▼
     ┌───────────────────────────────┐      ┌────────────────────────────────┐
     │  L2 检索与重排层               │      │  L1 数据摄入与切片层 (Stage 1) │
     │  • LLM 意图重写 (always-on)   │      │  • SDK 专轨: PyMuPDF 状态机    │
     │  • RRF 混合检索 (Dense+BM25)  │      │  • JAKA 专轨: MinerU×Qwen2-VL  │
     │  • 六大提权引擎 + Autocut     │      │  • 三重图片防线 + HTML表格清洗 │
     │  • 产品物理隔离 (where 过滤)  │      │  • OpenR6 目录噪声剔除 (☆过滤) │
     │  • HyDE SDK/JAKA 全线禁用     │      │  • KV 确定性属性库自动导出     │
     └──────────────┬────────────────┘      └──────────────┬─────────────────┘
                    │ 读取                                 │ 写入
                    ▼                                      ▼
     ┌─────────────────────────────────────────────────────┴────────────────┐
     │   Stage 2/3 存储与事实底座 (vector_store.py + attribute_kv.json)     │
     │   • ChromaDB 双集合: rag_v4_parent (宏观背景) / rag_v4_child (原子API) │
     │   • BM25Okapi 增量分词索引 (按 product_id 内存隔离)                  │
     │   • 确定性事实侧信道 (Modbus: 6502, 波特率: 9600, 管理员密码)       │
     └──────────────────────────────┬───────────────────────────────────────┘
                                    │ 检索命中上下文 + KV 物理注入
                                    ▼
     ┌──────────────────────────────────────────────────────────────────────┐
     │   L3 上下文组装与指令层 (rag_chain.py)                               │
     │   • System Prompt 极简 (~250 tok) │ 双轨 Markdown 填空模板底端锚定   │
     │   • 模板守卫三条件 → Fast-Path 确定性拒答直出                        │
     │   • 守卫命中 Context 代码脱敏 (_strip_code_from_context)              │
     └──────────────────────────────┬───────────────────────────────────────┘
                                    │ messages
                                    ▼
     ┌──────────────────────────────────────────────────────────────────────┐
     │   L4 生成控制与后处理层 (graph_rag.py)                               │
     │   • 极速流式穿透 (_stream_guardrail · TTFB < 2s)                     │
     │   • 代码围栏闭合状态机 (奇偶计数自动补 ```)                          │
     │   • SDK 自纠错重试回路 (set_前缀/CDLL/argtypes 检测 · retry≤2)       │
     │   • 8 项硬质量断言拦截 (防签名改写、防脑补、防泄露)                   │
     │   • 四层容灾金字塔: vLLM(:8001) → 智谱 API → 纯检索直出 → 硬拒答    │
     └──────────────────────────────┬───────────────────────────────────────┘
                                    │ SSE 流式穿透
                                    ▼
                      ┌───────────────────────────┐
                      │      前端 UI (:8501)      │
                      └───────────────────────────┘

```

---

### 三、 各层核心创新点与解决办法

#### L1 数据摄入与双轨解析层 (Stage 1)

* **SDK 专轨物理坐标流状态机**
* *问题*：传统 `pypdf` 会打乱双栏或复杂排版中的代码块，导致函数签名与参数定义跨节漂移；`ctypes` 类型名常被误识别为函数。
* *创新与解法*：基于 PyMuPDF (`fitz`) 的 `get_text("text", sort=True)` 物理坐标排序提取正文；通过 `_SDK_CHAPTER_BOUNDARY_RE` 行首编号状态机实现 27/30 章节原子切片；引入 `_CTYPES_BLACKLIST` 彻底消除类型名污染；开发 `_strip_openr6_toc()` 剥离 OpenR6 前置 1~29 项紧凑目录文本及 `☆` (U+2606) 符号噪声。


* **JAKA 专轨 MinerU × Qwen2-VL 多模态提纯**
* *问题*：工业 GUI 手册包含大量 UI 截图与复杂配置表格，传统 OCR 无法提纯截图中的关键 IP、端口和密码。
* *创新与解法*：采用 MinerU 离线提取大表；编写 `clean_html_tables()` 将 HTML `<table>` 规整为标准 GitHub Markdown；部署“几何过滤 (<80px / 比例>8) → 图注校验 (±100字) → Qwen2-VL 参数提纯”三重图片防线，自动提取 189 张截屏参数并生成 `data/jaka_manual_chunks.json` 提纯缓存。


* **AST-Lite 软装箱切片机制**
* *问题*：固定字符截断易切碎长表格、长代码块或 OCR 参数块。
* *创新与解法*：将文本划分为普通文本与受保护块（表格/代码/OCR）。普通文本按 `\n\n` 段落安全截断，受保护块允许超标整体装入（可扩展至 2600 字符），根除物理硬切断。



#### L2 混合检索与重排层 (Stage 2)

* **ChromaDB LangChain 单例规范与 Parent-Child 双层存储**
* *问题*：原生 `chromadb.PersistentClient` 混用引发文件锁与 Settings 冲突；单一切片难以兼顾宏观背景与原子精度。
* *创新与解法*：统一封装 LangChain `Chroma` 单例包装器；构建 Parent（章节背景）与 Child（原子 API/步骤）双层集合；检索时召回 Child 原子块，组装时按需补全 Parent 宏观上下文。


* **BM25Okapi 增量索引与复合词保护**
* *问题*：分词工具切碎 `snake_case` API 名及 `Ethernet/IP` 等工业网络协议词；每次增量上传全量重算开销大。
* *创新与解法*：基于结巴分词实现标识符保护与复合词原子化；维护内存级按产品隔离的 BM25 索引，支持 `bm25_upsert_product` 增量动态重算 IDF，并在启动时从 ChromaDB 秒级还原。


* **RRF 六大提权引擎与 Autocut 动态断崖截断**
* *问题*：纯向量检索难以保证核心 API 精准登顶，且固定 Top-K 会引入低分噪音。
* *创新与解法*：在 Dense + Sparse RRF 融合基础上叠加 Entity Anchor (+5.0)、API 精确匹配、代码块 3 倍加权与章节隔离提权；引入 `_autocut_knee()` 寻找相邻分数断崖差值点动态截断，过滤低质量长尾切片。



#### L3 上下文组装与指令层 (Stage 3)

* **Markdown 模板强约束 (Template Masking)**
* *问题*：传统通过 200+ 行 System Prompt 要求输出特定 JSON，推理耗时长且极易发生 JSON 解析溃败。
* *创新与解法*：System Prompt 压缩 83% 至 ~15 行 (~250 tokens)；将标准 Markdown 填空槽位模板置于 User Message 末尾，利用 LLM 的 Recency Bias 强控格式，实现开箱即用的 Markdown 规范流式输出。


* **KV 物理事实侧信道注入**
* *问题*：对于 Modbus 端口 6502、波特率 9600 等高敏参数，LLM 常出现免责套话或数值微调幻觉。
* *创新与解法*：在 `_build_messages` 之后、LLM 生成之前开辟物理侧信道，自动执行 `kv_extractor.lookup_attribute()` 读取 `attribute_kv.json`，在 Prompt 中强制前置硬注入真实属性，杜绝大模型猜错。


* **Fast-Path 确定性拒答与 Context 代码脱敏**
* *问题*：超纲提问容易导致模型抄录召回 Context 中的无关代码产生幻觉。
* *创新与解法*：L3 模板守卫命中（点名函数不在 Context、跨产品乱问）时，直接绕过 LLM 返回确定性拒答话术，根除拒答记忆中毒；同时执行 `_strip_code_from_context()` 剥离上下文代码块，彻底切断抄袭源头。



#### L4 生成控制与后处理层 (Stage 4)

* **极速流式穿透与代码围栏状态机**
* *问题*：后处理正则过滤会阻塞流式输出（TTFB 达 60~90s）；网络中断易导致前端代码块渲染破损。
* *创新与解法*：`_stream_guardrail` 零缓冲逐 Token 透传，将首字延迟 (TTFB) 降至 <2s；内置围栏状态机实时统计 ````` 奇偶性，在数据流结束若发现未闭合则自动补全闭合标记。


* **SDK 自纠错闭环与四层容灾金字塔**
* *问题*：小模型偶尔出现 Ctypes 传参丢失或裸调用问题。
* *创新与解法*：`sdk_verify_node` 执行静态签名检查，触发异常时启动回环重试（熔断次数 ≤2）；构建“本地 vLLM (:8001) → 智谱 GLM API → 纯检索直出 → 硬拒答”四层容灾金字塔，保障工业系统 NEVER-EMPTY 高可用。


* **8 项硬质量断言拦截**
* *创新*：评测与线上双重部署硬拦截策略（防 JSON 泄露、防段落重复、防跨轨套话、API 签名校验、提示词防泄露、零脑补、防代码截断与防 API 幻觉）。



---

### 四、 系统启动与操作指令

#### 1. 一键全自动启动（推荐）

```bash
chmod +x start_services.sh
./start_services.sh                  # 自动检测可用 GPU，依序拉起 vLLM(:8001) 与 FastAPI(:8000)

# 若需手动指定 GPU 卡号：
VLLM_GPU_ID=1 ./start_services.sh

```

#### 2. 分模块手动启动

* **终端 A — 启动 vLLM 推理后端**：
```bash
conda activate rag_agent
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --port 8001 \
    --gpu-memory-utilization 0.25 \
    --max-model-len 8192 \
    --enforce-eager \
    --quantization awq

```


* **终端 B — 启动 FastAPI 核心网关**：
```bash
conda activate rag_agent
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
python app.py

```


* **终端 C — 启动前端 Web UI**：
```bash
conda activate rag_agent
python frontend_server.py

```


*访问地址：前端 UI (`http://localhost:8501`) | Swagger API (`http://localhost:8000/docs`)*

#### 3. 知识库构建与增量管理

```bash
# 全量建库 (物理清理旧库 → 双轨切片解析 → 多模态提纯 → ChromaDB 双集合写入 + BM25 索引构建)
python src/rebuild_v4.py

# 增量摄入 (自动按扩展名路由：.pdf 走 SDK 专轨，.md 走 JAKA 专轨)
curl -X POST -F "file=@data/new_manual.pdf" http://localhost:8000/api/upload

```

#### 4. 自动化回归评测与质量审计

```bash
# 运行 35 个工业经典用例 × 8 项硬断言端到端评测
python tests/run_eval.py --verbose

# 快速冒烟评测（仅验证检索质量，不消耗 LLM 推理 Tokens）
python tests/run_eval.py --quick

# 执行 Stage 1 摄入与原子切片离线验收
python tests/test_stage1.py

# 执行向量库白盒质量审计 (零切片/垃圾切片/高敏实体存活率)
python tests/audit_ingestion.py

# 检查服务健康状态 (显存、端口与集合状态)
python check_status.py

```

#### 5. 内网穿透与安全停机

```bash
# 通过 ngrok 映射 FastAPI 后端至公网
python tunnel.py --token <YOUR_NGROK_TOKEN>

# 优雅关闭所有相关后台服务
fuser -k 8000/tcp 2>/dev/null        # 停止 FastAPI
fuser -k 8501/tcp 2>/dev/null        # 停止 前端 UI
pkill -f "vllm.entrypoints"          # 停止 vLLM 推理服务

```
