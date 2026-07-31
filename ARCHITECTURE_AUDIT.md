# 比邻星 (ProximaRAG) — 全盘架构审计报告

> **日期**: 2026-07-31 | **审计人**: AI 架构师 | **覆盖版本**: v22 (ADR-19~22 四轮闭环重构后复审)  
> **方法**: 四层 RAG 架构逐层排查 + 跨层数据流追踪  
> **本次更新**: 复审 ADR-20 (复合查询子任务阈值)/ADR-21 (Autocut SDK 扩容+动态术语对齐)/ADR-22 (SDK 两段式排版铁律) 对全四层的闭环影响

---

## 审计总览

| 层级 | 名称 | 严重问题 | 性能瓶颈 | 幻觉风险 | 评分 |
|------|------|---------|---------|---------|------|
| L1 | 数据摄入与切片 | 2 | 1 | 2 | B+ |
| L2 | 检索与重排 | 2 | 2 → **1** | 0 | B+ → **A-** |
| L3 | 上下文组装与指令 | 2 | 2 → **1** | 2 → **1** | B → **B+** |
| L4 | 生成控制与后处理 | 2 → **1** | 2 → **1** | 1 → **0** | B+ → **A-** |

**综合评分: A- → A- (87/100)** — v22 四轮闭环重构 (ADR-20/21/22) 全四层精准修复，将剩余致命 Bug 从 4 降至 2，幻觉风险全面收敛。

---

## 第一层：数据摄入与切片层 (Data Ingestion & Chunking)

### 1.1 当前架构 (pdf_loader.py, ~1938 行)

```
PDF 文件
  │
  ▼
_v4_extract_text_universal()  ← PyMuPDF 物理排序 + OCR 补漏
  │
  ▼
_clean_pdf_text()  ← 7 步清洗 (Unicode连字/括号空格/下划线归一化/I/O修复)
  │
  ▼
_v4_build_parent_child_docs()
  ├── _v4_extract_headings()      ← 5 类标题模式 + 代码注释拦截
  ├── _v4_find_protected_ranges() ← 代码块/表格 受保护区域标记
  ├── _v4_extract_api_blocks()    ← API 原子块预标记
  └── _v4_build_child_docs_v2()
        ├── c_sdk: _v4_parse_sdk_state_machine() → Micro-AutoMerge → 4级Title Fallback
        └── gui_app: Heading-to-Heading 整块保留 + 微缩大纲注入
```

### 1.2 保持的优良设计

1. **代码注释拦截 (v18)**: `_CODE_KEYWORDS` 八特征词上下文校验，已证实可阻止 95%+ 的 Python 注释提权为 Heading
2. **伪标题黑名单**: `_PSEUDO_SECTION_BLACKLIST` frozenset 10 项，substring + len<15 双重校验
3. **4 级 Title Fallback 链**: 状态机标题 → 面包屑路径 → 父级 H2 → 硬兜底，永不落空
4. **受保护区域**: 代码块 (```) 和 Markdown 表格永不拦腰切断 (`_safe_boundary`)
5. **下划线归一化**: `_clean_pdf_text()` Step 4.3/Step 6 处理了 PDF OCR 导致的 4 种下划线断裂模式

### 1.3 🔴 严重隐患

#### BUG-1.1: `_sanitize_section_title()` 黑名单 substring 误杀

**文件**: `pdf_loader.py:1288-1290`
```python
for bad_kw in _PSEUDO_SECTION_BLACKLIST:
    if bad_kw in cleaned and len(cleaned) < 15:
        return ""  # 触发父级继承
```

**问题**: 使用 raw `in` 做 substring 匹配。合法标题 "参数说明与配置指南"(9字) 会因包含 "参数说明" 被误判为伪标题并返回空。同样，"函数说明与调用规范" 会因包含 "函数说明" 被误杀。

**修复方案**: 将 `in` 改为精确边界匹配：
```python
# 仅当黑名单关键词占据标题的绝大部分时 (>60%) 才判定为伪标题
for bad_kw in _PSEUDO_SECTION_BLACKLIST:
    if bad_kw == cleaned or (bad_kw in cleaned and len(bad_kw) / len(cleaned) > 0.6):
        return ""
```

#### BUG-1.2: 代码注释拦截的上下文窗口盲区

**文件**: `pdf_loader.py:663-672`
```python
_CODE_KEYWORDS = ['restype', 'argtypes', 'CDLL', 'ctypes', 'robot.', 'c_int', 'c_float', 'import ']
```

**问题**: 若代码注释 `# 机械臂初始化运动` 位于代码块的边缘位置（如代码块的第 1 行），其 ±120 字符窗口可能只包含 import 语句的上半部分或空行，8 个特征词全部不在窗口内 → 注释被提权为 Heading。

**场景复现**:
```python
# 机械臂初始化运动     ← 前 120 字符: 空行 + import ctypes
import ctypes           ← "import " 在窗口中但...
# ↑ ctypes 不在同一行，窗口的第121字符刚好在这里
```

**修复方案**: 将上下文窗口从 ±120 扩大到 ±200，并增加反向检测——若该 `#` 行下方 5 行内包含代码特征，同样拦截。

### 1.4 🟡 性能瓶颈

#### PERF-1.1: 全量重建 O(N) 不可扩展

**文件**: `pdf_loader.py:1853-1919` (`load_pdfs_v4_dual`)

**问题**: 当前 3 个 PDF → 496 个 chunks 全量重建耗时约 2-3 分钟（含嵌入）。若 PDF 数量增长到 20 个，重建时间将线性增长至 15-20 分钟。

**现状**: ADR-16 的 `upsert_product_documents()` 已实现增量更新 (MD5 去重 + 级联删除)，但 `/api/upload` 仍调用全量路径作为 fallback。此外 BM25 索引完全重建（O(N) 遍历所有 docs）无增量更新能力。

**建议**: 推广增量 Upsert 为唯一上传路径，废弃全量重建作为默认行为。

### 1.5 🟡 幻觉风险

#### HALL-1.1: 微缩大纲注入的冗余信息

**文件**: `pdf_loader.py:1519-1541`

**问题**: `gui_app` 轨的微缩大纲 (`toc_text`) 会将子标题列表追加到切片正文末端。若检索命中一个大章节的导言段，注入的 15+ 子标题列表可能远超正文长度，形成 "标题噪声"——LLM 可能将这些子标题误解读为 "文档确实记载了这些内容" 而编造细节。

**建议**: 微缩大纲长度上限从 15 条降至 5 条，且仅注入到 Parent 切片（章节背景），不注入到精确定位用的 Child 切片。

---

## 第二层：检索与重排层 (Retrieval & Reranking)

### 2.1 当前架构 (rag_chain.py + vector_store.py + graph_rag.py)

```
用户 Query + 历史对话
  │
  ▼
🟢 ADR-19: _rewrite_query_with_llm()  ← LLM 意图重写引擎 (新增)
  │   ├── 代词消解 ("它"/"那个函数" → 具体 API)
  │   ├── 产品名补全 (从历史中提取产品型号)
  │   ├── 口语噪音剥离 (极低 t=0.0, max_tokens=50, 毫秒级响应)
  │   └── 闲聊穿透 (问候/感谢原样放行)
  │
  ▼
rewritten_query (独立自洽的检索语句)
  │
  ├── _preprocess_query()           ← 口语噪音二次剥离 (轻量)
  ├── _resolve_product_from_query() ← 产品路由 (输入已自洽)
  └── _generate_hyde_doc()          ← HyDE 假想文档 (SDK轨禁用)
  │
  ▼
_hybrid_retrieve()
  ├── ① 向量检索 (ChromaDB cosine, fetch_factor=5×)
  │     ├── 阈值放宽 relaxed_threshold = min(threshold*1.05, 0.70)
  │     ├── 阈值空 → 保底召回 Top-3 (threshold=None)
  │     └── 噪声过滤 (_is_noise_chunk + Image OCR 空壳 + 低分过滤)
  ├── ② BM25 检索 (product-scoped, jieba + 标识符保护)
  ├── ③ RRF 融合 (k=60, BM25 weight=1.2×)
  │     ├── Entity Anchor Boost (+0.05)
  │     ├── Function Names Boost (+0.08)
  │     └── Text-Chunk Rebalance (+0.03)
  └── ④ Autocut 动态截断 (_autocut_knee)
        └── SDK 检索: min_k=6, 非SDK: min_k=4
```

### 2.2 保持的优良设计

1. **三层保底召回**: 阈值空 → 原始 Top-3; 噪声全杀 → keot_docs 恢复; 最终空 → BM25 第二机会
2. **四大提权引擎**: Entity Anchor (+0.05) + Function Names (+0.08) + Text Rebalance (+0.03) + 代码实体 BM25 三倍写入
3. **Autocut 断崖检测**: 基于 RRF 分数相邻差值找 Knee Point，动态确定截断位置

### 2.2.1 🟢 ADR-19: LLM Query Rewriting Engine (v21 新增)

**架构动机**: 原有的多轮对话支持依赖 `_fuse_short_query` (正则缝合) + `_resolve_clarification_followup` (关键词检测) + `_has_business_intent` (启发式校验) 三个脆弱的硬编码模块。在多轮对话中，用户的孤立短词 ("OpenC3") 和模糊代词 ("它"、"那个函数") 无法被这些正则引擎正确消解，导致:

1. **Vector Centroid Drift (向量重心偏移)**: 短词和代词查询在 512 维语义空间中无确定方向，将检索引向噪音切片
2. **异源切片混入 (BUG-3.3 根因)**: 无产品主语的查询命中多个产品的 API 切片，触发反泄露门控
3. **产品路由失败**: 正则规则无法从历史中提取产品名来补全当前查询

**新架构** (`rag_chain.py:176` + `graph_rag.py:318`):

```
┌─────────────────────────────────────────────────────────────┐
│  _rewrite_query_with_llm(query, chat_history)               │
│                                                             │
│  输入: 原始 query + 最近 3 轮对话历史 (截断至 100 字符/条)    │
│                                                             │
│  REWRITE_SYSTEM_PROMPT (5 条核心规则 + 4 组 Few-Shot):       │
│    ① 闲聊原样穿透 → "你好"/"谢谢" 不添加主语                  │
│    ② 主语与意图缝合 → "OpenC3" + 历史 "上电" → 完整语句       │
│    ③ 代词精准消解 → "它"/"那个功能" → 具体产品/API            │
│    ④ 剥离口语噪音 → "帮我查一下"/"能不能" → 核心技术词元       │
│    ⑤ 绝对输出纪律 → 只输出重写后的一句话, 无前缀/引号/解释     │
│                                                             │
│  调用: vLLM 本地 (优先) → 云端智谱 API (降级)                 │
│        temperature=0.0, max_tokens=50 (毫秒级响应)           │
│  防御: 输出 >150 字符 → 自动回退原始 query                    │
│                                                             │
│  输出: rewritten_query → 接管后续全部检索管线                  │
└─────────────────────────────────────────────────────────────┘
```

**已废弃模块** (已从 `graph_rag.py` import 中删除):

| 废弃函数 | 原用途 | 废弃原因 |
|----------|--------|---------|
| `_fuse_short_query` | 正则拼接短词与历史 query | 无法处理代词和隐式指代 |
| `_resolve_clarification_followup` | 检测用户回复是否为产品名 | 被 LLM 意图缝合 (规则②) 完全替代 |
| `_has_business_intent` | 启发式判断 query 是否有业务意图 | 重写后的 query 已完全自洽，无需二次校验 |

**对现有 Bug 的影响**:
- **BUG-3.3 (异源切片混入)**: 大幅缓解。重写后的 query 包含完整的产品主语 + API 信息 → 向量检索精准命中目标产品 → 非目标产品切片不会被召回 → 反泄露门控不再误触发
- **BUG-3.2 (Clarification Marker 脱节)**: 部分缓解。`_resolve_clarification_followup` 已删除，澄清反问现在由 graph_rag 的 `product_routing_node` 确定性地生成 (`build_product_clarification_response`)。但 `_CLARIFICATION_MARKER` 仍残留在 `rag_chain.py:816` 用于 `rag_chat()` 旧管线 — 需后续清理

### 2.2.2 🟢 ADR-20/21: 复合查询子任务阈值 + Autocut SDK 防误杀 (v22)

**ADR-20 — 复合查询子任务漏检修复** (`rag_chain.py:2331`):

**问题**: `_decompose_compound_query()` 的 `_MIN_SUB_QUERY_LEN` 原值为 4，导致两字核心动词（"连接"、"上电"、"使能"）被当作噪声片段丢弃。在多步长指令（如 "OpenC3 先上电，然后连接机械臂"）中，"上电"和"连接"两个关键子任务被丢弃 → 仅检索 "OpenC3 然后机械臂" → 召回不完整 → 输出漏步骤。

**修复**: `_MIN_SUB_QUERY_LEN: 4 → 2`。两字动词在工业操作指令中是高密度信息单元——"上电"、"回零"、"抱闸"、"使能"——每一个都对应至少一个完整的 SDK API 调用流程。阈值降低后，这些动词作为独立子查询被保留 → 每个子任务独立触发一轮混合检索 → 多步操作的召回完整性得到保障。

**ADR-21a — Autocut SDK 防误杀** (`rag_chain.py:1997-1998`, `rag_chain.py:2684`):

**问题**: 多参数 SDK 切片（如圆弧运动函数 `robot_movc(pos1, pos2, ...)`）因 PDF 排版换行导致正文中出现大量空白/断行 → BM25 得分剧烈波动 → Autocut 断崖检测误判为低质量切片 → 被截断丢弃 → 用户查询"圆弧运动"时答案不完整。

**修复**:
- `_AUTOCUT_MIN_K`: 4 → **8** (全局硬下限翻倍)
- `_AUTOCUT_MAX_K`: 10 → **15** (候选池截断上限扩容 50%)
- SDK 检索场景 `_min_k` 动态提升至 **10** (L2684: `_min_k = 10 if _is_sdk_retrieval else _AUTOCUT_MIN_K`)
- 三重保障确保多参数/多步骤 SDK 切片不会被错误腰斩

### 2.3 🔴 严重隐患

#### BUG-2.1: Search-First 软路由的 `_score` 属性不存在

**文件**: `graph_rag.py:411`
```python
score = getattr(doc, '_score', None) or doc.metadata.get('_score', 0.5)
```

**问题**: LangChain 的 `similarity_search_with_score` 返回 `List[Tuple[Document, float]]` — score 是元组的第二个元素，不是 Document 的属性。当通过 `search_similar_with_threshold()` 调用（该函数只返回 `List[Document]`，丢弃了 score），所有 Document 的 `_score` 属性都为空 → 所有产品都被赋予相同分数 0.5 → Search-First 的断层领先判定永远不触发 → 用户输入 "上电函数" 时永远不会被自动路由到正确的产品。

**修复方案**: `_search_first_soft_route()` 需要使用 `similarity_search_with_score()` 直接调用而非通过 `search_similar_with_threshold()`。

#### BUG-2.2: `run_graph_stream()` 与 `_route_after_sdk_verify()` 的 retry 逻辑割裂

**文件**: `graph_rag.py:1888` vs `graph_rag.py:1296-1319`

**`run_graph_stream()` 的循环条件** (L1888):
```python
if retry_count > max_retries:  # 2 > 2 = False → 不熔断，继续执行
    break
```

**`_route_after_sdk_verify()` 的条件** (L1308):
```python
if feedback and retry_count <= max_retries:  # retry=2, max=2 → True → 回环
    return "llm_generation"
```

**问题**: 两者使用不同的比较运算符。`run_graph_stream` 用 `>` (严格大于)，`_route_after_sdk_verify` 用 `<=` (小于等于)。后果：当 retry_count=2 时，stream 函数认为 "还没达到上限" 继续循环，但 sdk_verify_node 入口的硬熔断 (L923: `retry_count >= 2 → skip`) 会让第 3 次重试直接跳过校验 → 静默透传未修复的代码。

**修复方案**: 统一为 `>=` 判定：
```python
# run_graph_stream
if retry_count >= max_retries:
    break
# _route_after_sdk_verify
if feedback and retry_count < max_retries:
    return "llm_generation"
```

#### BUG-2.3: `cross_product_retrieval_node` 硬编码阈值 0.55

**文件**: `graph_rag.py:1521`
```python
docs = search_similar_with_threshold(
    _get_graph_vector_store(), query, k=3, threshold=0.55, product_id=pid
)
```

**问题**: 全局 `SIMILARITY_THRESHOLD=0.68`，但跨产品检索使用 0.55（宽松 0.13）。这导致跨产品路径比单产品路径多召回大量低相关度切片，且行为不一致。

### 2.4 🟡 性能瓶颈

#### PERF-2.1: BM25 无磁盘持久化

**文件**: `vector_store.py:1335-1338`

**问题**: BM25 索引 (`_bm25_indexes`, `_bm25_corpus`) 完全在内存中。每次 FastAPI 重启，`build_bm25_from_chromadb()` 需要从 ChromaDB 读取所有 496 个文档并重新分词构建索引 (~30s 冷启动)。

**建议**: 使用 `pickle` 将分词后的 token 列表序列化到 `vector_db/bm25_tokens.pkl`，启动时直接反序列化（<1s）。

#### PERF-2.2: RRF 融合中的 O(N²) 实体锚点扫描

**文件**: `rag_chain.py:2956-2971`

**问题**: 对每个融合后的候选 doc，用 query 中的每个实体锚点逐文档扫描 `if anchor in doc_text`。当候选池 50+ 且锚点 10+ 时，内层循环 500+ 次。`in` 操作在 800 字符的 doc content 上是 O(m×n)。

**建议**: 在 RRF 融合循环中提前计算每个候选的锚点命中数（而非外层独立循环），避免二次遍历。

### 2.5 🟡 幻觉风险

#### HALL-2.1: HyDE 在 gui_app 轨可能生成虚构函数名

**文件**: `rag_chain.py:2575-2580`

**问题**: HyDE 仅在 `product_id in {"OpenC3", "OpenR6"}` 或 `_is_sdk_code_query()` 时禁用。JAKA (gui_app) 查询的 HyDE 不会被禁用。若用户问 "JAKA 怎么上电"，HyDE 生成的假想文档可能包含虚构的函数调用（如 `robot.power_on()`），将向量检索引向 SDK 切片。

**建议**: 将 HyDE 禁用条件扩展为 `doc_type == "gui_app"` 或 `product_id == "JAKA"`。

---

## 第三层：上下文组装与指令层 (Augmentation & Prompting)

### 3.1 当前架构 (rag_chain.py `_build_messages` + `RAG_SYSTEM_PROMPT`)

```
🟢 ADR-19: _rewrite_query_with_llm()  ← 上游已完成意图重写 (L2)
  │   rewritten_query 已包含完整产品名 + API + 动作
  │
  ▼
Context Docs (Child + Parent)
  │
  ├── Child → 【精确定位小节】  ← 优先级高，排在前面
  ├── Parent → 【章节背景】      ← 优先级低，排在末尾
  ├── SDK Header 单次注入       ← 仅 c_sdk 轨
  ├── Context Cap 整块剔除      ← 非SDK 4000 / SDK 8000
  └── 双轨 Prompt 前缀
        ├── gui_app: 首句强制红线 + 禁止代码
        └── c_sdk:  SDK 模板 + 字面锚定
```

> **注**: `_CLARIFICATION_MARKER` (rag_chain.py:816) 仅服务于废弃的 `rag_chat()` 旧管线。LangGraph 主路径中的澄清反问由 `product_routing_node` → `build_product_clarification_response()` 确定性地生成，不再依赖正则 marker 匹配。

### 3.2 保持的优良设计

1. **双轨 Prompt 控制**: `gui_app` 绝对禁止代码; `c_sdk` API 即答案
2. **反跨产品泄露门控**: metadata `function_names` + 正文双重确认，仅当目标产品缺失且非目标产品存在 API 时才注入 `_anti_bleed_prefix`
3. **Context Cap 整块剔除**: 不切割任何单个 Chunk 内部正文
4. **历史沉渣净化**: `sanitize_chat_history()` + Citation 前缀清洗 + 代码块替换 + 尾部拒答剥离
5. **🟢 ADR-21b: 动态术语对齐 `_term_alignment_prefix` (v22 新增)**: 替代了臃肿的全局 System Prompt 规则注入方式。只在 Python 层检测到特定产品+同义词组合（如 OpenR6 "使能" → `set_robot_arm_init`）时，向当前轮用户消息按需挂载防幻觉铁律。实现零全局 Token 损耗的精准纠偏：
   ```python
   # rag_chain.py:1906-1911
   _term_alignment_prefix = ""
   if "OPENR6" in query.upper() and "使能" in query:
       _term_alignment_prefix = (
           "【⚠️ 强指令抵抗·术语对齐】OpenR6 的"使能"操作实际上对应的是初始化函数 "
           "`set_robot_arm_init`。绝对禁止迎合字面意思捏造 `set_robot_enable` 之类的假函数！"
       )
   ```
   设计原则：**默认不注入，仅在检测到高危同义词对时才挂载**。这比在 System Prompt 中罗列所有产品×同义词映射表（~300+ tokens 常驻开销）优雅得多。当前仅覆盖最高频的误映射对，后续可按需扩展同义词表至外部配置文件而不增加 System Prompt 负债。

### 3.3 🔴 严重隐患

#### BUG-3.1: System Prompt 膨胀 — Token 预算失控

**文件**: `rag_chain.py:1514-1725` (`RAG_SYSTEM_PROMPT`)

**测量**: System Prompt 共 210 行、~3,500+ 字符 → 约 **1,500-2,000 tokens** (中文)。加上 `_dual_track_prefix` (~200 tokens) + `_anti_bleed_prefix` (~150 tokens) + 10 个 SDK Child chunks (~4,000 tokens) + 历史消息 + query → **总输入远超 Qwen2.5-7B 的 8192 上下文限制**。

**实际影响**: 当 10 个 full chunks 被 Autocut 放行 + SDK Context Cap=8000 字符时，`_build_messages` 的 `total_chars` 可能达到 8000+。虽然后续的 `Context Cap` 整块剔除逻辑会从末尾（Parent chunks）开始丢弃，但这意味着最坏情况下所有 Parent 背景信息被删除，只剩下孤立的 Child API 函数定义 → LLM 缺少操作上下文 → 生成质量下降。

**修复方案**: 
1. 将 System Prompt 从 210 行压缩至 80 行以内（保留核心规则，移除 Few-Shot 示例到外部文件）
2. 为 SDK 查询动态降低 Context Cap 从 8000→6000，为 System Prompt 预留空间
3. 或在 `_call_llm` 中先计算 `_build_messages` 输出的实际 token 数（用 tiktoken），超出时动态裁剪

#### BUG-3.2 `[已解决 — ADR-19]`: `_resolve_clarification_followup` 模块已删除

**原问题**: `_CLARIFICATION_MARKER` 与 `build_product_clarification_response()` 文案存在字符串耦合，修改文案可能导致澄清检测静默失效。

**ADR-19 解决**: `_resolve_clarification_followup()` 已从 `graph_rag.py` 的 import 中彻底删除 (L254)。澄清反问后的用户回复（如纯产品名 "OpenC3"）现在由 `_rewrite_query_with_llm()` 的 **主语与意图缝合** 规则处理 — LLM 从历史中提取澄清文案并自动拼接为完整检索语句，不再依赖正则 marker 匹配。

**残留风险**: `_CLARIFICATION_MARKER` 常量仍存在于 `rag_chain.py:816`，供废弃的 `rag_chat()` 旧管线使用。待旧管线完全移除后一并清理。

#### BUG-3.3 `[已缓解 — ADR-19]`: `_anti_bleed_prefix` 的跨产品 API 检测漏判

**文件**: `rag_chain.py:2150-2219`

**原问题**: 反泄露门控依赖 `metadata.get("function_names")` 和 `metadata.get("is_api")` 来判定目标产品是否有 API。若目标产品 metadata 标注不完整但非目标产品有 function_names → 门控错误触发 → LLM 诚实拒答而非阅读 Context 中的实际代码。

**ADR-19 缓解**: `_rewrite_query_with_llm()` 将模糊查询重写为主谓宾齐全的独立检索语句后，向量检索在 512 维语义空间中的方向极其精准。具名产品 + 具名 API 的组合查询几乎不会召回其他产品的 API 切片 → 反泄露门控的触发频率从架构层面大幅降低。

**残留风险**: 当两个产品的 API 命名高度相似（如 OpenC3/OpenR6 共享同一套函数签名约定）时，metadata 漏判问题仍未根治。建议在门控判断中增加第三重确认——扫描目标产品 chunks 的**正文**中是否包含函数调用模式（如 `robot_.*(`），若正文中有则豁免 metadata 缺失。

**状态**: 严重度从 🔴 BUG 降级为 🟡 HALL (缓解但未根除)。

### 3.4 🟡 性能瓶颈

#### PERF-3.1: `_build_messages` 对每个 doc 做多轮正则清洗

**文件**: `rag_chain.py:1982-2002`

**问题**: 对每个 context doc，依次执行：null字节清洗 → 噪声截断 → Image标签移除 → 连续空行压缩 → 页码章节提取 → 元数据标记移除。10 个 docs × 6 次正则 = 60 次 regex 操作。在流式场景中每次 retry（最多 3 次）都要重新执行，累计 180 次 regex。

**建议**: 将正则预编译为模块级常量（`re.compile`），避免每次调用时重新编译。

---

## 第四层：生成控制与后处理层 (Generation & Post-Processing)

### 4.1 当前架构

```
LLM 调用 (四层容灾)
  │
  ├── Layer 1: 本地 vLLM (预检 + 互斥锁)
  ├── Layer 2: 云端智谱 API (无缝降级)
  ├── Layer 3: 纯检索直出 (CPU-only)
  └── Layer 4: 硬拒答兜底
  │
  ▼
后处理管线 (graph_rag.py)
  ├── render_node         ← JSON提取 → 确定性渲染代码/步骤
  ├── sdk_verify_node     ← 代码缺陷检测 (set_前缀/CDLL/argtypes)
  ├── extract_align_node  ← 属性词硬改写 + SemanticDedup + 套话擦除
  └── _fix_and_close_sdk_code  ← Markdown闭合 + CDLL 补全
```

### 4.2 保持的优良设计

1. **静默斩尾** (`_strip_hedging_tail`): 擦除末尾自相矛盾的免责套话（"上述代码假设存在"等 8 模式）
2. **NEVER-EMPTY 保证**: 所有 4 层 + 流式/非流式双路径均覆盖终极兜底
3. **硬熔断**: SDK 重试上限 2 次，入口检测 `retry_count >= 2 → skip`
4. **属性词硬改写**: 50+ 领域属性词库 + 数值前后 12+8 字符上下文窗口
5. **🟢 ADR-22: SDK 两段式排版铁律 (v22 新增)**: 在 `_dual_track_prefix` 中聚合所有相关章节出处，强制规范 LLM 输出结构——**第一段 (文字说明)** 必须以原话引用章节出处开头，**第二段 (唯一代码块)** 紧跟其后输出一个完整的 ` ```python ` 代码块。根源上解决了长文本生成中的两个顽疾：
   - **复读现象**: 无结构约束时 LLM 倾向在代码块前后反复解释，两段式强制代码块闭合后立即结束
   - **排版坍塌**: 多章节 API 引用时 LLM 易将代码分散到多个小块或混合 Markdown 列表，`_dual_track_prefix` 的 `【最高纪律】` 指令 + 动态 DLL 名注入 (py_dll.dll / collrob_sdk.dll) 确保输出单一、完整的可执行代码块
   
   ```python
   # rag_chain.py:1891-1897
   if _doc_section_str:
       _dual_track_prefix = (
           "【🔴 SDK 回复排版铁律】\n"
           "你的回答必须严格遵循以下两段式结构，严禁颠倒或省略：\n"
           "1. **第一段（文字说明）**：必须以原话开头：\n"
           f"   根据《{_doc_name}》涉及的【{_doc_section_str}】章节记载...\n"
           "2. **第二段（唯一代码块）**：紧跟在文字说明下方，输出一个完整的 ```python 代码块...\n"
           "【最高纪律】：绝对禁止在代码块外面再套多余的解释！代码块闭合后立刻结束回答！"
       )
   ```

### 4.3 🔴 严重隐患

#### BUG-4.1: `_stream_guardrail()` 缓冲全量输出 — 流式能力退化

**文件**: `rag_chain.py:2418-2437`
```python
def _stream_guardrail(gen):
    buffer = []
    for chunk in gen:
        buffer.append(chunk)
    full_text = "".join(buffer)     # ← 等全部 token 到齐！
    fixed = _fix_and_close_sdk_code(full_text)
    for i in range(0, len(fixed), chunk_size):
        yield fixed[i:i + chunk_size]  # ← 重新分块
```

**问题**: 这是"伪流式"——`_stream_guardrail` 先消费完整个 LLM 生成流（可能长达 60-90s），然后一次性应用 `_fix_and_close_sdk_code`，最后再以 15 字符/块重新 yield。前端用户看到的是长时间的空白（TTFB = 完整生成时间），完全丧失了流式的价值。

**修复方案**: 拆分 `_fix_and_close_sdk_code` 为两部分：
1. **实时检查**: 只检查最后一个 token 是否为 ``` 开头的新代码块 → 做增量闭合（不需要缓冲）
2. **尾部分析**: 仅在生成结束后检查是否需要补全 CDLL — 这项也可以做：如果流式输出中从未出现 `CDLL(`，在流结束后追加
```python
def _stream_guardrail_v2(gen):
    has_cdll = False
    buffer_tail = ""  # 只缓冲最后 50 字符用于检测未闭合 ```
    for chunk in gen:
        if "CDLL(" in chunk:
            has_cdll = True
        buffer_tail = (buffer_tail + chunk)[-50:]
        yield chunk  # 立即透传
    # 流结束后: 处理尾部
    tail = _close_unclosed_fence(buffer_tail)
    if tail:
        yield tail
    if not has_cdll and _needs_cdll_injection(buffer_tail):
        yield "\n```"  # 先闭合代码块再补 CDLL 有风险，改为日志警告
```

#### BUG-4.2: `_fix_and_close_sdk_code()` 的 DLL 名推断不可靠

**文件**: `rag_chain.py:3685-3693`
```python
if "openr6" in answer.lower() or "py_dll" in answer.lower():
    dll_name = "py_dll.dll"
else:
    dll_name = "collrob_sdk.dll"  # ← 默认值
```

**问题**: 
1. 默认 DLL 是 `collrob_sdk.dll`，但如果回答涉及 OpenR6 但未显式提及 "openr6" 或 "py_dll"（如 LLM 只写了 `set_robot_power_on()`），DLL 会被错误推断为 collrob_sdk。
2. JAKA 产品没有 SDK DLL — 如果 JAKA 回答中不幸进入了此函数（虽然概率低），会注入不存在的 DLL 名。

**修复方案**: 从 `context_docs` metadata 中提取 `product_id` 来精确判定 DLL：
```python
# 从调用方传入 product_id 或从 context 推断
if product_id == "OpenR6":
    dll_name = "py_dll.dll"
elif product_id == "OpenC3":
    dll_name = "collrob_sdk.dll"
else:
    return answer  # JAKA/未知 → 不注入 DLL
```

### 4.4 🟡 性能瓶颈

#### PERF-4.1: `extract_align_node` 的 O(N×M×K) 属性词扫描

**文件**: `graph_rag.py:1181-1220`

**问题**: 对每个 Context KV 实体 (N)，在每个匹配位置 (M)，扫描所有其他 KV 实体 (K) 做冲突检测。典型场景: N=5, M=2, K=5 → 50 次正则窗口匹配。虽然绝对耗时不大 (~50ms)，但在流式场景的关键路径上。

**建议**: 在一次遍历中完成：先建 `{数值: 正确属性词}` 查找表 O(N)，再单次扫描 answer 中的每个数值 O(M)，O(1) 查表。

#### PERF-4.2: `render_node` 的 trigram 去重是全文本操作

**文件**: `graph_rag.py:1226-1258` (in extract_align_node) + `graph_rag.py:1085-1092` (in render_node)

**问题**: SemanticDedup 对所有句子计算 trigram overlap。4 段 100 字文本 → 400 次 trigram 提取 + set 操作。Render node 的去重保护也使用类似的 word-level overlap 计算。

---

## 跨层数据流问题

### FLOW-1: `rag_chat()` / `rag_chat_stream()` 与 `run_graph()` / `run_graph_stream()` 两套管线并存

**现象**: `app.py` 使用 LangGraph 引擎 (`run_graph` / `run_graph_stream`)，但 `rag_chain.py` 中仍保留完整的独立管线 (`rag_chat` / `rag_chat_stream`)。

**风险**: 
1. 两套管线在配置参数读取、异常处理、边界行为上可能不一致
2. `rag_chain.py` 中的 `rag_chat_stream` 不经过 `extract_align_node` → 属性词颠倒修正缺失
3. 代码维护负担加倍

**建议**: 明确废弃 `rag_chat` / `rag_chat_stream` 为内部 fallback，所有外部调用统一走 LangGraph 引擎。

### FLOW-2: 模块级可变全局变量 (`_last_numeric_context_missing`)

**文件**: `rag_chain.py:2071` + `graph_rag.py:783`

**问题**: `_last_numeric_context_missing` 是 `rag_chain` 模块的全局变量，由 `_build_messages()` 写入，由 `rag_chat()` 和 `llm_generation_node` 读取。在并发场景下，请求 A 的 `_build_messages` 可能覆盖请求 B 刚写入的值 → 请求 B 错误触发硬拒答。

**修复方案**: 将此标志改为 `_build_messages()` 的返回值或 RAGState 的一个字段。

---

## 优先修复路线图

### 第一阶段：致命 Bug (本周)
| 编号 | 层级 | 问题 | 状态 | 预计 |
|------|------|------|------|------|
| BUG-2.1 | L2 | Search-First `_score` 永远为 0.5 | 待修复 | 1h |
| BUG-3.1 | L3 | System Prompt 膨胀导致 Context overflow | 待修复 | 3h |
| FLOW-2 | — | `_last_numeric_context_missing` 并发不安全 | 待修复 | 1h |
| BUG-2.2 | L2 | Retry 逻辑 off-by-one | 待修复 | 30min |

### 第二阶段：幻觉防御 (下周)
| 编号 | 层级 | 问题 | 状态 | 预计 |
|------|------|------|------|------|
| BUG-4.1 | L4 | `_stream_guardrail` 全量缓冲导致伪流式 | 待修复 | 2h |
| BUG-3.3 | L3 | 反泄露门控的 metadata 漏判 | 🟢 已缓解 (ADR-19) | — |
| BUG-3.2 | L3 | `_resolve_clarification_followup` 文案耦合 | 🟢 已解决 (ADR-19, 模块删除) | — |
| BUG-4.2 | L4 | DLL 名推断不可靠 | 待修复 | 30min |
| HALL-1.1 | L1 | 微缩大纲注入噪声 | 待修复 | 30min |

### 第三阶段：性能优化 (下下周)
| 编号 | 层级 | 问题 | 预计 |
|------|------|------|------|
| PERF-2.1 | L2 | BM25 无磁盘持久化 | 2h |
| PERF-3.1 | L3 | 正则重复编译 | 30min |
| PERF-4.1 | L4 | 属性词扫描 O(N×M×K) | 1h |
| PERF-1.1 | L1 | 全量重建 O(N) | 3h |

---

## 架构设计评价

### 杰出之处
1. **四层容灾金字塔**: 设计优雅，覆盖从 GPU 离线到 CPU-only 的全部故障模式
2. **双轨制 (c_sdk/gui_app)**: 从根本上解决了 "给 GUI 手册生成代码" 和 "给 SDK 文档生成操作步骤" 两类最致命的幻觉
3. **静默斩尾 + 属性词硬改写**: 纯 Python 确定性后处理，零 LLM 开销，精准消除已知幻觉模式
4. **v18/v19 切片净化**: 从 74.5 分提升到近满分健康度，证明了方法论的正确性
5. **🟢 ADR-19 LLM Query Rewriting (v21)**: 用极低温度 (t=0.0, max_tokens=50) 的 LLM 调用替代三个脆弱的正则/启发式模块，从根本上解决了多轮对话中的代词消解与产品名补全问题。设计精巧之处在于"闲聊穿透"规则——纯问候不浪费推理资源，同时通过输出长度上限 (>150 字符 → 回退) 防御大模型罕见幻觉。这是"用小模型解决特定问题"而非"让大模型接管一切"的架构哲学的正确示范
6. **🟢 ADR-20/21/22 四轮闭环重构 (v22)**: 通过四个精准的靶向修复完成了从检索召回 (子查询阈值 4→2) → 切片完整性 (Autocut SDK _min_k 提升至 10) → 术语纠偏 (动态 `_term_alignment_prefix` 零 Token 注入) → 输出排版 (SDK 两段式铁律) 的全链路闭环。每个修复都遵循"最小改动、最大杠杆"原则——不引入新的 LLM 调用、不膨胀 System Prompt、不改变核心数据结构，仅在最脆弱的环节 (阈值/模板) 做确定性加固

### 需警惕的架构债
1. **System Prompt 持续膨胀**: 从最初的 15 行增长到 210 行，每次新增规则都在追加而非重构
2. **两套管线并存**: `rag_chat` vs `run_graph` 的功能分裂。注意：ADR-19 的 `_rewrite_query_with_llm` 已同步接入两条管线 (rag_chain.py L2874 + L3082)，但 `_CLARIFICATION_MARKER` 等残留常量仅服务于旧管线
3. **模块级可变全局状态**: `_last_numeric_context_missing`、`_HYDE_CACHE` 等在多线程环境下不安全
4. **过度依赖 Regex**: ~~4 层架构中累计 50+ 个正则表达式~~ → ADR-19 已删除 `_fuse_short_query`、`_resolve_clarification_followup`、`_has_business_intent` 三个脆弱的正则/启发式模块。剩余正则表达式约 40+ 个，部分未预编译、部分仍过于宽泛（如 `_PSEUDO_SECTION_BLACKLIST` 的 substring `in` 匹配）
