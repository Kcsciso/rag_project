# 比邻星 (ProximaRAG) — 全盘架构审计报告

> **日期**: 2026-07-30 | **审计人**: AI 架构师 | **覆盖版本**: v20  
> **方法**: 四层 RAG 架构逐层排查 + 跨层数据流追踪

---

## 审计总览

| 层级 | 名称 | 严重问题 | 性能瓶颈 | 幻觉风险 | 评分 |
|------|------|---------|---------|---------|------|
| L1 | 数据摄入与切片 | 2 | 1 | 2 | B+ |
| L2 | 检索与重排 | 3 | 2 | 1 | B |
| L3 | 上下文组装与指令 | 4 | 2 | 3 | B- |
| L4 | 生成控制与后处理 | 2 | 2 | 1 | B+ |

**综合评分: B+ (82/100)** — 架构设计优秀，细节工程有若干可修复隐患。

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
用户 Query
  │
  ├── _preprocess_query()           ← 口语噪音剥离
  ├── _normalize_punctuation()      ← 全半角归一化
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

### 3.2 保持的优良设计

1. **双轨 Prompt 控制**: `gui_app` 绝对禁止代码; `c_sdk` API 即答案
2. **反跨产品泄露门控**: metadata `function_names` + 正文双重确认，仅当目标产品缺失且非目标产品存在 API 时才注入 `_anti_bleed_prefix`
3. **Context Cap 整块剔除**: 不切割任何单个 Chunk 内部正文
4. **历史沉渣净化**: `sanitize_chat_history()` + Citation 前缀清洗 + 代码块替换 + 尾部拒答剥离

### 3.3 🔴 严重隐患

#### BUG-3.1: System Prompt 膨胀 — Token 预算失控

**文件**: `rag_chain.py:1514-1725` (`RAG_SYSTEM_PROMPT`)

**测量**: System Prompt 共 210 行、~3,500+ 字符 → 约 **1,500-2,000 tokens** (中文)。加上 `_dual_track_prefix` (~200 tokens) + `_anti_bleed_prefix` (~150 tokens) + 10 个 SDK Child chunks (~4,000 tokens) + 历史消息 + query → **总输入远超 Qwen2.5-7B 的 8192 上下文限制**。

**实际影响**: 当 10 个 full chunks 被 Autocut 放行 + SDK Context Cap=8000 字符时，`_build_messages` 的 `total_chars` 可能达到 8000+。虽然后续的 `Context Cap` 整块剔除逻辑会从末尾（Parent chunks）开始丢弃，但这意味着最坏情况下所有 Parent 背景信息被删除，只剩下孤立的 Child API 函数定义 → LLM 缺少操作上下文 → 生成质量下降。

**修复方案**: 
1. 将 System Prompt 从 210 行压缩至 80 行以内（保留核心规则，移除 Few-Shot 示例到外部文件）
2. 为 SDK 查询动态降低 Context Cap 从 8000→6000，为 System Prompt 预留空间
3. 或在 `_call_llm` 中先计算 `_build_messages` 输出的实际 token 数（用 tiktoken），超出时动态裁剪

#### BUG-3.2: `_CLARIFICATION_MARKER` 与 `build_product_clarification_response()` 文案脱节

**文件**: `rag_chain.py:714` vs `graph_rag.py:450-455`

**`rag_chain.py` 使用的 marker**:
```python
_CLARIFICATION_MARKER = "请问您询问的是哪一款产品呢"
```

**`graph_rag.py` 实际生成的文案**:
```python
f"请问您询问的是哪一款产品呢？（当前已支持：{products_str}）\n"
"不同产品的 SDK 接口与操作逻辑有所不同，请告知具体型号以便为您准确解答。"
```

**问题**: `_resolve_clarification_followup()` 用 `_CLARIFICATION_MARKER in last_assistant_msg` 检测上一轮是否为澄清反问。由于 marker 确实是澄清文案的子串，这在当前版本可以匹配。但若未来 `build_product_clarification_response()` 修改文案而 marker 未同步更新 → 澄清检测静默失效 → 用户回复产品名后不会被拼接。

**修复方案**: 将 marker 提取到 `config.py` 或使用更稳定的检测方式（如在返回的 dict 中增加 `is_clarification=True` 标记，通过历史消息的某个隐藏字段检测）。

#### BUG-3.3: `_anti_bleed_prefix` 的跨产品 API 检测漏判

**文件**: `rag_chain.py:2150-2219`

**问题**: 反泄露门控依赖 `metadata.get("function_names")` 和 `metadata.get("is_api")` 来判定目标产品是否有 API。但如果：
1. 目标产品的 Child chunks 确实含有 API 函数定义
2. 但 **metadata 未正确标注**（`function_names=""` 或 `is_api=False`）
3. 同时非目标产品的 chunks **有** function_names

→ 门控触发，错误地告诉 LLM "当前产品无 API"，导致 LLM 诚实拒答而非阅读 Context 中的实际代码。

**场景**: OpenR6 某个 API chunk 因边界合并导致 function_names 提取不完整，但正文中明确包含 `set_robot_power_on()`。此时若 query 命中了一个 OpenC3 的 chunk（有 function_names），门控错误触发 → LLM 拒答。

**修复方案**: 在门控判断中增加第三重确认——扫描目标产品 chunks 的**正文**（`page_content`）中是否包含函数调用模式（如 `robot_.*(`），若正文中有则豁免 metadata 缺失。

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
| 编号 | 层级 | 问题 | 预计 |
|------|------|------|------|
| BUG-2.1 | L2 | Search-First `_score` 永远为 0.5 | 1h |
| BUG-3.1 | L3 | System Prompt 膨胀导致 Context overflow | 3h |
| FLOW-2 | — | `_last_numeric_context_missing` 并发不安全 | 1h |
| BUG-2.2 | L2 | Retry 逻辑 off-by-one | 30min |

### 第二阶段：幻觉防御 (下周)
| 编号 | 层级 | 问题 | 预计 |
|------|------|------|------|
| BUG-4.1 | L4 | `_stream_guardrail` 全量缓冲导致伪流式 | 2h |
| BUG-3.3 | L3 | 反泄露门控的 metadata 漏判 | 1h |
| BUG-4.2 | L4 | DLL 名推断不可靠 | 30min |
| HALL-1.1 | L1 | 微缩大纲注入噪声 | 30min |

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

### 需警惕的架构债
1. **System Prompt 持续膨胀**: 从最初的 15 行增长到 210 行，每次新增规则都在追加而非重构
2. **两套管线并存**: `rag_chat` vs `run_graph` 的功能分裂
3. **模块级可变全局状态**: `_last_numeric_context_missing`、`_HYDE_CACHE` 等在多线程环境下不安全
4. **过度依赖 Regex**: 4 层架构中累计 50+ 个正则表达式，部分未预编译、部分过于宽泛
