# 比邻星 (ProximaRAG) — 全盘架构审计报告

> **日期**: 2026-08-06 | **审计人**: Staff Engineer (AI 架构师) | **覆盖版本**: v24 → v29 演进 + **v30 架构重构方案**  
> **方法**: 四层 RAG 架构逐层排查 + 跨层数据流追踪 + v24→v29 diff 深度审计 + v30 五大故障现象根因分析  
> **本次更新**: v29 架构瓶颈诊断 + v30 "状态机编排"统一架构方案。四层评分因系统性并发 Bug 和 OCR 参数丢失下调。

---

## 审计总览

| 层级 | 名称 | 严重问题 | 性能瓶颈 | 幻觉风险 | 评分 |
|------|------|---------|---------|---------|------|
| L1 | 数据摄入与切片 | 2 | 1 | 1 | **B+** (↓ from A-) |
| L2 | 检索与重排 | 1 | 1 | 1 | **B+** (↓ from A) |
| L3 | 上下文组装与指令 | 2 | 0 | 1 | **B** (↓ from A) |
| L4 | 生成控制与后处理 | 1 | 0 | 1 | **B+** (↓ from A+) |
| L0 | 🆕 统一门控编排 | — | — | — | **新增层** |

**综合评分: B/B+ (82/100)** — v25-v29 四轮迭代解决了 14→FAILED 的回归问题，但引入了三个结构性缺陷：(1) `_last_numeric_context_missing` 模块级全局变量的并发竞态，(2) Parent Chunk 暴力截断导致的 OCR 参数丢失，(3) 三种独立拒答机制的"三角混战"。v30 方案通过 L0 GuardOrchestrator 统一门控 + L1 语义边界保护 + L5 跨产品隔离墙，目标恢复 A 级评分。

---

## v25-v29 架构演进摘要

| 版本 | 核心主题 | 关键变更 | 引入的风险 |
|------|---------|---------|-----------|
| v25 | 回归攻坚 | 围栏闭合状态机 / 逃生舱条款 / JAKA 数字保护 / SemanticDedup 规范 / `extract_align_node` 入口接入 `_fix_and_close_sdk_code` | 逃生舱依赖 LLM 服从 |
| v26 | 最后一公里 | OCR 面积过滤重构 / BM25 复合词原子化 / 重写器 always-on / 逃生舱 `> [!WARNING]` 视觉加固 | CTM Y 归位污染切片 (v27 回退) |
| v27 | 回归反转 | 路由责任切分 / OCR 页尾追加 / 模板选择守卫 (A/B/C 三条件) / 动态 BM25 权重 / ~~CTM Y 归位~~ | 守卫三条件覆盖不全；动态权重对高频词稀释无效 |
| v28 | 切片状态机化 | 区域状态机标题提取 / line 级表格重建 / last_header 层级栈 / 守卫脱敏 / 重写指代泛化 | 表格重建仅 gui_app 轨；Parent 截断问题未解决 |
| v29 | 数据语义化+确定性拒答 | OCR 键值法 / 图片过滤重构 / 数字守卫复合词豁免 / **Fast-Path 短路** (侧信道 `(messages, refusal_flag)`) / 重写协议中立性 | `_last_numeric_context_missing` 全局变量竞态 (已承认未修复)；Fast-Path 与数字守卫状态不一致 |

---

# 🔴 v30 架构重构方案：从"补丁叠加"到"状态机编排"

## 零、问题总览：v29 架构的系统性塌陷

经过对 L1-L4 全链路代码的深度走查，v29 并非"个别 Bug"，而是**四种独立机制在四个层次上的互不理解**，产生了系统性故障。以下按五大故障现象逐一进行根因分析。

---

## 一、Fast-Path 物理短路失效 —— 三种拒答机制的"三角混战"

### 1.1 当前代码的真实结构

v29 存在 **三种独立运行的拒答判定机制**，各守各的门：

| 机制 | 位置 | 状态管理 | 线程安全 |
|------|------|---------|---------|
| **ABSTAIN 网关** | `graph_rag.py:709-739` | 硬编码实体列表 | ✅ 纯函数 |
| **模板守卫 (A/B/C)** | `rag_chain.py:1934-2001` | `_refusal_override` → 返回值 tuple | ✅ 线程安全 |
| **数字守卫** | `rag_chain.py:1734-1787` | `_last_numeric_context_missing` → **模块级全局变量** | ❌ **已知并发竞态** |

### 1.2 致命漏洞：`_last_numeric_context_missing` 的并发竞态

```python
# rag_chain.py:2221 — 模块级全局（FastAPI run_in_executor 线程池下竞态）
_last_numeric_context_missing = False
```

调用链还原：
1. 请求 A 调用 `_build_messages` → 设置 `_last_numeric_context_missing = True`（发现缺失实体）
2. 请求 B 调用 `_build_messages` → 设置 `_last_numeric_context_missing = False`（实体全部命中）
3. 请求 A 在 `graph_rag.py:801` 读取 `_rag_chain_mod._last_numeric_context_missing` → **已为 False！**
4. 请求 A 的数字守卫被**静默关闭** → LLM 收到含缺失实体的问题但不设防 → **幻觉输出**

v29 CLAUDE.md 自身已承认此风险（L3 表格 "`_last_numeric_context_missing` 线程安全" 行），但标记为"待修复为 State 字段"而从未执行。

### 1.3 第二个漏洞：逃生舱条款的 Prompt 依从性陷阱

即使模板守卫未触发（漏判），正常的双轨模板末尾带有 `> [!WARNING] ⛔🔴 绝密拦截` 逃生舱条款。v26 删除了"请明确说明"对冲行以增强 Recency Bias。但核心问题是：**逃生舱本身仍然是 Prompt 指令，依赖 LLM 服从**。当 Qwen2.5-7B 上下文窗口中有强信号代码块时，逃生舱指令可能被忽略。

### 1.4 第三个漏洞：ABSTAIN 网关覆盖不全

`graph_rag.py:710-714` 的 `_query_entities` 集合是**硬编码列表**，不包含新增的通用属性。例如查询"固件版本号"不会触发 ABSTAIN，需依赖后续的模板守卫。三个机制之间存在覆盖间隙。

### 1.5 第四个漏洞：流式路径中全局变量被多处读取

`run_graph_stream` (graph_rag.py:1818) 和 `llm_generation_node` (graph_rag.py:801) 均通过 `_rag_chain_mod._last_numeric_context_missing` 读取全局变量。两个并发流式请求可互相覆盖该值。

---

## 二、表格与离散数值召回失败 —— L1 截断 + L2 高频词稀释的"双重夹击"

### 2.1 L1 端：Parent Chunk 暴力截断是 OCR 参数的无声杀手

```python
# pdf_loader.py:1493-1499 — Parent 截断
if len(parent_text) > parent_chunk_size:  # GUI=2000
    cutoff = max(
        parent_text.rfind('\n\n', parent_chunk_size - 200, parent_chunk_size + 200),
        parent_text.rfind('\n', parent_chunk_size - 100, parent_chunk_size + 100),
        parent_chunk_size,
    )
    parent_text = parent_text[:cutoff].strip()
```

场景还原：
1. JAKA 手册某 H2 章节（如"Modbus 通讯设置"）包含正文 1500 字 + OCR 参数 500 字（端口/波特率等截图识别结果）
2. OCR 文本通过 `[本页图片解析参数: page=N]` 追加到页面末尾
3. 页面合并后进入该 H2 章节，总长度 2000+
4. **Parent Chunk 截断在段落边界处，恰好切在 OCR 参数的起始位置或中间**
5. OCR 提取的 6502/9600 等参数被整块丢弃

**关键**：GUI Child Chunk 在 `_split_text_into_children` 中确实不做截断（`pdf_loader.py:2008-2012`），但 **Parent 层做**。当 Parent 被召回（Chapter Isolation +20.0 提权），其中缺失 OCR 参数 → LLM 看不到参数 → 幻觉或拒答。

### 2.2 L2 端：高频词对 Dense 检索的隐性降权

```python
# rag_chain.py:2776 — 动态 BM25 权重
_BM25_WEIGHT = 3.0 if (len(query) <= 8 or _compound_re.search(query)) else 1.2
```

当用户查询 "JAKA Modbus 端口号 6502" 时：
- `len(query) > 8` 且不含复合词 → BM25 权重 = 1.2
- "JAKA" 是极高频词（遍布所有切片），Dense 向量对该词几乎无区分度
- Dense 相似度主要靠 "Modbus 端口号 6502" 匹配，但 6502 被 `jieba` 切碎
- 结果：Dense 得分被 "JAKA" 高频稀释，BM25 权重仅 1.2 无法有效纠偏
- RRF 融合后，目标切片排到第 5-8 名，被 Autocut（K=8~15）**边缘化**

**机制漏洞**：代码中**没有任何 IDF/词频惩罚机制**来抵消高频实体词对 Dense 检索的稀释效应。`_BM25_WEIGHT` 的动态条件只检查长度和复合词，不检测高频实体稀释。

### 2.3 L1 端（深层）：OCR 键值对被嵌入大文本块的 Dense 稀释

OCR 键值对 "端口：6502，波特率：9600" 被嵌入一个 1500 字的 child chunk 中，Dense 向量被正文稀释。BM25 对纯数字 token "6502" 权重被抵消——在 1500 字的文档中 BM25 的 TF 分量近乎为零。

---

## 三、长流程与宏观大纲丢失 —— Parent-Child 索引的"结构性失联"

### 3.1 Parent Chunk 的"假召回"

当前 Parent Chunk 包含 `[章节大纲参考]` 但正文被截断至 2000 字符。当用户提问"JAKA Modbus 通讯设置有哪些步骤"时：
- 检索召回 Parent Chunk（含大纲："- 配置端口参数 / - 设置从站地址 / - 测试通讯"）
- 但 Parent 正文中步骤 3-5 被截断（2000 字符限制）
- Child Chunks 包含完整步骤，但它们被索引为独立的小切片（每个 H3 一个 child）
- **单个 child 只包含 1-2 个步骤，缺乏完整流程**
- LLM 看到大纲但看不到后续步骤的正文 → 幻觉补全

### 3.2 章节大纲上限过严

```python
# pdf_loader.py:1863 — Child mini-TOC capped at 5 items
_MAX_TOC_ITEMS = 5
```

对于深度嵌套的章节（如 "3.1.5 Modbus 通讯设置" 下有 8 个子步骤），大纲只展示前 5 条 + "... (更多章节略)"。宏观提问时，LLM 看到 "..." 可能自行脑补。

### 3.3 宏观提权关键词覆盖有限

```python
# rag_chain.py:2843-2873 — 宏观提权 v2
if (_is_macro_intent or chunk_type == "parent" or _has_toc_in_content):
```

`_is_macro_intent` 关键词覆盖有限（"内容/总结/介绍/大意/结构"），对"有哪些步骤"、"包含什么功能"这类问法可能漏判。

---

## 四、多轮对话污染 —— 历史净化的"表面清洗"

### 4.1 净化不完整：技术数据不在代码块内就无法剥离

```python
# rag_chain.py:2093-2097 — 仅剥离代码块，不剥离技术数据
content = re.sub(r'```python[\s\S]*?```', '[已提供代码示例]', content, flags=re.DOTALL)
content = re.sub(r'```[\s\S]*?```', '[已提供代码块]', content, flags=re.DOTALL)
```

上一轮 LLM 输出了 "根据《JAKA手册》【Modbus通讯设置】的记载：端口号 6502，波特率 9600..."。这些技术数据**不在代码块内**，因此不被净化。下一轮用户问"OpenC3 怎么配置端口"，模型可能把上一轮的 JAKA 端口 6502 带入 OpenC3 的上下文中。

### 4.2 跨产品多轮对话缺少产品上下文隔离

`MAX_HISTORY_TURNS = 2` (只保留最近 4 条消息) 对于跨产品切换是**双刃剑**：
- 如果前 2 轮在讨论 JAKA，第 3 轮切换到 OpenC3：历史中仍有 JAKA 的参数/代码 → OpenC3 答案被污染
- 如果只保留 1 轮（2 条消息）：指代消解失效（"那它的参数呢？"无法解析"它"）

### 4.3 `_HARD_REFUSAL` 进历史后的复读风险

`sanitize_chat_history` 中的 `_TAIL_REFUSAL_RE` 能剥离大部分拒答句式，但 `_HARD_REFUSAL` 消息格式不完全匹配剥离正则——残余片段仍可能在下一轮被模型复读。

---

## 五、OCR 参数丢失与暴力截断 —— 终极根因

### 5.1 完整调用链还原

```
PDF Page (含表+截图)
  → PyMuPDF get_text("dict") → 按 line 级 y 聚类重建（gui_app 轨）
  → 字符密度检测 → 低密度页或 gui_app 强制扫图
  → RapidOCR 识别图片文字 → _ocr_kv_normalize_row + _ocr_merge_cross_line
  → ocr_lines 按图子块化（[图表内容包含：本页第N张截图]）
  → 追加到 page_parts：page_text + [本页图片解析参数: page=N] + OCR内容
  → 所有页面 "\n\n" 拼接 → full_text
  → _v4_extract_headings() 提取标题树
  → _v4_build_parent_child_docs() 构建 Parent + Child
    → Parent 按 H2 边界切 → 正文 > 2000 chars → **截断！OCR 参数在此丢失**
    → Child 按 H3 边界切 → _split_text_into_children(gui_app) → 整段保留（无截断）
```

### 5.2 为什么 Child Chunk 有完整内容但仍召回失败

1. **检索时 Child 的向量与 query 的相似度不足**：OCR 键值对被嵌入大文本，Dense 向量被稀释
2. **BM25 对纯数字不友好**：`jieba` 分词后 "6502" 作为独立 token 权重被稀释
3. **Parent 被召回但 OCR 内容缺失**：宏观或步骤类提问优先召回 Parent（Chapter Isolation +20），但 Parent 中的 OCR 内容已被截断
4. **结果**：召回排名靠前的是不含 OCR 参数的 Parent chunk → LLM 看不到参数 → 幻觉或拒答

---

# v30 统一架构方案：五层状态机编排

## 核心设计原则

> **从"打补丁"到"状态机编排"。每个故障现象由一个明确的"编排器 (Orchestrator)"在正确层次拦截，而不是分散在 4 层中的多个正则/全局变量/Prompt 指令。**

## 方案总览

```
                        ┌─────────────────────────────┐
                        │   L0: GuardOrchestrator     │  ← NEW — 统一拒答状态机
                        │   单一入口，唯一真相源        │
                        │   (替代 ABSTAIN + 模板守卫    │
                        │    + 数字守卫 三个分散机制)   │
                        └─────────────┬───────────────┘
                                      │ guard_result (dataclass, 不可变)
        ┌─────────────┬───────────────┼───────────────┬──────────────┐
        ▼             ▼               ▼               ▼              ▼
   L1: ChunkGuard  L2: RecallGuard  L3: Template    L4: Stream     L5: History
   语义边界保护    动态去噪提权     模板选择器       透传校验       跨轮隔离墙
   (替代暴力截断)  (替代固定权重)   (替代Prompt逃生舱) (替代正则清洗)  (替代正则净化)
```

### L0：GuardOrchestrator —— 统一拒答状态机（最高优先级 🔴 P0）

**问题**：当前三个拒答机制（ABSTAIN / 模板守卫 / 数字守卫）各自独立判定，状态分散在模块全局变量和函数返回值之间。

**方案**：

```python
# 设计意图：L0 是 _build_messages 调用前的单一判定节点
# 所有调用方（rag_chat / rag_chat_stream / llm_generation_node / run_graph_stream）
# 在调用 _build_messages 之前，必须先过 L0 门控

@dataclass(frozen=True)  # 不可变，线程安全
class GuardResult:
    decision: Literal["allow", "hard_refuse", "numeric_guard", "clarify"]
    reason: str
    # allow → 正常走 L1-L4
    # hard_refuse → 确定性拒答（跳过 L1-L4，直接返回 _HARD_REFUSAL）
    # numeric_guard → 数字实体缺失，触发 KV/BM25 第二机会
    # clarify → 多产品歧义，触发澄清反问

class GuardOrchestrator:
    """
    统一门控编排器。
    判定顺序（短路求值）：
      1. IMPOSSIBLE_COMBOS → hard_refuse (原 L4 检测提升至此)
      2. 闲聊/身份 → clarify (透传至 product_routing)
      3. 实体-上下文一致性检测 → numeric_guard 或 hard_refuse
      4. 模板守卫 A/B/C → hard_refuse (原 _refusal_override)
    所有判定均基于 (query, context_docs, product_id, doc_types) 四元组，
    零模块级全局变量。
    """
```

**关键变更**：
- **消除 `_last_numeric_context_missing` 全局变量**：改为 `GuardResult.decision == "numeric_guard"`
- **ABSTAIN 网关合并**：`graph_rag.py:709-739` 的硬编码实体列表迁移至 GuardOrchestrator 的实体-上下文一致性检测器
- **所有调用方在调用 LLM 前必须通过 L0**，GuardResult 作为不可变值沿调用链传递

### L1：ChunkGuard —— 语义边界保护替代暴力截断（🔴 P0）

**问题**：Parent Chunk 的 `parent_chunk_size` 暴力截断是 OCR 参数丢失的物理根因。

**方案**：不再按字符数截断，改为**语义边界保护**。

设计要点：
1. **受保护区域完全豁免**：OCR 补充块、代码块、Markdown 表格绝不截断
2. **Parent Chunk 改为"完整 H2 章节"或"受保护块 + 上下文窗口"**，不再设硬上限
3. 当 Parent 确实过大（> 4000 chars 的巨型章节）时，做语义分段：
   - 第一段：章节导言 + OCR 补充块 → `parent_core`
   - 后续段：按 H3 边界切 → 各为一个 `child_with_context`
4. Context Cap（`_MAX_CONTEXT_CHARS=8000`）移至 L3 TemplateSelector，基于检索排名做智能截断（优先保留含数字/表格/OCR 的切片）
5. **OCR 块标记为 "不可丢弃"**：在 Context Cap 裁剪时，含 `[本页图片解析参数]` 或 `[图表内容包含：]` 的切片具有最高保留优先级
6. **表格行密度标记**：`| cell1 | cell2 |` 格式的行自动标记为 "高信息密度"，检索时提权

### L2：RecallGuard —— 动态去噪提权替代固定权重（🟡 P1）

**问题**：
1. 高频词（"JAKA"）稀释 Dense 检索
2. `_BM25_WEIGHT` 的动态条件不检测高频实体稀释
3. BM25 对纯数字（6502）的 token 处理不理想

**方案**：

1. **高频实体稀释检测**：统计 query 中各 term 在语料库中的 DF (Document Frequency)；对 DF > 80% 的 term，降低其 Dense 向量维度权重，同时自动提升 BM25 权重
2. **数字 Token 强化**：检测 query 中的 ≥2 位数字 → 在 BM25 索引中做 n-gram 扩展（"6502" → ["6502", "650", "502"]），提升部分匹配容错；数字 token 在 RRF 融合中额外 +0.05 boost
3. **OCR 切片提权**：含 `[本页图片解析参数]` 或 `[图表内容包含：]` 的切片 → Dense 检索后额外 +0.03 RRF

### L3：TemplateSelector —— 模板选择器替代 Prompt 逃生舱（🟢 P2）

**问题**：当前双轨模板包含 `> [!WARNING]` 逃生舱条款，但仍然依赖 LLM 服从。

**方案**：将拒答决策从 Prompt 指令**前移**到 Python 层。

- 三种模板：`NORMAL_GUI` / `NORMAL_SDK` / `REFUSAL`
- **废除 `> [!WARNING]` 逃生舱**：拒答决策完全在 Python 层完成
- 所有模板均不含内部的条件分支（如 "如果...则输出拒答"）——条件分支已在 Python 层通过 GuardResult 完成
- 每条 User Message 末尾追加 `[输出指令: 仅输出以下格式]` 标记，用纯文本指令替代 Markdown 引用块

### L4：StreamGuard —— 流式透传校验替代全局变量依赖（🟢 P2）

**方案**：
- 流式路径不再读取 `_rag_chain_mod._last_numeric_context_missing`
- StreamGuard 接收 GuardResult，在流开始前即确定行为
- `_stream_guardrail` 的围栏闭合状态机保留（零缓冲的 ``` 奇偶计数）
- `render_node` 持续退化为纯透传（v24 方向正确）

### L5：HistoryIsolationWall —— 跨轮隔离墙替代正则净化（🟡 P1）

**问题**：正则清洗无法区分"JAKA 的有效技术数据"和"会污染 OpenC3 的跨产品泄露"。

**方案**：

1. **产品上下文标记**：每条 assistant 消息记录其回答所基于的 product_id
2. **跨产品切换检测**：当前 query 的 product_id 与历史中最近 assistant 消息的 product_id 不同 → 触发"产品上下文隔离"：历史中的技术数据（数字/代码/API名）被脱敏处理
3. **同产品延续**：不做额外处理，保留完整历史供指代消解
4. **拒答隔离**：assistant 消息若为 `_HARD_REFUSAL` / `_ESCAPE_REFUSAL` → 整条消息从历史中移除（而非仅清洗表面文本）
5. 跨产品脱敏规则：数字（≥2 位）→ `[数值]`；snake_case 函数名 → `[函数名]`；代码块 → `[代码已隔离]`

---

## 架构对比：v29 vs v30

| 维度 | v29 | v30 |
|------|-----|-----|
| 拒答判定 | 3 个独立机制 + 模块全局变量 | L0 GuardOrchestrator 单一入口 + 不可变 dataclass |
| 线程安全 | ❌ `_last_numeric_context_missing` 竞态 | ✅ GuardResult 不可变值传递 |
| Parent 截断 | 2000 chars 暴力截断 | 语义边界保护 + L3 Context Cap 智能裁剪 |
| OCR 保留 | Parent 截断可能丢弃 OCR | OCR 块标记 MUST_KEEP + 优先级保护 |
| BM25 权重 | 固定条件 `len<=8 或复合词` | DF 感知 + 数字 Token Boost |
| 历史净化 | 正则清洗（语法层面） | 跨产品隔离墙（语义层面） |
| 逃生舱 | Prompt `> [!WARNING]` 依赖 LLM 服从 | Python 层 TemplateSelector 确定性选择 |
| 调用链 | 4 个调用方各自处理 guard | 统一 `guard → _build_messages → LLM` 流水线 |

---

## 实施优先级与风险矩阵

| 优先级 | 模块 | 变更 | 影响面 | 风险 |
|--------|------|------|--------|------|
| **P0 🔴** | L0 GuardOrchestrator | 统一拒答状态机 + 消除 `_last_numeric_context_missing` 全局变量 | 4 个调用方 + graph_rag.py | 中：需重构 `_build_messages` 调用链 |
| **P0 🔴** | L1 ChunkGuard | 废除 Parent 暴力截断 → 语义边界保护 | pdf_loader.py `_v4_build_parent_child_docs` | 中：Parent 变大，Context Cap 压力增大 |
| **P1 🟡** | L5 HistoryIsolationWall | 跨产品隔离墙 | rag_chain.py `sanitize_chat_history` + graph_rag.py | 低：仅影响多轮对话 |
| **P1 🟡** | L2 RecallGuard | DF 感知权重 + 数字 Token Boost | rag_chain.py `_hybrid_retrieve_single` + vector_store.py | 低：仅改权重参数 |
| **P2 🟢** | L3 TemplateSelector | 废除 Prompt 逃生舱 → Python 层模板选择 | rag_chain.py `_build_messages` 模板部分 | 低：模板替换 |
| **P2 🟢** | L4 StreamGuard | 流式路径消除全局变量依赖 | rag_chain.py + graph_rag.py 流式路径 | 低：接口替换 |

### 新增风险与缓解

| 风险 | 缓解 |
|------|------|
| 🟡 L0 GuardOrchestrator 误判（如漏召回导致虚拒） | 保留 Condition A 的 BM25 第二机会；所有 `hard_refuse` 决定记录结构化日志供审计 |
| 🟡 Parent 不再截断 → 单 chunk 可达 8000+ chars | L3 Context Cap 仍上限 8000（整块丢弃），特大 Parent 通过语义拆分降级为 parent_core + overflow_children |
| 🟡 DF 统计需要额外索引 | DF 可近似自 ChromaDB metadata 的文档计数，无需全量重建 |
| 🟡 HistoryIsolationWall 的产品归属推断可能有误 | 优先用消息 metadata 中的 product_id（需在 LangGraph State 中新增字段）；fallback 到关键词推断 |
| 🟡 废除逃生舱可能导致边界 case 缺乏柔性 | TemplateSelector 保留 `numeric_guard` 中间状态（第二机会后再判定），非二元 allow/hard_refuse |

---

# 历史审计记录 (v24)

<details>
<summary>点击展开 v24 原始审计报告</summary>

## 🔴 v24 重构核心论述：为什么"放弃 JSON + 正则"转向"模板 + 流式"是对小模型的决定性胜利

### 问题诊断：旧架构的三重死锁

在 v23 及之前版本，ProximaRAG 的 L4 层遵循一条"大模型自由生成 → JSON 结构化提取 → 正则清洗修正"的后处理管线：

```
LLM 自由生成 (max_tokens=1024, 长上下文 8000 chars)
    │
    ▼
render_node: JSON 解析提取 → 结构化渲染
    │
    ▼
extract_align_node: 属性词硬改写 + SemanticDedup + 5 道物理清洗正则
    │
    ▼
_fix_and_close_sdk_code: Markdown 闭合 + CDLL 补全
    │
    ▼
_stream_guardrail: 全量缓冲 → 重新分块 → 伪流式输出
```

这套管线在理论上是自洽的——"让大模型放手生成，然后用确定性规则修正错误"。但在 1.5B/7B 级别的小模型实践中，暴露了三个致命缺陷：

#### 死锁一：注意力衰减 × 结构复杂度 = JSON 提取失败

小模型（尤其是 1.5B）在 8000 字符的长上下文中，注意力分布呈明显的"首尾偏置"（Primacy/Recency Bias）。当 Prompt 末尾要求输出特定 JSON 结构时，模型在生成中途已经"忘记"了 JSON Schema 的精确要求。结果：
- **JSON 格式错误率 ~15-20%**：缺失闭合引号/括号、多余逗号、字段名拼写偏差
- **`render_node` 的 JSON 解析频繁失败**：触发降级逻辑，整段输出被丢弃
- **恶性循环**：JSON 解析失败 → 重试 → 更多延迟 → 用户体验崩溃

#### 死锁二：正则清洗的"误杀-漏杀"跷跷板

L4 层的 5 道物理清洗正则 + `_fix_and_close_sdk_code` 中的函数名暴力替换，构成了一个脆弱的"补丁塔"：
- **误杀风险**：暴力替换可能破坏注释中的正常文本
- **漏杀风险**：正则无法覆盖 LLM 所有的创造性错误
- **维护噩梦**：每个新发现的 LLM 错误模式都需要新增一条正则规则

#### 死锁三：`_stream_guardrail` 的伪流式陷阱

旧版全量缓冲 + 重新分块导致：**TTFB = 完整生成时间（60-90s）**。前端用户看到的是长时间空白，然后瞬间吐出一大段文字——完全丧失了流式输出的用户体验价值。

### 新架构：Markdown 模板强约束 (Template Masking) + 极速流式穿透

v24 的核心理念转变是：**不再让小模型"自由创作然后修正"，而是给小模型一个精确的"填空模板"，将模型的自由度限制在模板的槽位（slot）内。**

#### 为什么模板约束对 1.5B/7B 小模型是决定性的

1. **注意力锚定效应 (Attention Anchoring)**：序列末尾的 token 对模型输出的影响权重最高（Recency Bias）。模板在 Prompt 底端 → 模型生成每个 token 时，模板约束始终在其注意力窗口的"热点区域"内
2. **自由度压缩 (Degrees-of-Freedom Compression)**：模板将"怎么说"的决策空间压缩到接近于零——释放认知资源给"说什么"
3. **错误模式可预测性 (Error Mode Predictability)**：自由格式的错误是发散的，模板约束下的错误是收敛的（只可能发生在槽位填充环节）
4. **流式穿透**：模板约束确保格式正确 → 不再需要等待完整输出后再用正则修正 → TTFB 降至 <2s

### v24 变更清单

| 文件 | 变更 | 类别 |
|------|------|------|
| `rag_chain.py` | `RAG_SYSTEM_PROMPT` 重写：从 210 行压缩至 ~30 行 | L3 重构 |
| `rag_chain.py` | `_doc_section_str` 仅取 Top-1 来源章节 | L3 优化 |
| `rag_chain.py` | `_stream_guardrail` 废除全量缓冲，恢复极速流式透传 | L4 重构 |
| `rag_chain.py` | `_fix_and_close_sdk_code` 函数名修正表保留但标注为"过渡期兜底" | L4 收敛 |
| `graph_rag.py` | `render_node` 退化为极简文本透传（废弃 JSON 解析） | L4 简化 |
| `graph_rag.py` | `extract_align_node` 删除"屠魔版"正则清洗逻辑 | L4 简化 |
| `graph_rag.py` | `run_graph_stream` 底部删除双重输出 Bug | Bug 修复 |

### v24 审计评分

| 层级 | 名称 | 严重问题 | 性能瓶颈 | 幻觉风险 | 评分 |
|------|------|---------|---------|---------|------|
| L1 | 数据摄入与切片 | 1 | 1 | 0 | A- |
| L2 | 检索与重排 | 2 → 1 | 1 | 0 | A |
| L3 | 上下文组装与指令 | 1 → 0 | 1 → 0 | 1 → 0 | A- → A |
| L4 | 生成控制与后处理 | 0 | 1 → 0 | 0 | A → A+ |

**综合评分: A → A+ (94/100)**

### v24 仍存在的隐患

- **BUG-2.1**: Search-First 软路由的 `_score` 属性为空 (`graph_rag.py:411`)
- **BUG-2.2**: Retry 逻辑 off-by-one (`run_graph_stream` 用 `>` 而 `_route_after_sdk_verify` 用 `<=`)
- **FLOW-2**: `_last_numeric_context_missing` 等模块级全局变量的并发安全问题

### v24 代码结构"体检"

| 文件 | 行数 | 职责数 | 问题 |
|------|------|--------|------|
| `rag_chain.py` | ~3,242 | 12+ | 上帝类反模式 |
| `graph_rag.py` | ~1,926 | 8+ | 图定义 + 9 个节点实现 + 条件路由 |
| `pdf_loader.py` | ~1,938 | 6+ | PDF 文本提取 + 清洗 + 标题解析 + 切片构建 + OCR |

### v24 优先修复路线图

| 阶段 | 编号 | 问题 | 预计 |
|------|------|------|------|
| 第一阶段 | REF-1 | `rag_chain.py` 拆分为 6 个子模块 | 8h |
| 第一阶段 | REF-2 | `graph_rag.py` 拆分为 4 个子模块 | 4h |
| 第一阶段 | FLOW-2 | `_last_numeric_context_missing` → RAGState 字段 | 1h |
| 第一阶段 | FLOW-1 | 废弃旧管线 `rag_chat` / `rag_chat_stream` | 2h |
| 第二阶段 | PERF-2.1 | BM25 磁盘持久化 (pickle) | 2h |
| 第二阶段 | BUG-2.1 | Search-First `_score` 属性修复 | 1h |
| 第二阶段 | BUG-2.2 | Retry 逻辑 off-by-one 修复 | 30min |
| 第三阶段 | L3-UP | 模板自动选择（基于 Query 意图分类） | 4h |
| 第三阶段 | L2-UP | Query-Aware Chunk Expansion | 3h |
| 第三阶段 | L4-UP | Slot-Level Factual Verifier | 4h |

</details>

---

# v30 优先修复路线图

### 第一阶段：架构安全 (v30 目标)
| 编号 | 问题 | 预计 | 影响 |
|------|------|------|------|
| **REF-3** | L0 GuardOrchestrator 实现 + 消除 `_last_numeric_context_missing` 全局变量 | 6h | 并发安全 + 拒答一致性 |
| **REF-4** | L1 ChunkGuard：废除 Parent 暴力截断 → 语义边界保护 + OCR MUST_KEEP 标记 | 4h | OCR 参数零丢失 |
| **REF-5** | L5 HistoryIsolationWall：跨产品上下文隔离墙 | 3h | 多轮跨产品污染根除 |
| REF-1 | `rag_chain.py` 拆分为 6 个子模块 (延续 v24 路线图) | 8h | 可维护性 |
| REF-2 | `graph_rag.py` 拆分为 4 个子模块 (延续 v24 路线图) | 4h | 可测试性 |
| FLOW-1 | 废弃旧管线 `rag_chat` / `rag_chat_stream` (延续 v24 路线图) | 2h | 维护负担 -50% |

### 第二阶段：检索质量
| 编号 | 问题 | 预计 |
|------|------|------|
| **REF-6** | L2 RecallGuard：DF 感知权重 + 数字 Token Boost + OCR 切片提权 | 4h |
| PERF-2.1 | BM25 磁盘持久化 (pickle) | 2h |
| BUG-2.1 | Search-First `_score` 属性修复 | 1h |
| BUG-2.2 | Retry 逻辑 off-by-one 修复 | 30min |

### 第三阶段：模板与流式
| 编号 | 问题 | 预计 |
|------|------|------|
| REF-7 | L3 TemplateSelector：废除 `> [!WARNING]` 逃生舱 → Python 层二元模板 | 3h |
| REF-8 | L4 StreamGuard：流式路径消除全局变量依赖 | 2h |

---

## 架构设计评价 (v30 更新)

### 杰出之处
1. **v24 模板约束策略**：对 1.5B/7B 小模型的理解深刻——不试图让模型"更聪明"，而是给模型"更窄的跑道"
2. **四层容灾金字塔**：设计优雅，历经 v24-v29 仍保持稳定
3. **双轨制 (c_sdk/gui_app)**：模板约束进一步强化了双轨差异
4. **v29 OCR 键值法 + Fast-Path 侧信道**：方向正确，`(messages, refusal_flag)` 元组设计消除了模板守卫本身的竞态
5. **v28 区域状态机标题提取**：保护区内跳过匹配——标题提取与区域保护首次解耦

### 需警惕的架构债
| 编号 | 问题 | 严重度 | 首次识别 |
|------|------|--------|---------|
| DEBT-1 | `_last_numeric_context_missing` 模块级全局变量并发竞态 | 🔴 致命 | v24 |
| DEBT-2 | Parent Chunk 暴力截断导致 OCR 参数丢失 | 🔴 致命 | **v30 新识别** |
| DEBT-3 | 三种拒答机制（ABSTAIN/模板守卫/数字守卫）各自独立运行 | 🔴 致命 | **v30 新识别** |
| DEBT-4 | 历史净化无法区分跨产品技术数据泄露 | 🟡 严重 | **v30 新识别** |
| DEBT-5 | 逃生舱依赖 LLM Prompt 服从（非确定性） | 🟡 严重 | v25 |
| DEBT-6 | `rag_chain.py` 3,242 行单文件上帝类 | 🟡 严重 | v24 |
| DEBT-7 | 两套管线并存 (`rag_chat` vs `run_graph`) | 🟡 严重 | v24 |
| DEBT-8 | Dense 检索无高频词稀释保护（IDF 缺失） | 🟡 严重 | **v30 新识别** |
