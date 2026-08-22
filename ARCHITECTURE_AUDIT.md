# 比邻星 (ProximaRAG) — 全量架构审查报告

> **日期**: 2026-08-22 | **覆盖**: Stage 1 → Stage 4 收口（v32 收口 + 本次修复） | **审计人**: AI 架构师
> **方法**: 四层逐层走查（4 路并行代码审查代理）+ git diff 深度审计 + 三套测试离线实测（test_stage1 18/18 · audit_ingestion · run_eval 25/35）+ ChromaDB 白盒探测 + OpenR6 文档重解析验证
> **上一版**: 2026-08-06 v24→v30 审计（归档于附录）

---

## 一、审计总览

| 层级 | 名称 | 核心资产 | 严重问题 | 已知债 | 评分 |
|------|------|---------|---------|--------|------|
| L1 | 数据摄入与切片 (Stage 1) | SDK 专轨 fitz 状态机 + JAKA 专轨 MinerU×Qwen2-VL 双轨 | 0 | 2 | **A-** |
| L2 | 检索与重排 (Stage 2) | LangChain Chroma 单例规范 + BM25Okapi 增量索引 | 0 | 4 | **B+** |
| L3 | 上下文组装与指令 (Stage 3) | Markdown 模板强约束 + KV 确定性侧信道 | 1 | 4 | **B+** |
| L4 | 生成控制与后处理 (Stage 4) | 极速流式穿透 + 围栏闭合 + render 透传 | 0 | 0 | **A-** |

**综合评分: B+/A- (85/100)**

结论：**Stage 1 的双轨摄入重构是决定性的架构胜利**——pdf_loader.py 从 ~2,900 行收敛至 ~660 行，`test_stage1.py` 18/18 全过；**Stage 2 的 LangChain 单例化根除了文件锁冲突**；**Stage 3 的 KV 侧信道为确定性数值提供了物理级保障**。但回归评测 71.4%（10/35 失败）暴露三个待闭环问题：① KV 注入粒度（整字典倾倒含噪声）② 数据目录噪声与 VLM 缓存缺失污染检索 ③ JAKA 轨微缩大纲未迁移（宏观提权失锚）。详见第五、六章。

---

## 二、Stage 1-4 重构落地盘点（架构同步）

### 2.1 L1 — SDK Fitz 物理坐标解析 + MinerU Qwen2-VL 双轨解析

```
                    ┌─────────────────────────────────────────────────┐
                    │        load_all_documents_v4_dual() 统一入口      │
                    └───────────────┬─────────────────┬───────────────┘
        ┌───────────────────────────┘                 └──────────────────────────┐
        ▼ SDK 专轨 (OpenC3/OpenR6)                                          ▼ JAKA 专轨
  fitz get_text("text", sort=True)                             MinerU Markdown (data/jaka_markdown/)
  物理坐标流排序 (废弃 pypdf)                                    _preprocess_all_images() 三重图片防线
        │                                                             (几何<80px/长宽比>8 → 图注±100字 → VLM)
  _clean_sdk_pdf_text() 断字/断行修复                                    │ :8005 Qwen2-VL 提纯 (缓存优先)
        │                                                             clean_html_tables() HTML→GitHub MD
  _strip_openr6_toc() 🔴 OpenR6 目录噪声剔除                         load_jaka_mineru_dual()
        │                                                             面包屑栈 + 1500ch 软装箱 + TOC 点线过滤
  _extract_sdk_header() + _v4_parse_sdk_state_machine()                        │
  _SDK_CHAPTER_BOUNDARY_RE 章节原子闭环切片                        export_kv_attributes() → kv_db/attribute_kv.json
        └───────────────────────────┬──────────────────────────────────────────┘
                                    ▼
                  Stage 2: create_dual_collections() → ChromaDB Parent+Child 双集合 + BM25
```

实测验收（`python tests/test_stage1.py` 18/18，2026-08-22 复测）：

| 产品轨 | 实测 | 备注 |
|--------|------|------|
| OpenC3 | 27 章原子切片（270~720 字符） | api_atomic 25 块全带函数名，CTypes 零污染 |
| OpenR6 | 30 真实章节 + 1 SDK 基础配置块（TOC 过滤修复后 31 切片） | api_atomic 28 块全带函数名，CTypes 零污染 |
| JAKA | 9 Parent（5 章+4 附录）+ 225 Child | HTML 表格 0 残留 / 444 行 Markdown 表格 |
| 统一入口 | 12P+284C（含 General 1P+1C 噪声，见 AUD-01） | 三产品线齐全 |

### 2.2 L2 — ChromaDB LangChain 单例 Parent-Child 双集合 + BM25Okapi 增量分词索引

| 组件 | 落地形态 | 位置 |
|------|---------|------|
| **LangChain Chroma 单例规范** | 全部读写统一走 `langchain_chroma.Chroma(persist_directory=CHROMA_PERSIST_DIR)`，**禁止**原生 `chromadb.PersistentClient` 与包装器混用同一 persist 目录（Settings 冲突/文件锁冲突根因）。写入口仅两个：`create_dual_collections()`（全量）/ `upsert_product_documents()`（增量） | `vector_store.py:939/1225` |
| **Parent-Child 双集合** | `rag_v4_parent` + `rag_v4_child`（hnsw cosine）；重置 = 全量 `get()` + `delete()` 后重写，替代旧的 delete_collection 重建 | `vector_store.py:959-1005` |
| **嵌入函数单例** | `get_embedding_function()` 模块级懒加载（HF bge-small-zh-v1.5 → ONNX 回退），进程内唯一实例 | `vector_store.py:268` |
| **BM25Okapi 增量分词索引** | 内存索引 `_bm25_indexes: {product_id: BM25Okapi}`；建库 `build_bm25_index()` 按产品分组全量构建；增量 `bm25_upsert_product()` corpus 扩展后整体重建重算 IDF；进程重启 `app.py` startup `build_bm25_from_chromadb()` 从 ChromaDB 全量恢复 | `vector_store.py:1176/1525/1573` |
| **分词规则** | jieba + `_IDENTIFIER_RE` 标识符保护 + `[CODE:xxx]` 三倍写入 + `_COMPOUND_RE` 复合词追加（排除 `.`）+ `_SPACE_SEP_RE` 空格归一化 + 3-5 位数字原子 token | `vector_store.py:1458` |
| **空 ID 拦截** | 文档 ID 生成 `or` 链兜底（4 处）：`metadata.get("chunk_id") or f"c_{product_id}_{i}"` — `get(key, default)` 对空串失效，`or` 识别空串触发兜底 | `vector_store.py:975/998/1288/1301` |
| **Metadata 清洗** | `_sanitize_metadata()`: list→逗号拼接 / dict→JSON / None→空串，写入前强制清洗 | `vector_store.py:879` |

### 2.3 L3 — KV 确定性属性侧信道注入

```
用户 query ──► _build_messages() ──► (messages, refusal_flag)
                                            │
                    ┌───────────────────────┤ _last_numeric_context_missing or _NUMERIC_QUERY_RE
                    │                       ▼
                    │   kv_extractor.lookup_attribute(query, product_id)
                    │   kv_db/attribute_kv.json（两级嵌套 {产品: {键: 值}} + _MANUAL_CALIBRATION 6502/9600）
                    │                       │ 命中 → 【⚠️ 系统属性库 — 高优先级已知事实】前置注入
                    │                       │ 未命中 → BM25 第二机会 → 硬拒答 (不调 LLM)
                    ▼
  4 个注入点（均位于 _build_messages 之后、LLM 调用之前）:
    • run_graph            → graph_rag.py:805  llm_generation_node 首条 user 消息前置
    • run_graph_stream     → graph_rag.py:1766 system 消息前置
    • rag_chat (legacy)    → rag_chain.py:3157 system 消息前置
    • rag_chat_stream      → rag_chain.py:3391 作为 Document 插入 context_docs[0] 后重建 messages
```

- **触发条件**: `_last_numeric_context_missing`（`_build_messages` 置位，rag_chain.py:2288 模块级）`or _NUMERIC_QUERY_RE.search(query)`（rag_chain.py:2280 模块级常量，graph_rag 共享）。
- **确定性保证**: 6502/9600/默认密码 等人工校准值不经过检索与生成，物理直入 Context，杜绝数字幻觉。
- **端到端实测**: `run_graph("JAKA Modbus 默认端口号是多少？")` → 回答含 "默认情况下，Modbus TCP 端口号为 6502" ✅（test.ipynb exec 7）。

### 2.4 L4 — 模板约束 + 透传兜底（v24 架构延续，零触碰）

`render_node` 纯文本透传 / `_stream_guardrail` 零缓冲透传 + ``` 奇偶计数自动闭合 / `extract_align_node` 属性对齐 + SemanticDedup + `_fix_and_close_sdk_code` 入口接入 / SDK 自纠错硬熔断 retry≤2 / 四层容灾金字塔（vLLM → 智谱 → 纯检索直出 → 硬拒答）。

---

## 三、六项 Bug 根因分析（本次修复）

### 3.1 BUG-1 参数缺失 → `load_single_sdk_pdf` / `upsert_product_documents` 显式参数化

- **现象**: 调用方无法显式指定 product_id / file_path，只能依赖文件名推断；文件不存在时在深层抛晦涩异常。
- **修复**: 两函数新增显式 `product_id` / `file_path` 参数 + `FileNotFoundError` 前置校验（`pdf_loader.py:532-547`，`vector_store.py:1233-1237`）。
- **验证**: `upsert_product_documents(..., product_id="OpenC3")` → `{'status': 'success', 1P, 27C}` ✅

### 3.2 BUG-2 文件锁冲突 → 废除原生 PersistentClient 混用

- **现象**: 建库/写入时 ChromaDB 报 Settings 冲突或 SQLite 锁异常——`create_dual_collections` 旧实现用 `chromadb.PersistentClient(path=..., settings=Settings(anonymized_telemetry=False))` 手工建集合，而读取侧（`load_vector_store`、`app.py`）走 LangChain 包装器（默认 Settings）。**两个客户端以不同 Settings 并发打开同一 persist 目录 → ChromaDB 拒绝服务/锁冲突**。
- **修复**: 写入侧全部改走 LangChain `Chroma` 包装器（`persist_directory`），删除手工 `Settings` 构造；「禁止原生 PersistentClient 混用」写入 CLAUDE.md 红线。
- **残留（只读例外，已备案）**: `check_status.py:310` 健康检查、`app.py:421/425` debug 检查接口、`vector_store.py:1104/1130` MD5 持久化；`vector_store.py:917/1158` 为死代码（见 AUD-12）。

### 3.3 BUG-3 空 ID 拦截 → `or` 链兜底

- **现象**: metadata 中 `chunk_id`/`parent_id` 为空字符串 `""` 时，`get(key, default)` 返回空串 → ChromaDB `add()` 收到空 ID 抛异常。
- **修复**: 4 处 ID 生成改 `or` 链（`""` 被识别为 False → 触发 `f"c_{product_id}_{i}"` 兜底）。注意 `get(key, default)` 的语义陷阱：default 仅在 **key 缺失**时生效，空值不生效。

### 3.4 BUG-4 TOC 噪声过滤失效 → 分隔线正则 `[☆★\*]{5,}` 修复

- **现象**: OpenR6 32 切片含 2 个目录页 TOC 噪声块（`29.机械臂通讯关闭` 重复块 + 前置残留），`_strip_openr6_toc` 形同虚设——**实测剥离 0 字符**。
- **根因**: 真实文档目录分隔线是 `☆☆☆☆…`（U+2606），正则 `[\*]{5,}` 只匹配 ASCII `*`。
- **修复**: `[\*]{5,}` → `[☆★\*]{5,}`（`pdf_loader.py:453-460`）。
- **实测验证**:
  | 指标 | 修复前 | 修复后 |
  |------|--------|--------|
  | TOC 剥离字符数 | 0 | **465** |
  | 状态机切分块数 | 119 | **61** |
  | OpenR6 最终切片 | 32 | **31**（30 真实章节 + SDK 基础配置块，内含"运动指令消息返回值类型判断"全局定义，符合保留设计） |
  | `29.机械臂通讯关闭` 重复噪声块 | 1 | **0** |
- **入库状态**: 当前生产 DB 仍为修复前构建（OpenR6 32），需全量重建生效。

### 3.5 BUG-5 /api/upload 双轨路由断裂 → 按扩展名路由

- **现象**: `upsert_product_documents` 旧实现 import 已删除的 `_v4_extract_text_universal`/`_v4_build_parent_child_docs`（Stage 1 遗留#1）→ 上传必 500。
- **修复**: 按扩展名路由 `.md` → `load_jaka_mineru_dual` / `.pdf` → `load_single_sdk_pdf(file_path, product_id=...)` + 写入前 `_sanitize_metadata` 清洗。

### 3.6 BUG-6 KV 注入路径悬空 → 重建 kv_extractor.py

- **现象**: kv_extractor.py 曾并入统一模块后删除，4 处懒加载 `try/except` 吞掉 ImportError（Stage 1 遗留#2）→ E05/E07 类确定性数值注入静默失效。
- **修复**: 新建 `src/kv_extractor.py`（轻量 KV 事实检索：文件缓存单例 + 两级嵌套匹配），恢复 4 注入点接线。

---

## 四、端到端验证结论（2026-08-22 实测）

| 验证项 | 命令 | 结果 |
|--------|------|------|
| Stage 1 离线验收 | `python tests/test_stage1.py` | ✅ **18/18 通过**（SDK 专轨 9 / JAKA 专轨 6 / KV 校准 1 / 统一入口 2） |
| 向量库白盒质检 | `python tests/audit_ingestion.py` | ⚠️ 未通过：规则 3b（6502 不在切片——实走 KV 侧信道）/ 4a（Parent 大纲注入 0/12）/ 4b（`[OCR补漏:]` 标记已被 VLM 替代）口径与 Stage 1 脱节 |
| 回归评测 | `python tests/run_eval.py --verbose` | ⚠️ **71.4% (25/35)**，10 用例失败 |
| KV 注入单测 | test.ipynb `lookup_attribute` | ✅ 端口 / 波特率双命中（6502 / 9600） |
| 增量 upsert | test.ipynb `upsert_product_documents(OpenC3)` | ✅ `{'status': 'success', parent_chunks: 1, child_chunks: 27}` |
| 全量重建 | test.ipynb `load_all_documents_v4_dual` + `create_dual_collections` | ✅ 11P+284C（当时）；TOC 修复后重解析预期 12P+284C（OpenR6 31） |
| 三产品线端到端 | test.ipynb `run_graph` × 3 | ✅ JAKA 功能IO（Markdown 表格+出处）/ OpenR6 `set_move_line`（POSE+CDLL+argtypes 完整代码）/ KV 端口注入（6502） |

**run_eval 质量指标**: Answer Keyword Recall 74.2% (49/66) · Product Isolation 97.1% · Format Cleanliness 100% · SDK 函数 4/4 · 安全注入防御 2/2 · 错别字容错 1/1。

**10 例失败归因**:

| 分组 | 用例 | 根因 |
|------|------|------|
| 数字参数 (3) | GT-1 / GT-6 / E05 | **AUD-08 KV 注入粒度缺陷**: `lookup_attribute` 产品级键（"JAKA"）命中时整体倾倒含噪声的整字典（"端口号: 填写端口号" 与 "Modbus TCP 端口号: 6502" 同时注入），模型受噪声干扰，6502 输出不稳定 |
| 拒答类 (3) | E09 / E21 / E25 | 模板守卫/逃生舱未按预期触发，需逐案复核；当前库 OpenR6 TOC 噪声 + ROS General 噪声块污染检索可能相关 |
| 召回/合成类 (4) | GT-4 / E13 / E22 / E26 | E26 根因明确：**AUD-10** Stage 1 JAKA 轨未迁移 `[章节大纲参考]` 微缩大纲注入（实测 0/12 Parent），宏观提权失去锚点；E13 依赖 VLM 多模态注入，**AUD-09** 缓存缺失 + :8005 离线（189 图全部快速失败被过滤） |

> ⚠️ 诚实声明: 回归评测与白盒质检未通过的项目**不因文档更新而消解**——上表归因与第六章路线图是下一步修复依据。

---

## 五、新识别问题与架构债（13 项）

| 编号 | 问题 | 层级 | 严重度 | 证据 |
|------|------|------|--------|------|
| AUD-01 | `data/ROS机器实践应用.pdf`（ROS 书籍）被统一入口误摄入为 `General` 产品线 c_sdk 切片（1P+1C） | L1/数据 | 🟡 中 | ChromaDB 白盒探测: product_id=General，section_title="SDK 全文" |
| AUD-02 | 增量 upsert 无旧切片级联删除——下标式 ID（`c_{pid}_{i}`）收缩时尾部旧 ID 残留（旧版 `delete_product_chunks`+`bm25_remove_product` 已被移除且无替代） | L2 | 🔴 高 | `vector_store.py:1225-1315` 无删除逻辑；32→31 切片收缩场景必现 |
| AUD-03 | `frontend_server.py` / `streamlit_app.py` 反向代理目标 `localhost:7860` 与 FastAPI 实际端口 8000 失配 → 前端 API 链路断裂 | 服务层 | 🔴 高 | `frontend_server.py:12`、`streamlit_app.py:10` |
| AUD-04 | `start_services.sh` 无 `--vllm-only`/`--fastapi-only`/`--gpu` CLI 参数（GPU 覆盖仅 `VLLM_GPU_ID` 环境变量），早期 CLAUDE.md/README 记载失实（已修正） | 服务层 | 🟢 低 | `start_services.sh:295-299` 无 argparse |
| AUD-05 | `_last_numeric_context_missing` 模块级全局（rag_chain.py:2288）被 graph_rag 跨模块读取——v29 已识别的并发竞态未销（FastAPI 线程池下多请求互相覆盖） | L3 | 🔴 高 | `graph_rag.py:801/1761` 读取 `_rag_chain_mod._last_numeric_context_missing` |
| AUD-06 | `attribute_tool.py` 死代码（0 导入者）；planner 的 `attribute_lookup` 子目标无执行器（`sub_results` 无写入方）——其 docstring 声称的接线点与代码不符 | L3 | 🟡 中 | grep 全项目 0 导入；`graph_rag.py:1299/1444` 仅字符串引用 |
| AUD-07 | BM25 corpus 类型不一致: build 路径存 Document、upsert 路径存 str → `bm25_search` 返回类型随路径而异 | L2 | 🟡 中 | `vector_store.py:1564/1604` vs `1197-1209` |
| AUD-08 | KV 注入粒度缺陷: 产品级键命中时 `str(整字典)` 整体注入（含 "端口号: 填写端口号" 等噪声键），干扰模型 → GT-1/GT-6/E05 三例失败 | L3 | 🔴 高 | `kv_extractor.py:72-78`；eval 实测 |
| AUD-09 | `data/jaka_manual_chunks.json` 提纯缓存缺失 + :8005 VLM 离线时 189 图全部快速失败被过滤（优雅降级但多模态参数丢失）→ 重建前必须先恢复 VLM 或生成缓存 | L1/运维 | 🟡 中 | 全盘 find 无缓存文件；重建实测 189/189 @750it/s 秒级完成=全部失败路径 |
| AUD-10 | Stage 1 JAKA 轨未迁移 `[章节大纲参考]` 微缩大纲注入（旧 L1 特性）→ 宏观提权失锚（E26 失败；audit 实测 0/12 Parent） | L1/L2 | 🟡 中 | `tests/audit_ingestion.py` 规则 4a 实测 0.0% |
| AUD-11 | `audit_ingestion.py` 规则 3b/4a/4b 断言口径与 Stage 1 产物脱节（6502 走 KV 侧信道、`[OCR补漏:]` 标记被 VLM 替代）→ 综合判定恒不通过 | 测试 | 🟡 中 | 实测 ❌ 审计未通过 |
| AUD-12 | vector_store 死代码残留: `_add_to_existing_collection`（内置 `hash()` 生成 ID，进程重启盐值漂移不稳定）/ `_embed_batched` / `search_dual_index` / `delete_product_chunks` / `bm25_remove_product` / `resolve_product_id` 均无调用方 | L2 | 🟢 低 | grep 调用方为 0 |
| AUD-13 | 文档与实现不一致: `_build_messages` docstring 声明返回 List 实为 `(messages, refusal_flag)`；`rag_chat_stream` KV 触发仅判 flag 缺 `_NUMERIC_QUERY_RE` 正则兜底（与其它 3 处不一致）；`check_status.py` docstring 端口 7860 | L3/杂项 | 🟢 低 | `rag_chain.py:1681/3389`、`check_status.py:9` |

---

## 六、修复路线图（按优先级）

### P0 — 本周必须（消除 🔴 高）

| 编号 | 任务 | 预计 | 依据 |
|------|------|------|------|
| FIX-1 | `lookup_attribute` 键级匹配改造：value 为 dict 时在内部做键名/拆分匹配，仅返回与 query 相关的具体键值对（禁止 `str(整字典)` 倾倒） | 2h | AUD-08，根治 GT-1/GT-6/E05 |
| FIX-2 | 恢复 upsert 级联删除：写入前按 product_id 删除双集合旧切片 + `bm25_remove_product`（或改为稳定 ID 方案：以 chunk_id 为主键自然覆盖） | 2h | AUD-02 |
| FIX-3 | `_last_numeric_context_missing` 迁移至 RAGState 字段 / GuardResult 不可变值（v30 方案 L0 原计划，从未执行） | 4h | AUD-05，并发竞态 |
| FIX-4 | 修复 `frontend_server.py`/`streamlit_app.py` 代理目标 7860 → 8000 | 30min | AUD-03 |

### P1 — 数据与验证闭环

| 编号 | 任务 | 预计 |
|------|------|------|
| FIX-5 | 移出 `data/ROS机器实践应用.pdf`（或配置产品映射白名单），随后全量重建（TOC 修复入库 + 重建前恢复 :8005 VLM 或生成 `jaka_manual_chunks.json` 缓存），重建后重跑三套测试 | 半天 |
| FIX-6 | JAKA 轨迁移 `[章节大纲参考]` 微缩大纲注入（对齐 E26 与 audit 规则 4a 口径） | 4h |
| FIX-7 | `audit_ingestion.py` 规则 3b/4a/4b 口径适配 Stage 1：6502 断言改查 KV 属性库、OCR 标记断言改查 `has_multimodal_data` metadata | 2h |
| FIX-8 | BM25 corpus 类型统一（upsert 路径改存 Document），`bm25_search` 返回签名固定 | 1h |

### P2 — 债务清理

| 编号 | 任务 | 预计 |
|------|------|------|
| FIX-9 | 删除 vector_store 死代码（含内置 `hash()` ID 的 `_add_to_existing_collection`）；`attribute_tool.py` 归档删除或标注废弃 | 2h |
| FIX-10 | `rag_chat_stream` KV 触发补 `_NUMERIC_QUERY_RE` 兜底（对齐其余 3 处）；`_build_messages` docstring 修正 | 30min |
| FIX-11 | 拒答类 3 例（E09/E21/E25）逐案复核——待 FIX-5 重建后重测，若仍失败则回溯 L3 守卫判定 | 2h |

---

## 附录: 历史审计归档

<details>
<summary>v24 架构审计（2026-08-04）与 v30 架构方案（2026-08-06）—— 点击展开</summary>

### v24: Markdown 模板强约束 (Template Masking) 核心论述

放弃"自由生成 → JSON 提取 → 正则清洗"的旧管线（JSON 错误率 15-20%、正则误杀/漏杀跷跷板、伪流式 TTFB 60-90s），转向"模板填空 + 流式透传"：
1. **注意力锚定**: 模板置于 User Message 末尾（Recency Bias 热点区）
2. **自由度压缩**: "怎么说"的决策空间压缩到接近零
3. **错误模式可预测性**: 自由格式错误发散，模板错误收敛于槽位
4. **流式穿透**: 格式正确性前置保证 → 零缓冲透传 → TTFB <2s

v24 评分: 综合 A→A+ (94/100)；遗留: BUG-2.1 Search-First `_score` 空属性、BUG-2.2 Retry off-by-one、FLOW-2 全局变量竞态（即 AUD-05 前身，**至今未销**）。

### v30: 五层状态机编排方案（L0 GuardOrchestrator）

v30 审计识别五大故障现象（三种拒答机制三角混战 / Parent 暴力截断杀 OCR 参数 / 宏观大纲失联 / 多轮历史表面清洗 / OCR 参数丢失终极根因），提出 L0 统一门控编排方案。**执行状态**: L0/L1/L5 方案未实施，但 Stage 1 重构以"数据语义化"路径部分解决了原问题（JAKA 参数提纯由 VLM 替代 OCR 键值法；Parent 软装箱由 JAKA 轨 1500ch 段落封箱承接）。`_last_numeric_context_missing` 竞态（DEBT-1）与 Parent 大纲注入（DEBT-2 变体，AUD-10）延续至本报告。

### 历史评分对照

| 版本 | 综合评分 | 关键变化 |
|------|---------|---------|
| v24 | A+ (94) | 模板约束 + 流式穿透 |
| v30 审计 | B/B+ (82) | 并发竞态 + OCR 丢失 + 拒答三角混战 |
| **本报告 (Stage 1-4)** | **B+/A- (85)** | 双轨摄入 + 单例化 + KV 侧信道落地；KV 粒度/数据噪声/大纲缺失为新的三个主矛盾 |

</details>
