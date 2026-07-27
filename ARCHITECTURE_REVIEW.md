# NewsPage RAG 系统 — 企业级架构审查报告

## 总体评估：B+ (82/100)

系统在架构设计（Plan-Execute-Synthesize）、检索策略（混合检索+双索引）、
容灾韧性（四层降级）方面已达到较高水准。但存在 **5 个 P0 级安全隐患** 和
**6 个 P1 级可靠性问题**，阻碍其直接部署至生产环境。

---

## 🔴 P0 — 安全红线（必须修复）

### 1. API Key 硬编码泄露 (config.py:30-33)
```python
DEEPSEEK_API_KEY = os.environ.get(
    "ZHIPU_API_KEY",
    "1fe4c37fd3264ffa9f535fec9d0fc96b.UtiuwWTVuFofYHnB"  # ← 已泄露到代码库
)
```
**风险**: 任何有代码仓库访问权限的人都能获取智谱 API Key。
**修复**: 删除默认值，改为启动时从环境变量强制读取，缺失时抛 `RuntimeError`。

### 2. 零认证/零鉴权 (app.py)
所有端点均无身份验证。**无会话管理、无 API Key 验证、无用户隔离、无速率限制**。
**修复**: API Key 中间件 + 速率限制（`slowapi` 或 nginx `limit_req`）。

### 3. 文件上传仅检查扩展名 (app.py:441)
```python
if not file.filename or not file.filename.lower().endswith(".pdf"):
```
**风险**: 不验证 MIME 类型或 magic bytes (`%PDF-`)。重命名的可执行文件可绕过。
**修复**: `file.read(5)` 检查前 5 字节是否为 `b'%PDF-'`。

### 4. Prompt 注入仅日志不拦截 (rag_chain.py:1862-1863)
`_contains_injection_pattern()` 检测到注入后**仅记录 WARNING，不拒绝请求**。
**修复**: 返回 `route_status="refuse"` 或抛出 `HTTPException(400)`。

### 5. 全局变量并发不安全
`_embedding_function`、`_product_md5_store`、`_bm25_indexes` 等全局状态在多请求并发下无锁保护。
**修复**: `threading.RLock` 或 `asyncio.Lock` 保护所有全局可变状态。

### 6. SSE 线程池泄漏 (app.py:343-367)
`loop.run_in_executor(None, _run_blocking_stream)` 的 Future 未追踪。
**修复**: 存储 Future 并在 `CancelledError` 中 `.cancel()`；设置 `max_workers`。

### 7. 错误信息泄露 (app.py:328, 414)
```python
raise HTTPException(status_code=500, detail=str(e))  # 泄露内部异常详情
```
**修复**: 返回通用错误消息；敏感上下文仅写入日志。

---

## 🟡 P1 — 可靠性（强烈建议）

### 8. 31+ 处裸 `except Exception` (src/ + app.py)
大量 `except Exception: pass` 静默吞噬所有异常。
**修复**: 至少 `logger.error(f"Unexpected: {e}", exc_info=True)`，已知异常用具体类型。

### 9. ChromaDB PersistentClient 重复创建 (vector_store.py)
`delete_product_chunks()`、`_add_to_existing_collection()`、`_init_md5_store_from_chroma()`、`_persist_md5_store()` 各自创建独立 `PersistentClient` → SQLite 连接泄漏。
**修复**: 模块级单例 `_chroma_client` + `with` 语句管理生命周期。

### 10. PyMuPDF fitz.Document 未在 finally 中关闭
`_ocr_image_from_page()` (multimodal_loader.py:151) 和 `_v4_inject_ocr_text()` (pdf_loader.py:565) 打开 `fitz.open()` 但异常路径可能跳过 `doc.close()`。
**修复**: `try: ... finally: doc.close()`。

### 11. vLLM 模型名缓存永不过期 (rag_chain.py:186)
`_resolved_vllm_model` 缓存一次后永不更新。若 vLLM 重启并加载不同模型，`_resolve_vllm_model()` 仍返回旧名。
**修复**: 健康检查失败时重置 `_resolved_vllm_model = None`。

### 12. BM25 增量更新非真正增量 (vector_store.py:1158-1161)
`bm25_upsert_product()` 通过 `老token + 新token → BM25Okapi(all_tokens)` 重建整个产品索引，O(N) 而非 O(n)。
**修复**: 仅对新增文档计算 token 并追加，保持已计算的 IDF。

### 13. 同步 `run_graph()` 阻塞 FastAPI 事件循环 (app.py:400)
非流式路径直接同步调用 `run_graph()`，阻塞整个 asyncio 事件循环。
**修复**: `await asyncio.to_thread(run_graph, ...)`。

---

## 🟢 P2 — 架构增强（远期）

| 项目 | 当前 | 建议 |
|------|------|------|
| **可观测性** | 无结构化日志/指标/追踪 | JSON 日志 + Prometheus `/metrics` + 请求 correlation ID |
| **响应缓存** | 无 | 相同查询的语义缓存 (Redis) |
| **多用户会话** | `chat_history` 前端透传 | Redis 会话存储 + `session_id` |
| **Docker 化** | 无 | Dockerfile + docker-compose |
| **配置热重载** | 需重启 | 环境变量或文件监听 |
| **评估数据集** | test_rag_eval.py 8 用例 | 100+ Ground Truth 标准化评估集 |
| **A/B 测试** | 无 | `model`/`chunk_mode` 参数切换 |
| **PDF 编码检测** | 无 | `chardet` 检测非 UTF-8 编码 |
| **向量版本管理** | 全局覆盖 | 版本化 Collection + 灰度切换 |

---

## 📊 代码质量总览

| 指标 | 数值 | 评估 |
|------|------|------|
| 总代码行数 (src/ + app.py) | ~5500 | 中等规模 |
| 重复代码块 | 5 处跨文件重复 | 🟡 应提取到公共模块 |
| 循环导入 | 0 (已通过模块引用缓解) | ✅ |
| 函数平均长度 | ~40 行 | ✅ |
| 最长函数 | `rag_chat()` ~250 行 | 🔴 需拆分 |
| 类型注解覆盖率 | ~60% | 🟡 目标 90%+ |
| 单元测试 | **0** (无 unittest/pytest) | 🔴 空白 |
| 集成测试 | 3 套件, 26 用例 | 🟢 |
| 魔术数字 | 8+ 处硬编码参数 | 🟡 应移入 config.py |

---

## 🔧 优先修复路线图

| 阶段 | 内容 | 预估 |
|------|------|------|
| **Week 1** | P0-1 删除 API Key + P0-2 API 鉴权 + P0-3 文件魔数 + P0-4 注入拦截 | 3d |
| **Week 2** | P0-5 并发锁 + P0-6 SSE 泄漏 + P0-7 错误信息 + P1-9 ChromaDB 连接池 + P1-10 fitz finally | 3d |
| **Week 3** | P1-8 异常处理 + P1-11 vLLM 缓存 + P1-12 BM25 增量 + P1-13 异步化 | 3d |
| **Week 4+** | P2: 结构化日志 + 指标 + 单元测试框架 + Docker | 按需 |
