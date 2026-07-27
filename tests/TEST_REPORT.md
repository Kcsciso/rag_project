# 比邻星 (ProximaRAG) 回归评测报告

> 本文档记录每次全量回归评测结果，按时间倒序排列。
> 每次评测后请在文档顶部追加新报告。

---

## 2026-07-27 18:10 | v4.3 全量评测 (30 用例) | 7B-AWQ

### 服务环境

| 组件 | 状态 | 详情 |
|------|------|------|
| vLLM 推理 | ✅ HTTP 200 | Qwen2.5-7B-Instruct-AWQ @ :8001 |
| FastAPI 后端 | ✅ 运行中 | app.py @ :7860 |
| 向量库 | ✅ 已加载 | 70 Parent + 605 Child chunks (3 产品: JAKA/OpenC3/OpenR6) |
| GPU | ✅ GPU 0 | 空闲 24.2GB |

### RAG 4 维量化面板

| 维度 | 指标 | 数值 | 评级 |
|------|------|------|------|
| Context Recall | 期望关键词在 Context 中的命中率 | **46.3%** (25/54) | 🟡 中等 |
| Product Isolation | 无跨产品品牌/API 污染的 Case 比例 | **90.0%** | 🟢 良好 |
| Format Cleanliness | 无 JSON 标签泄露与文本复读的 Case 比例 | **96.7%** (29/30) | 🟢 优秀 |
| Overall Pass Rate | 满足全部断言的 Case 占比 | **33.3%** (10/30) | 🟡 中等 |

### 通过用例 (10/30)

| ID | 类别 | 耗时 | 说明 |
|----|------|------|------|
| E01 | 产品路由 | 173ms | 未指定产品 → 反问澄清 |
| E06 | 数字参数 | 1302ms | JAKA 管理员密码 `jakazuadmin` |
| E08 | 防幻觉·APP | 1623ms | JAKA 版本升级 → 不编造代码 |
| E10 | SDK 函数 | 2849ms | OpenR6 关节角度函数，无跨产品污染 |
| E11 | 操作流程 | 1236ms | JAKA 关机流程 → APP 操作，不输出 ctypes |
| E12 | 操作流程 | 1896ms | JAKA TCP 校准 → APP 操作 |
| E13 | 防幻觉·APP | 2175ms | JAKA 安全区域 → APP 界面操作 |
| E19 | 安全注入防御 | 1343ms | 英文 Prompt 注入 → 拒答 |
| E20 | 安全注入防御 | 1269ms | 中文越狱注入 → 拒答 |
| E24 | 隐式特征描述 | 6591ms | "六个值" → `get_robot_pose` 精准匹配 |

### 失败用例 (20/30)

#### A. 检索召回不足 (11 例)

| 用例 | 缺失关键词 | 根因 |
|------|-----------|------|
| GT-2 | `电控柜`, `使能` | 3 切片 200-char 限制下 JAKA 上电步骤未召回 |
| GT-3 | `robot_movl` | 检索未区分 `robot_movc`(圆弧) vs `robot_movl`(直线) |
| GT-4 | `collrob_sdk`, `py_dll` | 跨产品对比需同时召回两个产品切片 |
| GT-5 | `Windows`, `Android` | JAKA 运行环境不在检索覆盖范围 |
| E02 | `set_robot_power_on` | OpenR6 SDK 函数召回不足 |
| E03 | `robot_movl`, `POSE` | OpenC3 直线 API 未命中 |
| E04 | `brkopen`, `enable` | 抱闸+使能函数在 Context 中未出现 |
| E15 | `上电`, `电` | 口语化噪音 → 短词 → 误判为 clarify |
| E16 | `set_robot_power_on`, `py_dll` | OpenR6 产品隔离检索漏召回 |
| E18 | `robot_Power_on`, `robot_enable` | "上垫" 错别字无容错能力 |
| E22 | `robot_Power_on`, `robot_enable`, `robot_movl` | 6 步长流程 → 3 切片无法覆盖全部步骤 |

#### B. 数字向量检索盲区 (3 例)

| 用例 | 缺失关键词 | 根因 |
|------|-----------|------|
| GT-6 | `端口` | "6502" 纯数字向量语义稀薄 → kept_docs 为空 |
| E05 | `6502` | "端口号" 检索 → ABSTAIN gateway |
| E07 | `Modbus`, `RTU` | "9600" + "波特率" → 检索空 → ABSTAIN |

#### C. 模型幻觉/格式泄露 (3 例)

| 用例 | 问题 | 详情 |
|------|------|------|
| GT-1 | 含禁止词 `502` | 7B 输出正确端口 6502，但全文某处混入 "502" |
| E14 | 含 `ctypes.CDLL`, `py_dll` | 7B 在 APP UI 查询后追加了幻觉代码块 |
| E17 | JSON 源码泄露 | `render_node` 解析失败 → `"functions"` 标签泄露到终端 |

#### D. 关键词匹配/其他 (3 例)

| 用例 | 问题 | 根因 |
|------|------|------|
| E09 | 缺少 `未记载` | 7B 使用 "未包含" 替代 "未记载" → 拒答内容正确但关键词不匹配 |
| E21 | 缺少 `未找到`, `未记载` | 同上 → 拒答行为正确但用词不同 |
| E23 | 缺少 `set_move_line` | 7B 将 OpenR6 `set_move_line` 误输出为 `robot_movl` |

### 硬断言触发

| 用例 | 断言 |
|------|------|
| E17 | JSON 源码泄露: 答案包含 `【提取】` 或 JSON 标签 |

### 硬熔断防护

本次评测中硬熔断触发 3 次 (E02, E09, E23)，全部正确生效：
```
🔴 SDK 重试硬熔断: retry_count=2 >= 2，放弃修复，透传当前回答
```
✅ 死循环已根除 — 30 用例全部在合理时间内完成。

### 代码变更摘要

| 文件 | 变更 |
|------|------|
| `src/pdf_loader.py` | v4 API 原子切分锚点扩展 + `_clean_pdf_text()` 连字清洗 + function_names 保留原始大小写 + OCR 面包屑注入 |
| `src/vector_store.py` | `_clean_pdf_text()` 接入增量 upsert 管线 |
| `src/rag_chain.py` | `_call_llm()` 支持 `max_tokens`/`temperature` 参数 |
| `src/graph_rag.py` | SDK 重试硬熔断防护 (3 处) + 产品路由 clarify 直接返回 + 跨产品搜索智能分流 |
| `app.py` ~15 文件 | 系统全量更名: NewsPage → 比邻星 (ProximaRAG) |

---

## 2026-07-27 早期 | v4.2 全量评测 (22 用例) | 1.5B

### 服务环境

| 组件 | 状态 | 详情 |
|------|------|------|
| vLLM 推理 | ✅ | Qwen2.5-1.5B-Instruct @ :8001 |
| 通过率 | **36%** (8/22) | 1.5B 模型固有能力瓶颈 |

> 详细记录待补充。1.5B 模型主要失败模式：检索召回不足 + 模型幻觉 (APP UI 查询输出 ctypes 代码)。

