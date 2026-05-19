# PA_Agent 代码与提示词审查报告

**项目**: D:\cl\PA_Agent — AI K线分析辅助工具  
**审查时间**: 2026-05-20  
**审查范围**: 全部核心源码、提示词文件、测试结构、架构设计  

---

## 一、严重问题 (Critical)

### 1. `deepseek_client.py` 每次调用新建 OpenAI client — 连接无法复用

**文件**: `pa_agent/ai/deepseek_client.py`  
**位置**: `chat()` 第79行、`stream_chat()` 第159行

```python
client = OpenAI(
    base_url=self._settings.base_url,
    api_key=self._settings.api_key,
)
```

每次 `chat()` / `stream_chat()` 都 `from openai import OpenAI` 并 `new OpenAI()`。`OpenAI` client 底层使用 `httpx.Client`，新建意味着：
- 每次建立新 TCP 连接，无法复用 HTTP/2 多路复用和 keep-alive
- TLS 握手开销（~50-100ms/次）
- 连接池无法共享，高频场景下浪费文件描述符

**修复建议**: 在 `__init__` 中创建 `self._client = OpenAI(...)` 并复用，添加 `close()` / `__enter__`/`__exit__` 生命周期管理。

---

### 2. `two_stage.py` 大量 `print()` 调试输出残留

**文件**: `pa_agent/orchestrator/two_stage.py`  
**位置**: Step 5/7/15/17 附近

```python
print("\n" + "="*80)
print("【Stage 1 发送的完整 Prompt】")
# ... 打印完整 system/user prompt 和 AI 响应
```

4处 `print()` 会将完整的 system prompt（含所有策略文件内容，可能数万字符）和 AI 完整响应输出到 stdout。问题：
- 生产环境下 console 污染严重
- 完整 prompt 含策略知识，输出到 terminal 可能泄露
- 与 `on_stage_prompt` callback 功能完全重复（callback 已经通过信号传递给 GUI debug tab）
- print() 不受 logging level 控制，无法关闭

**修复建议**: 全部替换为 `logger.debug()`，或直接删除（因为 `on_stage_prompt` callback 已提供同等功能）。

---

### 3. `paths.py` PROJECT_ROOT 硬编码绝对路径

**文件**: `pa_agent/config/paths.py`

```python
PROJECT_ROOT: Path = Path(r"D:\cl\PA_Agent")
```

项目只能从 `D:\cl\PA_Agent` 运行，任何其他路径部署都会失败。所有目录（prompt_engineering/、records/、config/、logs/）全部基于此硬编码路径。

**修复建议**: 改为相对路径推断：
```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # pa_agent/config/paths.py → PA_Agent/
```

---

### 4. `stream_chat()` 的 `first_chunk` 消费后 stream 迭代器状态不确定

**文件**: `pa_agent/ai/deepseek_client.py` 第176-190行

```python
stream = client.chat.completions.create(**stream_kwargs)
first_chunk = next(iter(stream), None)  # 消费第一个 chunk

# 如果异常，retry:
stream = client.chat.completions.create(**stream_kwargs)
first_chunk = next(iter(stream), None)  # 重新消费

chunks = ([first_chunk] if first_chunk is not None else [])
def _iter_all():
    yield from chunks
    yield from stream  # 继续迭代
```

问题：
- `next(iter(stream), None)` 在某些 httpx/stream 实现中可能消费了连接上的数据但创建了一个新迭代器，导致后续 `yield from stream` 丢失数据或抛出 `StreamConsumed` 错误
- 更严重的是：如果 `stream_options` 不被支持，first chunk 被**消费**后进入 except 分支，但此时**原始 stream 已经被部分消费**，再次 `create()` 是一次全新的请求（可能扣两次 token）

**修复建议**: 不要预消费 chunk 来检测 `stream_options`。改为：
1. 先尝试带 `stream_options` 创建，正常迭代
2. 如果第一个 chunk 抛出异常（`stream_options` 不支持），捕获后用无 `stream_options` 重试
3. 或者直接用 `try/except` 在 `create()` 级别判断（某些 API 会在 create 时就返回 400）

---

### 5. `_QT_AVAILABLE` 降级模式下 `pyqtSignal = None` 存在运行时风险

**文件**: `session_ledger.py`、`refresh_loop.py`

```python
if _QT_AVAILABLE:
    threshold_crossed = pyqtSignal(str, dict)
# 否则 threshold_crossed 不存在
```

当 `_QT_AVAILABLE = False` 时：
- `SessionTokenLedger` 和 `RefreshLoop` 的 `pyqtSignal` 属性根本不存在
- 任何代码 `obj.threshold_crossed.connect(...)` 或 `obj.threshold_crossed.emit(...)` 会抛 `AttributeError`
- `RefreshLoop(QThread)` 变成 `RefreshLoop(object)`，没有 `start()` / `quit()` / `wait()` 方法
- 虽然 PA_Agent 强依赖 PyQt6，但这种"半降级"比直接 ImportError 更危险——静默失败后可能在不相关的地方崩溃

**修复建议**: 要么去掉降级逻辑直接强制依赖 PyQt6（项目本来就是 PyQt6 GUI 应用），要么完整实现无信号版本的 stub 方法（`emit`/`connect` 为 no-op）。

---

## 二、重要问题 (Major)

### 6. `kline_buffer.append()` 的 forming→closed 提升逻辑存在竞态

**文件**: `pa_agent/data/kline_buffer.py`

当 `update_forming()` 更新了 forming bar 的 `ts_open`（新周期开始时 MT5 可能更新时间戳），旧 forming bar 的数据被覆盖而非追加到 closed deque。随后 `append(bars[1])` 追加的是新周期的 closed bar，**旧周期的最后一根 forming bar 就此丢失**。

**影响**: 在 1Hz 刷新 + MT5 tick 更新场景下，每根K线关闭时可能丢失一次快照数据。

**修复建议**: `update_forming()` 应检测 `ts_open` 变化，将旧 forming 先 append 到 closed 再更新为新 forming。

---

### 7. `app_context.py` 中 `configure_logging()` 被调用两次

**文件**: `pa_agent/main.py` + `pa_agent/app_context.py` bootstrap()

`main.py` 调用 `configure_logging()` 后，`bootstrap()` 内部又调用了一次。虽然重复调用不会崩溃，但会导致：
- 日志 handler 被重复添加（同一条日志输出两遍）
- `MaskingFormatter` 的 `update_api_key` 状态可能不一致

**修复建议**: 去掉 `main.py` 中的 `configure_logging()` 调用，仅在 `bootstrap()` 中统一初始化。

---

### 8. `settings.py` 中 api_key 在内存中明文存在

**文件**: `pa_agent/config/settings.py`

`load_settings()` 解密 `api_key_encrypted → api_key`，明文存储在 `Settings.provider.api_key` 中。`save_settings()` 加密后写入磁盘，但运行时内存中一直是明文。

虽然 `MaskingFormatter` 和 `_sanitize()` 防止了日志和文件泄漏，但：
- `AppContext` 传递 `settings.provider.api_key` 给 `PendingWriter`、`MainWindow` 等
- `MainWindow.__init__` 中直接 `_api_key = settings.provider.api_key`
- 任何 crash dump / memory dump 会暴露明文 key

**修复建议**: 使用 `SecretStr` 或只在需要时解密（如 `_get_api_key()` 方法），不在 dataclass 字段中长期持有明文。

---

### 9. `free_chat._derive_record_id` 与 `pending_writer._build_basename` 逻辑重复

**文件**: `orchestrator/free_chat.py` vs `records/pending_writer.py`

两个模块各自实现了根据时间戳生成记录 ID/basename 的逻辑，但实现细节不同（一个用 `isoformat`，一个用 `strftime`），可能导致同一会话的分析和后续对话记录无法关联。

**修复建议**: 统一为 `records/` 模块的一个工具函数，两处共享。

---

### 10. `yfinance_source.py` 4h resample 未考虑交易时段边界

**文件**: `pa_agent/data/yfinance_source.py`

`_resample_4h` 使用 `pandas.resample('4h')`，但外汇/黄金市场有日内休息时段（如 17:00-18:00 ET），resample 会跨休息时段聚合，产生包含 0 volume 的"虚假K线"。

**修复建议**: resample 后过滤 volume=0 的K线，或使用交易所 session-aware 的聚合方式。

---

### 11. `SessionTokenLedger` warn_pct 与 design.md 的 200K token 预算不匹配

**文件**: `ai/session_ledger.py` + `config/settings.py`

`design.md` 要求"单会话累计消耗超过200K token时提示用户"，但代码中：
- `context_window = 1_000_000`（1M）
- `warn_pct = 80.0` → 告警阈值 = 800K
- 200K 对应 20%，但 `warn_pct` 默认 80%

两者不一致。实际行为是 token 用到 800K 才告警，远超 200K 的设计要求。

**修复建议**: 将 `warn_pct` 默认改为 20.0，或增加独立的 `token_budget` 参数。

---

### 12. `router.py` 中性方向通道状态不加载策略文件

**文件**: `ai/router.py`

```python
if cp in _CHANNEL_STATES:
    if direction == "bullish":
        files.extend(_BULLISH_CHANNEL_FILES)
    elif direction == "bearish":
        files.extend(_BEARISH_CHANNEL_FILES)
    else:
        logger.warning("Channel state %r with neutral direction — no directional strategy files loaded", cp)
```

通道 + neutral 方向时只加载了宽窄策略文件，缺少方向性策略。但市场经常出现"方向不明但通道清晰"的状态，此时 AI 进入 Stage2 时没有任何方向性参考，大概率会"不下单"——这可能符合设计意图，但与 Al Brooks 体系"通道中惯性延续"的理念矛盾（上涨通道倾向继续上涨）。

**修复建议**: 考虑在 neutral 方向时同时加载多头和空头策略文件，让 AI 自行判断。

---

## 三、一般问题 (Minor)

### 13. `prompt_assembler.py` — Stage2 策略文件顺序可能影响 AI 权重

`stage2_prompt_txt_files()` 的顺序是：人设 → 策略文件 → 风控。但 `_WEDGE_FILE` 和 `_REVERSAL_FILE` 作为 overlay 追加在方向策略之后，风控之前。如果同时检测到 wedge + reversal，两个 overlay 文件会紧挨着，AI 可能过度关注这些模式。

**修复建议**: 在 overlay 文件之间插入分隔提示语。

---

### 14. `_STAGE2_OUTPUT_CONTRACT` 中 `diagnosis_confidence` 重复定义

Stage1 的 `_STAGE1_OUTPUT_REMINDER` 定义了 `diagnosis_confidence`（0-100），Stage2 的 `_STAGE2_OUTPUT_CONTRACT` 又定义了一次 `diagnosis_confidence`，且分档说明文字不同（Stage1 有 5 档，Stage2 也是 5 档但措辞差异），可能导致 AI 两次打分标准不一致。

**修复建议**: 统一措辞，或在 Stage2 提示中明确"此处沿用阶段一的评分"。

---

### 15. `json_validator.py` 的 `_strip_fences` 只匹配第一个 fence

```python
m = _FENCE_RE.search(text)
if m:
    return m.group(1).strip()
```

如果 AI 输出多个 ```json...``` 块（例如先解释再给 JSON），只取第一个。但如果 AI 在 JSON 之前输出了非 JSON 的 fenced block，会被错误提取。

**修复建议**: 改为匹配最后一个 fence，或增加 JSON 起始字符验证。

---

### 16. `EventBus` 无 `_QT_AVAILABLE` 降级

`event_bus.py` 直接 `from PyQt6.QtCore import QObject, pyqtSignal`，没有 try/except。如果 PyQt6 不可用，整个应用无法启动。而 `session_ledger.py` 和 `refresh_loop.py` 都做了降级。风格不一致。

**修复建议**: 统一策略——要么全部强制 PyQt6（推荐，因为这是 GUI 应用），要么全部降级。

---

### 17. `two_stage.py` 中 `model_copy(update={...})` 模式冗长

每个 checkpoint 都写一个巨大的 `model_copy(update={...})`，24步流程中至少 6 处，每处 10-20 行。大量重复字段（`stage1_messages`、`stage1_response` 等），容易遗漏。

**修复建议**: 使用 mutable dict 构建 record，只在最后一步构造 `AnalysisRecord`；或使用 builder 模式。

---

### 18. `deepseek_client.py` 中 `from openai import OpenAI` 延迟导入

每次 `chat()` / `stream_chat()` 都 `from openai import OpenAI`，虽然 Python 会缓存 import，但放在函数内部违反常规且增加静态分析难度。

**修复建议**: 移到模块顶层。

---

### 19. `two_stage.py` 缺少 Step 8（Step 7 后直接 Step 9）

代码注释从 Step 7 跳到了 Step 9，缺 Step 8。虽然不影响功能，但与 design doc 的 24 步编号不对应。

---

### 20. forming bar 不发送给 AI

`build_analysis_frame()` 丢弃 forming bar，AI 只分析已闭K线。这是设计决策，但可能导致 AI 在K线刚收盘时缺乏"最新未收盘K线"的信息（如当前正在形成的 pin bar、doji 等），错过实时信号。

**修复建议**: 考虑在 K线表格末尾附注"当前未收盘K线: O=xxx H=xxx L=xxx C=xxx"，供 AI 参考。

---

## 四、提示词问题

### P1. 提示词文件无版本标记

17个 txt 文件没有版本号或修改日期标记。AI 无法知道当前策略文件是否更新，调试时也无法追溯"某次分析用的是哪个版本的策略"。

**修复建议**: 在文件头部加 `# v1.2 2026-04-15` 注释，或在 prompt assembler 注入版本信息。

---

### P2. Stage1/Stage2 system prompt 过长

Stage1 system prompt = 人设 + 诊断框架 + K线信号 + 输出格式提醒，叠加后可能 15K-20K tokens。Stage2 system prompt = 人设 + 策略文件(2-4个) + 风控 + 经验库 + 输出契约，可能 20K-30K tokens。

对于 DeepSeek V4 Pro 的 1M context 来说还好，但：
- prompt 越长，AI 对中间部分的注意力越弱（Lost in the Middle 问题）
- 策略文件中的冗余信息可能稀释关键指令

**修复建议**: 
- 将人设文件精简，只保留核心思维框架
- 策略文件中重复的背景知识（如"什么是通道"）提取为共享片段，避免每次重复注入
- 考虑使用 RAG 检索替代全量注入

---

### P3. `diagnosis_confidence` 低于40时"不下单"规则仅在提示词中约束

design.md 要求 "diagnosis_confidence 低于40时默认不下单"，但这只是提示词中的软约束，代码侧（`json_validator.py`）没有做硬性校验。AI 可能在 confidence=30 时仍然下单。

**修复建议**: 在 `json_validator.py` 的 `_check_no_order_invariant` 中增加 confidence 低于阈值时的强制不下单检查。

---

### P4. 经验库格式是纯 JSON，缺乏结构化评分

经验库条目是 `ExperienceEntry`（Pydantic model），但 AI 看到的只是 JSON dump。没有"此案例是否成功"的显式标签，AI 需要从 JSON 内容自行推断。

**修复建议**: 在 `_render_experience` 中增加案例标签（✅成功 / ❌失败 / ⚠待确认），帮助 AI 快速区分。

---

### P5. `使用说明.txt` 未被任何代码加载

`prompt_engineering/使用说明.txt` 存在于目录中但未被 `STAGE1_PROMPT_TXT_FILES` 或 `STAGE2_BASE_PROMPT_TXT_FILES` 引用，也没有被 router 追加。如果包含重要信息则遗漏了。

---

## 五、架构建议

### A1. 引入 Repository 模式管理 records

当前 `PendingWriter` 直接操作文件系统（JSON + JSONL），经验库由 `ExperienceReader` 读取。建议抽象统一的 `RecordRepository` 接口，未来可切换到 SQLite 或远程存储。

### A2. 考虑异步化

当前 `TwoStageOrchestrator.submit()` 是同步阻塞的（在 QThread 中运行）。如果未来需要并行分析多个品种，需要重构为 async。建议提前定义 `async def submit_async()` 接口。

### A3. 配置热更新

当前修改设置需要重启应用。建议增加 `settings` 文件监听（`QFileSystemWatcher`），自动 reload。

### A4. K线数据序列化优化

`_build_empty_record` 中 `dataclasses.asdict(bar)` 对每个 KlineBar 做反射序列化，5000 根K线时有性能开销。建议 `KlineBar` 预定义 `to_dict()` 方法。

---

## 六、测试覆盖评估

| 层级 | 文件 | 覆盖情况 |
|------|------|----------|
| **Unit** | `test_kline_buffer.py` | ✅ 基本覆盖 |
| **Unit** | `test_deepseek_client.py` | ✅ mock 测试 |
| **Unit** | `test_prompt_assembler.py` | ✅ 消息构建 |
| **Unit** | `test_pending_writer_sanitize.py` | ✅ 安全性 |
| **Unit** | `test_router_determinism.py` | ✅ property |
| **Unit** | `test_secret_store_roundtrip.py` | ✅ |
| **Integration** | 8个场景（happy path/cancel/network/missing field 等） | ✅ 覆盖完整 |
| **Property** | 10个 property-based 测试 | ✅ |
| **E2E** | 4个 smoke test | ⚠ 需要真实 API |
| **缺失** | `test_json_validator.py` (unit) | ❌ 只有 property，缺 unit |
| **缺失** | `test_free_chat.py` (unit) | ❌ 只有 e2e |
| **缺失** | `test_app_context.py` | ❌ bootstrap 无测试 |
| **缺失** | `test_mt5_source.py` | ❌ 数据源无测试 |
| **缺失** | GUI 模块 | ❌ 无自动化测试 |

---

## 七、优先级排序

| 优先级 | 编号 | 问题 | 工作量 |
|--------|------|------|--------|
| P0 | #2 | print() 调试输出残留 | 0.5h |
| P0 | #3 | PROJECT_ROOT 硬编码 | 0.5h |
| P0 | #4 | stream first_chunk 消费风险 | 2h |
| P1 | #1 | OpenAI client 每次新建 | 1h |
| P1 | #5 | _QT_AVAILABLE 半降级 | 1h |
| P1 | #6 | forming→closed 竞态 | 2h |
| P1 | #7 | configure_logging 重复调用 | 0.5h |
| P1 | #11 | Token 预算阈值不匹配 | 0.5h |
| P2 | #8 | api_key 内存明文 | 3h |
| P2 | #9 | record ID 逻辑重复 | 1h |
| P2 | #10 | yfinance 4h resample | 2h |
| P2 | #12 | neutral 方向无策略 | 1h |
| P2 | P3 | confidence 低于40硬校验 | 1h |
| P3 | #13-20 | 一般代码问题 | 各0.5h |
| P3 | P1-P5 | 提示词优化 | 持续 |

---

**总结**: PA_Agent 整体架构清晰，两阶段分析流程设计合理，提示词体系基于 Al Brooks 价格行为体系具有专业深度。主要风险集中在：**调试输出泄漏**（P0）、**路径硬编码**（P0）、**流式处理竞态**（P0）三个即时问题，以及 **OpenAI client 连接复用**、**K线缓冲区竞态**两个性能/正确性问题。提示词方面建议增加版本管理和 confidence 硬校验。
