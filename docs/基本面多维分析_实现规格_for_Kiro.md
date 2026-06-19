# 实现规格：为 PA_Agent 增加「基本面 / 新闻 / 情绪 / 宏观」多维上下文（港股 + 美股，兼顾 A 股）

> 交付对象：实现方（Kiro）。本文是自包含规格，按此实现后由审阅方做验收。
> 代码风格：与现有代码一致（类型注解、`from __future__ import annotations`、中文 docstring/注释可接受、ruff/black line-length=100）。

---

## 0. 背景与目标

PA_Agent 是基于价格行为（Price Action）的 K 线分析工具：从 MT5 / TradingView / yfinance / 东方财富拉 OHLCV，本地算 EMA20/ATR14/几何特征，分两阶段（阶段一诊断 → 阶段二决策）发给大模型，输出结构化 JSON。

**目标**：让 AI 在做诊断时**额外看到**所分析标的的：
- 📊 **基本面**：市盈率(PE)、市值、EPS、营收/净利增速、ROE、股息率
- 🌍 **宏观快照**：相关大盘指数、利率、美元指数、VIX 等"市场环境"
- 🌡️ **情绪**：分析师评级分布 / 目标价（作为"舆情/情绪"的可得代理）
- 📰 **新闻**：最近若干条标题（Phase 2）

**覆盖市场**：**港股、美股**（用 yfinance）为主要诉求；**A 股**顺带激活（用项目已有的东财模块）。

**明确不做**：资金流、龙虎榜、盘口 / Level-2。（东财模块里虽然抓了这些，但本功能**不得**渲染/注入它们。）

**硬约束**：
1. 多维上下文是**附加**信息，**绝不能改变或破坏**现有的 K 线 / 价格行为分析流程。
2. 任何外部数据获取失败（超时、无数据、缺包）**必须静默降级**：跳过该维度，分析照常进行，不抛异常到主流程。
3. 必须**可配置开关**，默认行为见 §5。
4. 必须**缓存**，避免每次分析重复打网络。

---

## 1. 现状关键事实（已核对，务必先读）

### 1.1 已有但「未接线」的东财 A 股模块 —— 复用它，别重写
文件：`pa_agent/data/eastmoney_extended.py`（A 股专用，用 SH/SZ secid + 6 位代码）。

它**已经实现**且**自带 prompt 格式化 + GUI 分栏 + 60s 缓存**，但**全项目无任何调用**（死代码）。关键函数：
- `fetch_compact_stock_context(symbol, *, use_cache=True) -> dict` —— 并行抓取一篮子数据，带缓存。
- `format_compact_stock_context_for_prompt(ctx) -> str` —— 渲染成 markdown，供 prompt 注入。
- `format_compact_stock_context_sections(ctx) -> list[tuple[str, str]]` —— 拆成 (标题, 正文) 列表，供 GUI 标签展示。
- `clear_compact_stock_context_cache(symbol=None)` —— 清缓存。
- 估值/PE：`fetch_valuation_summary`；财务：`finance_main`；新闻：`fetch_stock_news`；研报评级：`fetch_research_reports`；舆情：千股千评 `stock_comment`、`market_focus`、`participation`。

> ⚠️ 该模块的 `format_compact_stock_context_for_prompt` 会渲染**资金流/龙虎榜**等小节。本功能注入时**必须过滤掉**这些小节（见 §3.3）。

### 1.2 注入位置（阶段一用户提示词）
文件：`pa_agent/ai/prompt_assembler.py`
- 类 `PromptAssembler`，构造器 `__init__(self, prompt_dir, experience_reader=None, *, prompt_settings=None)`（约 L839）。
  - 即：assembler 持有 `self._prompt_settings`（`PromptSettings`）和可选注入的 `self._experience_reader`。**新增依赖注入参照 `experience_reader` 的写法。**
- `build_stage1(frame, *, analysis_mode="original") -> list[dict]`（约 L971）。
- `_build_stage1_user_prompt(self, frame, *, analysis_mode="original") -> str`（约 L1196）：这是注入点。当前结尾结构为：
  ```
  ...K线几何特征表 {feature_table}...
  （在此插入「多维上下文」块）
  请根据以上数据，严格输出阶段一 JSON 诊断结果。
  {_STAGE1_TAIL_REMINDER}
  ```
  注入应放在 `feature_table` 之后、"请根据以上数据..."之前。
- 增量分析变体也需注入：`_build_incremental_stage1_user_prompt`（约 L1246）、`_build_incremental_stage1_continuation_user_prompt`（约 L1316）。三处共用同一段注入逻辑（抽成一个私有方法 `_fundamental_context_block(frame) -> str`）。
- `frame.symbol`、`frame.timeframe` 可用（`KlineFrame`）。

> KV-cache 注意：系统提示词是进程级缓存（静态），多维上下文是动态的，**必须放在 user prompt**（已是动态部分），不要放进 system prompt。

### 1.3 编排器
文件：`pa_agent/orchestrator/two_stage.py`，调用 `self._assembler.build_stage1(frame, analysis_mode=...)`（约 L420）。无需改它的调用方式（注入逻辑在 assembler 内部完成）。分析记录结构见 `pa_agent/records/schema.py`。

### 1.4 配置
文件：`pa_agent/config/settings.py`
- `PromptSettings`（约 L25）：现有字段 `stage2_load_full_strategy_library` / `experience_max_entries` / `experience_max_chars_per_entry` / `stage1_inject_pattern_briefs`。**新增字段加在这里**（见 §5）。
- `GeneralSettings.last_data_source: DataSourceKind`（`"mt5"|"tradingview"|"akshare"|"eastmoney"`）—— 可作为市场判定的辅助信号。

### 1.5 yfinance
`pa_agent/data/yfinance_source.py` 已用 `import yfinance as yf` / `yf.Ticker(symbol)`，但 yfinance **未列入 `pyproject.toml` 依赖**（靠手动 `pip install`）。本功能需把它加入可选依赖组（见 §6）。`yf.Ticker(x)` 提供 `.info`（含 trailingPE/forwardPE/marketCap/...）、`.news`、`.get_recommendations()` / `.recommendations`。

### 1.6 GUI 侧边栏
文件：`pa_agent/gui/ai_sidebar.py`（标签页容器）。现有"调试/原始/决策"等标签的 widget 在 `pa_agent/gui/`。新增标签参照现有标签写法（见 §4）。

---

## 2. 总体架构

新增包 `pa_agent/context/`：

```
pa_agent/context/
├── __init__.py
├── market_classifier.py      # 由 symbol(+数据源) 判定市场: a_share / hk / us / other
├── yfinance_fundamentals.py  # 港股/美股: 抓取 + 渲染 prompt + 拆 GUI 分栏 (+缓存)
├── macro_snapshot.py         # 宏观: 大盘/利率/美元/VIX 快照 (+缓存)
└── fundamental_context.py    # 统一入口: 按市场路由(东财/yfinance) + 拼宏观 + 降级 + 缓存
```

数据流：
```
build_stage1(frame)
  └─ _fundamental_context_block(frame)              [prompt_assembler]
       └─ fundamental_context.build_for_symbol(...)  [统一入口, 永不抛异常]
            ├─ market = classify(symbol, data_source)
            ├─ A股  → eastmoney_extended.fetch_compact_stock_context + 过滤渲染
            ├─ 港股/美股 → yfinance_fundamentals.*
            └─ + macro_snapshot.*
       → 返回 markdown 文本(空串=无内容, 注入时跳过)
GUI「基本面」标签 → 分析完成信号 → 同一入口(命中缓存)取 sections 展示
```

---

## 3. 详细任务

### 3.1 `market_classifier.py`
```python
from enum import Enum
class Market(str, Enum):
    A_SHARE = "a_share"; HK = "hk"; US = "us"; OTHER = "other"

def classify_market(symbol: str, data_source: str | None = None) -> Market: ...
```
规则（优先级从上到下）：
- 6 位纯数字（可带 `sh/sz` 前缀或 `.SH/.SZ` 后缀）→ `A_SHARE`。
- 形如 `HKEX:xxxx`、纯数字 1–5 位、或带 `.HK` → `HK`。（复用 `pa_agent/data/market_defaults.py` 里已有的 `normalize_hk_tv_code` / `_is_hk_tv_code` 判定，别重复造。）
- 纯字母 / 字母数字带 `.`（如 `AAPL`、`BRK.B`）→ `US`。
- 黄金/外汇/加密（`XAUUSD`/`EURUSD`/`BTC...` 等）、MT5 数据源的非股票 → `OTHER`（无个股基本面，仅可加宏观）。
- 数据源 `data_source` 作为辅助：`akshare/eastmoney` 倾向 A 股；`mt5` 倾向 OTHER/外汇。

### 3.2 `yfinance_fundamentals.py`（港股 + 美股）
镜像东财模块的三件套接口风格：
```python
def to_yf_symbol(symbol: str, market: Market) -> str: ...
def fetch_yf_fundamentals(symbol: str, market: Market, *, use_cache=True) -> dict: ...
def format_yf_fundamentals_for_prompt(ctx: dict) -> str: ...        # markdown, 空内容返回 ""
def format_yf_fundamentals_sections(ctx: dict) -> list[tuple[str,str]]: ...
def clear_yf_fundamentals_cache(symbol: str | None = None) -> None: ...
```
- **符号映射 `to_yf_symbol`**：
  - 港股：抽数字 → `zfill(4)` → 加 `.HK`（如 `700`/`HKEX:700`→`0700.HK`；`07709`→`07709.HK`）。
  - 美股：原样大写（`aapl`→`AAPL`）。
- **抓取 `fetch_yf_fundamentals`**：用 `yf.Ticker(yf_symbol)`：
  - 基本面（来自 `.info`，字段缺失要降级为 `None`）：`trailingPE`、`forwardPE`、`marketCap`、`trailingEps`、`revenueGrowth`、`earningsGrowth`、`returnOnEquity`、`dividendYield`、`profitMargins`、`sector`、`industry`、`longName`。
  - 情绪（分析师）：`recommendationKey`、`numberOfAnalystOpinions`、`targetMeanPrice`、`currentPrice`；如可得 `.recommendations` 的买/持/卖分布。
  - 新闻（Phase 2）：`.news` 前 N 条（标题 + 时间）。
  - 必须包 `try/except`，超时/无包/无字段都返回带 `None` 的 dict，绝不抛。
- **渲染 `format_*_for_prompt`**：紧凑 markdown，模块标题示例：`## 基本面与分析师观点（程序抓取，供参考）`，含「估值与基本面」「分析师评级」小节；数字格式化（市值用「亿/万亿」或「B/T」）。无任何有效字段时返回 `""`。
- **缓存**：仿东财 `_COMPACT_CTX_CACHE`，TTL 基本面建议 **6 小时**（变化慢），用 `time.monotonic()`。

### 3.3 A 股渲染过滤（复用东财，去掉不要的小节）
新增一个**过滤版渲染**，不要直接用东财的 `format_compact_stock_context_for_prompt`（它含资金流/龙虎榜）。两种实现方式择一：
- (推荐) 在 `fundamental_context.py` 里用 `format_compact_stock_context_sections(ctx)` 拿到分栏，再**白名单**保留以下标题，重新拼成 markdown：
  `估值与市值`、`机构盈利预测`、`主要财务`、`财务分析指标`、`主营构成`、`分红送转`、`股东户数`、`近期新闻`、`近期研报`、`千股千评`、`技术/资金评分`、`参与度`、`市场关注度`、`板块与概念`、`核心题材要点`、`公司重大事项`、`近期公告（API）`。
  **剔除**：含「资金流」「主力」「龙虎榜」「大宗交易」「融资融券」「沪深港通」「股东增减持」「席位」等的小节。
- 维护一个 `_AS_EXCLUDE_KEYWORDS` 关键词集合做剔除。

### 3.4 `macro_snapshot.py`（宏观）
```python
def fetch_macro_snapshot(market: Market, *, use_cache=True) -> dict: ...
def format_macro_for_prompt(snap: dict) -> str: ...
```
- 用 yfinance 拉指数最近 2 根日线算涨跌：
  - 港股：`^HSI`（恒指）+ `^GSPC` + `^TNX`（美债10Y）+ `DX-Y.NYB`（美元指数）。
  - 美股：`^GSPC`（标普）`^IXIC`（纳指）`^VIX`（恐慌）`^TNX` `DX-Y.NYB`。
  - A 股：`000001.SS`（上证）`399001.SZ`（深成）—— 也可用东财指数接口，二选一，能降级即可。
  - OTHER：仅 `DX-Y.NYB` + `^TNX`（外汇/黄金相关宏观）。
- 渲染 `## 宏观环境快照`，每行：名称 + 最新值 + 涨跌%。
- 缓存 TTL 建议 **1 小时**。失败降级返回 `""`。

### 3.5 `fundamental_context.py`（统一入口，核心）
```python
def build_for_symbol(
    symbol: str,
    *,
    data_source: str | None = None,
    settings: PromptSettings | Any = None,
    use_cache: bool = True,
) -> str:
    """返回注入用 markdown；任何失败返回 '' ；按 settings 开关裁剪维度。永不抛异常。"""

def build_sections_for_symbol(...) -> list[tuple[str, str]]:
    """GUI 用：返回 (标题, 正文) 列表。"""
```
- 读取 §5 配置开关，决定是否含基本面/新闻/宏观/情绪。
- `classify_market` → 路由：A 股走 §3.3 过滤渲染；港股/美股走 §3.2；都拼上 §3.4 宏观（按 market）。
- `OTHER` 市场：跳过个股基本面，仅尝试宏观。
- **整体用 `try/except` 包裹**，任何子步骤失败只丢该维度。
- 顶层加一句固定引导语，例如：
  `> 以下为程序抓取的基本面/宏观/情绪信息，作为价格行为分析的辅助。判断冲突时以价格行为为主；基本面/宏观仅用于"确认或背离"的加权（如技术看多但估值极高+宏观逆风→适度下调信心）。`

### 3.6 接线 `prompt_assembler.py`
- 构造器新增可选注入：`fundamental_provider: Any = None`（仿 `experience_reader`）。`app_context.py` 构建 assembler 时传入（默认就用 `pa_agent.context.fundamental_context` 模块的函数封装成一个轻对象，或直接传模块）。
- 新增私有方法 `_fundamental_context_block(self, frame) -> str`：
  - 若开关关闭或无 provider → 返回 `""`。
  - 否则调用 provider，套 `try/except` 兜底返回 `""`。
- 在 `_build_stage1_user_prompt` / 两个增量变体的 `feature_table` 之后注入该块（非空才插）。
- **不要**把它放进被缓存的 system prompt。

### 3.7 引导提示词文件（可选但推荐）
新增 `prompt_engineering/多维分析_基本面与宏观.txt`，写清楚"价格行为为主、基本面/宏观为辅"的权衡原则；若加入，需同步 `tests/unit/test_prompt_txt_files.py` 的清单（如果该测试断言文件集合）。

---

## 4. GUI（新增「基本面」标签，Phase 1 末或 Phase 2 均可）
- 在 `pa_agent/gui/ai_sidebar.py` 新增一个标签「基本面」，widget 参照现有只读文本标签（如 `prompt_files_panel.py` / `debug_widget.py` 的分节展示风格）。
- 数据来源：分析完成后，用 `fundamental_context.build_sections_for_symbol(当前symbol, ...)`（命中缓存，不会重复打网）渲染分栏。
- 无内容时显示占位文案（如"该标的无可用基本面/或未开启"）。
- 不阻塞 UI：抓取在后台线程（参照项目现有 `pa_agent/util/threading.py` / `snapshot_worker.py` 模式），完成后回主线程刷新。

---

## 5. 配置（`PromptSettings` 新增字段，全部带默认值，向后兼容）
```python
# 多维上下文总开关与分维度开关
enable_fundamental_context: bool = True      # 总开关
fundamental_include_news: bool = False       # 新闻(Phase 2)，默认关
fundamental_include_macro: bool = True       # 宏观快照
fundamental_include_sentiment: bool = True   # 分析师评级/情绪
fundamental_news_max_items: int = Field(default=3, ge=0, le=10)
fundamental_cache_ttl_minutes: int = Field(default=360, ge=1, le=1440)  # 基本面缓存
```
- `model_config = ConfigDict(extra="ignore")` 已在该类，旧 settings.json 不会报错。
- 设置对话框 UI 可后续补；Phase 1 允许仅支持改 json 文件。

---

## 6. 依赖（`pyproject.toml`）
在 `[project.optional-dependencies]` 新增组：
```toml
equity = [
    "yfinance>=0.2.40",
]
```
- 代码内 `import yfinance` 必须 `try/except ImportError`，缺包时该维度降级（与 §0 硬约束一致）。
- README 安装说明补一句：港股/美股基本面需 `pip install -e ".[equity]"`。

---

## 7. 测试要求（`tests/unit/`，必须随 PR 提交）
所有网络调用用 `unittest.mock` 打桩，**禁止**真实联网（与现有 `test_deepseek_client.py` 风格一致）。
1. `test_market_classifier.py`：`600519`→A股；`HKEX:700`/`700`/`07709`/`0700.HK`→港股；`AAPL`/`brk.b`→美股；`XAUUSD`/`EURUSD`→OTHER。
2. `test_yfinance_fundamentals.py`：
   - `to_yf_symbol`：`700`→`0700.HK`，`07709`→`07709.HK`，`aapl`→`AAPL`。
   - mock `yf.Ticker`，`.info` 含部分字段 → 渲染含 PE/市值；字段缺失 → 不报错且跳过该行。
   - `yf` 未安装（patch 成 ImportError）→ 返回空 dict / `""`，不抛。
3. `test_macro_snapshot.py`：mock 行情 → 渲染含指数涨跌；抓取异常 → 返回 `""`。
4. `test_fundamental_context.py`：
   - A 股路由 → 调东财（mock `fetch_compact_stock_context`）且渲染**不含**"资金流/龙虎榜/融资融券/沪深港通"等关键词（断言被过滤）。
   - 港股/美股路由 → 调 yfinance 分支。
   - 任一子步骤抛异常 → `build_for_symbol` 仍返回 `str`（不抛）。
   - 总开关关闭 → 返回 `""`。
5. `test_prompt_assembler_fundamental_injection.py`：
   - 注入 fake provider 返回固定文本 → `build_stage1` 的 user 消息包含该文本；增量两个变体同样包含。
   - provider 为 None 或抛异常 → user 消息正常生成、不含该块、不抛。
   - **回归**：provider=None 时，`build_stage1` 输出与未改动前一致（不破坏现有 KV-cache 前缀 / 现有断言）。

运行：`pytest -m unit` 全绿；不得新增联网测试到默认集合。

---

## 8. 验收标准（审阅方据此验收）
功能性：
1. `pytest -m unit` 全部通过；新增测试覆盖 §7 全部条目。
2. provider=None / 总开关关闭 / 抓取全失败 三种情况下，**原有两阶段分析行为不变**（K 线 prompt 与改动前逐字节一致——除非有意注入块）。
3. 港股（如 `0700.HK`/`700`）与美股（如 `AAPL`）能在阶段一 user prompt 中看到「基本面+分析师+宏观」块（可用 mock 演示）。
4. A 股渲染**确实剔除**了资金流/龙虎榜/盘口类小节（断言关键词不出现）。
5. 任意网络异常/缺 yfinance 包时，程序不崩、分析继续，仅日志 warning。

非功能性：
6. 每次分析对同一标的**不重复打网**（命中缓存）；缓存 TTL 生效。
7. 多维抓取在后台/带超时，不明显拖慢主流程（单次外部抓取总超时建议 ≤ 8s，超时即降级）。
8. 无明文密钥、无 PII 写日志；ruff/black 通过；新增 import 都有缺包降级。
9. 不引入对主流程的循环依赖：`pa_agent/context/` 可依赖 `data/`，但 `data/` 不反向依赖 `context/`。

交付物：
10. 新增 `pa_agent/context/` 4 个模块 + 测试；`prompt_assembler.py` 注入；`settings.py` 配置；`pyproject.toml` 依赖；（可选）GUI 标签；（可选）引导提示词文件 + README 一行说明。
11. PR 描述列出：改了哪些文件、默认开关状态、如何手动验证港股/美股各一个标的。

---

## 9. 实施顺序建议（分两期，便于增量验收）
- **Phase 1**：`market_classifier` + `macro_snapshot` + `yfinance_fundamentals`(基本面+情绪) + `fundamental_context` + 接线 + 配置 + 依赖 + 单测 + A 股过滤激活。（不含新闻、可暂不含 GUI 标签）
- **Phase 2**：新闻维度（`fundamental_include_news`）、GUI「基本面」标签、设置对话框 UI、引导提示词文件。

每期独立可跑、可验收。
