# PA Agent — AI K线分析辅助工具（桌面端）

**交流 QQ 群：871156180**

---

> 面向主观交易者的 **价格行为（Price Action）** AI 辅助决策工具：从 **MT5 / TradingView / yfinance / AkShare** 读取 K 线，将**结构化 K 线数据与程序预计算特征**送入大模型做**两阶段分析**（市场诊断 → 交易决策），**不是**截图识图，**不连接券商、不执行下单**。

---

## 目录

- [项目简介](#项目简介)
- [工作原理](#工作原理)
- [环境要求](#环境要求)
- [平台支持](#平台支持)
- [安装步骤](#安装步骤)
- [启动程序](#启动程序)
- [运行测试](#运行测试)
- [目录结构](#目录结构)
- [配置文件](#配置文件)
- [参与贡献与安全](#参与贡献与安全)
- [详细使用说明](#详细使用说明)
- [图表 K 线与分析快照说明](#图表-k-线与分析快照说明)
- [高级特性](#高级特性)
- [相关文档](#相关文档)
- [常见问题](#常见问题)

---

## 项目简介

PA Agent 是一款桌面辅助工具，帮助交易者按 Al Brooks 风格的价格行为框架理解当前图表，并把"看图"过程结构化成可复核的决策路径与字段。

程序会：

1. 通过 **MT5 / TradingView / yfinance / AkShare** 拉取选定品种、周期的 OHLCV K 线（可含当前未收盘 K，图表实时显示）
2. 本地计算 **EMA20、ATR14、K 线几何特征**（实体比、内包/外包、ii/iii、突破跟随等）
3. 将 **K 线文本表 + 特征表 + 提示词工程模块** 发给大模型（支持 DeepSeek、PackyAPI 等 OpenAI 兼容接口）
4. 经 **阶段一（诊断）** 与 **阶段二（决策）** 输出结构化 JSON，并在界面上绘制入场/止损/止盈参考线

**不会**把 K 线图截图发给 AI；模型读到的是与图表一致的数值化 K 线序列（K1 为最新已收盘棒）。

### 主要功能

- 📈 **多数据源**：MT5 实时 K 线、TradingView（匿名/登录）、yfinance（期货/加密货币）、AkShare（A 股） + 本地蜡烛图、EMA、序号标签
- 🧠 **两阶段 AI 分析**：闸门诊断 → 策略路由 → 交易决策（限价/突破/市价或不下单）
- 📋 **逐 K 摘要**（`bar_by_bar_summary`）与信号链校验，减少"口头看涨、JSON 做空"类矛盾
- 🌳 **决策树可视化**：赛博科幻风格的可交互流程图，自动播放闸门→策略路径动画
- 🔮 **未来走势预期**：AI 预测下一根 K 线方向和下一个市场周期位置
- 🔄 **增量分析与持续跟踪**：增量分析复用上次结论；`keep_analysis` 模式下新 K 线收盘自动触发新一轮分析
- 💬 **分析后自由追问**：完整的对话会话管理器，实时推理流 + 内容流、Token 用量进度条、对话历史持久化
- 📚 **经验库**：按周期位置检索历史案例供阶段二参考，可配置条目数和每条字数
- 📝 **完整落盘**：Prompt、原始响应、诊断/决策 JSON、Token 用量、追问记录
- 🛡️ **可配置校验体系**：JSON 校验、一致性检查、语义校验、截断修复、失败自动重试策略
- 🔒 **API Key** 本地 DPAPI 加密存储（Windows）/ cryptography 加密（macOS）

更完整的界面说明见仓库内 `[PA_Agent使用文档.md](PA_Agent使用文档.md)`。

---

## 工作原理

```text
数据源（MT5 / TradingView / yfinance / AkShare）
         │
         ▼
   本地缓冲 / 图表显示 ──► 提交分析（可选：等待当前 K 收盘）
                              │
         ┌────────────────────┴────────────────────┐
         ▼                                         ▼
   阶段一 · 市场诊断                          策略文件路由
   （周期/方向/闸门/逐K摘要）                  （按诊断加载 22 个策略文件）
         │                                         │
         └────────────────────┬────────────────────┘
                              ▼
                    阶段二 · 交易决策
                    （§9 信号链 / §10 风险收益 / §11 下单方式）
                              │
                              ▼
              JSON 校验 ──► 失败重试（格式 ≤3 次、语义 ≤1 次）
                              │
                              ▼
              图表叠加线 ──► 记录保存 ──► 可追问 ──► 决策树可视化
```


| 环节           | 说明                                                                                       |
| -------------- | ------------------------------------------------------------------------------------------ |
| 数据来源       | **MT5**（Windows）、**TradingView**（全平台）、**yfinance**（期货/加密货币）、**AkShare**（A 股，代码支持） |
| 送给 AI 的内容 | K 线表、几何特征表、阶段一诊断结果、路由后的策略提示词；阶段二另含决策树规则                                   |
| 图表作用       | 供肉眼确认；分析时图表可暂停刷新，避免与提交数据不一致                                                      |
| 输出           | 阶段一/二 JSON；阶段二含 `decision`、`decision_trace`、盈亏比等字段                                    |
| 边界           | **仅辅助分析，不连接券商下单**                                                                        |


---

## 环境要求


| 项目     | 要求                                                                    |
| -------- | ----------------------------------------------------------------------- |
| 操作系统 | Windows 10 / 11（主支持）、macOS 12+（TradingView 数据源，见[平台支持](#平台支持)） |
| Python   | 3.11+（推荐官方安装包，安装时勾选 Add to PATH）                                   |
| 数据源   | MT5 / TradingView / yfinance / AkShare **至少配置一种**（见[安装步骤](#安装步骤)） |
| 显卡     | 无特殊要求                                                                  |
| 网络     | 可访问所配置的 AI API（如 DeepSeek、PackyAPI 等）                        |


---

## 平台支持

| 平台            | 状态   | 可用数据源                          | 说明                                       |
| --------------- | ------ | ----------------------------------- | ------------------------------------------ |
| Windows 10 / 11 | ✅ 主支持 | MT5 / TradingView / yfinance / AkShare | 全部功能可用                                   |
| macOS 12+       | ✅ 可用   | TradingView / yfinance              | 无 MT5，默认 TradingView 数据源；部署指南见下文         |
| Linux           | ⚠️ 未验证 | TradingView                         | 理论上可运行，需手动适配                               |

### macOS 部署

macOS 用户请参照项目根目录的 **`MAC版本智能体部署方法.txt`**，内含完整的自动化部署步骤、依赖安装、故障排查和验收清单。

核心差异：
- macOS 不能使用 MT5，默认数据源为 **TradingView**
- `pyproject.toml` 已通过 `sys_platform == 'win32'` 条件自动跳过 `MetaTrader5` 和 `pywin32`
- 双击 `运行智能体.command` 即可启动

---

## 安装步骤

### 1. 安装 Python 3.11+

从 [python.org](https://www.python.org/downloads/) 下载并安装，勾选 **Add Python to PATH**。

```cmd
python --version
```

### 2. 配置数据源（至少一种）

**MT5（仅 Windows）**：安装并登录 MetaTrader 5 终端，确认「市场报价」中可见目标品种。

**TradingView（全平台）**：无需安装额外软件。支持匿名模式和登录模式（登录可提升请求频率限制）。支持 A 股（SSE/SZSE）、港股（HKEX）、外汇、期货、加密货币等。品种别名配置见 `config/tv_symbol_aliases.example.json`。

**yfinance（全平台）**：支持期货（`GC=F` 黄金、`CL=F` 原油、`ES=F` 标普500 等）和加密货币（`BTC-USD` 等）。注意：数据有 ~15 分钟延迟，日内数据仅支持近 60 天。GUI 暂未暴露此数据源，可通过代码配置。

> **港股/美股多维基本面**：如需在分析时注入港股/美股的基本面、资金面、宏观与分析师情绪上下文，请安装可选依赖：`pip install -e ".[equity]"`（基于 yfinance）。缺包时该功能自动静默降级，不影响 K 线分析。

**AkShare（仅代码支持）**：A 股数据源。GUI 已移除选项，但 `factory.py` 仍可实例化。如需启用，安装可选依赖：`pip install -e ".[ashare]"`。

### 3. 克隆或下载项目

```cmd
git clone <仓库地址>
cd PA_Agent
```

### 4. 创建虚拟环境（推荐）

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 5. 安装依赖

```cmd
pip install -e ".[dev]"
```

> 国内镜像示例：
>
> ```cmd
> pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

### 6. 配置 API

复制配置模板（首次克隆建议执行）：

```cmd
copy config\settings.example.json config\settings.json
```

启动程序后打开 **设置**，填写 **Base URL**、**模型名** 与 **API Key**（支持 DeepSeek 官方或第三方兼容网关）。Key 会加密写入 `config/settings.json`，不会以明文提交到 Git。字段说明见 `[config/README.md](config/README.md)`。

---

## 启动程序

```cmd
python -m pa_agent.main
```

或安装后：

```cmd
pa-agent
```

也可使用项目根目录的 `run.py`。

首次启动若提示数据源未连接：使用 MT5 请先确认终端已运行并登录；使用 TradingView 请点击工具栏「获取数据」按钮开始拉取。

---

## 运行测试

```cmd
pytest
```

跳过端到端 / GUI 测试：

```cmd
pytest -m "not e2e"
```

仅单元测试：

```cmd
pytest -m unit
```

仅属性测试（Hypothesis）：

```cmd
pytest -m property
```

仅集成测试：

```cmd
pytest -m integration
```

---

## 目录结构

```
PA_Agent/
├── pa_agent/                     # 主程序包
│   ├── main.py                   # 程序入口
│   ├── app_context.py            # 应用上下文（启动时 bootstrap 各组件）
│   ├── ai/                       # AI 核心
│   │   ├── prompts/schemas.py    # 阶段一/二 JSON Schema 定义
│   │   ├── prompt_assembler.py   # Prompt 组装引擎（97KB）
│   │   ├── decision_nodes.py     # 确定性决策节点引擎（92KB）
│   │   ├── decision_tree.py      # 决策树结构与路由
│   │   ├── deepseek_client.py    # API 客户端（流式 + batch）
│   │   ├── json_validator.py     # JSON 校验与错误报告
│   │   ├── pattern_routing.py    # 策略文件路由
│   │   ├── coherence_checks.py   # 阶段一/二交叉字段检查
│   │   ├── kline_features.py     # K 线几何特征计算
│   │   ├── cycle_enums.py        # 市场周期位置枚举
│   │   └── ...
│   ├── config/settings.py        # Pydantic 配置模型（含 Provider / General / Prompt / Validation 四组）
│   ├── data/                     # 数据源
│   │   ├── mt5.py                # MetaTrader 5 数据源
│   │   ├── tradingview.py        # TradingView 数据源（tvDatafeed）
│   │   ├── yfinance_source.py    # yfinance 数据源（期货/加密货币）
│   │   ├── akshare_source.py     # AkShare 数据源（A 股）
│   │   ├── factory.py            # 数据源工厂
│   │   └── refresh_loop.py       # K 线自动刷新循环
│   ├── gui/                      # PyQt6 界面
│   │   ├── main_window.py        # 主窗口（178KB，控制栏 / 图表 / 侧栏）
│   │   ├── ai_sidebar.py         # 右侧标签栏（7 个 Tab）
│   │   ├── decision_flow_viz.py  # 决策树可视化（44KB，赛博科幻流程图）
│   │   ├── future_trend_panel.py # 未来走势预期面板
│   │   ├── decision_panel.py     # 决策结果面板
│   │   ├── decision_tree_panel.py # 决策树文本面板
│   │   ├── conversation_widget.py # 追问对话历史时间线 UI
│   │   ├── ai_stream_window.py   # 实时推理流 / 内容流面板（含 Token 进度条）
│   │   ├── chart_widget.py       # K 线蜡烛图组件
│   │   ├── theme/                # 暗色主题（QSS）
│   │   └── widgets/              # 可复用 UI 组件
│   ├── orchestrator/             # 分析编排
│   │   ├── two_stage.py          # 两阶段分析编排器（46KB）
│   │   ├── free_chat.py          # 分析后自由追问会话管理器
│   │   └── validation_retry.py   # 校验失败自动重试
│   ├── records/                  # 分析记录读写
│   └── util/                     # 工具函数
├── prompt_engineering/           # 价格行为提示词模块
│   └── _reference/               # 22 个策略文件（通道/趋势/尖峰/震荡等）
├── experience/                   # 经验库案例（按周期位置分类）
│   ├── broad_channel/
│   ├── extreme_tr/
│   ├── micro_channel/
│   ├── spike/
│   ├── tight_channel/
│   ├── trading_range/
│   └── ...
├── tests/                        # 测试（unit / property / integration / e2e）
├── tools/                        # 诊断与辅助脚本
├── config/                       # 配置模板与说明
│   ├── settings.example.json
│   ├── tv_symbol_aliases.example.json
│   ├── exception_state.example.json
│   └── README.md
├── docs/                         # 补充文档
│   ├── 图表K线与分析快照说明.md
│   └── 获取数据功能说明.md
├── .github/workflows/            # CI（Windows + pytest）
├── logs/                         # 运行日志
├── records/                      # 分析记录（pending / 归档）
├── pyproject.toml
├── run.py
├── Makefile
└── README.md
```

---

## 配置文件

配置文件位于 `config/`，首次运行自动生成，**勿将含密钥的文件提交到 Git**。

| 文件                                    | 说明                                           |
| --------------------------------------- | ---------------------------------------------- |
| `config/settings.json`                  | 主配置（API Key 存为 `api_key_encrypted`）        |
| `config/settings.example.json`          | 无密钥的模板（复制为 `settings.json`）               |
| `config/tv_symbol_aliases.example.json` | TradingView 品种别名模板（复制为 `tv_symbol_aliases.json`） |
| `config/exception_state.example.json`   | 异常计数状态结构参考                                 |
| `config/exception_state.json`           | 运行时自动生成，不提交 Git                            |

配置分为 **四组**：

| 组            | 说明                                               |
| ------------- | -------------------------------------------------- |
| `provider.*`  | AI 提供商配置：模型、Base URL、API Key、Thinking、Effort |
| `general.*`   | 通用设置：数据源、品种、周期、K 线数、决策流播放、持续跟踪等        |
| `prompt.*`    | Prompt 组装调优：策略库加载、经验库条目数、模式 brief 注入      |
| `validation.*` | 校验行为：strict/lenient 模式、一致性检查、截断修复、重试策略    |

详细字段说明见 `[config/README.md](config/README.md)`。

### 防止密钥被 push 到 GitHub

1. 本机执行一次（可选）：
  ```powershell
   powershell -ExecutionPolicy Bypass -File tools\setup_git_secrets.ps1
  ```
2. 仅在 GUI「设置」或本地 `settings.json` 中配置 Key，不要写进 README / 测试用例。
3. 默认 `pytest` 不跑需真实网络的 `live` 测试。

---

## 参与贡献与安全

| 文档                                     | 说明               |
| ---------------------------------------- | ------------------ |
| `[CONTRIBUTING.md](CONTRIBUTING.md)`     | 开发环境、测试与 PR 约定   |
| `[SECURITY.md](SECURITY.md)`             | 漏洞与密钥泄露报告方式      |
| `[LICENSE](LICENSE)`                     | AGPL-3.0 许可证     |


---

## 详细使用说明

- 控制栏：**品种 / 周期 / K 线数**、**提交分析**、**增量分析**、**等待收盘**、**演示模式**
- 右侧标签（7 个）：**实时**（思考流 + 追问）、**决策树**（文本）、**决策树可视化**（可交互流程图 + 自动播放）、**决策**（入场/止损/止盈参考线）、**未来走势预期**（下一根K线 + 下一周期）、**原始**（Prompt / JSON 原始数据）、**调试**（策略文件清单等）
- **分析后自由追问**：分析完成后点击「发送」即可基于当前图表数据自由追问 AI，支持多轮对话、推理流实时显示、Token 用量进度条
- **增量分析**：新增 K 线 ≤ 阈值时自动走增量模式，无需从零开始

完整操作说明、交易倾向、策略路由表见：`[PA_Agent使用文档.md](PA_Agent使用文档.md)`

**图表为何在分析后少 1 根 K 线？** 见 `[docs/图表K线与分析快照说明.md](docs/图表K线与分析快照说明.md)`

**如何加速分析？** 见 `[分析速度优化操作指南.md](分析速度优化操作指南.md)`

---

## 高级特性

### 决策树可视化

右侧「**决策树可视化**」标签页展示赛博科幻风格的分支流程图，从 K 线数据闸门（§1.1）开始，逐节点展示方向判定、Always-In、信号链、下单方式等完整决策路径。支持鼠标拖拽缩放，默认自动播放（可在设置中关闭）。

### 未来走势预期

「**未来走势预期**」标签页显示 AI 生成的两类预测：
- **下一根 K 线预期**：看涨 / 看跌 / 中性的概率分布
- **下一个市场周期预期**：从 9 种周期位置（强涨 / 弱涨 / 震荡上行 / 震荡 / 震荡下行 / 弱跌 / 强跌 / 过渡 / 未定）中选择最可能的下一个状态

### 持续跟踪分析（Keep Analysis）

在设置中开启 `keep_analysis` 后，每当新 K 线收盘，程序会自动触发新一轮分析，无需手动点击「提交分析」。校验失败时可自动关闭持续跟踪（`cancel_keep_analysis_on_retry`）。

### 校验与重试机制

阶段一/二 JSON 输出经过多层校验：
- **语法校验**（category a）：JSON 格式错误 → 最多 3 次重试
- **语义校验**（category c）：逻辑矛盾、信号冲突 → 最多 1 次重试
- **截断修复**：流式输出被截断时自动尝试修复尾部 JSON
- **一致性检查**：默认关闭的可选跨字段交叉检查

校验策略通过 `validation.*` 配置组可调。

### 经验库

`experience/` 目录按市场周期位置（broad_channel、extreme_tr、spike 等）组织案例文件。阶段二分析时自动加载匹配当前周期的历史案例供 AI 参考。每条经验控制在 400 字符内，最多 10 条（`prompt.experience_max_entries` 可调）。

---

## 相关文档

| 文档                                         | 说明                         |
| -------------------------------------------- | ---------------------------- |
| `[PA_Agent使用文档.md](PA_Agent使用文档.md)`      | 完整操作界面说明                   |
| `[分析速度优化操作指南.md](分析速度优化操作指南.md)` | 分析速度优化（4 套方案 + 速查决策树）   |
| `[MAC版本智能体部署方法.txt](MAC版本智能体部署方法.txt)` | macOS 自动化部署指南          |
| `[docs/图表K线与分析快照说明.md](docs/图表K线与分析快照说明.md)` | 图表 K 线与分析快照行为说明        |
| `[docs/获取数据功能说明.md](docs/获取数据功能说明.md)`  | 数据获取功能详细说明               |
| `[config/README.md](config/README.md)`       | 配置文件字段完整说明               |

---

## 常见问题

### Q: 启动时提示 `ModuleNotFoundError: No module named 'pa_agent'`

在项目根目录激活虚拟环境后安装：

```cmd
.venv\Scripts\activate
pip install -e ".[dev]"
```

### Q: 提示 MT5 未连接或没有 K 线

1. 确认 MT5 终端已打开且已登录
2. 品种名与 MT5「市场报价」完全一致（含 `m` 等后缀）
3. 该品种在 MT5 中可正常显示 K 线
4. 若不使用 MT5，切换到 **TradingView** 数据源即可

### Q: TradingView 拉取数据失败或超时

1. 先确保 tvDatafeed 已知 Bug 已修复（见 `MAC版本智能体部署方法.txt` 的问题 3.5）
2. 小周期（1m/5m）更容易触发匿名限速 → 切换到 15m 以上测试
3. 长期解决方案：在 GUI 设置中填入 **TradingView 账号登录**（凭证加密存储）
4. 项目内置了防重叠调用和退避机制，切换品种/周期会自动中断挂起的请求

### Q: 程序是不是把截图发给 AI？

**不是。** 提交的是 K 线 OHLCV 文本表、程序算好的特征，以及提示词；图表仅供本地查看。

### Q: 分析时图表不刷新了？

分析进行中会**暂停图表自动刷新**，避免界面与提交数据不一致。可点 **图表实时更新** 恢复；追问发送时会先刷新一次再冻结，并以该时刻图表数据追问。

### Q: 如何加速分析速度？

参见 `[分析速度优化操作指南.md](分析速度优化操作指南.md)`，核心方法：
- 关闭 **Thinking**（3–5 倍提速，质量下降）
- 降低 **Reasoning Effort**（渐进提速，质量影响较小）
- 减少 **K 线数量**
- 使用 **增量分析**
- 换用更快的模型

### Q: 如何在 macOS 上使用？

参照 `[MAC版本智能体部署方法.txt](MAC版本智能体部署方法.txt)` 自动化部署，默认使用 TradingView 数据源。

### Q: API 调用失败

检查网络、Base URL、模型名与 API Key；若用代理需在系统或网关侧配置。

### Q: `config/settings.json` 损坏

删除后重启，程序会重建默认配置：

```cmd
del config\settings.json
```

### Q: 如何更新

```cmd
git pull
pip install -e ".[dev]"
```

### Q: 日志位置

`logs/` 目录下。

---

**免责声明**：本工具仅供学习与研究，不构成投资建议。交易有风险，决策后果自负。

---

本项目采用 [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE) 发布。

---

## 打赏与支持

如果你觉得这个程序对你有帮助的话，可以打赏激励作者继续优化程序，感谢你的支持和鼓励！

（作者会优先解决打赏人的问题，因为人太多了！回复不过来！）

<p align="center">
  <img src="1d935cac3a4a4575bb3e34beda997633.jpeg" alt="打赏二维码" width="420" />
</p>
