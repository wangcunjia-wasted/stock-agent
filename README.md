# Stock Agent

基于多智能体（Multi-Agent）架构的股票分析系统。输入股票名称或代码，自动输出技术面、基本面、风控三个维度的综合分析报告。

## 功能

- **多智能体协作**：技术面、基本面、风控三个 Agent 拥有独立的大模型 Prompt，独立分析。
- **并行执行**：基于 LangGraph 并行调度，大幅缩短整体分析耗时。
- **首席分析师汇总**：Manager Agent 综合三方意见，指出矛盾点，输出最终投资结论。
- **金融级数据清洗**：自动处理 `XD`、`XR`、`DR` 等除权除息前缀，支持智能模糊搜索。
- **本地缓存机制**：股票列表与历史数据双重缓存，避免频繁请求接口导致限流。

## 架构

本项目采用 **多智能体（Multi-Agent）协作架构**，通过 LangGraph 进行流程编排：

```text
run.py              # 入口：接收用户输入
graph.py            # LangGraph 编排：串联各个 Agent 与 Manager
manager.py          # 首席分析师：综合各方意见生成最终报告
agents/             # 智能体层：每个 Agent 都有独立的大模型 Prompt 和思考逻辑
├── base.py         # LLM 统一底座
├── technical.py    # 技术面 Agent
├── fundamental.py  # 基本面 Agent
├── risk.py         # 风控 Agent
└── news.py         # 新闻面 Agent（结合官方公告与媒体资讯）
tools/              # 工具层：纯数据计算函数（被 Agent 调用）
├── tech_tools.py   # 均线等计算
├── fund_tools.py   # 估值计算
└── risk_tools.py   # 波动率计算
data/               # 数据层：外部接口获取
├── realtime.py     # 新浪实时
├── valuation.py    # 腾讯 PE/PB
└── history.py      # Baostock 历史日线（带重试与缓存）
```

## 环境要求

- Python 3.11+
- DeepSeek API Key（用于 LLM 汇总）

## 安装

```bash
pip install baostock akshare langgraph openai
```

## 配置

设置环境变量：

```bash
set DEEPSEEK_API_KEY=你的key
```

## 运行

```bash
python run.py
```

输入股票名称或代码，例如：

```
请输入股票名称或代码（如：贵州茅台 / 600519）：贵州茅台
```

## 示例输出

> 最终结论：该股当前处于强势趋势与高估值、高波动相互拉扯的阶段。技术面多头动能与风控警示的下行风险形成明显矛盾：价格趋势强，但短线拉升偏急，追高性价比低；基本面估值已隐含较高增长预期，安全边际有限，若业绩兑现不及预期，估值回落可能放大波动。三者共振的风险点是高位追涨与回撤扩大，而非趋势立即反转。操作上，不建议追高；已持仓者应控制仓位并严格执行止损，跌破关键支撑优先减仓；未持仓者等待回调企稳或业绩验证后再考虑，风险承受能力不足者回避。

## 数据来源

- 实时行情：新浪财经
- 估值数据：腾讯财经
- 历史日线：Baostock
- 股票列表：AKShare

## 免责声明

本项目仅供学习研究，不构成任何投资建议。