# Stock Agent

基于多 Agent 架构的股票分析系统。输入股票名称或代码，自动输出技术面、基本面、风控三个维度的综合分析报告。

## 功能

- **技术面分析**：基于历史日线计算 MA5、MA20，判断趋势
- **基本面分析**：获取市盈率、市净率、总市值，判断估值水平
- **风控分析**：计算 60 日波动率、最大回撤，评估风险等级
- **LLM 汇总**：三个维度的结构化结果交由大模型生成自然语言报告

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
└── risk.py         # 风控 Agent
tools/              # 工具层：纯数据计算函数（被 Agent 调用）
├── tech_tools.py   # 均线等计算
├── fund_tools.py   # 估值计算
└── risk_tools.py   # 波动率计算
data/               # 数据层：外部接口获取
├── realtime.py     # 新浪实时
├── valuation.py    # 腾讯 PE/PB
└── history.py      # Baostock 历史日线
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

> 该股当前收盘价 1462.0 元，处于 MA5（1439.8）上方，但略低于 MA20（1466.31），技术面呈现震荡偏空格局。基本面方面，市盈率 20.16 倍、市净率 6.53 倍，总市值 16413.2 亿元，估值处于合理区间。风控维度显示，60 日波动率仅 1.03%，最大回撤 -6.85%，风险等级为低。综合来看，该股处于估值合理、低风险、但短期技术面偏弱的震荡整理状态。

## 数据来源

- 实时行情：新浪财经
- 估值数据：腾讯财经
- 历史日线：Baostock
- 股票列表：AKShare

## 免责声明

本项目仅供学习研究，不构成任何投资建议。