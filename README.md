# Stock Agent

基于多智能体（Multi-Agent）架构的智能股票分析系统。输入股票名称或代码，自动输出技术面、基本面、风控、新闻面、量化预测及红利专项的综合分析报告。

系统由两部分组成：

- **运行时多智能体分析**（LangGraph 编排，DeepSeek 驱动）：实时抓数据、各 Agent 并行分析、首席分析师汇总成报告；
- **训练侧量化管线**（`train/`）：在沪深300历史成分上做横截面机器学习（LightGBM），产出的 champion 模型供运行时 Forecast Agent 调用，给出个股在横截面中的相对位置。

## 功能

### 运行时（多智能体分析）

- **多智能体并行协作**：基于 LangGraph 并行调度，技术、基本、风控、新闻、量化预测 Agent 同时独立工作。
- **智能分诊路由（Triage）**：系统自动识别当前股票属性。若"股息率-国债利差"达标，自动唤醒【红利专项 Agent】进行深度防守分析。
- **双轨制红利计算**：结合"去年完整财年"与"滚动12个月"双重视角精准计算真实股息率，并引入十年期国债收益率进行利差定性。
- **量化预测（Forecast Agent）**：调用训练侧导出的 LightGBM 模型，在最新一期沪深300成分的横截面上打分，报告个股的排名/百分位/十分位及头部名单。只做相对排序，不做绝对涨跌预测。
- **金融级数据清洗**：自动处理 `XD`、`XR`、`DR` 等除权除息前缀，支持智能模糊搜索。
- **首席分析师汇总**：Manager Agent 综合各方意见，交叉验证（如技术面高位 vs 红利性价比），输出最终投资结论。
- **启动时自动维护**：后台线程自动"日更数据、月更模型"，失败自动冷却重试，不阻塞启动。

### 训练侧（量化管线，train/）

- **数据**：BaoStock 前复权日K（2015 年至今，快照法过滤时点成分防未来函数）+ 新浪季报基本面（法定披露截止日对齐，零前视）。
- **因子**：21 个量价因子（动量/波动/振幅/量价相关/流动性等）+ 6 个基本面因子（ROE/毛利率/净利率/负债率/营收与净利同比），32 列特征体系。
- **验证**：横截面 walk-forward（17 折），日度 RankIC/ICIR/逐日十分位、含成本多头回测（佣金+印花税+过户费+滑点）、置换检验（p=0.005）。
- **部署**：`export_champion.py` 全量训练导出模型到 `train/artifacts/`，训练/推理共用同一份因子定义（`factors.py`），保证分布一致。

## 架构

```text
run.py              # 入口：接收用户输入 + 启动后台自动维护
app.py              # Streamlit 网页版入口
graph.py            # LangGraph 编排：并行调度各 Agent 与 Manager
manager.py          # 首席分析师：综合各方意见生成最终报告
agents/             # 智能体层：每个 Agent 都有独立的大模型 Prompt 和思考逻辑
├── base.py         # LLM 统一底座
├── technical.py    # 技术面 Agent
├── fundamental.py  # 基本面 Agent（含历史估值分位）
├── risk.py         # 风控 Agent
├── news.py         # 新闻面 Agent（官方公告优先）
├── forecast.py     # 量化预测 Agent（叙述相对位置，硬约束不说绝对涨跌）
└── dividend.py     # 红利策略 Agent（股息-国债利差分析）
tools/              # 工具层：纯数据计算函数（被 Agent 调用）
├── tech_tools.py     # 均线等计算
├── fund_tools.py     # 估值计算
├── risk_tools.py     # 波动率计算
├── news_tools.py     # 巨潮公告与东财新闻抓取
├── dividend_tools.py # 股息率双轨计算与现金流排雷
├── forecast_tools.py # 横截面打分：读 panel 缓存 + champion 模型
└── auto_maintain.py  # 启动时自动维护（日更数据/月更模型，带锁与冷却）
data/               # 数据层：外部接口获取
├── realtime.py     # 新浪实时
├── valuation.py    # 腾讯 PE/PB
└── history.py      # Baostock 历史日线（带重试与缓存）
train/              # 训练侧：量化管线（与运行时通过 artifacts/ 衔接）
├── guard.py        # 守卫：阶段化日志 / FailureTracker / 异常体系
├── factors.py      # 因子定义（训练/推理唯一事实源）
├── fetch_panel.py            # 首次构建：沪深300成分日K面板（快照法时点过滤）
├── refresh_panel.py          # 日常增量：刷成分名单 + 逐股补数 + 接缝体检
├── fetch_fundamental_panel.py# 基本面季报面板
├── build_dataset_panel.py    # 因子+标签 → 面板 parquet（--horizon 参数化）
├── train_lgbm_panel.py       # walk-forward 训练（h5/h20）
├── evaluate_panel.py         # 逐日十分位 + 含成本多头回测
├── evaluate_turnover.py      # 换手优化：hold10+缓冲带等策略对比
├── blend_signal.py           # 双信号混合（固定/自适应权重）
├── permutation_test.py       # 置换检验
├── export_champion.py        # 全量训练导出 champion 模型（原子落盘）
└── build_dataset.py          # 单票数据集构造（被 panel 管线复用）
```

## 环境要求

- Python 3.11+
- DeepSeek API Key（用于 LLM 汇总）

## 安装

```bash
pip install baostock akshare langgraph openai pandas numpy requests lightgbm scikit-learn scipy pyarrow streamlit
```

## 配置

设置环境变量：

```bash
set DEEPSEEK_API_KEY=你的key
```

## 运行

命令行版：

```bash
python run.py
```

输入股票名称或代码，例如：

```
请输入股票名称或代码（如：贵州茅台 / 600519）：贵州茅台
```

或网页版：

```bash
streamlit run app.py
```

### 量化管线（可选，首次使用 Forecast Agent 需先训练）

```bash
cd train
python fetch_panel.py              # 首次构建日K面板（约几分钟）
python fetch_fundamental_panel.py  # 基本面面板
python build_dataset_panel.py      # 因子+标签
python train_lgbm_panel.py         # walk-forward 训练与评估
python export_champion.py          # 导出模型到 artifacts/
```

之后无需手动维护：`run.py` / `app.py` 启动时会自动日更数据、月更重训模型（见 `tools/auto_maintain.py`）。

## 数据来源

- 实时行情：新浪财经
- 估值与股息数据：腾讯财经、同花顺、AKShare
- 历史日线：Baostock
- 官方公告：巨潮资讯网
- 宏观无风险利率：中美国债收益率
- 基本面季报：新浪财经

## 免责声明

本项目仅供学习研究，不构成任何投资建议。模型输出为横截面相对排序，不构成对任何个股绝对涨跌的预测。
