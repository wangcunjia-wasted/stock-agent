from langgraph.graph import StateGraph, END
from typing import TypedDict
from agents.technical import run_technical_agent
from agents.fundamental import run_fundamental_agent
from agents.risk import run_risk_agent
from agents.news import run_news_agent
from agents.dividend import run_dividend_agent
from agents.forecast import run_forecast_agent
from tools.dividend_tools import get_dividend_data
from manager import summarize_reports


class State(TypedDict):
    code: str
    bs_code: str
    div_data: dict  # 分诊台查到的红利基础数据
    technical: str
    fundamental: str
    risk: str
    news: str
    dividend: str
    forecast: str   # 量化预测 Agent：沪深300横截面相对强弱（Forecast Agent）
    report: str


def node_technical(state: State):
    print("  -> [Graph] 技术面 Agent 启动...")
    return {"technical": run_technical_agent(state["bs_code"])}


def node_fundamental(state: State):
    print("  -> [Graph] 基本面 Agent 启动...")
    return {"fundamental": run_fundamental_agent(state["code"])}


def node_risk(state: State):
    print("  -> [Graph] 风控 Agent 启动...")
    return {"risk": run_risk_agent(state["bs_code"])}


def node_news(state: State):
    print("  -> [Graph] 新闻面 Agent 启动...")
    return {"news": run_news_agent(state["code"][-6:], state["bs_code"])}


def node_dividend(state: State):
    print("  -> [Graph] 系统识别为红利股，启动红利专项 Agent...")
    # 不要传 pre_fetched_data 了，让它自己去拉
    return {"dividend": run_dividend_agent(state["code"])}


def node_forecast(state: State):
    print("  -> [Graph] 量化预测 Agent 启动...")
    try:
        return {"forecast": run_forecast_agent(state["code"], state["bs_code"])}
    except Exception as e:
        # 量化预测是增量能力：模型/缓存缺失只降级，不拖垮整张图
        print(f"  -> [Graph] 量化预测失败（降级为空，不影响其他分析）: {e}")
        return {"forecast": ""}


def node_skip_dividend(state: State):
    print("  -> [Graph] 股息利差较低，跳过红利专项分析...")
    # 【修改这里】：不要给 Manager 传提示文本，直接传空
    return {"dividend": ""}

def node_report(state: State):
    final_report = summarize_reports(
        tech_report=state["technical"],
        fund_report=state["fundamental"],
        risk_report=state["risk"],
        news_report=state["news"],
        div_report=state["dividend"],
        forecast_report=state.get("forecast", "")
    )
    return {"report": final_report}


# ================= 构建图 =================
workflow = StateGraph(State)

workflow.add_node("technical", node_technical)
workflow.add_node("fundamental", node_fundamental)
workflow.add_node("risk", node_risk)
workflow.add_node("news", node_news)
workflow.add_node("dividend", node_dividend)
workflow.add_node("skip_dividend", node_skip_dividend)
workflow.add_node("forecast", node_forecast)
workflow.add_node("report", node_report)

# 1. 虚拟起点
workflow.add_node("start", lambda state: state)
workflow.set_entry_point("start")


# 2. 分诊判断逻辑（路由器）
def triage(state: State):
    div_info = get_dividend_data(state["code"])
    spread = div_info.get("spread", 0)

    if spread is not None and spread >= 2.0:
        return "dividend"
    else:
        return "skip_dividend"


workflow.add_conditional_edges(
    "start",
    triage,
    {
        "dividend": "dividend",
        "skip_dividend": "skip_dividend"
    }
)

# 3. 并行分发基础 Agent（它们不管是不是红利，都要跑）
workflow.add_edge("start", "technical")
workflow.add_edge("start", "fundamental")
workflow.add_edge("start", "risk")
workflow.add_edge("start", "news")
workflow.add_edge("start", "forecast")   # 量化预测：沪深300成分股横截面打分

# 4. 汇聚到 report
workflow.add_edge(["technical", "fundamental", "risk", "news", "forecast", "dividend"], "report")
workflow.add_edge(["technical", "fundamental", "risk", "news", "forecast", "skip_dividend"], "report")
workflow.add_edge("report", END)

app = workflow.compile()


if __name__ == "__main__":
    result = app.invoke({"code": "sh600519", "bs_code": "sh.600519"})
    print("\n最终报告\n")
    print(result["report"])