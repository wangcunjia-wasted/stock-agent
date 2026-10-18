from langgraph.graph import StateGraph, END
from typing import TypedDict
from agents.technical import run_technical_agent
from agents.fundamental import run_fundamental_agent
from agents.risk import run_risk_agent
from agents.news import run_news_agent
from manager import summarize_reports


class State(TypedDict):
    code: str
    bs_code: str
    technical: str
    fundamental: str
    risk: str
    news: str
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
    # 传两个参数进去
    return {"news": run_news_agent(state["code"][-6:], state["bs_code"])}

def node_report(state: State):
    final_report = summarize_reports(
        tech_report=state["technical"],
        fund_report=state["fundamental"],
        risk_report=state["risk"],
        news_report=state["news"]
    )
    return {"report": final_report}


workflow = StateGraph(State)

workflow.add_node("technical", node_technical)
workflow.add_node("fundamental", node_fundamental)
workflow.add_node("risk", node_risk)
workflow.add_node("news", node_news)
workflow.add_node("report", node_report)

# 虚拟起点
workflow.add_node("start", lambda state: state)
workflow.set_entry_point("start")

# 并行分发
workflow.add_edge("start", "technical")
workflow.add_edge("start", "fundamental")
workflow.add_edge("start", "risk")
workflow.add_edge("start", "news")

# 汇聚
workflow.add_edge(["technical", "fundamental", "risk", "news"], "report")
workflow.add_edge("report", END)

app = workflow.compile()


if __name__ == "__main__":
    result = app.invoke({"code": "sh600519", "bs_code": "sh.600519"})
    print("\n最终报告\n")
    print(result["report"])