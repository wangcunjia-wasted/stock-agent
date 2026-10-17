from langgraph.graph import StateGraph, END
from typing import TypedDict
from agents.technical import run_technical_agent
from agents.fundamental import run_fundamental_agent
from agents.risk import run_risk_agent
from manager import summarize_reports  # 引入 Manager


class State(TypedDict):
    code: str
    bs_code: str
    technical: str
    fundamental: str
    risk: str
    report: str


def node_technical(state: State):
    print("  -> [Graph] 调度技术面 Agent...")
    return {"technical": run_technical_agent(state["bs_code"])}

def node_fundamental(state: State):
    print("  -> [Graph] 调度基本面 Agent...")
    return {"fundamental": run_fundamental_agent(state["code"])}

def node_risk(state: State):
    print("  -> [Graph] 调度风控 Agent...")
    return {"risk": run_risk_agent(state["bs_code"])}

def node_report(state: State):
    # 把三个 Agent 的结果交给 Manager 汇总
    final_report = summarize_reports(
        tech_report=state["technical"],
        fund_report=state["fundamental"],
        risk_report=state["risk"]
    )
    return {"report": final_report}


# 构建图
workflow = StateGraph(State)

workflow.add_node("technical", node_technical)
workflow.add_node("fundamental", node_fundamental)
workflow.add_node("risk", node_risk)
workflow.add_node("report", node_report)

workflow.set_entry_point("technical")
workflow.add_edge("technical", "fundamental")
workflow.add_edge("fundamental", "risk")
workflow.add_edge("risk", "report")
workflow.add_edge("report", END)

app = workflow.compile()


if __name__ == "__main__":
    result = app.invoke({"code": "sh600519", "bs_code": "sh.600519"})
    print("\n最终报告\n")
    print(result["report"])