from langgraph.graph import StateGraph, END
from typing import TypedDict
from agents.technical import run_technical_agent
from agents.fundamental import run_fundamental_agent
from agents.risk import run_risk_agent
from manager import summarize_reports


class State(TypedDict):
    code: str
    bs_code: str
    technical: str
    fundamental: str
    risk: str
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

def node_report(state: State):
    print("  -> [Manager] 首席分析师正在汇总三位专家意见...")
    summary = summarize_reports(
        tech_report=state["technical"],
        fund_report=state["fundamental"],
        risk_report=state["risk"]
    )
    return {"report": summary}


# 构建图
workflow = StateGraph(State)

# 1. 添加节点
workflow.add_node("technical", node_technical)
workflow.add_node("fundamental", node_fundamental)
workflow.add_node("risk", node_risk)
workflow.add_node("report", node_report)

# 2. 设置并行：让三个节点作为图的入口
# 在 LangGraph 中，只要节点没有入边，并且被连接到同一个后继节点，它们就会并行执行。
# 我们可以用一个虚拟的 "start" 节点来分发。
workflow.add_node("start", lambda state: state)

workflow.set_entry_point("start")
workflow.add_edge("start", "technical")
workflow.add_edge("start", "fundamental")
workflow.add_edge("start", "risk")

# 3. 汇聚：三个节点的结果都送给 report
# 汇聚：等这三个节点全部执行完毕，统一触发一次 report
workflow.add_edge(["technical", "fundamental", "risk"], "report")

workflow.add_edge("report", END)

app = workflow.compile()


if __name__ == "__main__":
    result = app.invoke({"code": "sh600519", "bs_code": "sh.600519"})
    print("\n最终报告\n")
    print(result["report"])