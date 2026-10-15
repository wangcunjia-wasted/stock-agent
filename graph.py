from langgraph.graph import StateGraph, END
from typing import TypedDict
from agents.technical import analyze_technical
from agents.fundamental import analyze_fundamental
from agents.risk import analyze_risk
from agents.report import generate_report


class State(TypedDict):
    code: str
    bs_code: str
    technical: dict
    fundamental: dict
    risk: dict
    report: str


def node_technical(state: State):
    return {"technical": analyze_technical(state["bs_code"])}

def node_fundamental(state: State):
    return {"fundamental": analyze_fundamental(state["code"])}

def node_risk(state: State):
    return {"risk": analyze_risk(state["bs_code"])}

def node_report(state: State):
    return {"report": generate_report(state["code"], state["bs_code"])}


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
    print(result["report"])