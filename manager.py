# manager.py
from agents.base import ask_llm

MANAGER_PROMPT = """
你是一名首席投资官。
你的下属刚刚提交了针对某只股票的技术面、基本面和风控报告。
请综合这三份报告，写一段 200 字左右的最终结论。
要求：，最前面告知分析的标的名称，不要罗列数据，直接说结论！综合各维度指出当前的矛盾点或共振点。
"""


def summarize_reports(tech_report: str, fund_report: str, risk_report: str) -> str:
    """Manager 专用：接收三份报告，输出最终综合报告"""

    summary = ask_llm(
        system_prompt=MANAGER_PROMPT,
        user_content=f"技术面：{tech_report}\n基本面：{fund_report}\n风控：{risk_report}"
    )
    return summary