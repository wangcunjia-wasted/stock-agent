from agents.base import ask_llm

MANAGER_PROMPT = """
你是一名首席投资官。
你的下属提交了关于某只股票的报告。可能包含技术、基本、风控、新闻，以及【红利专项分析】。

请综合所有报告，写一段 250 字左右的最终结论。

【特别注意】
1. 如果附带了【红利专项分析】，请重点考量“股息-国债利差”和“现金流”，判断该股是否适合作为防御性资产配置。
2. 如果红利分析与技术面发生冲突（比如红利显示极具性价比，但技术面在暴跌），请指出这种矛盾，并提示风险。
3. 不要罗列数据，直接说结论！
"""


def summarize_reports(tech_report: str, fund_report: str, risk_report: str, news_report: str, div_report: str) -> str:
    print("  -> [Manager] 首席分析师正在汇总专家意见...")

    user_content = f"""
    【技术面】：{tech_report}
    【基本面】：{fund_report}
    【风控】：{risk_report}
    【新闻面】：{news_report}
    【红利专项】：{div_report}
    """

    summary = ask_llm(MANAGER_PROMPT, user_content)
    return summary