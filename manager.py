from agents.base import ask_llm

MANAGER_PROMPT = """
你是一名首席投资官。
你的四位下属（技术面、基本面、风控、新闻面）刚刚提交了报告。
请综合这四份报告，写一段 250 字左右的最终结论。

【特别注意】
1. 新闻面可能存在“滞后接盘”风险，请将新闻情绪与技术面、资金面交叉验证。如果技术面已经在高位，且新闻是一片利好，你要特别警惕“利好出尽”。
2. 不要罗列数据，直接说结论！指出当前的核心矛盾或共振点。
"""


def summarize_reports(tech_report: str, fund_report: str, risk_report: str, news_report: str) -> str:
    print("  -> [Manager] 首席分析师正在汇总四位专家意见...")

    summary = ask_llm(
        system_prompt=MANAGER_PROMPT,
        user_content=f"技术面：{tech_report}\n基本面：{fund_report}\n风控：{risk_report}\n新闻面：{news_report}"
    )
    return summary