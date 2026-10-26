from agents.base import ask_llm

MANAGER_PROMPT = """
你是一名首席投资官。
你的下属提交了关于某只股票的报告。可能包含技术、基本、风控、新闻、【红利专项分析】，以及【量化预测】。

请综合所有报告，写一段 250 字左右的最终结论。

【特别注意】
1. 如果附带了【红利专项分析】，请重点考量“股息-国债利差”和“现金流”，判断该股是否适合作为防御性资产配置。
2. 如果红利分析与技术面发生冲突（比如红利显示极具性价比，但技术面在暴跌），请指出这种矛盾，并提示风险。
3. 如果未提供【红利专项分析】，说明该股不是红利标的，请**绝对不要**提及“股息”、“红利”、“防御资产”等字眼，直接基于技术、基本面、风控和新闻面给出结论。
4. 如果附带了【量化预测】：它是量化模型在沪深300成分股横截面中的相对强弱排序（统计信号，非确定预测），可作为技术面的旁证；若与技术面方向矛盾，请简要指出这种分歧。
5. 如果未提供【量化预测】，**绝对不要**提及“模型”、“量化排序”等字眼。
6. 不要罗列数据，直接说结论！
"""


def summarize_reports(tech_report: str, fund_report: str, risk_report: str,
                      news_report: str, div_report: str,
                      forecast_report: str = "") -> str:
    print("  -> [Manager] 首席分析师正在汇总专家意见...")

    # 【核心修改】：动态拼装数据
    user_content = f"""
    【技术面】：{tech_report}
    【基本面】：{fund_report}
    【风控】：{risk_report}
    【新闻面】：{news_report}
    """

    # 只有在真的触发了红利分析，才把这段加进去
    if div_report and div_report.strip() != "":
        user_content += f"\n【红利专项】：{div_report}"

    # 量化预测（Forecast Agent）：横截面相对强弱信号，缺失时不出现
    if forecast_report and forecast_report.strip() != "":
        user_content += f"\n【量化预测】：{forecast_report}"

    summary = ask_llm(MANAGER_PROMPT, user_content)
    return summary