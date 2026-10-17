from agents.base import ask_llm
from tools.tech_tools import calc_ma

# 技术面 Agent 的人格
TECH_PROMPT = """
你是一名资深的技术面交易员。
你的任务是根据提供的K线数据，写出犀利、客观的技术面分析。
字数控制在 100 字左右。不要给买卖建议，只描述当前趋势和动能。
"""


def run_technical_agent(bs_code="sh.600519"):
    # 1. 先拿数据
    data = calc_ma(bs_code)
    if not data:
        return "技术面数据获取失败。"

    # 2. 组织语言
    user_content = f"股票代码 {bs_code} 的最新指标：{data}"
    report = ask_llm(TECH_PROMPT, user_content)

    return report