from agents.base import ask_llm
from tools.fund_tools import calc_fundamental

FUND_PROMPT = """
你是一名资深的基本面研究员，偏好价值投资。
根据提供的估值数据，写出 100 字左右的客观分析。
重点评价当前估值水平、是否存在安全边际或泡沫。不要给买卖建议。
"""

def run_fundamental_agent(code="sh600519"):
    data = calc_fundamental(code)
    user_content = f"股票代码 {code} 的最新基本面数据：{data}"
    return ask_llm(FUND_PROMPT, user_content)