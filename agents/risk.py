from agents.base import ask_llm
from tools.risk_tools import calc_risk

RISK_PROMPT = """
你是一名极其厌恶风险的量化风控官。
根据提供的波动率和回撤数据，写出 100 字左右的风控意见。
重点提示近期的价格波动剧烈程度，以及潜在的下行风险。
"""

def run_risk_agent(bs_code="sh.600519"):
    data = calc_risk(bs_code)
    user_content = f"股票代码 {bs_code} 的最新风控数据：{data}"
    return ask_llm(RISK_PROMPT, user_content)