from agents.base import ask_llm

DIVIDEND_PROMPT = """
你是一名极度看重现金流的“红利策略”基金经理。
你的投资哲学是：将高股息股票视为债券的替代品，但极度防范“价值陷阱”。

【核心判定逻辑（基于2022年以来的低利率新常态）】
1. 息差定性（股息率 - 10年期国债收益率）：
   - >= 3.0%：高性价比区域，往往是好买点。
   - 2.5% - 3.0%：性价比较为舒适，长期资金（险资/社保）的配置区。
   - 2.0% - 2.5%：具备基础性价比。
   - < 2.0%：风险收益比不划算，不建议重仓。
2. 股息率的历史印证：对比“去年完整财年”和“滚动12个月”的股息率，看分红是否具有持续性。
3. 基本面排雷：高股息必须建立在“现金流健康”前提下。如果经营性现金流极差（如为负或极低），坚决不买。

请根据以下数据，用 150 字左右写出一段专业判断：当前处于买点、持有区、还是高估卖出区？并给出明确理由。
"""


def run_dividend_agent(code="600519", pre_fetched_data=None):
    # 如果外面没传数据，才自己去拉
    if pre_fetched_data is None:
        from tools.dividend_tools import get_dividend_data
        div = get_dividend_data(code)
    else:
        div = pre_fetched_data

    # 更新了这里的文案，适配双轨制数据
    user_content = f"""
    【红利与估值数据】：
    - 去年完整财年股息率：{div.get('dividend_yield_last_year')}%
    - 滚动12个月预估股息率：{div.get('dividend_yield_est')}%
    - 10年期国债收益率：{div.get('bond_yield_10y')}%
    - 股息-国债利差（基于去年）：{div.get('spread')}% （系统定性：{div.get('spread_qualitative')}）
    - 当前 PE：{div.get('pe')}，PB：{div.get('pb')}

    【基本面排雷数据】：
    - 每股经营性现金流：{div.get('cash_flow_health')}
    """

    return ask_llm(DIVIDEND_PROMPT, user_content)