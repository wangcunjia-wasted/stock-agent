from agents.base import ask_llm

FUND_PROMPT = """
你是一名极其资深的“全赛道”基本面研究员。
你将收到某只股票的估值数据（PE/PB分位）和核心财务数据（ROE、同比、单季环比）。

【分析框架】
1. 判断风格：
   - 如果 PE 历史分位很高（>60%），市场大概率把它当“成长股”看。此时：单季环比的动能至关重要！如果单季环比下滑，哪怕同比很高，也是“戴维斯双杀”的前兆。
   - 如果 PE 历史分位很低（<30%），市场把它当“价值股”看。此时：更看重同比的稳定性和 ROE、股息的可持续性。
2. 盈利质量：ROE > 15% 视为优秀；< 8% 为较差。
3. 陷阱警报：
   - 高估值陷阱：PE分位高 + 营收/利润同比、环比双双放缓。
   - 价值陷阱：PE分位低 但 营收/利润同比负增长。
4. 毛利率：结合毛利率变化看竞争壁垒是否被削弱。
5. ROE 折算：如果是中报/三季报数据，注意将 ROE 和毛利率结合季节性因素进行年化考量，不要因为半年 ROE 只有 8% 就误判为低质量。

请用 180 字左右，综合评估该股当前是“绝佳击球区”、“合理持有区”、还是“价值陷阱/成长失速区”。结合数据给出深刻的商业洞察。
"""


def run_fundamental_agent(code="sh600519"):
    from tools.fund_tools import calc_fundamental
    fund = calc_fundamental(code)

    user_content = f"""
    【估值数据】：
    - 市盈率(PE)：{fund.get('市盈率')} （近5年分位：{fund.get('PE近5年分位')}%）
    - 市净率(PB)：{fund.get('市净率')}
    - 系统定性：{fund.get('估值定性')}

    【核心财务动能】：
    - 净资产收益率(ROE)：{fund.get('ROE')}%
    - 营收同比增速：{fund.get('营收同比增速')}%
    - 营收环比增速：{fund.get('营收环比增速')}%
    - 净利润同比增速：{fund.get('净利润同比增速')}%
    - 净利润环比增速：{fund.get('净利润环比增速')}%
    - 销售毛利率：{fund.get('毛利率')}%
    - 每股经营现金流：{fund.get('每股经营现金流')} 元
    """

    report = ask_llm(FUND_PROMPT, user_content)
    return report