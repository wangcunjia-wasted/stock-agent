from agents.base import ask_llm
from tools.forecast_tools import get_forecast

# 量化预测 Agent 的人格
FORECAST_PROMPT = """
你是多智能体投研系统里的量化预测分析师（Forecast Agent）。
你的依据是一个沪深300横截面选股模型（LightGBM，21个量价因子，预测未来5个
交易日的相对强弱）。它的样本外成绩：日度RankIC约+0.05、ICIR约0.33——
统计显著但强度温和。它擅长的是「同一天300只成分股里谁比谁强」的相对排序，
不擅长、也不做绝对涨跌预测。

写作要求：
1. 100字以内。客观陈述该股在最新一期成分股横截面中的相对位置
   （排名、百分位、十分位），并可点出头部名单里是否出现相关板块的联动信号。
2. 绝对不要说「该股将上涨/下跌」——模型只做相对排序。
   规范表述如「在300只成分股中排名靠后（后31%），模型对其5日相对表现偏谨慎」。
3. 排名处于前10%或后10%时可以适度强调相对倾向，但必须保留
   「统计信号、非确定预测」性质的措辞。
4. 不给任何买卖建议。注明评分数据日期。
"""


def run_forecast_agent(code: str = "sh600519", bs_code: str = "sh.600519") -> str:
    # 1. 拿数据（模型打分 + 横截面位置）
    data, err = get_forecast(bs_code)
    if err:
        return f"量化预测暂不可用：{err}"

    # 2. 组织语言
    user_content = (
        f"股票 {bs_code} 的量化横截面评分结果：\n"
        f"- 评分数据日期：{data['score_date']}（成分快照 {data['snap_date']}）\n"
        f"- 在 {data['total']} 只成分股中排名第 {data['rank']} 位"
        f"（前 {data['percentile']}%，十分位第 {data['decile']} 档，1=最强）\n"
        f"- 模型预测分：{data['target_score']}（横截面相对分数，无绝对含义）\n"
        f"- 模型最看好的前10只：{', '.join(data['top10'])}\n"
        f"- 预测视界：未来{data['horizon_days']}个交易日的相对表现"
    )
    report = ask_llm(FORECAST_PROMPT, user_content)
    return report
