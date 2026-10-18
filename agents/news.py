from agents.base import ask_llm
from tools.tech_tools import calc_ma
from tools.news_tools import fetch_recent_news
from datetime import datetime

NEWS_PROMPT = """
你是一名极其敏锐的金融情报分析师。
你将同时收到：某只股票的【当前技术面位置】和【近期新闻列表】。

你的任务是判断新闻对股价的潜在影响，并防范“接盘”风险。

【核心要求】
1. 识别“滞后与兑现”：如果新闻一片利好，但技术面已经处于高位（如短期涨幅极大、远离均线），请明确警告“利好出尽/高位接盘”风险。反之，如果技术面已在底部，利空新闻可能是“利空出尽”。
2. 判断情绪：结合技术位置，判断新闻整体对未来的影响是【偏多】、【偏空】还是【中性】。
3. 提炼核心催化剂：指出最关键的一两条新闻。
4. 注意对比新闻发布时间和当前时间的距离，如果新闻已经发布超过3天，请明确说明‘该新闻已发布数日，可能已被市场消化’。

字数控制在 150 字以内，语言要客观犀利。
"""


def run_news_agent(code="600519", bs_code="sh.600519"):
    # 1. 拿新闻
    news_data = fetch_recent_news(code)

    # 2. 拿技术面位置（判断是否处于高位）
    tech_data = calc_ma(bs_code)
    tech_position = "数据缺失"
    if tech_data:
        tech_position = f"当前收盘价 {tech_data['收盘价']}，MA5 {tech_data['MA5']}，MA20 {tech_data['MA20']}，趋势属于 {tech_data['趋势']}。"

    # 注入当前时间
    current_date = datetime.now().strftime('%Y-%m-%d')

    # 3. 一起喂给 LLM
    user_content = f"""
    【当前时间】：{current_date}
    【技术位置】：{tech_position}

    【近期新闻】：
    {news_data}
    """

    report = ask_llm(NEWS_PROMPT, user_content)
    return report