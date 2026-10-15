import time
from openai import OpenAI
from agents.technical import analyze_technical
from agents.fundamental import analyze_fundamental
from agents.risk import analyze_risk

client = OpenAI(
    api_key="DEEPSEEK_API_KEY",
    base_url="https://api.deepseek.com",
    timeout=60.0,
    max_retries=2
)


def generate_report(code="sh600519", bs_code="sh.600519"):
    t0 = time.time()
    tech = analyze_technical(bs_code)
    t1 = time.time()
    print(f"[计时] 技术面: {t1 - t0:.2f}秒")

    fund = analyze_fundamental(code)
    t2 = time.time()
    print(f"[计时] 基本面: {t2 - t1:.2f}秒")

    risk = analyze_risk(bs_code)
    t3 = time.time()
    print(f"[计时] 风控: {t3 - t2:.2f}秒")

    prompt = f"""你是一名投资研究分析师。以下是某只股票的结构化分析数据：

    【技术面】
    收盘价：{tech['收盘价']}
    MA5：{tech['MA5']}
    MA20：{tech['MA20']}
    趋势判断：{tech['趋势']}

    【基本面】
    市盈率：{fund['市盈率']}
    市净率：{fund['市净率']}
    总市值：{fund['总市值']}亿
    估值判断：{fund['估值判断']}

    【风控】
    60日波动率：{risk['60日波动率']}
    60日最大回撤：{risk['60日最大回撤']}
    风险等级：{risk['风险等级']}

    请用一段话（200字以内）综合以上三个维度，给出对该股票当前状态的判断。不要给买卖建议，只描述状态。
    """

    resp = client.chat.completions.create(
        model="deepseek-flash",
        messages=[{"role": "user", "content": prompt}]
    )
    t4 = time.time()
    print(f"[计时] LLM: {t4 - t3:.2f}秒")

    return resp.choices[0].message.content