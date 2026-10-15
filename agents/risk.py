from data.history import get_history


def analyze_risk(bs_code="sh.600519"):
    hist = get_history(bs_code)
    closes = hist['close']

    # 近60日波动率
    returns = closes.pct_change().dropna()
    volatility = returns.tail(60).std() * 100

    # 近60日最大回撤
    recent = closes.tail(60)
    peak = recent.cummax()
    drawdown = ((recent - peak) / peak).min() * 100

    if volatility > 3:
        level = "高风险"
    elif volatility > 1.5:
        level = "中风险"
    else:
        level = "低风险"

    return {
        "60日波动率": f"{volatility:.2f}%",
        "60日最大回撤": f"{drawdown:.2f}%",
        "风险等级": level,
    }