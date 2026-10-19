from data.history import get_history


def calc_risk(bs_code="sh.600519"):
    hist = get_history(bs_code)

    # 【新增】：如果历史数据为空，直接返回默认值，别让它崩
    if hist is None or hist.empty or 'close' not in hist.columns:
        return {
            "60日波动率": "数据缺失",
            "60日最大回撤": "数据缺失",
            "风险等级": "无法评估"
        }

    closes = hist['close']
    returns = closes.pct_change().dropna()

    volatility = returns.tail(60).std() * 100
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
        "风险等级": level
    }