from data.history import get_history


def calc_ma(bs_code="sh.600519"):
    """纯计算工具：返回均线和趋势数据"""
    hist = get_history(bs_code)
    if hist.empty:
        return None

    ma5 = hist['close'].rolling(5).mean().iloc[-1]
    ma20 = hist['close'].rolling(20).mean().iloc[-1]
    current = hist['close'].iloc[-1]

    if current > ma5 > ma20:
        trend = "强势多头"
    elif current > ma20:
        trend = "震荡偏多"
    elif current < ma5 < ma20:
        trend = "弱势空头"
    else:
        trend = "震荡偏空"

    return {
        "收盘价": round(current, 2),
        "MA5": round(ma5, 2),
        "MA20": round(ma20, 2),
        "趋势": trend
    }