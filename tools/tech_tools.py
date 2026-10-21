from data.history import get_history


# 在 tools/tech_tools.py 里扩充 calc_ma 函数：
def calc_ma(bs_code="sh.600519"):
    hist = get_history(bs_code)
    if hist.empty: return None

    closes = hist['close']

    # 1. 均线
    ma5 = closes.rolling(5).mean().iloc[-1]
    ma20 = closes.rolling(20).mean().iloc[-1]
    current = closes.iloc[-1]

    # 2. 计算 RSI (14天)
    delta = closes.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs.iloc[-1]))

    # 3. 计算 MACD(指标有所改动，相较默认数值更敏感)
    exp12 = closes.ewm(span=10, adjust=False).mean()
    exp26 = closes.ewm(span=20, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=5, adjust=False).mean()
    macd_val = macd.iloc[-1]
    signal_val = signal.iloc[-1]

    # 判断
    trend = "多头" if current > ma20 else "空头"
    rsi_status = "超买" if rsi > 80 else ("超卖" if rsi < 20 else "正常")
    macd_status = "金叉多头" if macd_val > signal_val else "死叉空头"

    return {
        "收盘价": round(current, 2),
        "MA5": round(ma5, 2),
        "MA20": round(ma20, 2),
        "趋势": trend,
        "RSI(14)": round(rsi, 2),
        "RSI状态": rsi_status,
        "MACD状态": macd_status
    }