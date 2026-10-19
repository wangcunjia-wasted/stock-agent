import akshare as ak
import pandas as pd
from data.valuation import get_valuation


def _calc_percentile(series: pd.Series, current_value: float) -> float:
    """内部辅助函数：计算当前值在历史序列中的分位数"""
    if series is None or series.empty or current_value is None:
        return None
    clean_series = series.dropna()
    if clean_series.empty:
        return None
    percentile = (clean_series < current_value).mean() * 100
    return round(percentile, 1)


def calc_fundamental(code="sh600519"):
    pure_code = code[-6:]

    # 1. 拿当前的 PE/PB
    val = get_valuation(code)
    current_pe = val["市盈率"]
    current_pb = val["市净率"]

    result = {
        "市盈率": current_pe,
        "市净率": current_pb,
        "总市值": val["总市值"],
        "PE近5年分位": None,
        "PB近5年分位": None,
        "估值定性": "未知"  # 先占位，后面算完分位再填
    }

    # 2. 拿历史数据算分位（PE/PB）
    try:
        # 换成百度股市通的接口
        df_pe = ak.stock_zh_valuation_baidu(symbol=pure_code, indicator="市盈率(TTM)", period="近五年")
        if not df_pe.empty:
            current_pe = result["市盈率"]
            if current_pe:
                result["PE近5年分位"] = _calc_percentile(df_pe['value'], current_pe)

        df_pb = ak.stock_zh_valuation_baidu(symbol=pure_code, indicator="市净率", period="近五年")
        if not df_pb.empty:
            current_pb = result["市净率"]
            if current_pb:
                result["PB近5年分位"] = _calc_percentile(df_pb['value'], current_pb)
    except Exception as e:
        print(f"[警告] 获取历史估值分位失败: {e}")

    # 【重写定性逻辑】：优先用分位数判断，没有分位数再用绝对值兜底
    pe_pct = result["PE近5年分位"]
    if pe_pct is not None:
        if pe_pct <= 20:
            result["估值定性"] = "处于历史底部区（低估）"
        elif pe_pct >= 80:
            result["估值定性"] = "处于历史高位区（高估）"
        else:
            result["估值定性"] = "处于历史合理区间"
    else:
        # 如果分位数没算出来（比如次新股），再用死板的绝对值兜底
        if current_pe is None:
            result["估值定性"] = "数据缺失"
        elif current_pe < 15:
            result["估值定性"] = "绝对数值偏低"
        elif current_pe < 30:
            result["估值定性"] = "绝对数值合理"
        else:
            result["估值定性"] = "绝对数值偏高"

    return result