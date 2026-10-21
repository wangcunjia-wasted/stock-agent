import akshare as ak
import pandas as pd
from data.valuation import get_valuation


def _calc_percentile(series: pd.Series, current_value: float) -> float:
    if series is None or series.empty or current_value is None:
        return None
    clean_series = series.dropna()
    if clean_series.empty:
        return None
    return round((clean_series < current_value).mean() * 100, 1)


def calc_fundamental(code="sh600519"):
    pure_code = code[-6:]
    prefix = "sh" if pure_code.startswith("6") else "sz"

    val = get_valuation(prefix + pure_code)
    current_pe = val.get("市盈率")
    current_pb = val.get("市净率")

    result = {
        "市盈率": current_pe, "市净率": current_pb, "总市值(亿)": val.get("总市值"),
        "PE近5年分位": None, "PB近5年分位": None, "估值定性": "未知",
        "ROE": None, "营收同比增速": None, "营收环比增速": None,
        "净利润同比增速": None, "毛利率": None, "每股经营现金流": None
    }

    # 1. 估值分位
    try:
        df_pe = ak.stock_zh_valuation_baidu(symbol=pure_code, indicator="市盈率(TTM)", period="近五年")
        if not df_pe.empty and current_pe:
            result["PE近5年分位"] = _calc_percentile(df_pe['value'], current_pe)
        df_pb = ak.stock_zh_valuation_baidu(symbol=pure_code, indicator="市净率", period="近五年")
        if not df_pb.empty and current_pb:
            result["PB近5年分位"] = _calc_percentile(df_pb['value'], current_pb)
    except:
        pass

    def safe_float(val):
        if pd.isna(val) or val == '--' or val is None or val is False: return None
        try:
            s = str(val).replace('%', '')
            if '亿' in s: return float(s.replace('亿', '')) * 100000000
            if '万' in s: return float(s.replace('万', '')) * 10000
            return float(s)
        except:
            return None

    # 2. 同花顺接口（排序后取最新）
    try:
        df_ths = ak.stock_financial_abstract_ths(symbol=pure_code, indicator="按报告期")
        if not df_ths.empty:
            # 按报告期降序排列，保证最新的在第一行
            df_ths = df_ths.sort_values(by='报告期', ascending=False).reset_index(drop=True)
            latest = df_ths.iloc[0]

            result["ROE"] = safe_float(latest.get('净资产收益率'))
            result["毛利率"] = safe_float(latest.get('销售毛利率'))
            result["每股经营现金流"] = safe_float(latest.get('每股经营现金流'))
            result["营收同比增速"] = safe_float(latest.get('营业总收入同比增长率'))
            result["净利润同比增速"] = safe_float(latest.get('净利润同比增长率'))

            # 手算单季环比（用最新数据）
            report_date = str(latest.get('报告期'))
            year = report_date[:4]
            curr_cum = safe_float(latest.get('营业总收入'))

            if '06-30' in report_date:
                q1_row = df_ths[df_ths['报告期'].str.contains(f"{year}-03-31")]
                if not q1_row.empty:
                    q1_cum = safe_float(q1_row.iloc[0].get('营业总收入'))
                    if curr_cum and q1_cum and q1_cum > 0:
                        result["营收环比增速"] = round(((curr_cum - q1_cum - q1_cum) / q1_cum) * 100, 2)
            elif '09-30' in report_date:
                q2_row = df_ths[df_ths['报告期'].str.contains(f"{year}-06-30")]
                q1_row = df_ths[df_ths['报告期'].str.contains(f"{year}-03-31")]
                if not q2_row.empty and not q1_row.empty:
                    q2_cum = safe_float(q2_row.iloc[0].get('营业总收入'))
                    q1_cum = safe_float(q1_row.iloc[0].get('营业总收入'))
                    if curr_cum and q2_cum and q1_cum and (q2_cum - q1_cum) > 0:
                        q3_single = curr_cum - q2_cum
                        q2_single = q2_cum - q1_cum
                        result["营收环比增速"] = round(((q3_single - q2_single) / q2_single) * 100, 2)
            elif '12-31' in report_date:
                q3_row = df_ths[df_ths['报告期'].str.contains(f"{year}-09-30")]
                q2_row = df_ths[df_ths['报告期'].str.contains(f"{year}-06-30")]
                if not q3_row.empty and not q2_row.empty:
                    q3_cum = safe_float(q3_row.iloc[0].get('营业总收入'))
                    q2_cum = safe_float(q2_row.iloc[0].get('营业总收入'))
                    if curr_cum and q3_cum and q2_cum and (q3_cum - q2_cum) > 0:
                        q4_single = curr_cum - q3_cum
                        q3_single = q3_cum - q2_cum
                        result["营收环比增速"] = round(((q4_single - q3_single) / q3_single) * 100, 2)
    except Exception as e:
        print(f"[警告] 同花顺财务获取失败: {e}")

    # 3. 东财接口兜底（也加上排序）
    if result["营收同比增速"] is None or result["净利润同比增速"] is None:
        try:
            df_em = ak.stock_financial_analysis_indicator(symbol=pure_code)
            if not df_em.empty:
                df_em = df_em.sort_values(by='日期', ascending=False).reset_index(drop=True)
                latest_em = df_em.iloc[0]

                if result["ROE"] is None: result["ROE"] = safe_float(latest_em.get('净资产收益率(%)'))
                if result["毛利率"] is None: result["毛利率"] = safe_float(latest_em.get('销售毛利率(%)'))
                if result["营收同比增速"] is None: result["营收同比增速"] = safe_float(
                    latest_em.get('主营业务收入增长率(%)'))
                if result["净利润同比增速"] is None: result["净利润同比增速"] = safe_float(
                    latest_em.get('净利润增长率(%)'))
                if result["每股经营现金流"] is None: result["每股经营现金流"] = safe_float(
                    latest_em.get('每股经营性现金流(元)'))
        except Exception as e:
            print(f"[警告] 东财财务获取失败: {e}")

    # 4. 估值定性
    pe_pct = result["PE近5年分位"]
    if pe_pct is not None:
        if pe_pct <= 20:
            result["估值定性"] = "历史底部（低估）"
        elif pe_pct >= 80:
            result["估值定性"] = "历史高位（高估）"
        else:
            result["估值定性"] = "历史合理区间"
    else:
        if current_pe is None:
            result["估值定性"] = "数据缺失"
        elif current_pe < 15:
            result["估值定性"] = "绝对数值偏低"
        elif current_pe < 30:
            result["估值定性"] = "绝对数值合理"
        else:
            result["估值定性"] = "绝对数值偏高"

    return result