import akshare as ak
import pandas as pd
from data.valuation import get_valuation
from data.realtime import get_realtime


def calc_percentile(series: pd.Series, current_value: float) -> float:
    if series is None or series.empty or current_value is None:
        return None
    clean_series = series.dropna()
    if clean_series.empty:
        return None
    return round((clean_series < current_value).mean() * 100, 1)


def get_dividend_data(code="600519"):
    pure_code = code[-6:]
    prefix = "sh" if pure_code.startswith("6") else "sz"

    result = {
        "current_price": None,
        "dividend_yield_last_year": None,  # 去年完整财年股息率
        "dividend_yield_est": None,  # 滚动12个月预估年化股息率
        "bond_yield_10y": None,
        "spread": None,
        "spread_qualitative": "未知",
        "pe": None, "pb": None, "cash_flow_health": "未知"
    }

    # 1. 拿当前股价
    try:
        rt = get_realtime(prefix + pure_code)
        price = rt.get("当前价")
        if price and price > 0:
            result["current_price"] = price
    except Exception:
        pass

    # 2. 拿估值
    try:
        val = get_valuation(prefix + pure_code)
        result["pe"] = val.get("市盈率")
        result["pb"] = val.get("市净率")
    except:
        pass

    # 3. 【双轨制】计算股息率
    try:
        df_div = ak.stock_history_dividend_detail(symbol=pure_code, indicator="分红")
        if not df_div.empty:
            df_div['除权除息日'] = pd.to_datetime(df_div['除权除息日'], errors='coerce')
            df_div['派息'] = pd.to_numeric(df_div['派息'], errors='coerce').fillna(0)
            valid_divs = df_div[df_div['派息'] > 0]

            if not valid_divs.empty and result["current_price"]:
                current_year = pd.Timestamp.now().year
                last_year = current_year - 1

                # 轨道一：计算去年（完整财年）的总派息
                last_year_divs = valid_divs[
                    (valid_divs['除权除息日'] >= f"{last_year}-01-01") &
                    (valid_divs['除权除息日'] <= f"{last_year}-12-31")
                    ]
                if not last_year_divs.empty:
                    total_last_year = last_year_divs['派息'].sum() / 10.0
                    result["dividend_yield_last_year"] = round((total_last_year / result["current_price"]) * 100, 2)

                # 轨道二：计算滚动近12个月的总派息（更准的年化预估）
                latest_date = valid_divs.iloc[0]['除权除息日']
                if pd.notna(latest_date):
                    one_year_ago = latest_date - pd.Timedelta(days=365)
                    recent_12m_divs = valid_divs[valid_divs['除权除息日'] > one_year_ago]
                    if not recent_12m_divs.empty:
                        total_12m = recent_12m_divs['派息'].sum() / 10.0
                        result["dividend_yield_est"] = round((total_12m / result["current_price"]) * 100, 2)

                # 兜底：如果去年没算出来，用今年预估的
                if result["dividend_yield_last_year"] is None and result["dividend_yield_est"] is not None:
                    result["dividend_yield_last_year"] = result["dividend_yield_est"]

    except Exception as e:
        print(f"[警告] 股息率双轨计算失败: {e}")

    # 4. 国债与利差定性（用去年完整财年的股息率来算利差，最扎实）
    try:
        df_bond = ak.bond_zh_us_rate(start_date="20240101")
        bond_y = float(df_bond.iloc[-1].get('中国国债收益率10年', 0))
        result["bond_yield_10y"] = bond_y

        base_yield = result["dividend_yield_last_year"]
        if base_yield and bond_y > 0:
            spread = base_yield - bond_y
            result["spread"] = round(spread, 2)

            if spread >= 3.0:
                result["spread_qualitative"] = "高性价比区（较为低估）"
            elif spread >= 2.5:
                result["spread_qualitative"] = "性价比较为舒适区（具配置价值）"
            elif spread >= 2.0:
                result["spread_qualitative"] = "具备基础性价比"
            elif spread > 0:
                result["spread_qualitative"] = "性价比偏低（风险收益比差）"
            else:
                result["spread_qualitative"] = "股息率低于国债（极度高估）"
    except Exception:
        pass

    # 5. 现金流排雷（改用同花顺接口 + 新浪兜底）
    try:
        df_fin = ak.stock_financial_abstract_ths(symbol=pure_code, indicator="按报告期")
        if not df_fin.empty:
            cash = df_fin.iloc[0].get('每股经营现金流', None)
            if pd.notna(cash):
                result["cash_flow_health"] = cash
    except Exception:
        try:
            df_fin2 = ak.stock_financial_analysis_indicator(symbol=pure_code)
            if not df_fin2.empty:
                cash2 = df_fin2.iloc[0].get('每股经营性现金流(元)', None)
                if pd.notna(cash2):
                    result["cash_flow_health"] = cash2
        except:
            pass

    return result