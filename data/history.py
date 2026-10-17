import baostock as bs
import pandas as pd
from datetime import datetime, timedelta
import time

_logged_in = False
_cache = {}


def _ensure_login():
    global _logged_in
    if not _logged_in:
        try:
            bs.login()
            _logged_in = True
        except Exception as e:
            print(f"[警告] Baostock 登录失败: {e}")


def get_history(code="sh.600519"):
    _ensure_login()

    if code in _cache:
        return _cache[code]

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')

    # 最多重试 3 次
    for attempt in range(3):
        try:
            rs = bs.query_history_k_data_plus(code,
                                              "date,close,volume,high,low",
                                              start_date=start_date,
                                              end_date=end_date,
                                              frequency="d",
                                              adjustflag="2")
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())

            # 如果返回空数据，说明代码不对或者停牌了
            if not data_list:
                print(f"[警告] {code} 未获取到历史数据")
                return pd.DataFrame()

            df = pd.DataFrame(data_list, columns=rs.fields)
            for col in ['close', 'volume', 'high', 'low']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])

            _cache[code] = df
            return df

        except Exception as e:
            print(f"[警告] 获取 {code} 历史数据失败 (第 {attempt + 1} 次尝试): {e}")
            if attempt == 2:
                print("[错误] Baostock 接口彻底崩溃，返回空数据。")
                return pd.DataFrame()
            time.sleep(1)  # 等待 1 秒后重试