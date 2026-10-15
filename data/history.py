import baostock as bs
import pandas as pd

_logged_in = False

def _ensure_login():
    global _logged_in
    if not _logged_in:
        bs.login()
        _logged_in = True

def get_history(code="sh.600519"):
    _ensure_login()
    rs = bs.query_history_k_data_plus(code,
        "date,close,volume",
        start_date='2025-01-01', end_date='2025-10-15',
        frequency="d", adjustflag="3")
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    df = pd.DataFrame(data_list, columns=rs.fields)
    df['close'] = pd.to_numeric(df['close'])
    return df