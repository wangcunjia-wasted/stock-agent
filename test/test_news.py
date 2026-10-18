import akshare as ak
df = ak.stock_zh_a_disclosure_report_cninfo(symbol="601668", start_date="20260901", end_date="20261018")
print(df.head())