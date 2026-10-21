import akshare as ak
import pandas as pd


def check_dates():
    code = "300502"  # 新易盛
    print(f"========== 检查 {code} 的财报日期 ==========\n")

    # 1. 检查同花顺接口
    print("【同花顺接口原始数据】")
    try:
        df_ths = ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")
        if not df_ths.empty:
            # 我们先按你代码里的方式排序
            df_ths = df_ths.sort_values(by='报告期', ascending=False).reset_index(drop=True)

            # 打印前三行最关键的列
            cols_to_show = ['报告期', '营业总收入', '营业总收入同比增长率', '净利润同比增长率']
            # 过滤出存在的列，防止部分股票缺列报错
            show_cols = [c for c in cols_to_show if c in df_ths.columns]

            print(df_ths[show_cols].head(3))
            print("\n  --> 取到的第一行报告期是:", df_ths.iloc[0]['报告期'])
    except Exception as e:
        print("同花顺接口报错:", e)

    print("\n" + "=" * 40 + "\n")

    # 2. 检查东财接口
    print("【东方财富接口原始数据】")
    try:
        df_em = ak.stock_financial_analysis_indicator(symbol=code)
        if not df_em.empty:
            df_em = df_em.sort_values(by='日期', ascending=False).reset_index(drop=True)

            cols_to_show = ['日期', '主营业务收入增长率(%)', '净利润增长率(%)', '每股经营性现金流(元)']
            show_cols = [c for c in cols_to_show if c in df_em.columns]

            print(df_em[show_cols].head(3))
            print("\n  --> 取到的第一行日期是:", df_em.iloc[0]['日期'])
    except Exception as e:
        print("东财接口报错:", e)


if __name__ == "__main__":
    check_dates()
