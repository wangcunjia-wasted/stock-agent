import os
import time
import pandas as pd
import akshare as ak
import baostock as bs
from graph import app

_stock_list = None
CACHE_FILE = "stock_list_cache.csv"


def find_code(user_input):
    global _stock_list
    user_input = user_input.strip()

    # 1. 如果已经是代码，直接返回
    if user_input.startswith(("sh", "sz")) or user_input.isdigit():
        return user_input

    # 2. 加载股票列表（带每日缓存）
    if _stock_list is None:
        need_download = False
        if not os.path.exists(CACHE_FILE):
            need_download = True
        else:
            mtime = os.path.getmtime(CACHE_FILE)
            if time.strftime('%Y-%m-%d', time.localtime(mtime)) != time.strftime('%Y-%m-%d'):
                need_download = True

        if need_download:
            print("正在更新最新的股票列表...")
            try:
                _stock_list = ak.stock_info_a_code_name()
            except Exception:
                print("全量接口卡顿，尝试使用沪深备用接口...")
                sh_list = ak.stock_info_sh_name_code()
                sz_list = ak.stock_info_sz_name_code()
                _stock_list = pd.concat([sh_list, sz_list], ignore_index=True)
                _stock_list = _stock_list.rename(columns={"证券代码": "code", "证券简称": "name"})

            _stock_list.to_csv(CACHE_FILE, index=False)
        else:
            _stock_list = pd.read_csv(CACHE_FILE)

    # 3. 匹配逻辑（绝对不能漏！）
    match = _stock_list[_stock_list["name"] == user_input]
    if match.empty:
        match = _stock_list[_stock_list["name"].str.contains(user_input, na=False)]

    if match.empty:
        print(f"未找到股票：{user_input}")
        return None

    # 4. 提取代码并转换格式
    code = match.iloc[0]["code"]

    # 如果从 CSV 读出来的是数字，转成字符串并补齐 6 位（防止 000001 变成 1）
    code = str(code).zfill(6)

    if code.startswith("6"):
        return "sh" + code
    elif code.startswith(("0", "3")):
        return "sz" + code
    else:
        return "bj" + code  # 北交所


if __name__ == "__main__":
    raw = input("请输入股票名称或代码（如：贵州茅台 / 600519）：").strip()

    t0 = time.time()
    code = find_code(raw)
    t1 = time.time()
    print(f"[计时] 查代码: {t1 - t0:.2f}秒")

    if code:
        bs_code = code[:2] + "." + code[2:]
        result = app.invoke({"code": code, "bs_code": bs_code})
        t2 = time.time()
        print(f"[计时] 图流转+LLM: {t2 - t1:.2f}秒")
        print("\n最终报告\n")
        print(result["report"])

    # 程序结束时登出 Baostock，关闭 socket
    bs.logout()