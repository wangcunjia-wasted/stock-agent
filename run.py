import os
import time
import re
import pandas as pd
import akshare as ak
import baostock as bs
from graph import app

_stock_list = None
CACHE_FILE = "stock_list_cache.csv"


def find_code(user_input):
    global _stock_list
    user_input = user_input.strip().upper()

    # 1. 如果是带字母的代码，或者纯数字代码，直接返回
    if user_input.startswith(("SH", "SZ")) or user_input.isdigit():
        return user_input.lower()

    # 2. 清洗前缀：把开头的 XD, XR, DR, N, C 去掉
    clean_input = re.sub(r'^(XD|XR|DR|N|C)', '', user_input).strip()
    if not clean_input:
        clean_input = user_input

    # 3. 加载股票列表（极简版：有缓存就直接用）
    if _stock_list is None:
        if os.path.exists(CACHE_FILE):
            _stock_list = pd.read_csv(CACHE_FILE)
        else:
            print("首次运行，正在拉取全市场股票列表...")
            try:
                _stock_list = ak.stock_info_a_code_name()
            except Exception:
                print("全量接口卡顿，尝试使用沪深备用接口...")
                sh_list = ak.stock_info_sh_name_code()
                sz_list = ak.stock_info_sz_name_code()
                _stock_list = pd.concat([sh_list, sz_list], ignore_index=True)
                _stock_list = _stock_list.rename(columns={"证券代码": "code", "证券简称": "name"})

            _stock_list.to_csv(CACHE_FILE, index=False)

    # 4. 匹配逻辑（智能模糊搜索）
    # 第一层：精确匹配
    match = _stock_list[_stock_list["name"] == clean_input]

    # 第二层：前缀匹配（比如输入"中国平"，匹配"中国平安"）
    if match.empty:
        match = _stock_list[_stock_list["name"].str.startswith(clean_input, na=False)]

    # 第三层：包含匹配（比如输入"平安"，匹配"中国平安"）
    if match.empty:
        match = _stock_list[_stock_list["name"].str.contains(clean_input, na=False)]

    # 第四层：逆向截断匹配（解决除权截断，比如"中国巨石"去匹配表里的"XD中国巨"）
    if match.empty:
        for length in range(len(clean_input) - 1, 1, -1):
            short_input = clean_input[:length]
            match = _stock_list[_stock_list["name"].str.contains(short_input, na=False)]
            if not match.empty:
                break

    # 终极判定：没找到
    if match.empty:
        print(f"未找到股票：{user_input}")
        return None

    # 终极判定：匹配到多个，让用户自己选（防止瞎猜）
    if len(match) > 1:
        print(f"您输入的 '{user_input}' 匹配到了多只股票，请指明具体是哪一只：")
        for idx, row in match.head(5).iterrows():
            print(f"  - {row['name']} ({row['code']})")
        return None

    # 5. 提取代码并格式化
    raw_code = match.iloc[0]["code"]

    # 强制转换为浮点数再转整数，彻底消灭可能存在的 .0
    try:
        code_int = int(float(raw_code))
        code = str(code_int).zfill(6)
    except:
        code = str(raw_code).split('.')[0].zfill(6)

    if code.startswith("6"):
        return "sh" + code
    elif code.startswith(("0", "3")):
        return "sz" + code
    else:
        return "bj" + code


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
        print("\n================ 最终报告 ================\n")
        print(result["report"])

    # 程序结束时登出 Baostock，安全保护
    try:
        bs.logout()
    except Exception:
        pass