import akshare as ak
from graph import app
import baostock as bs
import time

_stock_list = None


def find_code(user_input):
    global _stock_list
    user_input = user_input.strip()

    if user_input.startswith(("sh", "sz")) or user_input.isdigit():
        return user_input

    if _stock_list is None:
        print("正在加载股票列表...")
        _stock_list = ak.stock_info_a_code_name()

    match = _stock_list[_stock_list["name"] == user_input]
    if match.empty:
        match = _stock_list[_stock_list["name"].str.contains(user_input, na=False)]

    if match.empty:
        print(f"未找到股票：{user_input}")
        return None

    code = match.iloc[0]["code"]
    return ("sh" if code.startswith("6") else "sz") + code


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