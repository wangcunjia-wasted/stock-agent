"""
P1 数据层：沪深300 成分股 panel 拉取

两件事：
  1. 成分名单：每个季度首日查一次 BaoStock 的沪深300成分（历史时点名单！）
     —— 全期取并集入库，训练时按"当日是否在成分内"过滤，杜绝幸存者偏差
  2. 逐股日K：并集里每只股票拉 2015-01-01~2026-10-23 日线，落 data/panel/{code}.csv

用法：
    python fetch_panel.py
预计 15~40 分钟（约 400~500 只股票 × 逐只请求），中断可重跑（已有的文件自动跳过）。
"""
import json
import os

import baostock as bs
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
PANEL_DIR = os.path.join(DATA_DIR, "panel")
START, END = "2015-01-01", "2026-10-23"

STOCK_FIELDS = ("date,code,open,high,low,close,preclose,"
                "volume,amount,adjustflag,turn,tradestatus,pctChg,isST")


def quarter_starts(trade_days: list[str]) -> list[str]:
    """取每个季度第一个交易日作为成分查询时点。"""
    seen, out = set(), []
    for d in trade_days:
        key = (d[:4], (int(d[5:7]) - 1) // 3)      # 年 + 季度序号
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def main():
    os.makedirs(PANEL_DIR, exist_ok=True)

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {lg.error_msg}")
    try:
        # ---- 1. 交易日历 → 季度首日序列
        rs = bs.query_trade_dates(start_date=START, end_date=END)
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        trade_days = [r[0] for r in rows if r[1] == "1"]
        q_starts = quarter_starts(trade_days)
        print(f">>> 交易日 {len(trade_days)} 个，成分查询时点 {len(q_starts)} 个（每季首日）")

        # ---- 2. 历史时点成分名单 → 并集 + 成分映射表
        union: set[str] = set()
        membership = []                     # [(date, code), ...] 当日为成分股
        for i, d in enumerate(q_starts):
            rs = bs.query_hs300_stocks(date=d)
            if rs.error_code != "0":
                print(f"  ⚠️ {d} 成分查询失败: {rs.error_msg}，跳过")
                continue
            codes = []
            while rs.next():
                row = rs.get_row_data()
                codes.append(row[1])        # 字段0=updateDate 1=code
            union.update(codes)
            membership.extend((d, c) for c in codes)
            print(f"  [{i+1}/{len(q_starts)}] {d}: 成分 {len(codes)} 只，累计并集 {len(union)}")
        mem = pd.DataFrame(membership, columns=["date", "code"])
        mem_path = os.path.join(DATA_DIR, "hs300_membership.csv")
        mem.to_csv(mem_path, index=False, encoding="utf-8-sig")
        print(f"✅ 成分映射表 {len(mem)} 行 → {mem_path}")

        # ---- 3. 逐股拉日K（跳过已存在的文件，断点续跑）
        codes = sorted(union)
        done, fail = 0, []
        for i, code in enumerate(codes):
            out_path = os.path.join(PANEL_DIR, f"{code}.csv")
            if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
                done += 1
                continue
            # 逐股打印并 flush：BaoStock 无超时机制，若卡死可立刻定位卡在谁身上
            print(f"  [{done}/{len(codes)}] fetching {code} ...", flush=True)
            try:
                rs = bs.query_history_k_data_plus(
                    code, STOCK_FIELDS, start_date=START, end_date=END,
                    frequency="d", adjustflag="2")
                if rs.error_code != "0":
                    fail.append((code, rs.error_msg))
                    continue
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                df = pd.DataFrame(rows, columns=rs.fields)
                for c in ("open", "high", "low", "close", "preclose",
                          "volume", "amount", "turn", "pctChg"):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df.to_csv(out_path, index=False, encoding="utf-8-sig")
                done += 1
            except Exception as e:          # 单股失败不中断全局
                fail.append((code, str(e)))
        print(f"\n✅ panel 完成：{done}/{len(codes)} 只落盘于 {PANEL_DIR}")
        if fail:
            print(f"⚠️ 失败 {len(fail)} 只（重跑本脚本可续）：")
            for c, msg in fail[:10]:
                print(f"   {c}: {msg}")

        with open(os.path.join(DATA_DIR, "panel_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"start": START, "end": END, "codes": codes,
                       "n_membership_rows": len(mem)}, f, ensure_ascii=False, indent=2)
    finally:
        bs.logout()


if __name__ == "__main__":
    main()
