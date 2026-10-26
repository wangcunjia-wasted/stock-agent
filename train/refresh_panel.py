"""
panel 缓存增量刷新（Forecast Agent 的日常弹药补给）

与 fetch_panel.py 的分工：
  fetch_panel.py  —— 初次建库（668 只全历史，一次性）；
  本脚本          —— 日常增量：只补"缓存最后日期之后"的 K 线 + 季度调样时补新成分。

两个关键设计（都是防脏数据的闸）：
1. 接缝体检：缓存用前复权价（adjustflag=2），分红除权后整个历史的价格基准会变。
   直接追加会在接缝处制造假跳空。因此每只股票追加前，检查"新增首行 preclose
   == 旧末行 close"——不等说明期间发生过除权，整股重拉（一两秒），绝不对着
   错位基准拼接。
2. 成分名单先于行情刷新：季度调样后新成分股没有 panel 文件，先补它的全历史，
   横截面才不会缺人。

用法：
    python refresh_panel.py               # 刷新最新一期成分（约 300 只，2~5 分钟）
    python refresh_panel.py --all         # 刷新全部 668 只并集（训练侧重建前用）
"""
import argparse
import json
import os
import sys
import time
from datetime import date, datetime

import baostock as bs
import pandas as pd

from guard import DataError, FailureTracker, PanelError, QuantError, get_logger, step

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
PANEL_DIR = os.path.join(DATA_DIR, "panel")
MEM_PATH = os.path.join(DATA_DIR, "hs300_membership.csv")
META_PATH = os.path.join(DATA_DIR, "panel_meta.json")
START = "2015-01-01"
TODAY = date.today().isoformat()
DATA_READY_HOUR = 18     # BaoStock 当日行情约傍晚就绪；之前当日不算"已完成交易日"

STOCK_FIELDS = ("date,code,open,high,low,close,preclose,"
                "volume,amount,adjustflag,turn,tradestatus,pctChg,isST")
NUM_COLS = ("open", "high", "low", "close", "preclose",
            "volume", "amount", "turn", "pctChg")
SEAM_RTOL = 1e-3        # 接缝体检的相对容差（前复权四舍五入的噪声量级）


def _atomic_csv(df: pd.DataFrame, path: str):
    """原子写盘：先写 tmp 再 replace，中途被杀不会留下半截 CSV。"""
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def _fetch_kline(code: str, start: str, end: str) -> pd.DataFrame:
    """单股日K（含数值转换），与 fetch_panel.py 完全同源同参。"""
    rs = bs.query_history_k_data_plus(
        code, STOCK_FIELDS, start_date=start, end_date=end,
        frequency="d", adjustflag="2")
    if rs.error_code != "0":
        raise RuntimeError(f"BaoStock {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@step("交易日历与成分名单刷新")
def refresh_membership() -> list:
    """补查当前季度的成分快照（季度调样时名单才会长出新日期），返回最新成分。"""
    rs = bs.query_trade_dates(start_date=START, end_date=TODAY)
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    trade_days = [r[0] for r in rows if r[1] == "1"]
    # 当前季度首个交易日（与 fetch_panel 的 quarter_starts 同逻辑）
    seen, q_starts = set(), []
    for d in trade_days:
        key = (d[:4], (int(d[5:7]) - 1) // 3)
        if key not in seen:
            seen.add(key)
            q_starts.append(d)

    mem = pd.read_csv(MEM_PATH)
    existing_snaps = set(mem["date"])
    pending = [d for d in q_starts if d not in existing_snaps]
    if pending:
        print(f">>> 发现 {len(pending)} 个未入库的季度快照日：{pending}")
        new_rows = []
        for d in pending:
            rs = bs.query_hs300_stocks(date=d)
            if rs.error_code != "0":
                raise DataError(f"成分查询失败 {d}: {rs.error_msg}")
            codes = []
            while rs.next():
                codes.append(rs.get_row_data()[1])
            new_rows.extend((d, c) for c in codes)
            print(f"    {d}: {len(codes)} 只")
        mem = pd.concat([mem, pd.DataFrame(new_rows, columns=["date", "code"])],
                        ignore_index=True)
        _atomic_csv(mem, MEM_PATH)
        print(f">>> 成分名单已更新：{len(mem)} 行")
    else:
        print(">>> 成分名单已是最新（无新增季度快照）")

    latest_snap = mem["date"].max()
    members = sorted(mem.loc[mem["date"] == latest_snap, "code"].unique())
    print(f">>> 最新快照 {latest_snap}：{len(members)} 只成分")
    return members, trade_days


@step("逐股增量补数")
def refresh_klines(codes: list, trade_days: list) -> dict:
    """逐股补 last_date 之后的 K 线；接缝错位则整股重拉。单股失败记账容忍。"""
    tracker = FailureTracker("panel增量刷新", max_fail_rate=0.2)
    # "已完成交易日"：今天的行情傍晚才就绪，收盘前当日不算（否则会对未产生的
    # 数据空拉一轮，还把正常股票误标成"停牌"）
    cutoff = trade_days[-1] if (datetime.now().hour >= DATA_READY_HOUR
                                and trade_days[-1] == TODAY) else \
        max((d for d in trade_days if d < TODAY), default=TODAY)

    t0 = time.time()
    stat = {"fresh": 0, "appended": 0, "refetched": 0, "new_file": 0, "suspended": 0}
    for i, code in enumerate(codes):
        try:
            path = os.path.join(PANEL_DIR, f"{code}.csv")
            if os.path.exists(path) and os.path.getsize(path) > 1000:
                old = pd.read_csv(path)
                last = old["date"].max()
                if last >= cutoff:
                    tracker.record_ok(code)
                    stat["fresh"] += 1
                    continue
                new = _fetch_kline(code, (pd.Timestamp(last) + pd.Timedelta(days=1))
                                   .strftime("%Y-%m-%d"), TODAY)
                if new.empty:
                    tracker.record_ok(code)
                    stat["suspended"] += 1               # 区间内停牌，无新行
                    continue
                # ---- 接缝体检：新增首行 preclose 必须接上旧末行 close
                old_last_close = float(old["close"].iloc[-1])
                new_first_pre = float(new["preclose"].iloc[0])
                if abs(new_first_pre - old_last_close) > SEAM_RTOL * max(old_last_close, 1e-9):
                    combined = _fetch_kline(code, START, TODAY)   # 除权 → 全量重拉
                    stat["refetched"] += 1
                else:
                    combined = pd.concat([old, new], ignore_index=True)
                    stat["appended"] += 1
                combined = (combined.drop_duplicates(subset="date", keep="last")
                            .sort_values("date").reset_index(drop=True))
                _atomic_csv(combined, path)
            else:
                # 新成分股：没有 panel 文件，补全历史
                df = _fetch_kline(code, START, TODAY)
                if df.empty:
                    raise DataError("全历史查询为空")
                _atomic_csv(df, path)
                tracker.record_ok(code)
                stat["new_file"] += 1
        except Exception as e:
            tracker.record_fail(code, e)
            print(f"  ⚠️ {code}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  进度 {i+1}/{len(codes)}（{time.time()-t0:.0f}s，"
                  f"失败 {len(tracker.failed)}）")
    tracker.check()
    stat["last_completed"] = cutoff
    return stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="刷新全部历史并集（默认只刷最新一期成分）")
    args = ap.parse_args()

    lg = bs.login()
    if lg.error_code != "0":
        raise PanelError(f"BaoStock 登录失败: {lg.error_msg}")
    try:
        members, trade_days = refresh_membership()
        if args.all:
            codes = sorted({c for c in pd.read_csv(MEM_PATH)["code"].unique()})
            print(f">>> --all 模式：刷新并集 {len(codes)} 只")
        else:
            codes = members
        stat = refresh_klines(codes, trade_days)

        print("\n===== 刷新统计 =====")
        print(f"已是最新 {stat['fresh']} | 追加增量 {stat['appended']} | "
              f"除权重拉 {stat['refetched']} | 停牌无新行 {stat['suspended']} | "
              f"新建文件 {stat['new_file']}")
        # meta 更新 end 日期，供训练侧感知数据新鲜度
        try:
            with open(META_PATH, encoding="utf-8") as f:
                meta = json.load(f)
            meta["end"] = stat.get("last_completed", TODAY)
            meta["refreshed_at"] = str(date.today())
            with open(META_PATH, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        print(">>> refresh_panel.py 完成")
        get_logger().info(f"panel 刷新完成 {stat}")
    finally:
        bs.logout()


if __name__ == "__main__":
    try:
        main()
    except QuantError as e:
        print(f"\n✗ refresh_panel 失败：{e}")
        sys.exit(1)
