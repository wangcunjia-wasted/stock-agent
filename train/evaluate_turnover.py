"""
降换手实验：同一 champion 信号（21因子 OOF 预测分），三种组合构造对比

背景：champion 回测平均换手 70.9%/5日 × 往返成本 0.3019% ≈ 每年被成本吃掉
约 11 个点，是费后超额（+4.89%）的最大敌人。本实验固定信号不动，只动"怎么拿"：

  A. baseline ：每 5 日全量调仓，买入当日前 10%（champion 现状）
  B. hold10   ：持有期 5→10 日。信号在 10 日尺度上是否衰减，换手成本是否省过衰减
  C. buffer   ：缓冲带。每 5 日体检持仓，跌出前 20% 才卖，前 10% 未持有的买入。
                经典降换手手艺：排名在小幅波动时不触发交易

回测纪律与 evaluate_panel 完全一致：等权、往返成本率 0.3019%（佣金
max(万2.5,5元)+印花税卖出0.05%+过户费万0.1+滑点0.1%）、基准沪深300。
B 需要 10 日前瞻收益——从面板 close 现算（这是已实现收益，非信号，无前视问题）。

用法：
  python evaluate_turnover.py                                # champion h5 全部策略
  python evaluate_turnover.py --oof oof_h20_price21.csv --strategies BD
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from guard import PanelError, QuantError, step
from evaluate_panel import DATA_DIR, IDX_PATH, OOF_PATH, round_trip_cost_rate

PANEL_PATH = os.path.join(DATA_DIR, "panel_dataset.parquet")

PORT_NOTIONAL = 1_000_000.0
N_STOCKS = 30
COST_RT = round_trip_cost_rate(PORT_NOTIONAL / N_STOCKS)
TOP_PCT = 0.10          # 买入阈值：当日排名前 10%
BUFFER_PCT = 0.20       # 缓冲带：跌出前 20% 才卖


@step("数据加载与10日前瞻收益")
def load_all(oof_path: str = None):
    """fwd10 从 h5 面板现算（close 相同，h5 面板尾部截断更少，覆盖更全）。"""
    if oof_path and not os.path.isabs(oof_path):
        oof_path = os.path.join(DATA_DIR, oof_path)
    oof = pd.read_csv(oof_path or OOF_PATH, parse_dates=["date"])
    panel = pd.read_parquet(PANEL_PATH, columns=["date", "code", "close"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["code", "date"])
    # 每只股票内部 shift(-10) 算 10 个交易日后的收益（已实现收益，回测度量用）
    panel["fwd10"] = panel.groupby("code")["close"].shift(-10) / panel["close"] - 1
    oof = oof.merge(panel[["date", "code", "fwd10"]], on=["date", "code"], how="left")
    idx = pd.read_csv(IDX_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    idx_pos = {d: i for i, d in enumerate(idx["date"].to_numpy())}
    idx_close = idx["close"].to_numpy()
    return oof, idx_pos, idx_close


def bench_ret(idx_pos, idx_close, d0, h):
    j = idx_pos.get(np.datetime64(d0))
    if j is None or j + h >= len(idx_close):
        return np.nan
    return idx_close[j + h] / idx_close[j] - 1


def stats(net, bench, h):
    """年化费后、年化超额、最大回撤。"""
    bt = pd.DataFrame({"net": net, "bench": bench}).dropna()
    n_years = len(bt) * h / 244
    eq = (1 + bt["net"]).cumprod()
    ann = eq.iloc[-1] ** (1 / n_years) - 1
    dd = (eq / eq.cummax() - 1).min()
    ann_b = (1 + bt["bench"]).prod() ** (1 / n_years) - 1
    excess = (bt["net"] - bt["bench"]).mean() * 244 / h
    return ann, dd, ann_b, excess, len(bt)


@step("策略A：baseline 每5日全量调仓")
def strategy_baseline(oof):
    """champion 现状：每5日买入当日 top10%，收益 = 持仓5日已实现收益。"""
    daily = {d: g for d, g in oof.groupby("date")}
    dates = np.array(sorted(oof["date"].unique()))
    starts = dates[::5]
    gross_l, turn_l, bench_l = [], [], []
    prev = set()
    for d0 in starts:
        g = daily.get(d0)
        if g is None:
            continue
        k = max(1, int(len(g) * TOP_PCT))
        cur = set(g.nlargest(k, "p_lgbm")["code"])
        turn = 1.0 - (len(cur & prev) / len(cur) if prev else 0.0)
        gross_l.append(g[g["code"].isin(cur)]["y_ret"].mean())
        turn_l.append(turn)
        bench_l.append(bench_ret(_IDX_POS, _IDX_CLOSE, d0, 5))
        prev = cur
    return gross_l, turn_l, bench_l


@step("策略B：持有期拉长到10日")
def strategy_hold10(oof):
    """每10日调仓，收益改用 10 日前瞻收益（成本窗口减半，换手自然上升一点）。"""
    daily = {d: g for d, g in oof.groupby("date")}
    dates = np.array(sorted(oof["date"].unique()))
    starts = dates[::10]
    gross_l, turn_l, bench_l = [], [], []
    prev = set()
    for d0 in starts:
        g = daily.get(d0)
        if g is None:
            continue
        k = max(1, int(len(g) * TOP_PCT))
        cur = set(g.nlargest(k, "p_lgbm")["code"])
        turn = 1.0 - (len(cur & prev) / len(cur) if prev else 0.0)
        gross_l.append(g[g["code"].isin(cur)]["fwd10"].mean())
        turn_l.append(turn)
        bench_l.append(bench_ret(_IDX_POS, _IDX_CLOSE, d0, 10))
        prev = cur
    return gross_l, turn_l, bench_l


@step("策略C：缓冲带（跌出前20%才卖）")
def strategy_buffer(oof, h: int = 5, ret_col: str = "y_ret"):
    """每 h 日体检：持仓中跌出当日 top20% 的卖出；当日 top10% 未持有的买入。
    持仓收益 = 各持仓已实现收益均值；换手 = 实际替换比例。"""
    daily = {d: g for d, g in oof.groupby("date")}
    dates = np.array(sorted(oof["date"].unique()))
    starts = dates[::h]
    gross_l, turn_l, bench_l = [], [], []
    held = set()
    for d0 in starts:
        g = daily.get(d0)
        if g is None:
            continue
        g = g.copy()
        # rank(pct=True) 是升序（分低→pct小），看多的一端是 pct 大的一端：
        # 前10% = pct >= 0.90。这里是升序陷阱，写反就会买进 dec1（已踩过）
        g["pct"] = g["p_lgbm"].rank(ascending=False, pct=True)
        top10 = set(g[g["pct"] <= TOP_PCT]["code"])
        top20 = set(g[g["pct"] <= BUFFER_PCT]["code"])
        sell = {c for c in held if c not in top20}     # 跌出缓冲带才卖
        buy = top10 - held                              # 补入新的前10%
        cur = (held - sell) | buy
        if not cur:
            continue
        n_ref = max(len(cur), 1)
        turn = max(len(sell), len(buy)) / n_ref
        gross_l.append(g[g["code"].isin(cur)][ret_col].mean())
        turn_l.append(turn)
        bench_l.append(bench_ret(_IDX_POS, _IDX_CLOSE, d0, h))
        held = cur
    return gross_l, turn_l, bench_l


_IDX_POS, _IDX_CLOSE = None, None


def main():
    global _IDX_POS, _IDX_CLOSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default=None,
                    help="OOF csv（默认 data/oof_panel.csv=champion h5；"
                         "可换 oof_h20_price21.csv / oof_h20_fund30.csv）")
    ap.add_argument("--strategies", default="ABCD", help="要跑的策略字母，如 BD")
    args = ap.parse_args()

    oof, _IDX_POS, _IDX_CLOSE = load_all(args.oof)
    src = os.path.basename(args.oof) if args.oof else "oof_panel.csv（champion h5）"
    print(f">>> OOF {src} | {len(oof):,} 行 | 往返成本率 {COST_RT:.4%}\n")

    plans = {"A": ("A baseline 5日全量", lambda o: strategy_baseline(o), 5),
             "B": ("B hold10", lambda o: strategy_hold10(o), 10),
             "C": ("C buffer 5日+缓冲带", lambda o: strategy_buffer(o, h=5), 5),
             "D": ("D hold10+缓冲带", lambda o: strategy_buffer(o, h=10, ret_col="fwd10"), 10)}

    rows = []
    for k in args.strategies.upper():
        name, fn, h = plans[k]
        gross, turn, bench = fn(oof)
        net = [g - t * COST_RT for g, t in zip(gross, turn)]
        ann, dd, ann_b, excess, n_win = stats(net, bench, h)
        cost_drag = np.mean(turn) * COST_RT * 244 / h
        rows.append({"策略": name, "窗口数": n_win,
                     "平均换手": f"{np.mean(turn):.1%}",
                     "成本拖累/年": f"-{cost_drag:.1%}",
                     "费前年化": f"{ann + cost_drag:+.2%}",
                     "费后年化": f"{ann:+.2%}",
                     "基准年化": f"{ann_b:+.2%}",
                     "费后超额/年": f"{excess:+.2%}",
                     "最大回撤": f"{dd:.2%}"})

    print(f"===== 降换手对比表（OOF 来源：{src}）=====")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n看点：费后超额列。B 赌信号在10日尺度衰减小于成本节约；"
          "C 赌排名在小幅波动时不触发交易。")


if __name__ == "__main__":
    try:
        main()
    except QuantError as e:
        print(f"\n✗ evaluate_turnover 失败：{e}")
        sys.exit(1)
