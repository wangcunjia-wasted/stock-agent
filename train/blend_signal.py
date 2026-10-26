"""
混合信号实验：量价（进攻）× 基本面（粘性）

动机（2026-10-25 实验发现）：h20_fund30 信号在 hold10+缓冲带下换手仅 36.3%
（h5 量价信号 66.8%）——季报一季才变一次，排名"粘"；而量价信号进攻性强但
排名天天跳。假设：两者混合能在几乎不牺牲收益的前提下压换手、提费后超额。

方法：
  1. 两份 OOF（h5 champion 量价、h20 fund30）按 (date, code) 内连接对齐
  2. 各自在当天横截面内转百分位排名（rank ascending=False，分高=看多）
  3. 混合分 s = w * rank_价 + (1-w) * rank_基本面，扫 w ∈ {1.0, 0.75, 0.5, 0.25, 0.0}
  4. 统一评估框架（消除持有期错配）：hold10 + 20%缓冲带，fwd10 已实现收益，
     成本全额计提；IC 也统一对 fwd10 算（信号层与组合层对齐）
  5. 对照组：h20 price21（预测期效应，不含基本面）同样混合

判读要点：
  - w=1.0 行应复现 h5+D ≈ +9.25%（管线自检）
  - 看 IC10 与费后超额随 w 的曲线：基本面是稀释还是互补，"粘性"有没有换来净收益
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from guard import QuantError
from evaluate_turnover import (COST_RT, DATA_DIR, BUFFER_PCT, TOP_PCT,
                               load_all, stats)

H = 10                    # 统一持有期：10 个交易日
REBAL = 10


def daily_rank(s: pd.Series) -> pd.Series:
    """当天横截面内百分位排名。约定：原始分越高 → rank 越大 → 越看多。
    （升序 rank；回测侧按 score 降序取前排，两者必须同一约定——
    本次首跑就因两个函数约定打架，精准买入了最差的一批，靠 w=1.0 自检行揪出）"""
    return s.groupby(s.index.get_level_values("date")).rank(pct=True)


def generic_backtest(m: pd.DataFrame, idx_pos, idx_close):
    """对已带 score 列的面板跑 hold10+缓冲带回测（score 由调用方按任意 w 规则合成）。"""
    daily = {d: g for d, g in m.groupby("date")}
    dates = np.array(sorted(m["date"].unique()))
    starts = dates[::REBAL]

    gross_l, turn_l, bench_l = [], [], []
    held = set()
    for d0 in starts:
        g = daily.get(d0)
        if g is None:
            continue
        g = g.sort_values("score", ascending=False)
        n = len(g)
        top10 = set(g["code"].iloc[:max(1, int(n * TOP_PCT))])
        top20 = set(g["code"].iloc[:max(1, int(n * BUFFER_PCT))])
        sell = {c for c in held if c not in top20}
        buy = top10 - held
        cur = (held - sell) | buy
        if not cur:
            continue
        turn = max(len(sell), len(buy)) / max(len(cur), 1)
        gross_l.append(g[g["code"].isin(cur)]["fwd10"].mean())
        turn_l.append(turn)
        j = idx_pos.get(np.datetime64(d0))
        bench_l.append(idx_close[j + H] / idx_close[j] - 1
                       if j is not None and j + H < len(idx_close) else np.nan)
        held = cur

    # 日度 RankIC（对统一基准 fwd10，而非各自训练标签）
    ic = m.groupby("date").apply(
        lambda g: spearmanr(g["score"], g["fwd10"]).statistic
        if len(g) >= 30 else np.nan).dropna()

    return np.array(gross_l), np.array(turn_l), np.array(bench_l), ic


def blended_backtest(m: pd.DataFrame, w: float, idx_pos, idx_close):
    score = w * m["r_price"] + (1 - w) * m["r_fund"]
    m = m.assign(score=score)
    return generic_backtest(m, idx_pos, idx_close)


def adaptive_weights(m: pd.DataFrame, window: int = 252, quarter: int = 63,
                     clip: tuple = (0.2, 0.8), min_obs: int = 126) -> pd.Series:
    """季度滚动自适应权重：每季度起点用"过去一年滚动日度IC"定 w。

    防前视关键：日度 IC(d) = spearman(信号_d, fwd10_d) 用到了 d+10 的收益，
    所以季度起点 q 只允许使用 d_pos <= q_pos - 10 的日子——fwd10 在 q 时点
    必须已完全实现。历史不足 min_obs 个有效日 → w=0.5（中性起步）。
    w = clip(IC价/(IC价+IC基本面), 0.2, 0.8)：谁近期排得准谁权重高。
    返回：date → w（每个交易日的 w 由其所属季度决定）。
    """
    daily_ic = m.groupby("date").apply(
        lambda g: pd.Series({
            "ic_p": spearmanr(g["r_price"], g["fwd10"]).statistic if len(g) >= 30 else np.nan,
            "ic_f": spearmanr(g["r_fund"], g["fwd10"]).statistic if len(g) >= 30 else np.nan,
        })).dropna()
    dates = np.array(sorted(m["date"].unique()))
    pos = {d: i for i, d in enumerate(dates)}
    ic_pos = {d: i for i, d in enumerate(daily_ic.index)}

    w_map = {}
    for q0 in dates[::quarter]:
        q_pos = pos[q0]
        # 可用 IC 日：位置 < q_pos - H（fwd10 完全实现），取最近 window 个
        usable = daily_ic.index[daily_ic.index.map(ic_pos).to_numpy() < q_pos - H][-window:]
        if len(usable) < min_obs:
            w_map[q0] = 0.5
            continue
        sub = daily_ic.loc[usable]
        mp, mf = sub["ic_p"].mean(), sub["ic_f"].mean()
        if not np.isfinite(mp + mf) or mp + mf <= 0:
            w_map[q0] = 0.5
        else:
            w_map[q0] = float(np.clip(mp / (mp + mf), *clip))

    # 展开到每个交易日：date 属于它之前最近的季度起点
    qs = np.array(sorted(w_map))
    day_pos = np.searchsorted(qs, dates, side="right") - 1
    return pd.Series([w_map[qs[max(0, i)]] for i in day_pos], index=dates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fund-oof", default="oof_h20_fund30.csv",
                    help="基本面信号 OOF（粘性来源）")
    ap.add_argument("--price-oof", default="oof_panel.csv",
                    help="量价信号 OOF（进攻来源，champion h5）")
    ap.add_argument("--weights", default="1.0,0.75,0.5,0.25,0.0",
                    help="固定权重扫描网格，逗号分隔（如 0.4,0.45,0.5,0.55,0.6）")
    ap.add_argument("--adaptive", action="store_true",
                    help="附加季度滚动自适应权重版（trailing 252日IC 定权）")
    args = ap.parse_args()
    weights = [float(x) for x in args.weights.split(",")]

    oof, idx_pos, idx_close = load_all()          # 自带 fwd10（h5 面板现算）
    fund = pd.read_csv(os.path.join(DATA_DIR, args.fund_oof), parse_dates=["date"])
    price = pd.read_csv(os.path.join(DATA_DIR, args.price_oof), parse_dates=["date"])

    # 内连接对齐 + 各自当日横截面百分位排名
    m = price[["date", "code", "p_lgbm"]].rename(columns={"p_lgbm": "s_price"}).merge(
        fund[["date", "code", "p_lgbm"]].rename(columns={"p_lgbm": "s_fund"}),
        on=["date", "code"], how="inner")
    m = m.merge(oof[["date", "code", "fwd10"]], on=["date", "code"], how="left")
    m = m.set_index(["date", "code"])
    m["r_price"] = daily_rank(m["s_price"])
    m["r_fund"] = daily_rank(m["s_fund"])
    m = m.reset_index()
    n_overlap = len(m)
    print(f">>> 对齐后样本 {n_overlap:,} 行（量价 {len(price):,} × 基本面 {len(fund):,} 的交集）"
          f" | 统一框架：hold{H} + 缓冲带{BUFFER_PCT:.0%} | 往返成本 {COST_RT:.4%}\n")

    rows = []
    for w in weights:
        gross, turn, bench, ic = blended_backtest(m, w, idx_pos, idx_close)
        rows.append(_row(f"w={w:g}", w, None, turn, gross, bench, ic))
    if args.adaptive:
        w_ser = adaptive_weights(m)
        m2 = m.copy()
        w_by_date = m2["date"].map(w_ser)
        m2["score"] = w_by_date * m2["r_price"] + (1 - w_by_date) * m2["r_fund"]
        gross, turn, bench, ic = generic_backtest(m2, idx_pos, idx_close)
        rows.append(_row("adaptive(季度)", None, w_ser, turn, gross, bench, ic))
        print(">>> 自适应 w 轨迹（每季度一个值）：")
        print(w_ser.describe().to_string(float_format=lambda x: f"{x:.3f}"))

    print(f"===== 混合信号扫描（r = w·量价 + (1-w)·基本面，全部 hold{H}+缓冲带）=====")
    print(pd.DataFrame(rows).to_string(index=False))
    if not args.adaptive:
        print("\n判读：w=1.0 应≈h5+D（管线自检）；往 w=0 方向走，看 IC10 与费后超额"
              "的形状——单调变差=基本面纯稀释，先平后降=有互补成分，"
              "换手显著下降=粘性成立。")


def _row(name, w, w_ser, turn, gross, bench, ic) -> dict:
    net = gross - turn * COST_RT
    ann, dd, ann_b, excess, n_win = stats(net, bench, H)
    cost_drag = np.mean(turn) * COST_RT * 244 / H
    return {"方案": name,
            "w_量价": f"{w:g}" if w is not None
                      else f"{w_ser.mean():.2f}均值[{w_ser.min():.2f}~{w_ser.max():.2f}]",
            "IC10均值": f"{ic.mean():+.4f}",
            "ICIR": f"{ic.mean()/ic.std():+.3f}",
            "平均换手": f"{np.mean(turn):.1%}",
            "成本拖累/年": f"-{cost_drag:.1%}",
            "费前年化": f"{ann + cost_drag:+.2%}",
            "费后年化": f"{ann:+.2%}",
            "费后超额/年": f"{excess:+.2%}",
            "最大回撤": f"{dd:.2%}"}


if __name__ == "__main__":
    try:
        main()
    except QuantError as e:
        print(f"\n✗ blend_signal 失败：{e}")
        sys.exit(1)
