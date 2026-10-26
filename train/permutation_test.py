"""
置换检验（permutation test）：给 champion 的费后超额上"数学公证"

回答的问题（存嘉 2026-10-25 提出）：回测收益到底是模型学得好，还是组合构造/
运气的功劳？

原理——随机对照实验的量化版：
  把每天的预测分在当天横截面内部随机打乱（保持每天股票数、预测分布形状、
  调仓规则、成本模型全部不变，只破坏"分数与个股的对应关系"），重跑完整
  回测 N 次，得到"无信号世界"的超额收益分布。
  - 真实超额显著落在分布右端 → 收益来自模型信号本身（p 值 = 被 null 超过的频率）
  - 真实超额落在分布中间 → 收益来自组合层构造偏差或运气，模型没干活
打乱限制在"每天内部"是关键：跨天打乱会破坏每天 300 只的横截面结构，
那样 null 就不是公平的对照了。

产出：
  - 无信号世界的年化超额分布（均值/分位数）
  - 真实超额的 p 值与超越分位
  - 顺带给日度 RankIC 做同样检验（IC 的 null 应以 0 为中心，作 sanity check）
"""
import os
import sys

import numpy as np
import pandas as pd

from guard import QuantError, step
from evaluate_panel import (DATA_DIR, OOF_PATH, REBAL, IDX_PATH,
                            round_trip_cost_rate)

N_PERM = 200              # 置换次数：null 分布 200 个点，p 值分辨率 1/201
PORT_NOTIONAL = 1_000_000.0
N_STOCKS = 30             # 与 evaluate_panel 同一组合假设
SEED = 42


@step("置换检验")
def permutation_test(oof: pd.DataFrame, idx: pd.DataFrame):
    rng = np.random.default_rng(SEED)

    # ---- 基准：沪深300 同窗口收益（与 evaluate_panel.long_backtest 完全一致）
    idx_dates = idx["date"].to_numpy()
    idx_close = idx["close"].to_numpy()
    idx_pos = {d: i for i, d in enumerate(idx_dates)}

    dates = np.array(sorted(oof["date"].unique()))
    starts = dates[::REBAL]
    bench = []
    for d0 in starts:
        j = idx_pos.get(np.datetime64(d0))
        bench.append(idx_close[j + REBAL] / idx_close[j] - 1
                     if j is not None and j + REBAL < len(idx_close) else np.nan)
    bench = np.array(bench)
    valid = ~np.isnan(bench)
    starts_v, bench_v = starts[valid], bench[valid]
    n_years = len(starts_v) * REBAL / 244

    cost_rt = round_trip_cost_rate(PORT_NOTIONAL / N_STOCKS)

    # ---- 按日期分组缓存（score 数组 + 5日收益数组），置换与回测都在组内做
    groups = {}
    for d, g in oof.groupby("date"):
        groups[d] = (g["code"].to_numpy(), g["score"].to_numpy(),
                     g["y_ret"].to_numpy())

    def run_backtest(shuffle: bool) -> tuple:
        """返回 (年化费后超额, 日度IC)。shuffle=True 时组内打乱分数。"""
        nets, ics = [], []
        prev_set = set()
        for d0 in starts_v:
            g = groups.get(d0)
            if g is None:
                continue
            codes, scores, rets = g
            if shuffle:
                scores = rng.permutation(scores)
            k = int(np.ceil(N_STOCKS * 10 / len(codes)))       # 前10%对应的rank数
            order = np.argsort(-scores)[:max(1, len(codes) // 10)]
            cur = set(codes[order])
            overlap = len(cur & prev_set) / len(cur) if prev_set else 0.0
            turnover = 1.0 - overlap
            gross = rets[order].mean()
            nets.append(gross - turnover * cost_rt)
            # 日度 RankIC（spearman 的秩近似：argsort 两次）
            r1 = np.argsort(np.argsort(scores)).astype(float)
            r2 = np.argsort(np.argsort(rets)).astype(float)
            ics.append(np.corrcoef(r1, r2)[0, 1])
            prev_set = cur
        nets = np.array(nets)
        excess = nets - bench_v[:len(nets)]
        ann_excess = excess.mean() * 244 / REBAL
        return ann_excess, np.mean(ics)

    # ---- 真实世界
    real_excess, real_ic = run_backtest(shuffle=False)
    print(f">>> 真实世界：费后年化超额 {real_excess:+.2%} | 日度RankIC {real_ic:+.4f}")

    # ---- 无信号世界：N_PERM 次"当天内部打乱"
    perm_excess, perm_ic = [], []
    for i in range(N_PERM):
        e, ic = run_backtest(shuffle=True)
        perm_excess.append(e)
        perm_ic.append(ic)
        if (i + 1) % 50 == 0:
            print(f"  置换进度 {i+1}/{N_PERM}")
    perm_excess = np.array(perm_excess)
    perm_ic = np.array(perm_ic)

    # ---- 判决
    p_excess = (1 + (perm_excess >= real_excess).sum()) / (N_PERM + 1)
    p_ic = (1 + (perm_ic >= real_ic).sum()) / (N_PERM + 1)
    q_excess = (perm_excess < real_excess).mean()          # 真实值超越 null 的分位

    print(f"\n===== 置换检验结果（{N_PERM} 次当天内部打乱，seed={SEED}）=====")
    print(f"无信号世界 费后年化超额：均值 {perm_excess.mean():+.2%} | "
          f"标准差 {perm_excess.std():.2%} | "
          f"P5 {np.quantile(perm_excess, .05):+.2%} | "
          f"P50 {np.quantile(perm_excess, .50):+.2%} | "
          f"P95 {np.quantile(perm_excess, .95):+.2%}")
    print(f"真实世界 费后年化超额 {real_excess:+.2%} → 超越 null 分布的 "
          f"{q_excess:.1%} 分位 | p 值 {p_excess:.4f}")
    print(f"\n无信号世界 日度RankIC：均值 {perm_ic.mean():+.5f}（应以 0 为中心）| "
          f"标准差 {perm_ic.std():.5f}")
    print(f"真实 RankIC {real_ic:+.4f} → p 值 {p_ic:.4f}")
    verdict = ("模型信号真实存在（真实值远超无信号世界）"
               if p_excess <= 0.05 else
               "不能排除组合层/运气解释（真实值未显著超出 null）")
    print(f"\n>>> 判决：{verdict}")
    return real_excess, perm_excess, p_excess


def main():
    oof = pd.read_csv(OOF_PATH, parse_dates=["date"])
    oof = oof.rename(columns={"p_lgbm": "score"})
    idx = pd.read_csv(IDX_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    permutation_test(oof, idx)
    print("\n>>> permutation_test.py 完成")


if __name__ == "__main__":
    try:
        main()
    except QuantError as e:
        print(f"\n✗ permutation_test 失败：{e}")
        sys.exit(1)
