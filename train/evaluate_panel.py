"""
评估层（P1 横截面版）：分位数信号体检 + 含成本组合回测

三个分析，回答三个问题：
  A. 逐日十分位表：每天在成分内按预测分十组，看各组真实5日收益是否单调
     （P0 是全期混着切 qcut——不同时期预测分布不同会互相污染，横截面必须逐日）
  B. 多头组合回测：每 5 个交易日调仓一次，等权买入当日预测前 10%，扣真实成本，
     基准为同期沪深300。成本按换手率计——组合成员没换就不用付费
  C. 多空价差（LS）：每日"前10%均值 - 后10%均值"，市场中性视角的信号价值
     （A股现货不能做空，仅作信息参考）

成本模型（存嘉定版，与 P0 一致）：佣金 max(金额×万2.5, 5元)、印花税卖出0.05%、
过户费万0.1双向、滑点单边0.1%。组合假设 100 万、约30只 → 单只约3.3万/边，
最低佣金 5 元基本不被触发（这是与 P0 单票大额成交的重要区别）。

纪律：调仓只用当日横截面排序，收益是 T→T+5 的已实现收益；无任何未来信息进信号。
用法：
  python evaluate_panel.py                                   # champion OOF + 5日调仓
  python evaluate_panel.py --oof oof_xxx.csv --rebal 20      # 长预测期版
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from build_dataset import HORIZON
from guard import PanelError, QuantError, get_logger, step

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
OOF_PATH = os.path.join(DATA_DIR, "oof_panel.csv")
IDX_PATH = os.path.join(DATA_DIR, "hs300_daily.csv")

# ---------------- 成本参数（与 P0 evaluate.py 完全一致）
COMMISSION = 2.5e-4      # 佣金万2.5
MIN_COMMISSION = 5.0     # 单笔最低5元
STAMP_TAX = 5e-4         # 印花税，卖出
TRANSFER_FEE = 1e-5      # 过户费，双向
SLIPPAGE = 1e-3          # 滑点，单边
PORT_NOTIONAL = 1_000_000.0   # 组合总资金假设（单只 ≈ 3.3万/边）
REBAL = 5                     # 调仓周期（交易日）；main() 里按 --rebal 覆盖


def round_trip_cost_rate(notional: float) -> float:
    """一次建仓+一次平仓的总成本率（与 P0 同公式，notional 决定最低佣金是否触发）。"""
    def one_side(amount, is_sell):
        c = max(amount * COMMISSION, MIN_COMMISSION) + amount * TRANSFER_FEE
        if is_sell:
            c += amount * STAMP_TAX
        return c
    buy_amt = notional * (1 + SLIPPAGE)
    sell_amt = notional * (1 - SLIPPAGE)
    total = one_side(buy_amt, False) + one_side(sell_amt, True) + (buy_amt + sell_amt) * SLIPPAGE
    return total / notional


@step("OOF 与基准数据加载")
def load_data(oof_path: str = OOF_PATH):
    if not os.path.exists(oof_path):
        raise PanelError(f"OOF 不存在：{oof_path}（先跑 train_lgbm_panel.py）")
    oof = pd.read_csv(oof_path, parse_dates=["date"])
    idx = pd.read_csv(IDX_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return oof, idx


def daily_decile(oof: pd.DataFrame) -> pd.DataFrame:
    """逐日横截面十分位：dec = 当日预测排名的十分位（1=最看空 … 10=最看多）。"""
    pct = oof.groupby("date")["p_lgbm"].rank(pct=True)
    oof = oof.copy()
    oof["dec"] = np.ceil(pct * 10).clip(1, 10).astype(int)
    return oof


@step("十分位信号体检")
def decile_report(oof: pd.DataFrame):
    t = oof.groupby("dec").agg(
        n=("y_ret", "size"),
        mean_ret=("y_ret", "mean"),
        up_ratio=("y_bin", "mean"),
    )
    print("===== A. 逐日十分位体检表（当日成分内排名 → 真实5日收益）=====")
    print(t.to_string(float_format=lambda x: f"{x:.4f}"))
    print("看点：mean_ret 从 dec1 到 dec10 大体单调递增 = 排序力落到收益上\n")

    # 多空价差按年看：信号是否年年都在（而不是某段行情撑全场）
    top = oof[oof["dec"] == 10].groupby(oof["date"].dt.year)["y_ret"].mean()
    bot = oof[oof["dec"] == 1].groupby(oof["date"].dt.year)["y_ret"].mean()
    spread = (top - bot).dropna()
    by_year = pd.DataFrame({
        "多头组": top, "空头组": bot, "多空价差": spread})
    print("===== A2. 前10% - 后10% 日度收益差（按年，5日持有）=====")
    print(by_year.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"全期日均多空价差 {spread.mean():+.4f}（年化约 {spread.mean()*244/REBAL:+.1%}，费前）\n")
    return spread.mean()


@step("含成本多头组合回测（每5日调仓）")
def long_backtest(oof: pd.DataFrame, idx: pd.DataFrame):
    dates = np.array(sorted(oof["date"].unique()))
    starts = dates[::REBAL]

    # 快查表：date → 当日前10%成员及其5日收益
    top_map = {d: g for d, g in oof[oof["dec"] == 10].groupby("date")}
    idx_dates = idx["date"].to_numpy()
    idx_close = idx["close"].to_numpy()
    idx_pos = {d: i for i, d in enumerate(idx_dates)}

    per_stock_notional = PORT_NOTIONAL / 30          # 约30只的等权假设
    cost_rt = round_trip_cost_rate(per_stock_notional)

    rows = []
    prev_set = set()
    for k, d0 in enumerate(starts):
        g = top_map.get(d0)
        if g is None or len(g) == 0:
            continue
        cur = set(g["code"])
        overlap = len(cur & prev_set) / len(cur) if prev_set else 0.0
        turnover = 1.0 - overlap                      # 换掉的仓位比例
        gross = g["y_ret"].mean()
        # 建仓换手 turnover 比例 → 卖旧买新各一份 = turnover 次往返；
        # 首窗 turnover=1 记满一次往返（略保守：真实首窗只买不卖）
        net = gross - turnover * cost_rt
        # 基准：沪深300 同窗口（d0 → d0 后第5个交易日）
        j = idx_pos.get(np.datetime64(d0))
        bench = idx_close[j + REBAL] / idx_close[j] - 1 if j is not None and j + REBAL < len(idx_close) else np.nan
        rows.append({"start": d0, "n_stocks": len(cur), "turnover": turnover,
                     "gross": gross, "net": net, "bench": bench})
        prev_set = cur

    bt = pd.DataFrame(rows)
    n_years = len(bt) * REBAL / 244

    def stats(ret):
        equity = (1 + ret).cumprod()
        ann = equity.iloc[-1] ** (1 / n_years) - 1
        dd = (equity / equity.cummax() - 1).min()
        return ann, dd, equity

    ann_n, dd_n, eq = stats(bt["net"])
    ann_g, dd_g, _ = stats(bt["gross"])
    b = bt["bench"].dropna()
    ann_b = (1 + b).prod() ** (1 / n_years) - 1
    excess = bt["net"] - bt["bench"]

    print(f"===== B. 含成本多头回测（前10%等权，每{REBAL}日调仓，"
          f"资金 {PORT_NOTIONAL:,.0f}，往返成本率 {cost_rt:.4%}）=====")
    print(f"调仓窗口 {len(bt)} 个 | 平均持仓 {bt['n_stocks'].mean():.0f} 只 | "
          f"平均换手 {bt['turnover'].mean():.1%}")
    print(f"策略 费前年化 {ann_g:+.2%} | 费后年化 {ann_n:+.2%} | 最大回撤 {dd_n:.2%}")
    print(f"基准 沪深300 同期年化 {ann_b:+.2%} | 费后超额年化 {excess.mean()*244/REBAL:+.2%}")
    print("(超额收益才是横截面策略的成绩单——它不承担市场方向，赚的是选股的钱)")
    return bt


def ls_reference(oof: pd.DataFrame):
    """多空价差参考（市场中性视角，费前；A股现货无法做空，仅供理解信号价值）。"""
    top = oof[oof["dec"] == 10].groupby("date")["y_ret"].mean()
    bot = oof[oof["dec"] == 1].groupby("date")["y_ret"].mean()
    ls = (top - bot).dropna()
    ann = ls.mean() / REBAL * 244
    cost2 = 2 * round_trip_cost_rate(PORT_NOTIONAL / 30) / REBAL * 244
    print(f"\n===== C. 多空价差参考（每日 前10%-后10%，费前）=====")
    print(f"日均价差 {ls.mean():+.4f} | 年化约 {ann:+.1%} | "
          f"若双腿各扣成本约 -{cost2:.1%}/年 → 费后约 {ann - cost2:+.1%}")
    print("(这是信号理论价值的天花板估计；实盘受限做空与成本，取其一部分已属不易)")
    return ls


def main():
    global REBAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default=OOF_PATH, help="OOF csv 路径")
    ap.add_argument("--rebal", type=int, default=5, help="调仓周期（交易日）")
    args = ap.parse_args()
    REBAL = args.rebal
    oof_path = args.oof if os.path.isabs(args.oof) else os.path.join(DATA_DIR, args.oof)

    oof, idx = load_data(oof_path)
    oof = daily_decile(oof)
    decile_report(oof)
    long_backtest(oof, idx)
    ls_reference(oof)
    get_logger().info("evaluate_panel 完成")
    print("\n>>> evaluate_panel.py 完成")


if __name__ == "__main__":
    try:
        main()
    except QuantError as e:
        print(f"\n✗ evaluate_panel 失败：{e}")
        sys.exit(1)
