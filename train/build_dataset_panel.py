"""
P1 数据集构造：逐股算因子+标签 → 成分过滤 → 横截面大表

用法：
  python build_dataset_panel.py                 # HORIZON=5，落 panel_dataset.parquet
  python build_dataset_panel.py --horizon 20    # 20日标签，落 panel_dataset_h20.parquet
                                                # （P2 长预测期实验：给慢变量因子翻身仗用）

三道关（在 P0 的防前视闸之上，横截面新增的两条纪律）：
  1. 因子/标签：逐股走 P0 同一套 compute_factors + build_labels，公式零改动
  2. 成分过滤：样本日期必须"当时在沪深300内"（asof 对齐季度名单，防幸存者偏差）
  3. 可投资性过滤：剔除停牌日（tradestatus≠1）与 ST 状态行

守卫接入（guard.py 的第一个示范）：
  - @step 给每个阶段计时+写日志（data/logs/）
  - FailureTracker 记逐股成败，结束统一清算：
    失败率超 20% 或全部失败 → ThresholdError，附原因分布
  - 顶层 except QuantError 只拦守卫抛的错，意外异常原样穿透不吞

产出：data/panel_dataset.parquet（pyarrow 缺失时退回 csv）
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

from factors import FACTOR_DICT, FUNDAMENTAL_FACTORS
from build_dataset import build_dataset
from guard import (FailureTracker, PanelError, QuantError, get_logger, step)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
PANEL_DIR = os.path.join(DATA_DIR, "panel")
FUND_DIR = os.path.join(DATA_DIR, "fundamental")

# 季报原始字段（百分数）→ 因子名；可用日 = 法定披露截止日（零前视：宁晚勿早）
FUND_MAP = {
    "净资产收益率(%)":        "F_roe",
    "销售毛利率(%)":          "F_gpm",
    "销售净利率(%)":          "F_npm",
    "资产负债率(%)":          "F_debt",
    "主营业务收入增长率(%)":  "F_rev_yoy",
    "净利润增长率(%)":        "F_ni_yoy",
}
# 每股指标（元）→ 估值因子原料（TTM 在下面构造）
PER_SHARE_SRC = {"eps_ttm": "摊薄每股收益(元)",
                 "cfps_ttm": "每股经营性现金流(元)",
                 "bps": "每股净资产_调整前(元)"}


def avail_date(d: pd.Timestamp) -> pd.Timestamp:
    """报告期 → 法定披露截止日（所有上市公司必然在此日前披露完毕）。
    Q1→4/30，中报→8/31，三季报→10/31，年报→次年4/30。"""
    if d.month == 3:
        return pd.Timestamp(year=d.year, month=4, day=30)
    if d.month == 6:
        return pd.Timestamp(year=d.year, month=8, day=31)
    if d.month == 9:
        return pd.Timestamp(year=d.year, month=10, day=31)
    if d.month == 12:
        return pd.Timestamp(year=d.year + 1, month=4, day=30)
    return pd.NaT


def _add_ttm(f: pd.DataFrame) -> pd.DataFrame:
    """每股收益/每股经营现金流是"年初至今累计值"，TTM（滚动4季）=
    去年年报累计 + 今年本期累计 - 去年同期累计；年报期本身即全年值。
    上下游任一期缺数据 → NaN（不硬凑），LGBM 原生吃。"""
    f = f.sort_values("stat").copy()
    f["year"], f["md"] = f["stat"].dt.year, f["stat"].dt.strftime("%m-%d")
    for src, dst in (("摊薄每股收益(元)", "eps_ttm"),
                     ("每股经营性现金流(元)", "cfps_ttm")):
        lookup = {(y, m): v for y, m, v in zip(f["year"], f["md"], f[src])}
        annual = {y: v for y, m, v in zip(f["year"], f["md"], f[src]) if m == "12-31"}
        out = []
        for y, m, cur in zip(f["year"], f["md"], f[src]):
            if m == "12-31":
                out.append(cur)
                continue
            pa, ps = annual.get(y - 1), lookup.get((y - 1, m))
            if pa is None or ps is None or pd.isna(pa) or pd.isna(ps) or pd.isna(cur):
                out.append(np.nan)
            else:
                out.append(pa + cur - ps)
        f[dst] = out
    return f.rename(columns={"每股净资产_调整前(元)": "bps"})


def load_fundamental(code: str) -> pd.DataFrame:
    """季报基本面 → 可用日对齐表。返回按 avail 升序的
    [avail, 6个因子字段, eps_ttm, cfps_ttm, bps]。"""
    path = os.path.join(FUND_DIR, f"{code}.csv")
    if not os.path.exists(path):
        return None
    f = pd.read_csv(path).rename(columns={"日期": "stat"})
    f["stat"] = pd.to_datetime(f["stat"])
    f["avail"] = f["stat"].apply(avail_date)
    f = f.dropna(subset=["avail"])
    if f.empty:
        return None
    for c in list(FUND_MAP) + list(PER_SHARE_SRC.values()):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    for c in FUND_MAP:
        f[c] = f[c] / 100.0                                    # 百分数 → 小数
    f = _add_ttm(f)
    # 同一可用日（如 4/30 同时是年报和一季报截止）→ 报告期新的排后面，
    # searchsorted side='right'-1 取到的就是最新一期
    cols = ["avail"] + list(FUND_MAP) + list(PER_SHARE_SRC)
    return f.sort_values(["avail", "stat"])[cols]


@step("成分名单加载")
def load_membership() -> pd.DataFrame:
    """季度成分名单：date, code。date 解析为 datetime 供快照对齐。"""
    path = os.path.join(DATA_DIR, "hs300_membership.csv")
    if not os.path.exists(path):
        raise PanelError(f"成分名单不存在：{path}（先跑 fetch_panel.py）")
    mem = pd.read_csv(path)
    mem["date"] = pd.to_datetime(mem["date"])
    return mem.sort_values("date").reset_index(drop=True)


@step("逐股因子计算")
def compute_all(files: list, horizon: int = 5, verbose_every: int = 50) -> pd.DataFrame:
    """逐股算因子+标签并拼接。单股失败记账容忍，系统性失败由 tracker 清算时炸出。"""
    get_logger().info(f"panel 文件 {len(files)} 个，逐股算因子+标签"
                      f"（horizon={horizon}，因子公式与 P0 完全一致）")
    tracker = FailureTracker("panel因子计算", max_fail_rate=0.2)

    feat_cols = list(FACTOR_DICT.keys()) + list(FUNDAMENTAL_FACTORS.keys())
    parts = []
    t0 = time.time()
    for i, f in enumerate(files):
        code = os.path.splitext(os.path.basename(f))[0]
        try:
            # 关3 在源头做：读入即剔停牌/ST，再进因子管线
            raw = pd.read_csv(f)
            if "tradestatus" in raw.columns:
                raw = raw[raw["tradestatus"] == 1]
            if "isST" in raw.columns:
                raw = raw[raw["isST"] == 0]
            ds = build_dataset(f, horizon=horizon, raw=raw)   # factors + labels + 尾部截断

            # 季报基本面：按"法定披露可用日"向后对齐（searchsorted，与成分过滤同手艺）
            # 对齐映射：(fund 返回表中的列名 → 因子名)。fund 只含派生列
            # （eps_ttm/cfps_ttm/bps），原始中文列已被 load_fundamental 裁剪
            align_pairs = list(FUND_MAP.items()) + [(c, c) for c in PER_SHARE_SRC]
            fund = load_fundamental(code)
            if fund is not None:
                dts = pd.to_datetime(ds["date"]).to_numpy()
                idx = np.searchsorted(fund["avail"].to_numpy(), dts, side="right") - 1
                valid = idx >= 0
                for src, fcol in align_pairs:
                    col = np.full(len(ds), np.nan)
                    col[valid] = fund[src].to_numpy()[idx[valid]]
                    ds[fcol] = col
            else:
                for _, fcol in align_pairs:
                    ds[fcol] = np.nan
            # 估值三件套：每股指标 / 当日收盘价（分母用当日价，分子只用已披露的季报）
            close_safe = ds["close"].replace(0, np.nan)
            ds["F_ep"] = ds["eps_ttm"] / close_safe
            ds["F_bp"] = ds["bps"] / close_safe
            ds["F_npcf"] = ds["cfps_ttm"] / close_safe
            ds = ds.drop(columns=list(PER_SHARE_SRC))

            # 瘦身为最终大表（BaoStock 原始 CSV 自带 code 列与冗余字段）
            ds = ds[["date", "close"] + feat_cols + ["y_ret", "y_bin"]].copy()
            ds.insert(0, "code", code)
            parts.append(ds)
            tracker.record_ok(code)
        except Exception as e:                  # 单股异常记账后继续，不中断
            tracker.record_fail(code, e)
            print(f"  ⚠️ {code}: {e}")
        if (i + 1) % verbose_every == 0:
            print(f"  因子进度 {i+1}/{len(files)}（{time.time()-t0:.0f}s，"
                  f"失败 {len(tracker.failed)}）")

    tracker.check()                             # 清算：失败率超 20% 在这里炸，原因分布一目了然

    panel = pd.concat(parts, ignore_index=True)
    print(f">>> 因子完成：{len(panel):,} 行 × {len(feat_cols)} 因子，"
          f"覆盖 {panel['code'].nunique()} 只")
    if panel.empty:
        raise PanelError("因子计算结果为空，无法继续")
    return panel


@step("成分过滤（对齐季度快照名单）")
def filter_membership(panel: pd.DataFrame) -> pd.DataFrame:
    """关2：样本只保留"处于最近一期快照名单内"的行（防幸存者偏差）。

    修复记录：最初用 merge_asof(by=code, direction=backward)，但名单表只含有
    成员期间的记录——asof 会命中"该股上一次在名单里的季度"，导致调出指数后
    的样本永远被误判为在成分内（实测日均成分 492、单日最多 633，而真实为 300）。
    改为快照法：样本日 → 向前找最近快照日 → 该日名单必须真的包含此 code。
    """
    mem = load_membership()
    panel["date"] = pd.to_datetime(panel["date"])

    snap_dates = np.sort(mem["date"].unique())            # 48 个季度快照日
    pos = np.searchsorted(snap_dates, panel["date"].to_numpy(), side="right") - 1
    valid = pos >= 0                                      # 早于首个名单日的样本无法验证 → 剔除
    panel = panel[valid].copy()
    panel["snap"] = snap_dates[pos[valid]]

    members = mem[["date", "code"]].rename(columns={"date": "snap"})
    merged = panel.merge(members, on=["snap", "code"], how="left", indicator=True)
    n0 = len(merged)
    merged = merged[merged["_merge"] == "both"].drop(columns=["_merge", "snap"])
    print(f">>> 成分过滤：{n0:,} → {len(merged):,} 行"
          f"（剔除 {n0-len(merged):,} 行非成分期样本）")
    if merged.empty:
        raise PanelError("成分过滤后样本为空——检查 hs300_membership.csv 的 code 格式"
                         "是否与 panel 文件名一致（如 sz.300454）")
    return merged.reset_index(drop=True)


def main() -> pd.DataFrame:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5,
                    help="预测期（交易日）；5=champion 复现，20=长预测期实验")
    ap.add_argument("--out", default=None,
                    help="输出文件名；默认 panel_dataset.parquet（h20 时 panel_dataset_h20.parquet）")
    args = ap.parse_args()
    out_name = args.out or (f"panel_dataset_h{args.horizon}.parquet"
                            if args.horizon != 5 else "panel_dataset.parquet")

    files = sorted(glob.glob(os.path.join(PANEL_DIR, "*.csv")))
    if not files:
        raise PanelError(f"panel 目录为空：{PANEL_DIR}（先跑 fetch_panel.py）")
    panel = filter_membership(compute_all(files, horizon=args.horizon))

    # ---- 横截面健康检查
    per_day = panel.groupby("date")["code"].nunique()
    print("\n=== 横截面规模 ===")
    print(f"总行数 {len(panel):,} | 股票数 {panel['code'].nunique()} | "
          f"日均成分数 {per_day.mean():.0f}（最少 {per_day.min()} / 最多 {per_day.max()}）")
    # 闸2自检：每期名单恰好 300 只，任何一天的在成分数都不可能超过 300
    if per_day.max() > 300:
        raise PanelError(f"单日成分数达到 {per_day.max()} > 300，成分过滤存在泄漏，"
                         f"数据不可用于训练")
    by_year = panel.groupby(panel["date"].dt.year).agg(
        n=("y_bin", "size"), up_ratio=("y_bin", "mean"))
    print("\n=== 按年标签分布 ===")
    print(by_year.to_string(float_format=lambda x: f"{x:.3f}"))

    # ---- 落盘
    out_parquet = os.path.join(DATA_DIR, out_name)
    try:
        panel.to_parquet(out_parquet, index=False)
        print(f"\n>>> 落盘 {out_parquet}")
    except Exception:
        out_csv = os.path.join(DATA_DIR, out_name.rsplit(".", 1)[0] + ".csv")
        panel.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\n>>> 未装 pyarrow，退回 CSV：{out_csv}"
              f"（建议 pip install pyarrow，后续读写快十倍）")
    print(">>> build_dataset_panel.py 完成")
    return panel


if __name__ == "__main__":
    try:
        main()
    except QuantError as e:                     # 只拦守卫认识的自定义错误
        print(f"\n✗ build_dataset_panel 失败：{e}")
        sys.exit(1)
