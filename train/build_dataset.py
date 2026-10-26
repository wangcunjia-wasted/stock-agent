"""
训练集构造：因子矩阵 + 标签（未来 N 日涨跌）

职责边界：
  - factors.py   只管"给定截至 T 的数据，算 T 时刻的因子"（向后看）
  - build_dataset.py  管"T 时刻的因子 配 T 时刻可知的标签"（向前打标，但只在训练集构造时用未来数据——
    这不构成前视：模型在 T 时刻推理时只输入因子，标签只在训练/评估时被看见）
  - train_lgbm.py 管"怎么切窗口、怎么训"（下一块砖）

防前视三道闸（本文件负责第二道）：
  1. 因子 rolling 向后看           —— factors.py 的截断断言（已过）
  2. 标签对齐 + 尾部窗口截断       —— 本文件的 label_alignment 断言
  3. 时序切分禁随机                —— train_lgbm.py 的 walk-forward（下一步）
"""
import os

import numpy as np
import pandas as pd

from factors import FACTOR_DICT, compute_factors

HORIZON = 5                      # 预测未来 5 个交易日的收益
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# 时序切分（仅预览用；真正训练用 walk-forward 滚动窗口）
SPLIT_TRAIN_END = "2022-12-31"
SPLIT_VALID_END = "2024-12-31"


def build_labels(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """
    标签定义：T 日收盘买入，持有 horizon 个交易日的收益。
      y_ret : close[T+h] / close[T] - 1     （回归目标）
      y_bin : y_ret > 0                      （分类目标）
    用 shift(-h) 一次向量化，等价于逐行取 close[T+h]，最后 h 行无未来 → 置 NaN 后丢弃。
    """
    out = df.copy()
    out["y_ret"] = out["close"].shift(-horizon) / out["close"] - 1
    out["y_bin"] = (out["y_ret"] > 0).astype(float)
    out.loc[out["y_ret"].isna(), "y_bin"] = np.nan     # 尾部 h 行标签无效
    return out


def build_dataset(csv_path: str, horizon: int = HORIZON, raw: pd.DataFrame = None) -> pd.DataFrame:
    """一只股票的完整数据集：date + 21 因子 + 标签。逐股调用，P1 扩池时循环 concat。
    raw：可选，调用方预过滤后的原始数据（如剔除停牌/ST 行）。"""
    if raw is None:
        raw = pd.read_csv(csv_path)
    feat = compute_factors(raw)                        # date, close, F_*
    ds = build_labels(feat, horizon)
    n_before = len(ds)
    ds = ds.dropna(subset=["y_ret"]).reset_index(drop=True)   # 闸2：尾部无标签行必须丢弃
    assert len(ds) == n_before - horizon, (
        f"标签截断异常：丢了 {n_before - len(ds)} 行，应为 {horizon}")
    return ds


def label_alignment_check(ds: pd.DataFrame, horizon: int = HORIZON, n_probe: int = 5):
    """闸2核心断言：随机抽 n 行，手工重算未来收益必须与 shift 版完全一致。"""
    rng = np.random.default_rng(42)
    probe = rng.choice(np.arange(len(ds) - horizon), size=n_probe, replace=False)
    raw = pd.read_csv(os.path.join(DATA_DIR, "600519_daily.csv"))
    closes = raw["close"].to_numpy()
    dates = raw["date"].to_numpy()
    for i in probe:
        d = ds.loc[i, "date"]
        j = int(np.where(dates == d)[0][0])
        manual = closes[j + horizon] / closes[j] - 1
        assert abs(manual - ds.loc[i, "y_ret"]) < 1e-12, f"标签错位 @ {d}"
    return True


def split_preview(ds: pd.DataFrame):
    """时序三段切分的数量预览（不用随机，按日期边界）。"""
    tr = ds[ds["date"] <= SPLIT_TRAIN_END]
    va = ds[(ds["date"] > SPLIT_TRAIN_END) & (ds["date"] <= SPLIT_VALID_END)]
    te = ds[ds["date"] > SPLIT_VALID_END]
    return tr, va, te


if __name__ == "__main__":
    csv_path = os.path.join(DATA_DIR, "600519_daily.csv")
    ds = build_dataset(csv_path)
    print(f">>> 数据集：{len(ds)} 行（因子 {len(FACTOR_DICT)} 个 + y_ret/y_bin）")
    print(f">>> 日期范围 {ds['date'].min()} ~ {ds['date'].max()}（尾部 {HORIZON} 行无标签已丢弃）")

    # ---- 闸2：标签对齐断言
    ok = label_alignment_check(ds)
    print(f"\n=== 标签对齐断言（手工重算 == shift 计算）→ {ok}（必须为 True）===")

    # ---- 类别平衡（分类目标若严重偏斜，accuracy 不可信，须看 AUC）
    up_ratio = ds["y_bin"].mean()
    print(f"\n=== 标签分布：上涨占比 {up_ratio:.3f}（5日二分类，0.5 附近为健康）===")
    by_year = ds.groupby(ds["date"].str[:4])["y_bin"].agg(["mean", "count"])
    by_year.columns = ["up_ratio", "n"]
    print(by_year.to_string(float_format=lambda x: f"{x:.3f}"))

    # ---- NaN 情况：LightGBM 原生吃 NaN，只统计不填充
    feat_cols = list(FACTOR_DICT.keys())
    nan_per_row = ds[feat_cols].isna().mean(axis=1)
    print(f"\n=== 因子缺失：整行全 NaN 的行数 {int((nan_per_row == 1).sum())}，"
          f"平均缺失率 {nan_per_row.mean():.3f}（前 250 日窗口不满所致，保留不填充）===")

    # ---- 时序切分预览
    tr, va, te = split_preview(ds)
    print(f"\n=== 时序切分预览（walk-forward 之前先看静态边界）===")
    print(f"train {len(tr):4d} 行  {tr['date'].min()}~{tr['date'].max()}  up_ratio {tr['y_bin'].mean():.3f}")
    print(f"valid {len(va):4d} 行  {va['date'].min()}~{va['date'].max()}  up_ratio {va['y_bin'].mean():.3f}")
    print(f"test  {len(te):4d} 行  {te['date'].min()}~{te['date'].max()}  up_ratio {te['y_bin'].mean():.3f}")

    out_path = os.path.join(DATA_DIR, "dataset_600519.csv")
    ds.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n>>> 落盘 {out_path}")
    print(">>> build_dataset.py 自检通过")
