"""
LightGBM + 横截面 walk-forward 训练与评估（P1 面板版）

与 P0 单票版的区别：
  - 滚动单位从"行"变为"交易日"：每天是一个横截面（约 300 只成分股）
  - 评估核心是逐日横截面 RankIC：每一天对当天成分股做"预测分 vs 未来5日收益"
    的 spearman 相关，汇总出 IC 均值 / ICIR / t 值——这是横截面选股信号的标准体检
  - 模型纪律与 P0 完全一致：扩展窗（样本只增不减）、超参零改动（可比性）、
    LightGBM 原生吃 NaN，LR 基线用训练折统计量填补（统计量不来自测试折）
  - 基线只保留 LR：问题本质是"非线性树模型 vs 线性模型在横截面上差多少"，
    RF 在 84 万行上又慢又不提供额外信息增量（P0 已验证 RF≈LGBM）

防前视复核（三道闸沿用）：
  1. 因子 rolling 向后看（factors.py 截断断言）
  2. 标签对齐（build_dataset 断言）
  3. 时序切分禁随机（本文件 walk-forward：模型只见过测试窗之前的数据）
用法：
  python train_lgbm_panel.py                        # champion：21量价因子 + 5日标签
  python train_lgbm_panel.py --features fund30 --panel data/panel_dataset_h20.parquet
                                                    # 长预测期实验：21量价+9基本面，20日标签
"""
import argparse
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from factors import FACTOR_DICT, FUNDAMENTAL_FACTORS
from guard import PanelError, QuantError, get_logger, step

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
PANEL_PATH = os.path.join(DATA_DIR, "panel_dataset.parquet")

# ---------------- 特征集（P2 三组对照实验的裁决记录，--features 选择）
#   price21：21 个量价因子（champion，5日预测）
#   fund30 ：price21 + 估值三件套(F_ep/F_bp/F_npcf) + 基本面6（ROE/毛利率/净利率/
#            负债率/营收同比/净利同比）。慢变量救不了 5 日快预测，但 20 日+ 未见得——
#            这是长预测期实验要回答的问题。
FEATURE_SETS = {
    "price21": [c for c in FACTOR_DICT if c not in ("F_size", "F_liq")],
    "fund30": [c for c in FACTOR_DICT if c not in ("F_size", "F_liq")]
              + list(FUNDAMENTAL_FACTORS.keys()),
}

# ---------------- walk-forward 配置（单位：交易日，不是行）
MIN_TRAIN_DAYS = 750     # 首折至少 3 年训练数据
STEP_DAYS = 126          # 每半年滚动一次

# ---------------- LightGBM 参数：与 P0 完全一致，保证单票/横截面可比
LGB_PARAMS = dict(
    objective="binary", n_estimators=300, learning_rate=0.05,
    num_leaves=15, max_depth=4, min_child_samples=30,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=1.0, random_state=42, verbose=-1,
)
MIN_STOCKS_PER_DAY = 30  # 少于 30 只的横截面不做日度 IC（排序无意义）


@step("面板数据加载")
def load_panel(panel_path: str = PANEL_PATH, feat_key: str = "price21") -> pd.DataFrame:
    if not os.path.exists(panel_path):
        raise PanelError(f"面板数据不存在：{panel_path}（先跑 build_dataset_panel.py）")
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "code"]).reset_index(drop=True)

    feat_cols = FEATURE_SETS[feat_key]
    missing = [c for c in feat_cols if c not in panel.columns]
    if missing:
        raise PanelError(f"面板缺因子列：{missing}")
    print(f">>> 读入面板 {len(panel):,} 行 | {panel['code'].nunique()} 只 | "
          f"{panel['date'].nunique()} 个交易日 "
          f"（{panel['date'].min():%Y-%m-%d} ~ {panel['date'].max():%Y-%m-%d}）"
          f" | 特征集 {feat_key}（{len(feat_cols)} 因子）")
    return panel


@step("walk-forward 训练（扩展窗，按交易日滚动）")
def walk_forward(panel: pd.DataFrame, feat_key: str = "price21") -> pd.DataFrame:
    """
    [0, t) 个交易日训练 → 预测 [t, t+STEP)；t 每次 +STEP，样本只增不减。
    返回 OOF 明细（date, code, y_ret, y_bin, p_lgbm, p_lr, fold）。
    """
    feat_cols = FEATURE_SETS[feat_key]
    X = panel[feat_cols]
    y = panel["y_bin"].astype(int)

    days = panel["date"].sort_values().unique()        # 交易日轴（不是行轴）
    n_days = len(days)
    day_pos = pd.Series(np.arange(n_days), index=days)  # date → 交易日序号
    p_days = day_pos.reindex(panel["date"]).to_numpy()  # 每行的交易日序号

    oof_parts = []
    fold_id = 0
    t = MIN_TRAIN_DAYS
    while t < n_days:
        te_end = min(t + STEP_DAYS, n_days)
        tr_mask, te_mask = p_days < t, (p_days >= t) & (p_days < te_end)
        X_tr, y_tr = X[tr_mask], y[tr_mask]
        X_te = X[te_mask]

        # --- LightGBM：NaN 原生处理（与 P0 同参）
        m_lgb = lgb.LGBMClassifier(**LGB_PARAMS)
        m_lgb.fit(X_tr, y_tr)
        p_lgb = m_lgb.predict_proba(X_te)[:, 1]

        # --- LR 基线：训练折中位数填补 + 标准化（统计量只来自训练折）
        imp = SimpleImputer(strategy="median").fit(X_tr)
        m_lr = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=0.5))
        m_lr.fit(imp.transform(X_tr), y_tr)
        p_lr = m_lr.predict_proba(imp.transform(X_te))[:, 1]

        part = panel.loc[te_mask, ["date", "code", "y_ret", "y_bin"]].copy()
        part["p_lgbm"], part["p_lr"] = p_lgb, p_lr
        part["fold"] = fold_id
        oof_parts.append(part)
        print(f"  fold {fold_id:2d} | train ≤{days[t-1]:%Y-%m-%d}（{t} 日）"
              f" | test {days[t]:%Y-%m-%d}~{days[te_end-1]:%Y-%m-%d}"
              f" | n_test {te_mask.sum():,} 行")
        fold_id += 1
        t = te_end

    return pd.concat(oof_parts, ignore_index=True)


def daily_rank_ic(oof: pd.DataFrame) -> pd.DataFrame:
    """逐日横截面 RankIC：每天对成分股做 spearman(预测, 未来5日收益)。
    这是横截面信号的标准体检——P0 的整体 spearman 会把不同日期混在一起，
    掩盖"某天排序对、某天全错"的真实结构。"""
    rows = []
    for d, g in oof.groupby("date"):
        if len(g) < MIN_STOCKS_PER_DAY:
            continue
        rows.append({
            "date": d, "n": len(g),
            "ic_lgbm": spearmanr(g["p_lgbm"], g["y_ret"]).statistic,
            "ic_lr": spearmanr(g["p_lr"], g["y_ret"]).statistic,
        })
    return pd.DataFrame(rows).set_index("date")


def report(oof: pd.DataFrame):
    lg = get_logger()
    ic = daily_rank_ic(oof)

    print("\n===== 样本外总成绩（OOF 拼接）=====")
    pooled = []
    for col in ("p_lgbm", "p_lr"):
        pooled.append({
            "model": col,
            "AUC": roc_auc_score(oof["y_bin"], oof[col]),
            "日度RankIC均值": ic[f"ic_{col.split('_')[1]}"].mean(),
        })
    pooled.append({"model": "all-up(先验)", "AUC": 0.5, "日度RankIC均值": np.nan})
    rep = pd.DataFrame(pooled).set_index("model")
    print(rep.to_string(float_format=lambda x: f"{x:.4f}"))

    print(f"\n===== 逐日横截面 RankIC 体检（{len(ic)} 个有效交易日）=====")
    for col in ("ic_lgbm", "ic_lr"):
        m, s = ic[col].mean(), ic[col].std()
        t_stat = m / s * np.sqrt(len(ic[col]))
        pos_ratio = (ic[col] > 0).mean()
        print(f"{col:8s} | 均值 {m:+.4f} | 标准差 {s:.4f} | ICIR {m/s:+.3f} | "
              f"t值 {t_stat:+.2f} | IC>0 占比 {pos_ratio:.1%}")
    print("   参考系：|日度IC| 稳定 >0.02 且 t>2 算可用信号；ICIR>0.3 优秀；"
          "绝对值 <0.01 基本是噪声")

    print("\n===== 分折 AUC（p_lgbm，看稳定性）=====")
    per_fold = []
    for f, g in oof.groupby("fold"):
        per_fold.append({
            "fold": f, "n": len(g),
            "AUC": roc_auc_score(g["y_bin"], g["p_lgbm"]),
            "RankIC": spearmanr(g["p_lgbm"], g["y_ret"]).statistic,
            "up_ratio": g["y_bin"].mean(),
        })
    pf = pd.DataFrame(per_fold).set_index("fold")
    print(pf.to_string(float_format=lambda x: f"{x:.3f}"))

    lg.info(f"OOF AUC(lgbm)={rep.loc['p_lgbm', 'AUC']:.4f} "
            f"日度IC均值={ic['ic_lgbm'].mean():+.4f} ICIR={ic['ic_lgbm'].mean()/ic['ic_lgbm'].std():+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=PANEL_PATH, help="面板 parquet 路径")
    ap.add_argument("--features", default="price21", choices=list(FEATURE_SETS))
    ap.add_argument("--oof-out", default=None, help="OOF 输出文件名（默认 oof_panel.csv）")
    args = ap.parse_args()

    panel = load_panel(args.panel, args.features)
    oof = walk_forward(panel, args.features)
    report(oof)

    out_name = args.oof_out or ("oof_panel.csv" if args.features == "price21"
                                and args.panel == PANEL_PATH
                                else f"oof_{args.features}_{os.path.basename(args.panel).split('.')[0]}.csv")
    out_csv = os.path.join(DATA_DIR, out_name)
    oof.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n>>> OOF 预测落盘 {out_csv}（evaluate_panel.py 的分位数回测+成本用）")
    print(">>> train_lgbm_panel.py 完成")


if __name__ == "__main__":
    try:
        main()
    except QuantError as e:                     # 只拦守卫认识的自定义错误
        print(f"\n✗ train_lgbm_panel 失败：{e}")
        sys.exit(1)
