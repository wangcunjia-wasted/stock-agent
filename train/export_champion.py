"""
把横截面 champion（price21 特征集 + 5日标签）在全量面板上训练最终模型，
落盘 artifacts/ 供运行时 Forecast Agent 推理（tools/forecast_tools.py）。

与 walk-forward 的关系：walk-forward 产出的是 OOF（评估用），每个 fold 一个
临时模型、用完即弃；本脚本用「全部数据」训练一个最终模型——部署的标准做法
（上线时当然要把所有历史数据都用上）。它的样本外成绩以 OOF 为准，本脚本
不重复评估，只做训练集内 sanity check（训练 AUC 应高于 OOF，属正常过拟合）。

用法：python export_champion.py
"""
import json
import os
import sys
from datetime import date

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

from train_lgbm_panel import FEATURE_SETS, LGB_PARAMS, PANEL_PATH
from guard import DataError, QuantError, get_logger, step

BASE = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(BASE, "artifacts")
MODEL_PATH = os.path.join(ART_DIR, "lgbm_panel.txt")
META_PATH = os.path.join(ART_DIR, "lgbm_panel.meta.json")


@step("全量面板加载")
def load_full_panel() -> pd.DataFrame:
    if not os.path.exists(PANEL_PATH):
        raise DataError(f"面板不存在：{PANEL_PATH}（先跑 build_dataset_panel.py）")
    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel


@step("最终模型训练与导出")
def export_model(panel: pd.DataFrame):
    feat_cols = FEATURE_SETS["price21"]

    # 标签完整的样本才可训练（最后 5 个交易日因子在、标签 NaN，自动剔除）
    df = panel.dropna(subset=["y_bin"]).copy()
    X, y = df[feat_cols], df["y_bin"].astype(int)

    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X, y)

    os.makedirs(ART_DIR, exist_ok=True)
    # 原子落盘：先写临时文件再 os.replace，运行时打分进程（forecast_tools）
    # 与本脚本并发时不会读到半截模型文件
    model.booster_.save_model(MODEL_PATH + ".tmp")
    os.replace(MODEL_PATH + ".tmp", MODEL_PATH)

    meta = {
        "features": feat_cols,               # 顺序即训练顺序，推理必须一致
        "horizon_days": 5,                   # 标签 = 未来5个交易日涨跌（y_bin）
        "feature_set": "price21",
        "params": LGB_PARAMS,
        "n_train_rows": int(len(df)),
        "n_stocks": int(df["code"].nunique()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "up_ratio": float(y.mean()),
        "train_auc_insample": round(roc_auc_score(y, model.predict_proba(X)[:, 1]), 4),
        "oof_reference": {                   # 样本外成绩以 walk-forward OOF 为准
            "auc": 0.5230, "daily_rank_ic": 0.0485, "icir": 0.325,
            "source": "train_lgbm_panel.py walk-forward 17折（2026-10-25）",
        },
        "universe": "沪深300历史成分（时点快照过滤）",
        "exported_at": str(date.today()),
    }
    with open(META_PATH + ".tmp", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(META_PATH + ".tmp", META_PATH)

    print(f">>> 模型落盘 {MODEL_PATH}")
    print(f">>> 元数据落盘 {META_PATH}")
    print(f"    训练样本 {len(df):,} 行（{meta['date_min']} ~ {meta['date_max']}）"
          f" | {len(feat_cols)} 因子 | 上涨占比 {meta['up_ratio']:.1%}")
    print(f"    训练集内 AUC {meta['train_auc_insample']:.4f}"
          f"（OOF 参考 0.5230，训练集内更高属正常过拟合，勿当成绩宣传）")


def main():
    panel = load_full_panel()
    export_model(panel)
    get_logger().info("champion 模型已导出到 artifacts/lgbm_panel.txt")
    print(">>> export_champion.py 完成")


if __name__ == "__main__":
    try:
        main()
    except QuantError as e:
        print(f"\n✗ export_champion 失败：{e}")
        sys.exit(1)
