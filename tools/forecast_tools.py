"""
量化预测运行时工具（Forecast Agent 的数据与打分层）

职责：给定 bs_code（如 sh.600519），在「最新一期沪深300成分」的横截面内，
用训练侧导出的 champion 模型（train/artifacts/lgbm_panel.txt）给全部成分股打分，
返回目标股的相对位置（排名/百分位/十分位）与头部名单。

设计要点：
1. 训练/推理共用同一份因子定义：直接 import train/factors.py 的 compute_factors，
   保证"训练时的20日波动率 = 推理时的20日波动率"，一个字符都不能差；
2. 数据源：train/data/panel/*.csv（BaoStock 日K缓存）。横截面打分需要全成分股
   的因子值，逐股现拉不现实，用缓存；缓存最后日期即"评分日"，报告中如实标注；
3. 与训练分布对齐：同样在源头剔停牌/ST；单日横截面上打分（不跨日期混排）；
4. 本工具只做"相对排序"，不做任何绝对涨跌预测——这是模型的数学边界。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_TOOLS_DIR)
TRAIN_DIR = os.path.join(ROOT, "train")
if TRAIN_DIR not in sys.path:
    sys.path.insert(0, TRAIN_DIR)

from factors import compute_factors  # noqa: E402  训练/推理唯一事实源

if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
from auto_maintain import wait_refresh_lock  # noqa: E402  数据刷新期间打分等待

ART_DIR = os.path.join(TRAIN_DIR, "artifacts")
PANEL_DIR = os.path.join(TRAIN_DIR, "data", "panel")
MEMBERSHIP_PATH = os.path.join(TRAIN_DIR, "data", "hs300_membership.csv")

MIN_CROSS_SECTION = 200   # 评分日横截面少于 200 只视为数据异常，拒绝出分
MAX_STALE_DAYS = 7        # 个股停牌容忍：最后交易日距评分日超过 7 个自然日则剔除


def _load_model():
    """加载导出的 champion 模型与元数据（特征顺序必须与训练一致）。"""
    with open(os.path.join(ART_DIR, "lgbm_panel.meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    import lightgbm as lgb
    booster = lgb.Booster(model_file=os.path.join(ART_DIR, "lgbm_panel.txt"))
    return meta, booster


def _latest_members():
    """最新一期成分快照（fetch_panel.py 的季度时点名单）。"""
    mem = pd.read_csv(MEMBERSHIP_PATH)
    mem["date"] = pd.to_datetime(mem["date"])
    snap = mem["date"].max()
    codes = sorted(mem.loc[mem["date"] == snap, "code"].unique())
    return snap, codes


def get_forecast(bs_code: str):
    """
    返回 (result_dict, None) 或 (None, 失败原因)。
    result_dict 字段：score_date / snap_date / n_scored / rank / total /
    percentile / decile / top10 / target_score。
    """
    try:
        # 后台日更任务正在刷新 panel 缓存时，等它结束再打分，
        # 避免横截面一半新一半旧（最多等 10 分钟，超时按旧数据兜底）
        wait_refresh_lock(timeout=600)
        meta, booster = _load_model()
    except Exception as e:
        return None, f"模型加载失败（先跑 train/export_champion.py）：{e}"

    feat_cols = meta["features"]
    try:
        snap_date, codes = _latest_members()
    except Exception as e:
        return None, f"成分名单读取失败：{e}"
    if bs_code not in codes:
        return None, (f"{bs_code} 不在最新一期沪深300成分中"
                      f"（快照日 {snap_date:%Y-%m-%d}），横截面模型无法对其排序")

    # ---- 逐股取"最后一条有效交易日"的因子值（源头剔停牌/ST，与训练对齐）
    rows = {}
    for code in codes:
        path = os.path.join(PANEL_DIR, f"{code}.csv")
        if not os.path.exists(path):
            continue
        try:
            raw = pd.read_csv(path)
            if "tradestatus" in raw.columns:
                raw = raw[raw["tradestatus"] == 1]
            if "isST" in raw.columns:
                raw = raw[raw["isST"] == 0]
            if raw.empty or len(raw) < 120:      # 窗口不满的次新股，因子全是 NaN
                continue
            fac = compute_factors(raw)
            rows[code] = fac.iloc[-1]
        except Exception:
            continue                              # 单股失败容忍，清算看数量

    if len(rows) < MIN_CROSS_SECTION:
        return None, (f"有效成分股仅 {len(rows)} 只（< {MIN_CROSS_SECTION}），"
                      f"疑似 panel 缓存缺失（先跑 fetch_panel.py）")
    cross = pd.DataFrame(rows).T
    cross["date"] = pd.to_datetime(cross["date"])

    # ---- 评分日 = 缓存中最新交易日；只保留当日有行情的股票（单日横截面，不混日期）
    score_date = cross["date"].max()
    cross = cross[cross["date"] == score_date].copy()
    if bs_code not in cross.index:
        return None, (f"{bs_code} 在评分日 {score_date:%Y-%m-%d} 无行情"
                      f"（可能停牌），本期无法排序")

    X = cross[feat_cols].astype(float)
    cross["p"] = booster.predict(X)

    # ---- 排名统计（p 越大越被看好；rank 1 = 横截面最强）
    cross = cross.sort_values("p", ascending=False)
    total = len(cross)
    rank = int((cross["p"] > cross.loc[bs_code, "p"]).sum()) + 1
    pct = rank / total
    decile = int(np.ceil(pct * 10))
    top10 = list(cross.index[:10])

    return {
        "score_date": str(score_date.date()),
        "snap_date": str(snap_date.date()),
        "n_scored": total,
        "rank": rank,
        "total": total,
        "percentile": round(pct * 100, 1),
        "decile": decile,
        "top10": top10,
        "target_score": round(float(cross.loc[bs_code, "p"]), 4),
        "horizon_days": meta["horizon_days"],
    }, None


# ---------------- 自检：python forecast_tools.py ----------------
if __name__ == "__main__":
    import time
    t0 = time.time()
    res, err = get_forecast("sh.600519")
    if err:
        print(f"✗ {err}")
        sys.exit(1)
    print(f">>> 耗时 {time.time()-t0:.1f}s")
    print(json.dumps(res, ensure_ascii=False, indent=2))
