"""
P2 基本面数据拉取（akshare/新浪财经，单股一次调用拿全历史，断点续跑）

data/fundamental/{code}.csv：报告期 + 原始字段（百分数/元）
  净资产收益率 / 销售毛利率 / 销售净利率 / 资产负债率
  / 主营业务收入增长率 / 净利润增长率
  / 摊薄每股收益 / 每股净资产 / 每股经营性现金流   ← 估值因子（EP/BP/PCF）的原料

时点纪律（本项目的命根子）：季报的"报告期"≠"可得日"。
  拉数阶段只存报告期，对齐阶段（build_dataset_panel）统一用法定披露截止日
  作为可用日：Q1→4/30、中报→8/31、三季报→10/31、年报→次年4/30。
  所有公司在截止日前必然披露完毕 → 零前视（宁可晚用一个月，绝不早用一天）。

弃用说明：原 Phase A 走 BaoStock 拉 peTTM 等日频估值字段，实测 ~45s/股
  （668 只 ≈ 8 小时）不可接受；估值改为季报每股指标派生（快 30 倍以上，
  时点同样干净），F_sp 因无干净数据源暂弃。
"""
import glob
import os
import sys
import time

import akshare as ak
import pandas as pd

from guard import FailureTracker, PanelError, QuantError, get_logger, step

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
FUND_DIR = os.path.join(DATA_DIR, "fundamental")

FUND_COLS = ["日期",
             "净资产收益率(%)", "销售毛利率(%)", "销售净利率(%)",
             "资产负债率(%)", "主营业务收入增长率(%)", "净利润增长率(%)",
             "摊薄每股收益(元)", "每股净资产_调整前(元)", "每股经营性现金流(元)"]


def codes_from_panel() -> list:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "panel", "*.csv")))
    if not files:
        raise PanelError("panel 目录为空（先跑 fetch_panel.py）")
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def atomic_csv(df: pd.DataFrame, path: str):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


@step("季报基本面拉取（akshare，668只）")
def fetch_fundamental():
    os.makedirs(FUND_DIR, exist_ok=True)
    codes = codes_from_panel()
    tracker = FailureTracker("fundamental拉取", max_fail_rate=0.2)
    t0 = time.time()
    for i, code in enumerate(codes):
        out = os.path.join(FUND_DIR, f"{code}.csv")
        if os.path.exists(out) and os.path.getsize(out) > 50:
            tracker.record_ok(code)
        else:
            symbol = code.split(".")[1]           # sz.300750 -> 300750
            try:
                df = ak.stock_financial_analysis_indicator(
                    symbol=symbol, start_year="2014")
                missing = [c for c in FUND_COLS if c not in df.columns]
                if missing:
                    raise RuntimeError(f"缺字段 {missing}")
                keep = df[FUND_COLS].copy()
                if keep.empty:
                    raise RuntimeError("akshare 返回空")
                atomic_csv(keep, out)
                tracker.record_ok(code)
            except Exception as e:
                tracker.record_fail(code, e)
                print(f"  ⚠️ {code}: {e}", flush=True)
        if (i + 1) % 25 == 0:
            print(f"  基本面进度 {i+1}/{len(codes)}（{time.time()-t0:.0f}s，"
                  f"失败 {len(tracker.failed)}）", flush=True)
    tracker.check()


def main():
    fetch_fundamental()
    print("\n>>> fetch_fundamental_panel.py 完成")


if __name__ == "__main__":
    try:
        main()
    except QuantError as e:
        print(f"\n✗ fetch_fundamental_panel 失败：{e}")
        sys.exit(1)
