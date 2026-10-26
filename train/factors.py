"""
因子工程 —— 唯一事实源（Single Source of Truth）

训练（train/）和推理（tools/forecast_tools.py）都 import 本模块计算因子，
保证「训练时 '过去20日波动率' 的定义 = 推理时 '过去20日波动率' 的定义」，
一个字符都不能差。

设计纪律：
1. 所有因子只用「当日及之前」的数据 —— rolling() 天然向后看，绝不 shift(-n)；
2. 未来收益（标签）不在这里算，那是 build_dataset.py 的事；
3. 缺失值一律保留 NaN 交给 LightGBM 处理，不许 fillna(0) 造假数据；
4. 输入输出统一用英文列名（BaoStock 原始列名），与 data/history.py 对齐。

输入 df 必需列：date, open, high, low, close, volume, amount, turn, pctChg
输出 df：原列 + F_ 前缀的因子列（F_ 前缀是为了和原始列一眼区分开）
"""

import numpy as np
import pandas as pd

# 因子清单：name -> 中文说明（供特征元数据落盘与日志打印）
FACTOR_DICT = {
    # ---- 动量类 ----
    "F_ret_1d":   "日收益率",
    "F_ret_5d":   "过去5日累计收益（短线动量）",
    "F_ret_20d":  "过去20日累计收益（月度动量）",
    "F_ret_60d":  "过去60日累计收益（季度动量）",
    # ---- 均线偏离类 ----
    "F_bias_5":   "收盘价对MA5偏离",
    "F_bias_20":  "收盘价对MA20偏离",
    "F_bias_60":  "收盘价对MA60偏离",
    # ---- 波动类 ----
    "F_vol_20d":  "20日收益率标准差（月波动率）",
    "F_vol_60d":  "60日收益率标准差（季波动率）",
    "F_amp_20d":  "20日平均振幅(high/low-1)",
    "F_atr_14":   "ATR(14)/收盘价（归一化真实波幅）",
    # ---- 量能类 ----
    "F_vol_ratio":   "成交量/20日均量（量比）",
    "F_turn_ratio":  "换手率/20日均换手（换手异动）",
    "F_turn_trend":  "5日均换手/20日均换手（换手趋势）",
    "F_pv_corr_20":  "20日量价相关系数（放量滞涨/缩量上涨信号）",
    # ---- 价格位置类 ----
    "F_pos_20d":   "20日价格位置 (close-low20)/(high20-low20)",
    "F_pos_60d":   "60日价格位置",
    "F_pos_250d":  "年线位置（距52周高点）",
    # ---- 技术指标类 ----
    "F_rsi_14":    "RSI(14)",
    "F_macd_dif":  "MACD DIF/价格（归一化）",
    "F_macd_hist": "MACD 柱/价格（归一化）",
    # ---- 规模/流动性（P2）----
    "F_size": "ln(流通市值)，由 close×volume/turn% 反推",
    "F_liq":  "ln(20日均成交额)",
}

# P2 基本面因子：季报数据，panel 层按"法定披露可用日"asof 对齐后拼接，
# 不在 compute_factors 内计算（它们的时点纪律是公告日，不是交易日 rolling）。
# 估值三件套由季报每股指标派生：F_ep=TTM每股收益/close、F_bp=每股净资产/close、
# F_npcf=TTM每股经营现金流/close（TTM 在 build_dataset_panel 内构造）
FUNDAMENTAL_FACTORS = {
    "F_ep":      "盈利收益率 TTM每股收益/收盘价",
    "F_bp":      "账面收益率 每股净资产/收盘价",
    "F_npcf":    "现金流收益率 TTM每股经营现金流/收盘价",
    "F_roe":     "净资产收益率(季报)",
    "F_gpm":     "销售毛利率(季报)",
    "F_npm":     "销售净利率(季报)",
    "F_debt":    "资产负债率(季报)",
    "F_rev_yoy": "主营业务收入同比",
    "F_ni_yoy":  "净利润同比",
}


def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入单只股票的日线数据，返回带 F_ 因子列的 DataFrame。
    行数与输入一致、按 date 升序；前 N 行（窗口不满）为 NaN 属正常。
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    c, h, l = df["close"], df["high"], df["low"]
    v, t = df["volume"], df["turn"]
    r = df["pctChg"] / 100.0          # BaoStock 的 pctChg 是百分数

    def ma(s, n):
        return s.rolling(n, min_periods=n).mean()

    def std(s, n):
        return s.rolling(n, min_periods=n).std()

    # ---------------- 动量 ----------------
    df["F_ret_1d"] = r
    for n in (5, 20, 60):
        df[f"F_ret_{n}d"] = c / c.shift(n) - 1

    # ---------------- 均线偏离 ----------------
    for n in (5, 20, 60):
        df[f"F_bias_{n}"] = c / ma(c, n) - 1

    # ---------------- 波动 ----------------
    df["F_vol_20d"] = std(r, 20)
    df["F_vol_60d"] = std(r, 60)
    df["F_amp_20d"] = (h / l - 1).rolling(20, min_periods=20).mean()
    # ATR(14)：真实波幅 = max(高-低, |高-昨收|, |低-昨收|)
    pre_c = df["preclose"] if "preclose" in df.columns else c.shift(1)
    tr = pd.concat([h - l, (h - pre_c).abs(), (l - pre_c).abs()], axis=1).max(axis=1)
    df["F_atr_14"] = tr.rolling(14, min_periods=14).mean() / c

    # ---------------- 量能 ----------------
    df["F_vol_ratio"] = v / ma(v, 20)
    df["F_turn_ratio"] = t / ma(t, 20)
    df["F_turn_trend"] = ma(t, 5) / ma(t, 20)
    # 量价相关：未来当日收益与当日成交额的相关性，20 日窗
    df["F_pv_corr_20"] = r.rolling(20, min_periods=20).corr(np.log(v + 1))

    # ---------------- 价格位置 ----------------
    for n in (20, 60):
        lo, hi = l.rolling(n, min_periods=n).min(), h.rolling(n, min_periods=n).max()
        df[f"F_pos_{n}d"] = (c - lo) / (hi - lo)
    hi_250 = h.rolling(250, min_periods=120).max()   # 次新股窗口不满时降低要求
    df["F_pos_250d"] = c / hi_250 - 1

    # ---- 技术指标 ----------------
    # RSI(14)：用 Wilder 平滑的简化版（ewm alpha=1/n）
    gain = r.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-r.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["F_rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    # MACD(12,26,9)，除以价格归一化（不同价位的股票可比）
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    df["F_macd_dif"] = dif / c
    df["F_macd_hist"] = (dif - dea) / c

    # ---------------- 规模 / 流动性（P2 新增，来源：现有量价列派生）----------------
    # 流通市值 = 收盘价 × 流通股本；流通股本 = 成交量 / (换手率/100)
    #   （turn 是百分数：换手率% = volume/流通股本×100）
    float_mv = c * v * 100.0 / t.replace(0, np.nan)
    df["F_size"] = np.log(float_mv.replace(0, np.nan))
    # 流动性：20 日均成交额（amount 单位：元），取对数压缩长尾
    amt = pd.to_numeric(df["amount"], errors="coerce") if "amount" in df.columns \
        else pd.Series(np.nan, index=df.index)
    df["F_liq"] = np.log(amt.rolling(20, min_periods=20).mean().replace(0, np.nan))

    return df


def factor_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """每个因子的非缺失率与样本数 —— 自检用，覆盖率异常低说明窗口/数据有问题"""
    stat = pd.DataFrame({
        "coverage": [df[col].notna().mean() for col in FACTOR_DICT],
        "n_valid": [int(df[col].notna().sum()) for col in FACTOR_DICT],
    }, index=FACTOR_DICT.keys())
    return stat


# ============================================================
# 自检：直接运行本文件，用现有茅台数据算一遍因子
# ============================================================
if __name__ == "__main__":
    import os

    # 训练资产自包含：数据在 train/data/，由 fetch_data.py 生成；缺失时现拉兜底
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "600519_daily.csv")
    if os.path.exists(csv_path):
        raw = pd.read_csv(csv_path)
        print(f">>> 读取本地缓存 {csv_path}：{len(raw)} 条")
    else:
        import baostock as bs
        bs.login()
        fields = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg"
        rs = bs.query_history_k_data_plus("sh.600519", fields,
                                          start_date="2020-01-01", end_date="2026-10-24",
                                          frequency="d", adjustflag="2")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        bs.logout()
        raw = pd.DataFrame(rows, columns=rs.fields)
        for col in ("open", "high", "low", "close", "volume", "amount", "turn", "pctChg"):
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        print(f">>> BaoStock 现拉：{len(raw)} 条")

    out = compute_factors(raw)
    print(f">>> 因子计算完成：{out.shape[0]} 行 × {len(FACTOR_DICT)} 个因子")

    stat = factor_coverage(out)
    print("\n=== 因子覆盖率自检（coverage 应接近 1，前 60 行低是窗口不满，属正常）===")
    print(stat.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n=== 最新 3 行因子值抽样 ===")
    cols = ["date", "close"] + list(FACTOR_DICT.keys())
    print(out[cols].tail(3).to_string(index=False))

    # 前视 sanity check：把数据砍到第 T 行，因子值必须与全量计算的第 T 行完全一致
    # （rolling 只向后看，砍尾不影响历史行 —— 这条断言过了，防前视才算焊死）
    cut = len(out) - 100
    out_trunc = compute_factors(raw.iloc[:cut])
    fcols = list(FACTOR_DICT.keys())
    same = np.allclose(out[fcols].iloc[cut - 1], out_trunc[fcols].iloc[cut - 1],
                       equal_nan=True)
    print(f"\n=== 前视断言：截断重算 == 全量计算 → {same}（必须为 True）===")
    assert same, "因子存在前视！检查是否有 shift(-n) 或用了未来数据"
    print(">>> factors.py 自检通过")
