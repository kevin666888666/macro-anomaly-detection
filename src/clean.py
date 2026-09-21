# -*- coding: utf-8 -*-
"""
clean.py — 数据清洗与派生指标

用法:
    python src/clean.py

输入:  data/raw/macro_raw.csv      (由 fetch_wind.py 生成)
输出:  data/processed/macro_clean.csv
       data/processed/macro_features.csv

清洗内容:
    1. 统一为月末日期
    2. 补齐缺失的月份(生成完整月度索引)
    3. 标记(不删除)已知的结构性冲击时点
    4. 生成派生指标: M1/M2 剪刀差
    5. 生成检测用特征: 滚动均值/标准差、变化率
"""

import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw", "macro_raw.csv")
OUT_DIR = os.path.join(BASE, "data", "processed")
CLEAN = os.path.join(OUT_DIR, "macro_clean.csv")
FEAT = os.path.join(OUT_DIR, "macro_features.csv")

# 已知的重大冲击时点(用于后面的评估,不参与检测)
# 这些事件是公开共识的,用来检验检测方法是否有效
KNOWN_EVENTS = {
    "2008-09-30": "全球金融危机 (雷曼兄弟破产)",
    "2008-11-30": "中国四万亿刺激计划公布",
    "2015-06-30": "A股股灾",
    "2020-02-29": "新冠疫情冲击",
    "2020-04-30": "疫情后货币大幅宽松",
    "2022-04-30": "上海封控 / 供应链冲击",
}

ROLL_WINDOW = 36        # 滚动窗口长度(月) — 用于 Z-score


def load_raw():
    if not os.path.exists(RAW):
        raise FileNotFoundError(
            f"找不到 {RAW}\n请先运行: python src/fetch_wind.py"
        )
    df = pd.read_csv(RAW, index_col=0, parse_dates=True)
    print(f"[读取] {RAW}")
    print(f"       原始形状: {df.shape}")
    print(f"       时间范围: {df.index.min().date()} ~ {df.index.max().date()}")
    return df


def to_month_end(df):
    """
    把索引统一到月末。
    Wind 的宏观月度数据日期有时是月初,有时是月末,必须统一,
    否则后面按月对齐会错位。
    """
    idx = pd.to_datetime(df.index)
    idx = idx + pd.offsets.MonthEnd(0)
    df = df.copy()
    df.index = idx
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


def fill_month_gaps(df):
    """
    生成完整的月度索引。
    注意: 这里只保证时间轴连续,不填补数值(缺失仍然是 NaN)。
    填补数值会凭空造数据,后面会让"异常检测"结果失真。
    """
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="ME")
    df = df.reindex(full_idx)
    df.index.name = "date"
    return df


def add_derived(df):
    """派生指标。"""
    if "M1_YoY" in df.columns and "M2_YoY" in df.columns:
        # M1/M2 剪刀差 = M1同比 - M2同比
        # 这是市场常用的经济活力指标: 剪刀差回升通常意味着企业活期存款增加,经济活跃度上升
        df["M1_M2_scissors"] = df["M1_YoY"] - df["M2_YoY"]
        print("[派生] M1_M2_scissors = M1_YoY - M2_YoY")
    return df


def interpolate_small_gaps(df, max_gap=3):
    """
    只对连续不超过 max_gap 个月的缺失做线性插值。
    大段缺失保持 NaN —— 因为早期数据缺失不是"随机缺失",
    插值会造出不存在的规律,让异常检测误报。
    """
    report = []
    for col in df.columns:
        s = df[col]
        n_missing = s.isna().sum()
        if n_missing == 0:
            report.append((col, 0, 0))
            continue
        # 找出连续缺失段的长度
        isna = s.isna()
        groups = (isna != isna.shift()).cumsum()
        filled = 0
        for _, grp in s.groupby(groups):
            if len(grp) <= max_gap and grp.isna().all():
                filled += len(grp)
        if filled > 0:
            df[col] = s.interpolate(method="linear", limit=max_gap, limit_area="inside")
        report.append((col, n_missing, filled))
    print("\n[缺失处理] 列名                 原始缺失   已插值")
    for col, nm, fl in report:
        print(f"           {col:20s} {nm:6d} {fl:8d}")
    return df


def add_rolling_features(df, window=ROLL_WINDOW):
    """
    生成检测用的滚动特征。
    全部用 shift(1) —— 只能看过去,不能看未来。
    这一点很重要: 如果用当期值参与计算均值和标准差,就是"未来信息泄露",
    检测结果会虚高,实际用不了。
    """
    out = df.copy()
    for col in ["M1_YoY", "M2_YoY", "M1_M2_scissors", "PPI_YoY", "CPI_YoY"]:
        if col not in df.columns:
            continue
        past = df[col].shift(1)
        out[f"{col}_roll_mean"] = past.rolling(window, min_periods=12).mean()
        out[f"{col}_roll_std"] = past.rolling(window, min_periods=12).std()
    return out


def main():
    print("=" * 68)
    print("数据清洗")
    print("=" * 68)

    df = load_raw()
    df = to_month_end(df)
    print(f"\n[月末对齐] 完成, {len(df)} 行")

    df = fill_month_gaps(df)
    print(f"[补齐月份] 完成, {len(df)} 行 "
          f"({df.index.min().date()} ~ {df.index.max().date()})")

    df = add_derived(df)
    df = interpolate_small_gaps(df)

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(CLEAN, encoding="utf-8-sig")
    print(f"\n[保存] {CLEAN}")

    # 特征表
    feats = add_rolling_features(df)
    feats.to_csv(FEAT, encoding="utf-8-sig")
    print(f"[保存] {FEAT}")

    # 已知事件时点
    ev = pd.Series(KNOWN_EVENTS, name="event")
    ev.index = pd.to_datetime(ev.index) + pd.offsets.MonthEnd(0)
    ev_path = os.path.join(OUT_DIR, "known_events.csv")
    ev.to_csv(ev_path, encoding="utf-8-sig", header=["event"])
    print(f"[保存] {ev_path}  ({len(ev)} 个已知事件)")

    print("\n" + "=" * 68)
    print("清洗后各列缺失情况:")
    print(df.isna().sum().to_string())
    print("\n清洗后最后 5 行:")
    print(df.tail(5).to_string())
    print("\n[提示] 下一步: python src/detect.py")


if __name__ == "__main__":
    main()

