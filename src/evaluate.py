# -*- coding: utf-8 -*-
"""
evaluate.py — 评估检测结果

用法:
    python src/evaluate.py

输入:  results/anomalies_all.csv
       data/processed/known_events.csv
       data/processed/macro_clean.csv
输出:  results/evaluation_summary.csv
       results/consistency.csv
       figures/fig4_sensitivity.png

评估做三件事:
    1. 已知事件检验 —— 检测方法能否抓到公认的经济冲击时点?
    2. 方法一致性   —— 三种方法的结果重合度如何?
    3. 敏感性分析   —— 改变参数后结果稳不稳定?
"""

import os
import sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
CLEAN = os.path.join(BASE, "data", "processed", "macro_clean.csv")
EVENTS = os.path.join(BASE, "data", "processed", "known_events.csv")
FIG = os.path.join(BASE, "figures")

TOLERANCE_MONTHS = 3       # 异常点距已知事件 3 个月内,视为"命中"


def load_all():
    p = os.path.join(RES, "anomalies_all.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"找不到 {p}\n请先运行: python src/detect.py")
    anomalies = pd.read_csv(p, parse_dates=["date"])
    events = None
    if os.path.exists(EVENTS):
        events = pd.read_csv(EVENTS, index_col=0, parse_dates=True)
    df = pd.read_csv(CLEAN, index_col=0, parse_dates=True) if os.path.exists(CLEAN) else None
    return anomalies, events, df


def evaluate_known_events(anomalies, events, tol=TOLERANCE_MONTHS):
    """
    已知事件检验。

    方法: 对每个已知事件时点,看其前后 tol 个月内有没有异常点被检出。
    这是"召回率"的思路 —— 能不能抓到我们知道确实发生过的事。

    注意: 这只检验"抓到了没有",不检验"抓错的没有"。
    因为真实的宏观数据没有标注好的"真值",没法算精确的准确率。
    这一点要在 README 的 Limitations 里写清楚。
    """
    print("\n" + "=" * 68)
    print(f"评估 1: 已知事件检验 (±{tol} 个月视为命中)")
    print("=" * 68)

    if events is None or events.empty:
        print("  [跳过] 没有已知事件文件")
        return pd.DataFrame()

    rows = []
    anom_dates = anomalies["date"].sort_values().unique()
    anom_dates = pd.to_datetime(anom_dates)

    for ev_date, row in events.iterrows():
        ev_date = pd.Timestamp(ev_date)
        lo = ev_date - pd.DateOffset(months=tol)
        hi = ev_date + pd.DateOffset(months=tol)
        hits = anom_dates[(anom_dates >= lo) & (anom_dates <= hi)]

        # 命中的方法
        methods = set()
        for h in hits:
            m = anomalies.loc[anomalies["date"] == h, "method"].unique()
            methods.update(m)

        rows.append({
            "event_date": ev_date,
            "event": row["event"],
            "n_anomalies_nearby": len(hits),
            "detected": len(hits) > 0,
            "methods": ", ".join(sorted(methods)) if methods else "",
        })

    out = pd.DataFrame(rows)
    n_total = len(out)
    n_hit = int(out["detected"].sum())
    print(f"\n  已知事件总数: {n_total}")
    print(f"  检出的:       {n_hit}  ({n_hit/n_total:.0%})")
    print()
    for _, r in out.iterrows():
        mark = "✓" if r["detected"] else "✗"
        print(f"    {mark} {r['event_date'].date()}  {r['event'][:28]:30s} "
              f"附近 {r['n_anomalies_nearby']:2d} 个异常  [{r['methods']}]")

    out.to_csv(os.path.join(RES, "evaluation_summary.csv"),
               index=False, encoding="utf-8-sig")
    print(f"\n  [保存] {os.path.join(RES, 'evaluation_summary.csv')}")
    return out


def evaluate_consistency(anomalies):
    """
    方法一致性。

    思路: 如果三种方法(统计、时序、机器学习)互相独立地指向同一个时点,
    那么这个时点是真的异常的可能性更高。
    """
    print("\n" + "=" * 68)
    print("评估 2: 三种方法的一致性")
    print("=" * 68)

    pivot = anomalies.pivot_table(
        index="date", columns="method", values="series",
        aggfunc="count", fill_value=0
    )
    pivot["n_methods"] = (pivot > 0).sum(axis=1)
    pivot = pivot.sort_values(["n_methods", "date"], ascending=[False, True])

    print(f"\n  涉及时点总数: {len(pivot)}")
    for k in [3, 2, 1]:
        n = int((pivot["n_methods"] == k).sum())
        print(f"    {k} 种方法同时标记: {n:3d} 个时点")

    print("\n  三种方法同时标记的时点(最可信):")
    top = pivot[pivot["n_methods"] == 3]
    for d, _ in top.iterrows():
        print(f"    {pd.Timestamp(d).date()}")

    pivot.to_csv(os.path.join(RES, "consistency.csv"), encoding="utf-8-sig")
    print(f"\n  [保存] {os.path.join(RES, 'consistency.csv')}")
    return pivot


def sensitivity_analysis(df, thresholds=(2.0, 2.5, 3.0), windows=(24, 36, 48)):
    """
    敏感性分析。

    为什么重要: 如果一个方法的结果对参数极度敏感,说明它不可靠 ——
    换一个窗口长度就得出完全不同结论,那这个结论没有价值。

    做法: 对不同窗口长度和阈值,重新算一遍 Z-score,看异常点数量的变化。
    """
    print("\n" + "=" * 68)
    print("评估 3: 敏感性分析 (Z-score 的窗口长度与阈值)")
    print("=" * 68)

    if df is None:
        print("  [跳过] 没有原始数据")
        return None

    col = "M1_YoY" if "M1_YoY" in df.columns else df.columns[0]
    s = df[col].dropna()

    rows = []
    for w in windows:
        past = s.shift(1)
        mean = past.rolling(w, min_periods=12).mean()
        std = past.rolling(w, min_periods=12).std()
        z = (s - mean) / std.replace(0, np.nan)
        for t in thresholds:
            rows.append({
                "window": w,
                "threshold": t,
                "n_anomalies": int((z.abs() > t).sum()),
                "pct": round(float((z.abs() > t).mean() * 100), 2),
            })

    out = pd.DataFrame(rows)
    print(f"\n  指标: {col}")
    print(out.to_string(index=False))

    # 判断稳定性: 同一阈值下,不同窗口的异常数差异
    print("\n  稳定性判断:")
    for t in thresholds:
        sub = out[out["threshold"] == t]["n_anomalies"]
        spread = sub.max() - sub.min()
        verdict = "稳定" if spread <= max(2, sub.mean() * 0.3) else "不稳定"
        print(f"    阈值 {t}: 异常数 {sub.min()}~{sub.max()} (极差 {spread})  -> {verdict}")

    out.to_csv(os.path.join(RES, "sensitivity.csv"),
               index=False, encoding="utf-8-sig")

    # 画图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass

        os.makedirs(FIG, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        for t in thresholds:
            sub = out[out["threshold"] == t]
            ax.plot(sub["window"], sub["n_anomalies"], marker="o", label=f"|Z| > {t}")
        ax.set_xlabel("Rolling window (months)")
        ax.set_ylabel("Number of anomalies detected")
        ax.set_title(f"Sensitivity of anomaly count to window length ({col})")
        ax.set_xticks(list(windows))
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        p = os.path.join(FIG, "fig4_sensitivity.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"\n  [保存] {p}")
    except ImportError:
        print("\n  [跳过画图] 未安装 matplotlib")

    return out


def main():
    print("=" * 68)
    print("检测结果评估")
    print("=" * 68)

    anomalies, events, df = load_all()
    print(f"[读取] {len(anomalies)} 条异常记录, "
          f"方法: {', '.join(anomalies['method'].unique())}")

    evaluate_known_events(anomalies, events)
    evaluate_consistency(anomalies)
    sensitivity_analysis(df)

    print("\n" + "=" * 68)
    print("评估完成。")
    print("=" * 68)
    print("\n把上面的结论写进 README 的 'What I found' 一节。")


if __name__ == "__main__":
    main()
