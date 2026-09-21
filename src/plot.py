# -*- coding: utf-8 -*-
"""
plot.py — 生成项目图表

用法:
    python src/plot.py
    python src/plot.py 1      # 只画图 1

输入:  data/processed/macro_clean.csv
       results/*.csv
输出:  figures/fig1_main.png        主指标 + 三层方法标出的异常
       figures/fig2_scissors.png    M1/M2 剪刀差 + 结构突变点
       figures/fig3_methods.png     三种方法一致性对比
       (fig4 由 evaluate.py 生成)
"""

import os
import sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(BASE, "data", "processed", "macro_clean.csv")
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")

COLORS = {
    "zscore": "#D9534F",
    "stl": "#F0AD4E",
    "iforest": "#5BC0DE",
}


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    # 中文字体 —— 如果系统没有,退回英文标签也能出图
    for f in ["Microsoft YaHei", "SimHei", "PingFang SC", "DejaVu Sans"]:
        try:
            plt.rcParams["font.sans-serif"] = [f]
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def load():
    df = pd.read_csv(CLEAN, index_col=0, parse_dates=True)
    def rd(n):
        p = os.path.join(RES, n)
        if os.path.exists(p):
            d = pd.read_csv(p, parse_dates=["date"] if n.startswith("anomalies") else None)
            return d
        return pd.DataFrame()
    return df, rd("anomalies_all.csv"), rd("structural_breaks.csv")


def fig1_main(df, anomalies, plt):
    """
    图 1: 主指标序列 + 三层方法标出的异常点。
    这是项目的门面图 —— README 里放这一张。
    """
    cols = [c for c in ["M1_M2_scissors", "PPI_YoY", "M2_YoY"] if c in df.columns]
    if not cols:
        cols = df.columns[:3].tolist()

    fig, axes = plt.subplots(len(cols), 1, figsize=(11, 3.0 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        s = df[col].dropna()
        ax.plot(s.index, s.values, color="#333333", lw=1.1, label=col, zorder=1)

        if not anomalies.empty:
            sub = anomalies[anomalies["series"] == col]
            for m in sub["method"].unique():
                pts = sub[sub["method"] == m]
                ax.scatter(pts["date"], pts["value"], s=34,
                           color=COLORS.get(m, "#888888"),
                           edgecolor="white", linewidth=0.5,
                           label=m, zorder=3, alpha=0.9)

        ax.set_ylabel(col)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    axes[0].set_title("Chinese macro indicators with anomalies detected by three methods")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    p = os.path.join(FIG, "fig1_main.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  [保存] {p}")


def fig1b_iforest(df, anomalies, plt):
    """
    图 1b: Isolation Forest 的多变量检测结果。

    为什么要单独一张图?
        Isolation Forest 是"多变量"方法 —— 它一次看全部指标,
        输出的是"哪些月份整体异常",不归属到某一个指标。
        如果把它硬塞进图 1(单指标曲线图),它哪条线都匹配不上,
        结果就是图例里有它、图上却看不到任何点。
        单列一张图才能正确表达:这些月份是"组合起来异常"。
    """
    col = "M1_M2_scissors" if "M1_M2_scissors" in df.columns else df.columns[0]
    sub = anomalies[anomalies["method"] == "iforest"] if not anomalies.empty else pd.DataFrame()
    if sub.empty:
        print("  [跳过] 没有 Isolation Forest 结果")
        return

    dates = pd.to_datetime(sub["date"]).sort_values()

    fig, ax = plt.subplots(figsize=(11, 4.2))
    s = df[col].dropna()
    ax.plot(s.index, s.values, color="#333333", lw=1.0, label=col, zorder=2)

    for i, d in enumerate(dates):
        ax.axvspan(d - pd.Timedelta(days=15), d + pd.Timedelta(days=15),
                   color="#5BC0DE", alpha=0.55, zorder=1,
                   label="flagged by Isolation Forest" if i == 0 else None)

    ax.set_title(f"Multivariate anomalies (Isolation Forest) against {col}")
    ax.set_ylabel(col)
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    p = os.path.join(FIG, "fig1b_iforest.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  [保存] {p}  ({len(dates)} 个时点)")


def fig2_scissors(df, breaks, plt):
    """
    图 2: M1/M2 剪刀差 + 结构突变点。
    突出"行为模式改变"的时点 —— 这是比单点异常更重要的发现。
    """
    col = "M1_M2_scissors"
    if col not in df.columns:
        print(f"  [跳过] 数据里没有 {col}")
        return

    s = df[col].dropna()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(s.index, s.values, color="#1F4E79", lw=1.3, label="M1 YoY growth minus M2 YoY growth")
    ax.fill_between(s.index, 0, s.values, where=(s.values >= 0),
                    color="#1F4E79", alpha=0.12, interpolate=True)
    ax.fill_between(s.index, 0, s.values, where=(s.values < 0),
                    color="#C00000", alpha=0.12, interpolate=True)
    ax.axhline(0, color="#666666", lw=0.8, ls="--")

    if breaks is not None and not breaks.empty:
        sub = breaks[breaks["series"] == col].sort_values("date")
        ymin, ymax = float(s.min()), float(s.max())
        span = ymax - ymin
        # 标签分三层错开高度 —— 否则相邻的突变点标签会叠在一起看不清
        levels = [0.96, 0.80, 0.64]
        for i, (_, r) in enumerate(sub.iterrows()):
            d = pd.Timestamp(r["date"])
            ax.axvline(d, color="#C00000", lw=1.3, ls=":", alpha=0.8)
            ypos = ymin + span * levels[i % len(levels)]
            ax.annotate(d.strftime("%Y-%m"),
                        xy=(d, ypos), fontsize=7.2, color="#C00000", ha="center",
                        va="center", rotation=0,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec="#C00000", lw=0.5, alpha=0.92))
        ax.set_ylim(ymin - span * 0.10, ymax + span * 0.16)

    ax.set_title("M1/M2 YoY growth rate gap with detected structural breaks")
    ax.set_ylabel("Percentage points")
    ax.set_xlabel("Date")
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    p = os.path.join(FIG, "fig2_scissors.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  [保存] {p}")


def fig3_methods(anomalies, plt):
    """
    图 3: 三种方法检测结果的对比。
    用"每年各方法检出数量"的堆叠柱状图 + 重合度统计。
    """
    if anomalies.empty:
        print("  [跳过] 没有异常数据")
        return

    a = anomalies.copy()
    a["year"] = pd.to_datetime(a["date"]).dt.year

    pivot = a.pivot_table(index="year", columns="method",
                          values="series", aggfunc="count", fill_value=0)
    methods = [m for m in ["zscore", "stl", "iforest"] if m in pivot.columns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4),
                                   gridspec_kw={"width_ratios": [2.4, 1]})

    bottom = np.zeros(len(pivot))
    for m in methods:
        ax1.bar(pivot.index, pivot[m].values, bottom=bottom,
                color=COLORS.get(m, "#888888"), label=m, width=0.8)
        bottom += pivot[m].values
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of anomalies")
    ax1.set_title("Anomalies per year by method")
    ax1.legend(fontsize=8.5)

    # 右图: 重合度
    counts = a.groupby("date")["method"].nunique()
    dist = counts.value_counts().sort_index()
    labels = [f"{int(k)} method(s)" for k in dist.index]
    vals = dist.values
    colors = ["#B0B0B0", "#F0AD4E", "#C00000"][:len(vals)]
    ax2.barh(labels, vals, color=colors, height=0.55)
    for i, v in enumerate(vals):
        ax2.text(v + max(vals) * 0.02, i, str(int(v)), va="center", fontsize=9)
    ax2.set_xlabel("Number of time points")
    ax2.set_title("Agreement between methods")
    ax2.grid(axis="x", alpha=0.25)
    ax2.grid(axis="y", visible=False)

    fig.tight_layout()
    p = os.path.join(FIG, "fig3_methods.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  [保存] {p}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    plt = setup_matplotlib()
    os.makedirs(FIG, exist_ok=True)

    print("=" * 68)
    print("生成图表")
    print("=" * 68)

    try:
        df, anomalies, breaks = load()
    except FileNotFoundError as e:
        print(f"[错误] {e}")
        return

    print(f"[读取] {len(df)} 个月数据, {len(anomalies)} 条异常, {len(breaks)} 个突变点\n")

    if which in ("all", "1"):
        fig1_main(df, anomalies, plt)
        fig1b_iforest(df, anomalies, plt)
    if which in ("all", "2"):
        fig2_scissors(df, breaks, plt)
    if which in ("all", "3"):
        fig3_methods(anomalies, plt)

    print("\n完成。图 4(敏感性分析) 由 evaluate.py 生成。")


if __name__ == "__main__":
    main()
