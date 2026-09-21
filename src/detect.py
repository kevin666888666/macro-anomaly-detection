# -*- coding: utf-8 -*-
"""
detect.py — 三层异常检测 + 结构突变识别

用法:
    python src/detect.py              # 跑全部四层
    python src/detect.py zscore       # 只跑第 1 层
    python src/detect.py stl          # 只跑第 2 层
    python src/detect.py iforest      # 只跑第 3 层
    python src/detect.py breaks       # 只跑第 4 层

输入:  data/processed/macro_clean.csv
       data/processed/macro_features.csv
输出:  results/anomalies_zscore.csv
       results/anomalies_stl.csv
       results/anomalies_iforest.csv
       results/structural_breaks.csv
       results/anomalies_all.csv       (合并结果)
"""

import os
import sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(BASE, "data", "processed", "macro_clean.csv")
FEAT = os.path.join(BASE, "data", "processed", "macro_features.csv")
RES = os.path.join(BASE, "results")

TARGET_COLS = ["M1_YoY", "M2_YoY", "M1_M2_scissors", "PPI_YoY", "CPI_YoY"]

Z_THRESHOLD = 2.5          # Z-score 异常阈值
STL_THRESHOLD = 3.0        # STL 残差异常阈值(以残差标准差为单位)
IF_CONTAMINATION = 0.05    # Isolation Forest 预期异常比例


def load():
    if not os.path.exists(CLEAN):
        raise FileNotFoundError(f"找不到 {CLEAN}\n请先运行: python src/clean.py")
    df = pd.read_csv(CLEAN, index_col=0, parse_dates=True)
    feats = pd.read_csv(FEAT, index_col=0, parse_dates=True) if os.path.exists(FEAT) else df
    return df, feats


def save(df, name):
    os.makedirs(RES, exist_ok=True)
    p = os.path.join(RES, name)
    df.to_csv(p, encoding="utf-8-sig")
    print(f"    [保存] {p}")


# ---------------------------------------------------------------------------
# 第 1 层: 滚动窗口 Z-score
# ---------------------------------------------------------------------------
def detect_zscore(df, feats, threshold=Z_THRESHOLD):
    """
    对每个指标,用过去 36 个月的均值和标准差算当前值的 Z-score。

    为什么用滚动窗口而不是全样本?
        全样本均值包含了未来信息。用全样本算 2008 年的 Z 值,
        等于用 2009-2025 年的数据去判断 2008 年是否异常 —— 这在实盘里做不到。
        滚动窗口只用过去的数据,是"当时就能算出来"的。

    判断标准: |Z| > threshold 视为异常
    """
    print("\n" + "=" * 68)
    print(f"第 1 层: 滚动窗口 Z-score (阈值 |Z| > {threshold})")
    print("=" * 68)

    records = []
    for col in TARGET_COLS:
        if col not in df.columns:
            continue
        mean_col = f"{col}_roll_mean"
        std_col = f"{col}_roll_std"

        if mean_col in feats.columns and std_col in feats.columns:
            mean = feats[mean_col]
            std = feats[std_col]
        else:
            past = df[col].shift(1)
            mean = past.rolling(36, min_periods=12).mean()
            std = past.rolling(36, min_periods=12).std()

        z = (df[col] - mean) / std.replace(0, np.nan)

        mask = z.abs() > threshold
        n = int(mask.sum())
        print(f"  {col:20s} 检测到 {n:3d} 个异常点")

        for dt in df.index[mask]:
            records.append({
                "date": dt,
                "series": col,
                "value": df.loc[dt, col],
                "score": z.loc[dt],
                "method": "zscore",
            })

    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values(["date", "series"]).reset_index(drop=True)
    save(out, "anomalies_zscore.csv")
    return out


# ---------------------------------------------------------------------------
# 第 2 层: STL 分解 + 残差检验
# ---------------------------------------------------------------------------
def detect_stl(df, threshold=STL_THRESHOLD):
    """
    把序列分解成 趋势 + 季节 + 残差,在残差上找异常。

    为什么需要这一层?
        宏观指标有明显的季节性(春节、季末).
        直接对原序列做 Z-score,会把每年春节的规律性下跌误判为异常。
        STL 先把季节性剔除,再在残差上看异常 —— 这才是"真正意外"的部分。

    要求: 至少 2 个完整周期(月度即至少 24 个月),否则跳过该指标。
    """
    print("\n" + "=" * 68)
    print(f"第 2 层: STL 分解 + 残差检验 (阈值 {threshold} 倍残差标准差)")
    print("=" * 68)

    try:
        from statsmodels.tsa.seasonal import STL
    except ImportError:
        print("  [跳过] 未安装 statsmodels。安装命令: pip install statsmodels")
        return pd.DataFrame()

    records = []
    for col in TARGET_COLS:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) < 24:
            print(f"  {col:20s} 样本不足 ({len(s)} 个月),跳过")
            continue

        try:
            res = STL(s, period=12, robust=True).fit()
        except Exception as e:
            print(f"  {col:20s} STL 失败: {e}")
            continue

        resid = pd.Series(res.resid, index=s.index)
        # 用 MAD(中位数绝对偏差)估计尺度,比标准差更抗异常值本身的影响
        # 如果用标准差,异常值自己会把标准差抬高,导致检测不到自己
        mad = np.median(np.abs(resid - np.median(resid)))
        scale = 1.4826 * mad if mad > 0 else resid.std()
        if scale == 0 or np.isnan(scale):
            print(f"  {col:20s} 残差尺度为 0,跳过")
            continue

        z = resid / scale
        mask = z.abs() > threshold
        n = int(mask.sum())
        print(f"  {col:20s} 检测到 {n:3d} 个异常点 "
              f"(季节强度 {res.seasonal.std()/s.std():.2f})")

        for dt in s.index[mask]:
            records.append({
                "date": dt,
                "series": col,
                "value": s.loc[dt],
                "score": z.loc[dt],
                "method": "stl",
            })

    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values(["date", "series"]).reset_index(drop=True)
    save(out, "anomalies_stl.csv")
    return out


# ---------------------------------------------------------------------------
# 第 3 层: Isolation Forest (多变量)
# ---------------------------------------------------------------------------
def detect_iforest(df, feats, contamination=IF_CONTAMINATION):
    """
    多变量异常检测。

    和前两层的区别:
        前两层是单变量 —— 一次只看一个指标。
        这一层同时看多个指标 —— 能发现"单个指标看起来正常,但组合起来不正常"的情况。
        这才是系统性风险的视角。

    特征: 当前值 + 滚动均值 + 滚动标准差 + 变化率
    算法: Isolation Forest —— 用随机切分隔离样本,异常点更容易被"孤立"出来,
          因此平均切分深度更短。不需要假设数据分布。
    """
    print("\n" + "=" * 68)
    print(f"第 3 层: Isolation Forest (预期异常比例 {contamination:.0%})")
    print("=" * 68)

    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  [跳过] 未安装 scikit-learn。安装命令: pip install scikit-learn")
        return pd.DataFrame()

    cols = [c for c in TARGET_COLS if c in df.columns]
    X = df[cols].copy()

    # 加变化率特征: 环比变化能捕捉"水平不高但突然跳变"的异常
    for c in cols:
        X[f"{c}_diff"] = df[c].diff()

    X = X.dropna()
    if len(X) < 60:
        print(f"  样本不足 ({len(X)} 行),跳过")
        return pd.DataFrame()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,          # 固定随机种子,保证结果可复现
        n_jobs=-1,
    )
    labels = model.fit_predict(Xs)          # 1 = 正常, -1 = 异常
    scores = model.decision_function(Xs)    # 越小越异常

    mask = labels == -1
    n = int(mask.sum())
    print(f"  共 {len(X)} 个时点,检测到 {n} 个异常时点")

    out = pd.DataFrame({
        "date": X.index[mask],
        "series": "ALL (multi-variable)",
        "value": np.nan,
        "score": scores[mask],
        "method": "iforest",
    })
    for c in cols:
        out[c] = X.loc[mask, c].values

    out = out.sort_values("date").reset_index(drop=True)
    save(out, "anomalies_iforest.csv")
    return out


# ---------------------------------------------------------------------------
# 第 4 层: 结构突变识别
# ---------------------------------------------------------------------------
def detect_breaks(df, window=36, z_threshold=2.0):
    """
    识别统计性质(均值)是否发生持久改变。

    和前几层的区别:
        异常检测找的是"某个时点的偏离"。
        结构突变找的是"序列的行为模式变了" —— 这是更重要的发现,
        因为它意味着过去的规律不再适用。

    方法: 比较每个时点前后各 window 个月的均值。
        如果差异(benchmark 化后)超过阈值,认为该点发生了均值突变。
        这是一种简化的 CUSUM 思路,优点是直观、可解释。

    局限: 这是简化方法,不是严格的统计检验。
        严格的检验(Chow test / Bai-Perron)需要指定断点或做更复杂的搜索。
        本项目用简化方法定位候选点,再用 t 检验验证。
    """
    print("\n" + "=" * 68)
    print(f"第 4 层: 结构突变识别 (前后各 {window} 个月均值对比)")
    print("=" * 68)

    try:
        from scipy import stats as sps
    except ImportError:
        print("  [跳过] 未安装 scipy。安装命令: pip install scipy")
        return pd.DataFrame()

    records = []
    for col in TARGET_COLS:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) < window * 2 + 12:
            print(f"  {col:20s} 样本不足,跳过")
            continue

        found = []
        for i in range(window, len(s) - window):
            before = s.iloc[i - window:i]
            after = s.iloc[i:i + window]
            # 用 t 检验判断前后均值差异是否显著
            t_stat, p_val = sps.ttest_ind(before, after, equal_var=False)
            if p_val < 0.001:          # 严格阈值,避免选出太多假突变
                found.append({
                    "date": s.index[i],
                    "series": col,
                    "mean_before": before.mean(),
                    "mean_after": after.mean(),
                    "change": after.mean() - before.mean(),
                    "p_value": p_val,
                })

        # 同一段持续突变会产生连续多个点,只保留变化量最大的那个
        if found:
            fdf = pd.DataFrame(found)
            fdf["year"] = fdf["date"].dt.year
            keep = fdf.loc[fdf.groupby("year")["change"].apply(
                lambda x: x.abs().idxmax()
            ).values]
            print(f"  {col:20s} 检测到 {len(keep)} 个结构突变时点")
            records.extend(keep.to_dict("records"))
        else:
            print(f"  {col:20s} 未检测到显著突变")

    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values("date").reset_index(drop=True)
    save(out, "structural_breaks.csv")
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    print("=" * 68)
    print("宏观指标异常检测")
    print("=" * 68)

    df, feats = load()
    print(f"[读取] {len(df)} 个月的数据, {df.shape[1]} 列")
    print(f"       指标: {', '.join([c for c in TARGET_COLS if c in df.columns])}")

    z = st = ifo = br = pd.DataFrame()

    if which in ("all", "zscore"):
        z = detect_zscore(df, feats)
    if which in ("all", "stl"):
        st = detect_stl(df)
    if which in ("all", "iforest"):
        ifo = detect_iforest(df, feats)
    if which in ("all", "breaks"):
        br = detect_breaks(df)

    # 合并所有异常结果
    parts = [x for x in [z, st, ifo] if not x.empty]
    if parts:
        allanom = pd.concat(parts, ignore_index=True)
        # 统计每个时点被几种方法标为异常
        counts = allanom.groupby("date")["method"].nunique().rename("n_methods")
        allanom = allanom.merge(counts, left_on="date", right_index=True, how="left")
        allanom = allanom.sort_values(["date", "series"]).reset_index(drop=True)
        save(allanom, "anomalies_all.csv")

        print("\n" + "=" * 68)
        print("汇总")
        print("=" * 68)
        print(f"  总异常记录: {len(allanom)} 条")
        print(f"  涉及时点:   {allanom['date'].nunique()} 个")
        print(f"  三种方法都标记的时点: {(counts == 3).sum()} 个")
        print(f"  两种方法都标记的时点: {(counts == 2).sum()} 个")
        print(f"  只有一种方法标记的时点: {(counts == 1).sum()} 个")
        print("\n  被三种方法同时标记的时点(可信度最高):")
        top = counts[counts == 3].index.sort_values()
        for d in top:
            print(f"    {d.date()}")

    print("\n[提示] 下一步: python src/evaluate.py")


if __name__ == "__main__":
    main()

