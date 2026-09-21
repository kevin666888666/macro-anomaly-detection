# -*- coding: utf-8 -*-
"""
fetch_wind.py — 从 Wind 金融终端提取宏观经济指标

用法:
    1. 确保 Wind 金融终端已打开并登录
    2. python src/fetch_wind.py
    3. 数据保存到 data/raw/macro_raw.csv

注意:
    WindPy 必须在 Wind 终端运行时才能工作。
    本脚本会先做一次连接测试，失败时打印诊断信息。
"""

import os
import sys
import time
from datetime import datetime

import pandas as pd

# ---------------------------------------------------------------------------
# 指标清单: (Wind 代码, 中文名, 英文名用于列名)
# ---------------------------------------------------------------------------
INDICATORS = [
    ("M0001385", "M2同比",            "M2_YoY"),
    ("M0001386", "M0同比",            "M0_YoY"),
    ("M0001387", "M1同比",            "M1_YoY"),
    ("M0000612", "CPI当月同比",        "CPI_YoY"),
    ("M0001227", "PPI全部工业品同比",   "PPI_YoY"),
    ("M0000556", "制造业PMI",          "PMI"),
    ("M0001429", "社会融资规模存量同比", "TSF_YoY"),
    ("M0000545", "工业增加值同比",      "IP_YoY"),
]

START_DATE = "2000-01-31"          # 样本期起点
END_DATE = datetime.today().strftime("%Y-%m-%d")

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
OUT_FILE = os.path.join(OUT_DIR, "macro_raw.csv")


def connect():
    """连接 Wind 终端。返回 w 对象，失败时退出。"""
    try:
        from WindPy import w
    except ImportError:
        print("[错误] 找不到 WindPy 模块。")
        print("       请确认 Wind 终端已安装，且 Python 位数与终端一致（通常都是 64 位）。")
        print("       安装 WindPy 的常见方式：在 Wind 终端里运行「修复插件」或「Python接口」安装。")
        sys.exit(1)

    print("正在连接 Wind 终端 ...")
    ret = w.start(waitTime=30)

    if ret.ErrorCode != 0:
        print(f"[错误] 连接失败，ErrorCode = {ret.ErrorCode}")
        print("       常见原因：")
        print("       1. Wind 金融终端没有打开，或没有登录")
        print("       2. 账号没有 API 权限（需要单独开通）")
        print("       3. 终端版本过旧")
        sys.exit(1)

    print("连接成功。")
    return w


def test_one_series(w):
    """先用一个指标试取，确认权限和代码都正确。"""
    print("\n[测试] 试取 M1同比 (M0001387) ...")
    d = w.wsd("M0001387", "close", "2024-01-31", "2025-12-31", "")

    if d.ErrorCode != 0:
        print(f"[错误] 试取失败，ErrorCode = {d.ErrorCode}")
        print("       ErrorCode 常见含义:")
        print("       -40522017 / -40521009 : 无该数据权限")
        print("       -40520007            : 代码不存在")
        print("       请联系 Wind 客服或改用有权限的指标代码。")
        sys.exit(1)

    vals = [v for v in d.Data[0] if v is not None and str(v) != "nan"]
    print(f"[测试] 成功。取到 {len(vals)} 个观测值。")
    if vals:
        print(f"       最近一个值: {vals[-1]}  (日期 {d.Times[-1]})")
        print(f"       数值范围参考: M1同比 通常在 0 到 30 之间。若明显不符，说明代码可能不对。")
    return True


def fetch_series(w, code, name_en):
    """取单个指标的历史序列。返回 Series 或 None。"""
    d = w.wsd(code, "close", START_DATE, END_DATE, "")

    if d.ErrorCode != 0:
        print(f"  [跳过] {code} ({name_en})  ErrorCode={d.ErrorCode}")
        return None

    s = pd.Series(d.Data[0], index=pd.to_datetime(d.Times), name=name_en)
    s = pd.to_numeric(s, errors="coerce")
    print(f"  [成功] {code:12s} {name_en:10s} {len(s):4d} 个观测 "
          f"({s.index.min().date()} ~ {s.index.max().date()})")
    return s


def main():
    print("=" * 68)
    print("Wind 宏观经济指标提取")
    print(f"样本期: {START_DATE} ~ {END_DATE}")
    print("=" * 68)

    w = connect()
    test_one_series(w)

    print("\n[提取] 开始逐个取数 ...")
    frames = []
    failed = []

    for code, name_cn, name_en in INDICATORS:
        s = fetch_series(w, code, name_en)
        if s is not None:
            frames.append(s)
        else:
            failed.append((code, name_cn))
        time.sleep(0.3)          # 轻微限速，避免请求过密

    if not frames:
        print("\n[错误] 一个指标都没取到。请检查权限。")
        sys.exit(1)

    df = pd.concat(frames, axis=1)
    df = df.sort_index()

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_FILE, encoding="utf-8-sig")

    print("\n" + "=" * 68)
    print(f"[完成] 共 {df.shape[1]} 个指标，{df.shape[0]} 行")
    print(f"       保存到: {OUT_FILE}")
    print(f"       时间范围: {df.index.min().date()} ~ {df.index.max().date()}")
    if failed:
        print(f"\n[注意] 以下 {len(failed)} 个指标未取到:")
        for code, nm in failed:
            print(f"       {code}  {nm}")

    print("\n[数据概览] 各指标缺失值数量:")
    print(df.isna().sum().to_string())
    print("\n[数据概览] 各指标最后 3 个值:")
    print(df.tail(3).to_string())

    w.close()
    print("\n[提示] 下一步: python src/clean.py")


if __name__ == "__main__":
    main()
