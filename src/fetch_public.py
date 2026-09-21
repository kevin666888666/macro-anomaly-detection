# -*- coding: utf-8 -*-
"""
fetch_public.py — 用公开数据源获取宏观指标（不依赖 Wind）

为什么有这个脚本:
    原方案用 Wind 终端取数(src/fetch_wind.py),但 Wind 终端需要在有授权
    的机器上运行。这个脚本用公开 API 取同样的指标,让项目在没有 Wind
    的环境下也能完整复现。

    如果后来拿到 Wind 权限,改用 fetch_wind.py 即可 —— clean.py 之后的所有
    脚本都只读 data/raw/macro_raw.csv,不关心数据是从哪来的。

数据源:
    东方财富数据中心公开接口 (datacenter-web.eastmoney.com)
    - RPT_ECONOMY_CURRENCY_SUPPLY  货币供应量 (M0 / M1 / M2 同比)
    - RPT_ECONOMY_CPI              居民消费价格指数 (同比)
    - RPT_ECONOMY_PPI              工业生产者出厂价格指数 (同比)
    - RPT_ECONOMY_PMI              制造业采购经理指数

用法:
    python src/fetch_public.py
    python src/fetch_public.py --check    # 只测试连通性,不写文件
"""

import json
import os
import sys
import time
import urllib.request
from datetime import datetime

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "data", "raw")
OUT_FILE = os.path.join(OUT_DIR, "macro_raw.csv")

API = "https://datacenter-web.eastmoney.com/api/data/v1/get"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36",
    "Referer": "https://data.eastmoney.com/",
}

START_YEAR = 2000


def call_api(report_name, page_size=500, page=1):
    """调用东方财富数据中心的通用接口。"""
    url = (f"{API}?reportName={report_name}&columns=ALL"
           f"&pageSize={page_size}&pageNumber={page}&sortColumns=REPORT_DATE&sortTypes=-1")
    req = urllib.request.Request(url, headers=HEADERS)
    raw = urllib.request.urlopen(req, timeout=25).read().decode("utf-8")
    js = json.loads(raw)
    if not js.get("success"):
        raise RuntimeError(f"接口返回失败: {js.get('message')}")
    return js["result"]["data"] if js.get("result") else []


def fetch_all_pages(report_name, max_pages=10):
    """翻页取全,直到没有更多数据。"""
    out, page = [], 1
    while page <= max_pages:
        rows = call_api(report_name, page_size=500, page=page)
        if not rows:
            break
        out.extend(rows)
        if len(rows) < 500:
            break
        page += 1
        time.sleep(0.4)
    return out


def to_series(rows, value_field, name):
    """把接口返回的行转成按月末索引的 Series。"""
    recs = {}
    for r in rows:
        d = r.get("REPORT_DATE")
        v = r.get(value_field)
        if not d or v is None:
            continue
        try:
            dt = pd.Timestamp(d)
        except Exception:
            continue
        if dt.year < START_YEAR:
            continue
        # 统一到月末
        dt = dt + pd.offsets.MonthEnd(0)
        try:
            recs[dt] = float(v)
        except (TypeError, ValueError):
            continue
    if not recs:
        print(f"  [警告] {name}: 没有取到有效数据 (字段 {value_field})")
        return None
    s = pd.Series(recs, name=name).sort_index()
    print(f"  [成功] {name:12s} {len(s):4d} 个观测 "
          f"({s.index.min().date()} ~ {s.index.max().date()})  字段={value_field}")
    return s


def main():
    check_only = "--check" in sys.argv

    print("=" * 70)
    print("公开数据源取数 — 中国宏观指标")
    print(f"数据源: 东方财富数据中心   样本期: {START_YEAR} 年至今")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. 货币供应量: M0 / M1 / M2 同比
    #
    # ⚠️ 字段名与实际指标完全错位 —— 绝不能按字面理解。
    #
    # 核对依据: 中国人民银行 2026-08 金融统计数据(经中宏网转载):
    #     M2 余额 356.81 万亿元, 同比  7.5%
    #     M1 余额 115.77 万亿元, 同比  4.1%
    #     M0 余额  14.83 万亿元, 同比 11.2%
    #
    # 与本接口 2026-08 的返回逐项对应:
    #     BASIC_CURRENCY     = 3,568,083.60 亿元 = 356.81 万亿 -> 实际是 M2
    #     CURRENCY           = 1,157,741.43 亿元 = 115.77 万亿 -> 实际是 M1
    #     FREE_CASH          =   148,311.98 亿元 =  14.83 万亿 -> 实际是 M0
    #
    # 因此同比字段的正确映射是:
    #     BASIC_CURRENCY_SAME -> M2 同比   (7.5%  ✓)
    #     CURRENCY_SAME       -> M1 同比   (4.1%  ✓)
    #     FREE_CASH_SAME      -> M0 同比  (11.2%  ✓)
    #
    # 若按字面理解(把 FREE_CASH_SAME 当 M2),序列会出现大幅假跳变。
    # ------------------------------------------------------------------
    print("\n[1/4] 货币供应量 ...")
    try:
        rows = fetch_all_pages("RPT_ECONOMY_CURRENCY_SUPPLY")
        print(f"       原始 {len(rows)} 行")
        m0 = to_series(rows, "FREE_CASH_SAME", "M0_YoY")
        m1 = to_series(rows, "CURRENCY_SAME", "M1_YoY")
        m2 = to_series(rows, "BASIC_CURRENCY_SAME", "M2_YoY")
    except Exception as e:
        print(f"  [失败] {e}")
        m0 = m1 = m2 = None

    # ------------------------------------------------------------------
    # 2. CPI 同比
    # ------------------------------------------------------------------
    print("\n[2/4] 居民消费价格指数 ...")
    try:
        rows = fetch_all_pages("RPT_ECONOMY_CPI")
        print(f"       原始 {len(rows)} 行")
        cpi = to_series(rows, "NATIONAL_SAME", "CPI_YoY")
    except Exception as e:
        print(f"  [失败] {e}")
        cpi = None

    # ------------------------------------------------------------------
    # 3. PPI 同比
    # ------------------------------------------------------------------
    print("\n[3/4] 工业生产者出厂价格指数 ...")
    try:
        rows = fetch_all_pages("RPT_ECONOMY_PPI")
        print(f"       原始 {len(rows)} 行")
        ppi = to_series(rows, "BASE_SAME", "PPI_YoY")
    except Exception as e:
        print(f"  [失败] {e}")
        ppi = None

    # ------------------------------------------------------------------
    # 4. 制造业 PMI
    # ------------------------------------------------------------------
    print("\n[4/4] 制造业采购经理指数 ...")
    try:
        rows = fetch_all_pages("RPT_ECONOMY_PMI")
        print(f"       原始 {len(rows)} 行")
        pmi = to_series(rows, "MAKE_INDEX", "PMI")
    except Exception as e:
        print(f"  [失败] {e}")
        pmi = None

    # ------------------------------------------------------------------
    series = [s for s in [m0, m1, m2, cpi, ppi, pmi] if s is not None]
    if not series:
        print("\n[错误] 一个指标都没取到。请检查网络。")
        sys.exit(1)

    df = pd.concat(series, axis=1).sort_index()

    print("\n" + "=" * 70)
    print(f"共 {df.shape[1]} 个指标, {df.shape[0]} 行")
    print(f"时间范围: {df.index.min().date()} ~ {df.index.max().date()}")
    print("\n各列缺失值:")
    print(df.isna().sum().to_string())
    print("\n最后 5 行:")
    print(df.tail(5).to_string())

    if check_only:
        print("\n[--check] 只测试连通性,未写入文件。")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_FILE, encoding="utf-8-sig")
    print(f"\n[保存] {OUT_FILE}")
    print("\n[提示] 下一步: python src/clean.py")


if __name__ == "__main__":
    main()
