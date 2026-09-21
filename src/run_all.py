# -*- coding: utf-8 -*-
"""
run_all.py — 一次跑完整条流程

用法:
    python src/run_all.py              # 全部(会用 Wind 重新取数,如果终端开着)
    python src/run_all.py --no-fetch   # 跳过取数,用已有的 CSV

如果 Wind 终端没开,会自动跳过取数,用 data/raw/macro_raw.csv 继续跑。
"""

import os
import sys
import subprocess
import time
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "src")
RAW_CSV = os.path.join(BASE, "data", "raw", "macro_raw.csv")


def run(script, desc, optional=False):
    """运行一个脚本,失败时询问(或按 optional 跳过)。"""
    path = os.path.join(SRC, script)
    print("\n" + "=" * 70)
    print(f"  运行 {script}  —  {desc}")
    print("=" * 70)

    t0 = time.time()
    result = subprocess.run([sys.executable, path], cwd=BASE)
    dt = time.time() - t0

    if result.returncode != 0:
        print(f"\n[失败] {script} 返回码 {result.returncode},耗时 {dt:.1f}s")
        if optional:
            print("       这一步是可选的,继续下一步。")
            return False
        print("       流程中止。请把上面的报错发出来。")
        sys.exit(1)

    print(f"\n[完成] {script}  耗时 {dt:.1f}s")
    return True


def main():
    no_fetch = "--no-fetch" in sys.argv

    print("=" * 70)
    print("  中国宏观经济指标异常检测 — 完整流程")
    print(f"  开始时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    steps = []

    if no_fetch:
        print("\n[跳过] 取数步骤 (--no-fetch)")
        if not os.path.exists(RAW_CSV):
            print(f"[错误] 没有 {RAW_CSV},无法跳过取数。")
            print("       请先在有 Wind 终端的环境下运行 python src/fetch_wind.py")
            sys.exit(1)
    else:
        # 先用公开数据源(不需要任何订阅);失败再试 Wind
        ok = run("fetch_public.py", "从公开数据源提取宏观数据", optional=True)
        if not ok:
            print("[重试] 公开源失败,尝试 Wind ...")
            ok = run("fetch_wind.py", "从 Wind 提取宏观数据", optional=True)
        if not ok and os.path.exists(RAW_CSV):
            print(f"[继续] 取数失败,改用已有的 {RAW_CSV}")
        elif not ok:
            print("[中止] 取数失败且没有已有数据。")
            sys.exit(1)

    steps = [
        ("clean.py",    "数据清洗与派生指标", False),
        ("detect.py",   "四层异常检测",       False),
        ("evaluate.py", "评估与敏感性分析",   True),
        ("plot.py",     "生成图表",           True),
    ]

    for script, desc, optional in steps:
        run(script, desc, optional=optional)

    print("\n" + "=" * 70)
    print("  全部完成")
    print("=" * 70)
    print(f"  结束时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("\n  产出:")
    for d in ["data/processed", "results", "figures"]:
        p = os.path.join(BASE, d)
        if os.path.isdir(p):
            files = [f for f in os.listdir(p) if not f.startswith(".")]
            print(f"    {d}/  ({len(files)} 个文件)")
            for f in sorted(files):
                print(f"        {f}")
    print("\n  下一步: 把 results/ 和 figures/ 里的结果写进 README 的 'What I found' 一节。")


if __name__ == "__main__":
    main()
