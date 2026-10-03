#!/usr/bin/env python3
"""
server_run_matching.py — 服务器端运行 ACDL×ERA5 从头匹配的独立入口

在服务器(ACDL 数据所在机器)上运行:读取 ACDL 原始廓线 + ERA5 气压层 .mat,
按日产出匹配样本(见 Match_ACDL_ERA5_FromScratch.py)。

特点
  - 路径等参数全部走命令行,不用改源码(适合服务器/本地两处共用)
  - 运行前做预检:依赖、目录、ACDL 文件数、ERA5 月度文件是否齐、磁盘可写
  - 支持 --skip-existing 断点续跑
  - 支持 --jobs N 多进程:把日期区间切成 N 段并行跑(各进程写不同日文件,互不冲突)
  - 支持 --log 写日志、nohup 后台运行

用法
  # 0) 环境自检(不需数据)
  python server_run_matching.py --selftest

  # 1) 预检 + 试运行(只统计不写盘)
  python server_run_matching.py --acdl-dir /data/ACDL/ProfileMat --era5-root /data/era5_mat \
      --out-dir /home/user/matchdata --start 20220601 --end 20220630 --dry-run

  # 2) 正式跑(整月),后台 + 日志
  nohup python server_run_matching.py --acdl-dir /data/ACDL/ProfileMat --era5-root /data/era5_mat \
      --out-dir /home/user/matchdata --start 20220101 --end 20221231 --case 4 \
      --skip-existing --log match_2022.log > /dev/null 2>&1 &

  # 3) 多进程(4 个进程并行,各自跑一个季度)
  python server_run_matching.py ... --start 20220101 --end 20221231 --jobs 4 --skip-existing

依赖:numpy, scipy, h5py, hdf5storage
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 让日志在服务器无 BOM/编码问题
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


HERE = Path(__file__).resolve().parent
MATCHER = HERE / "Match_ACDL_ERA5_FromScratch.py"


# ============================================================
# 站点默认值(服务器路径写这里;命令行参数优先级更高)
# ============================================================
SITE_DEFAULTS = {
    "acdl_dir":  "/media/data61/lichengen/Production/Retrieval/Mat",
    "era5_root": "/media/data61/ZhangYi/FangMingZhao/era5_mat",
    "out_dir":   "/media/data61/ZhangYi/FangMingZhao/Matchoutput",
    "res":       "0.25deg",
    "prefix":    "ACDL11_",
    "case":      4,
    "time_tol":  30.0,
    "cov_min":   0.0,
    "buffer_m":  1000.0,
    "thick_m":   1000.0,
    "thick_cod": 0.3,
    "max_node_dist": 0.18,   # 廓线到所属 ERA5 节点的最大球面距离(度);见匹配脚本 CONFIG 注释
}


class Tee:
    """同时输出到终端与日志文件。"""

    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def load_matcher():
    spec = importlib.util.spec_from_file_location("acdl_match", MATCHER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def month_list(start: str, end: str) -> list[str]:
    d0 = datetime.strptime(start, "%Y%m%d")
    d1 = datetime.strptime(end, "%Y%m%d")
    out, y, m = [], d0.year, d0.month
    while (y, m) <= (d1.year, d1.month):
        out.append(f"{y:04d}{m:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def preflight(m, args) -> bool:
    """运行前检查:依赖、目录、数据齐不齐、可写。返回是否可以继续。"""
    ok = True
    print("=" * 78)
    print("预检 PREFLIGHT")
    print(f"  python        : {sys.version.split()[0]}  ({sys.executable})")
    for lib in ("numpy", "scipy", "h5py", "hdf5storage"):
        try:
            mod = __import__(lib)
            print(f"  {lib:13s} : {getattr(mod, '__version__', 'ok')}")
        except Exception as exc:
            print(f"  {lib:13s} : ✗ 缺失({exc})")
            ok = False
    print(f"  匹配脚本      : {MATCHER}  {'✓' if MATCHER.exists() else '✗ 不存在'}")

    acdl = Path(args.acdl_dir)
    era5 = Path(args.era5_root)
    out = Path(args.out_dir)
    for label, p in (("ACDL 目录", acdl), ("ERA5 根目录", era5)):
        if not p.exists():
            print(f"  {label:13s}: ✗ 不存在 {p}")
            ok = False
        else:
            print(f"  {label:13s}: ✓ {p}")

    # ACDL 文件计数
    if acdl.exists():
        files = sorted(acdl.glob(f"{args.prefix}*.mat"))
        in_range = [f for f in files
                    if any(d in f.name for d in _day_strings(args.start, args.end))]
        print(f"  ACDL 文件      : 共 {len(files)} 个;落在区间内 {len(in_range)} 个")
        if not files:
            print(f"                 ⚠ 未匹配到 {args.prefix}*.mat,请确认 --prefix / 目录")

    # ERA5 月度文件
    if era5.exists():
        missing = []
        for ym in month_list(args.start, args.end):
            base = era5 / "era5_pressure_levels" / args.res / ym
            for name in (f"temperature_{ym}.mat", f"relative_humidity_{ym}.mat",
                         f"geopotential_{ym}.mat"):
                if not (base / name).exists():
                    missing.append(str(base / name))
        if missing:
            print(f"  ERA5 月度文件  : ✗ 缺 {len(missing)} 个,例如 {missing[0]}")
            ok = False
        else:
            print(f"  ERA5 月度文件  : ✓ {len(month_list(args.start, args.end))} 个月齐备")

    # 输出目录可写
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".write_probe"
        probe.write_text("ok")
        probe.unlink()
        print(f"  输出目录      : ✓ 可写 {out}")
    except Exception as exc:
        print(f"  输出目录      : ✗ 不可写 {out} ({exc})")
        ok = False

    n_days = (datetime.strptime(args.end, "%Y%m%d") - datetime.strptime(args.start, "%Y%m%d")).days + 1
    print(f"  日期区间      : {args.start}~{args.end} ({n_days} 天)  QC_CASE={args.case}"
          f"  jobs={args.jobs}  skip_existing={args.skip_existing}  dry_run={args.dry_run}")
    print("=" * 78)
    return ok


def _day_strings(start: str, end: str) -> list[str]:
    d0 = datetime.strptime(start, "%Y%m%d")
    d1 = datetime.strptime(end, "%Y%m%d")
    out, d = [], d0
    while d <= d1:
        out.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return out


def apply_config(m, args) -> None:
    m.CONFIG.update({
        "ACDL_DIR": str(args.acdl_dir),
        "ACDL_PREFIX": args.prefix,
        "ERA5_MAT_ROOT": str(args.era5_root),
        "ERA5_RES": args.res,
        "OUT_DIR": str(args.out_dir),
        "QC_CASE": args.case,
        "TIME_TOL_MIN": args.time_tol,
        "COV_MIN": args.cov_min,
        "CLOUD_BUFFER_M": args.buffer_m,
        "CLOUD_THICK_M": args.thick_m,
        "CLOUD_THICK_COD": args.thick_cod,
        "MAX_NODE_DIST_DEG": args.max_node_dist,
    })


def run_parallel(args) -> int:
    """把日期区间切成 jobs 段,各起一个子进程跑(互不重叠)。"""
    d0 = datetime.strptime(args.start, "%Y%m%d")
    d1 = datetime.strptime(args.end, "%Y%m%d")
    total_days = (d1 - d0).days + 1
    jobs = max(1, min(args.jobs, total_days))
    step = (total_days + jobs - 1) // jobs

    print(f"[PARALLEL] 拆成 {jobs} 段,每段约 {step} 天")
    procs = []
    for k in range(jobs):
        s = d0 + timedelta(days=k * step)
        e = min(d0 + timedelta(days=(k + 1) * step - 1), d1)
        if s > e:
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--acdl-dir", args.acdl_dir, "--era5-root", args.era5_root,
               "--out-dir", args.out_dir, "--res", args.res, "--prefix", args.prefix,
               "--start", s.strftime("%Y%m%d"), "--end", e.strftime("%Y%m%d"),
               "--case", str(args.case), "--time-tol", str(args.time_tol),
               "--cov-min", str(args.cov_min), "--buffer-m", str(args.buffer_m),
               "--thick-m", str(args.thick_m), "--thick-cod", str(args.thick_cod),
               "--max-node-dist", str(args.max_node_dist),
               "--jobs", "1"]
        if args.skip_existing:
            cmd.append("--skip-existing")
        if args.with_stats:
            cmd.append("--with-stats")
        if args.no_cache:
            cmd.append("--no-cache")
        if args.no_super_levels:
            cmd.append("--no-super-levels")
        if args.dry_run:
            cmd.append("--dry-run")
        log = Path(args.out_dir) / f"_worker_{k+1}_{s.strftime('%Y%m%d')}_{e.strftime('%Y%m%d')}.log"
        print(f"  [worker {k+1}] {s:%Y%m%d}~{e:%Y%m%d} → {log.name}")
        fh = open(log, "w", encoding="utf-8")
        procs.append((subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT), fh, log))
    rc = 0
    for p, fh, log in procs:
        code = p.wait()
        fh.close()
        if code != 0:
            rc = 1
            print(f"  [worker] ✗ 退出码 {code},见 {log}")
    print(f"[PARALLEL] 全部结束,退出码 {rc}")
    return rc


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="服务器端 ACDL×ERA5 从头匹配入口(调用 Match_ACDL_ERA5_FromScratch.py)")
    ap.add_argument("--acdl-dir", required=False, help="ACDL .mat 所在目录")
    ap.add_argument("--era5-root", required=False, help="ERA5 产物根目录(含 era5_pressure_levels/)")
    ap.add_argument("--out-dir", required=False, help="输出目录")
    ap.add_argument("--res", default=None, help="ERA5 分辨率子目录名(默认 0.25deg)")
    ap.add_argument("--prefix", default=None, help="ACDL 文件名前缀(默认 ACDL11_)")
    ap.add_argument("--start", default=None, help="起始日期 YYYYMMDD")
    ap.add_argument("--end", default=None, help="结束日期 YYYYMMDD(含)")
    ap.add_argument("--case", type=int, choices=(1, 2, 3, 4), default=None,
                    help="QC Case:1严格/2宽松/3缓冲/4自适应(默认 4)")
    ap.add_argument("--time-tol", type=float, default=None, help="时间窗口(分钟,默认 30)")
    ap.add_argument("--cov-min", type=float, default=None, help="廓线级最小覆盖(默认 0)")
    ap.add_argument("--buffer-m", type=float, default=None, help="云底缓冲带厚度 m(默认 1000)")
    ap.add_argument("--thick-m", type=float, default=None, help="厚云判定:云块厚度 m(默认 1000)")
    ap.add_argument("--thick-cod", type=float, default=None, help="厚云判定:COD 阈值(默认 0.3)")
    ap.add_argument("--max-node-dist", type=float, default=None,
                    help="廓线到所属 ERA5 节点的最大球面距离(度,默认 0.18)。"
                         "设成很大的值(如 999)= 关闭限制,退回到'域外廓线吸附到边界节点'的旧行为")
    ap.add_argument("--jobs", type=int, default=1, help="并行进程数(默认 1;>1 时切分日期区间)")
    ap.add_argument("--skip-existing", action="store_true", help="跳过已存在的输出文件")
    ap.add_argument("--with-stats", action="store_true",
                    help="输出完整 264 列(含测试用统计/QC 列);默认 lite 132 列")
    ap.add_argument("--no-cache", action="store_true", help="不使用日级缓存(强制解析 .mat)")
    ap.add_argument("--no-super-levels", action="store_true",
                    help="不并入 ERA5 单层列(BLH/TCWV/Z_sfc),输出 132 列")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写盘")
    ap.add_argument("--log", default=None, help="把日志同时写入该文件")
    ap.add_argument("--selftest", action="store_true", help="运行匹配脚本的离线自检")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    log_fh = open(args.log, "a", encoding="utf-8") if args.log else None
    if log_fh:
        sys.stdout = Tee(sys.__stdout__, log_fh)
        sys.stderr = Tee(sys.__stderr__, log_fh)
    print(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} server_run_matching 启动 =====")

    m = load_matcher()

    if args.selftest:
        return m.selftest()

    # 参数解析优先级:命令行 > SITE_DEFAULTS(本文件顶部)> 匹配脚本 CONFIG
    def pick(cli, key, cfg_key):
        if cli is not None:
            return cli
        if key in SITE_DEFAULTS and SITE_DEFAULTS[key] is not None:
            return SITE_DEFAULTS[key]
        return m.CONFIG[cfg_key]

    args.acdl_dir = pick(args.acdl_dir, "acdl_dir", "ACDL_DIR")
    args.era5_root = pick(args.era5_root, "era5_root", "ERA5_MAT_ROOT")
    args.out_dir = pick(args.out_dir, "out_dir", "OUT_DIR")
    args.res = pick(args.res, "res", "ERA5_RES")
    args.prefix = pick(args.prefix, "prefix", "ACDL_PREFIX")
    args.start = pick(args.start, "start", "START")
    args.end = pick(args.end, "end", "END")
    args.case = pick(args.case, "case", "QC_CASE")
    args.time_tol = pick(args.time_tol, "time_tol", "TIME_TOL_MIN")
    args.cov_min = pick(args.cov_min, "cov_min", "COV_MIN")
    args.buffer_m = pick(args.buffer_m, "buffer_m", "CLOUD_BUFFER_M")
    args.thick_m = pick(args.thick_m, "thick_m", "CLOUD_THICK_M")
    args.thick_cod = pick(args.thick_cod, "thick_cod", "CLOUD_THICK_COD")
    args.max_node_dist = pick(args.max_node_dist, "max_node_dist", "MAX_NODE_DIST_DEG")

    if not preflight(m, args):
        print("[ABORT] 预检未通过,请先修复上面标 ✗/⚠ 的项。")
        return 2

    if args.jobs and args.jobs > 1 and not args.dry_run:
        rc = run_parallel(args)
        print(f"===== 结束 {datetime.now():%Y-%m-%d %H:%M:%S} rc={rc} =====")
        return rc

    apply_config(m, args)
    m.CONFIG["OUTPUT_STATS"] = bool(args.with_stats)
    m.CONFIG["USE_CACHE"] = not args.no_cache
    if args.no_super_levels:
        m.CONFIG["SUPER_LEVELS"] = ()
    rows = m.run(args.start, args.end, args.dry_run, skip_existing=args.skip_existing)
    print(f"[SUMMARY] 区间 {args.start}~{args.end} 共产出样本行数 {rows}"
          f"  (QC_CASE={args.case}, {'DRY-RUN' if args.dry_run else 'written'})")
    print(f"===== 结束 {datetime.now():%Y-%m-%d %H:%M:%S} rc=0 =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
