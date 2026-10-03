
"""
add_era5_single_level_columns.py   【仅用于旧的 132 列产物】
  ⚠️ 2026-09-21 起,匹配脚本 match_acdl_era5_from_scratch.py 已直接写出这三列
     (SUPER_LEVELS / --super-levels),新匹配无需本脚本;此处保留仅为给已有旧产物后补。
============================================================
把 ERA5 单层场(默认:BLH 边界层高度、TCWV 整层水汽、Z_sfc 地表位势高度)
**按 (ERA5 格点, 整点) 精确对齐**,追加到已有的匹配样本上,输出到新目录。

为什么能精确对齐:
  匹配产物里存的 ERA5_Lon/Lat 就是 0.25° 网格节点值、ERA5_Time 就是整点 datenum,
  而单层场用的是同一套网格与同一套整点时间 → 直接按 (格点, 时次) 索引即可,无需插值。

输入:
  <MATCH_DIR>/ACDL_ERA5_MatchV2_YYYYMMDD.mat        (struct MatchV2)
  <ERA5_SINGLE_ROOT>/<RES>/YYYYMM/<var>_YYYYMM.mat  (struct ECMWF,由 era5_grib_to_mat.py 生成)
      boundary_layer_height_YYYYMM.mat / total_column_water_vapour_YYYYMM.mat / geopotential_YYYYMM.mat

输出:
  <OUT_DIR>/ACDL_ERA5_MatchV2_YYYYMMDD.mat(默认 OUT_DIR = MATCH_DIR + "_plus")
  列 = 原列 + 追加列(顺序见 --vars);VarNames / Level / Meta 同步扩展。

用法(路径已写在脚本顶部 CONFIG 里,直接运行即可):
  python add_era5_single_level_columns.py                 # 用 SITE 指定的站点路径
  python add_era5_single_level_columns.py --match-dir ...  # 需要时仍可用命令行覆盖
============================================================
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import h5py
import hdf5storage


# ============================================================
# 配置(直接改这里,平时无需命令行参数)
# ============================================================
SITE = "local"          # "local"(本机)或 "server"(服务器);改这一个词即可切换

SITE_PATHS = {
    "local": {
        "MATCH_DIR":        r"D:/matchdata",
        "ERA5_SINGLE_ROOT": r"D:/era5_mat/era5_single_levels",
        "OUT_DIR":          r"D:/matchdata_plus",
    },
    "server": {
        "MATCH_DIR":        r"/media/data61/ZhangYi/FangMingZhao/Matchoutput",
        "ERA5_SINGLE_ROOT": r"/media/data61/ZhangYi/FangMingZhao/era5_mat/era5_single_levels",
        "OUT_DIR":          r"/media/data61/ZhangYi/FangMingZhao/Matchoutput_plus",
    },
}

CONFIG = {
    **SITE_PATHS[SITE],
    "VARS":         "blh,tcwv,zsfc",   # 要追加的列:blh / tcwv / zsfc(逗号分隔)
    "RES":          "0.25deg",
    "TIME_TOL_MIN": 30.0,
    "OVERWRITE":    False,
}


# 单层变量定义:名称 → (文件名前缀, 单位, 是否需除以 g0)
VAR_DEFS = {
    "blh":  ("boundary_layer_height", "m", False),
    "tcwv": ("total_column_water_vapour", "kg m^-2", False),
    "zsfc": ("geopotential", "m", True),          # 地表位势 → /g0 = 地形高度(m)
}
G0 = 9.80665


# ============================================================
# 读取
# ============================================================
def decode_varnames(f, ds) -> list[str] | None:
    try:
        refs = np.asarray(ds[()])
        return ["".join(chr(int(c)) for c in np.asarray(f[r][()]).ravel()) for r in refs.ravel()]
    except Exception:
        return None


class SingleLevelMonth:
    """按月惰性打开单层变量;把 Data 统一成 (time, nlat, nlon) 的索引顺序。"""

    def __init__(self, root: Path, res: str, yyyymm: str, vars_: list[str]):
        base = root / res / yyyymm
        self.arrs: dict[str, np.ndarray] = {}
        self.lat2d = self.lon2d = None
        self.time = None
        for v in vars_:
            prefix, _unit, _divg = VAR_DEFS[v]
            p = base / f"{prefix}_{yyyymm}.mat"
            if not p.exists():
                raise FileNotFoundError(f"缺少单层文件: {p}")
            with h5py.File(p, "r") as f:
                g = f["ECMWF"]
                D = np.asarray(g["Data"][()], dtype=np.float64)       # 磁盘 (time, lon, lat)
                lat_d = np.asarray(g["Lat"][()], dtype=np.float64)
                lon_d = np.asarray(g["Lon"][()], dtype=np.float64)
                t = np.asarray(g["Time"][()], dtype=np.float64).ravel()
            if self.time is None:
                self.time = t
                self.lat2d = lat_d.T if lat_d.shape == D.shape[1:] else lat_d     # → 逻辑 (nlat, nlon)
                self.lon2d = lon_d.T if lon_d.shape == D.shape[1:] else lon_d
                self.nlat, self.nlon = self.lat2d.shape
                self._node_of = {}
                for r in range(self.nlat):
                    for c in range(self.nlon):
                        self._node_of[(round(float(self.lon2d[r, c]), 4),
                                       round(float(self.lat2d[r, c]), 4))] = (r, c)
            elif not np.array_equal(t, self.time):
                raise ValueError(f"{p.name}: Time 与其它单层变量不一致")
            # → (time, lat, lon)
            self.arrs[v] = np.transpose(D, (0, 2, 1)) if D.shape == (D.shape[0], self.nlon, self.nlat) else D

    def node_index(self, lon: float, lat: float):
        """返回 (r, c);命中不到返回 None。"""
        return self._node_of.get((round(float(lon), 4), round(float(lat), 4)))

    def time_index(self, datenum: float) -> tuple[int, float]:
        i = int(np.argmin(np.abs(self.time - datenum)))
        return i, (self.time[i] - datenum) * 1440.0          # 偏移(分钟)


# ============================================================
# 追加一列
# ============================================================
def append_columns(match_file: Path, out_file: Path, sl_root: Path, res: str,
                   vars_: list[str], time_tol_min: float) -> dict:
    with h5py.File(match_file, "r") as f:
        g = f["MatchV2"]
        D = np.asarray(g["Data"][()], dtype=np.float64)
        names = decode_varnames(f, g["VarNames"])
        level = np.asarray(g["Level"][()]).ravel() if "Level" in g else None
        meta = {}
        if "Meta" in g:
            try:
                meta = {k: (v.decode() if isinstance(v, (bytes, np.bytes_)) else v)
                        for k, v in g["Meta"].attrs.items()}
            except Exception:
                meta = {}
    if D.ndim != 2:
        raise ValueError(f"{match_file.name}: Data 维度异常 {D.shape}")
    if D.shape[1] not in (132, 264) and (names is None or D.shape[1] != len(names)):
        # 磁盘列主序 → 逻辑 (N, C)
        if D.shape[0] in (132, 133, 134, 135, 264):
            D = D.T
    n = D.shape[0]
    lon, lat, t_dn = D[:, 0], D[:, 1], D[:, 2]
    yyyymm = match_file.stem.split("_")[-1][:6]
    sl = SingleLevelMonth(sl_root, res, yyyymm, vars_)

    new_cols = {v: np.full(n, np.nan) for v in vars_}
    n_node_miss = 0
    n_time_miss = 0
    max_off = 0.0
    for i in range(n):
        rc = sl.node_index(lon[i], lat[i])
        if rc is None:
            n_node_miss += 1
            continue
        ti, off = sl.time_index(t_dn[i])
        max_off = max(max_off, abs(off))
        if abs(off) > time_tol_min:
            n_time_miss += 1
            continue
        r, c = rc
        for v in vars_:
            val = sl.arrs[v][ti, r, c]
            if VAR_DEFS[v][2]:
                val = val / G0
            new_cols[v][i] = val

    add_names = []
    for v in vars_:
        add_names.append({"blh": "BLH_m", "tcwv": "TCWV_kgm2", "zsfc": "Z_sfc_m"}[v])
    Data_out = np.column_stack([D] + [new_cols[v] for v in vars_])
    names_out = (names if names is not None else [f"col{i}" for i in range(D.shape[1])]) + add_names

    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {"Data": Data_out, "VarNames": names_out}
    if level is not None:
        payload["Level"] = level.reshape(-1, 1)
    meta_out = dict(meta) if isinstance(meta, dict) else {}
    meta_out["SingleLevelColumns"] = ",".join(add_names)
    meta_out["SingleLevelSource"] = f"{sl_root}/{res}/{yyyymm}"
    payload["Meta"] = meta_out
    hdf5storage.savemat(str(out_file), {"MatchV2": payload}, fmt="7.3",
                        store_python_metadata=False, appendmat=False,
                        matlab_compatible=True, oned_as="column",
                        action_for_matlab_incompatible="error")
    return {"file": match_file.name, "rows": n, "cols_out": Data_out.shape[1],
            "node_miss": n_node_miss, "time_miss": n_time_miss, "max_off_min": max_off,
            "nan": {v: int(np.isnan(new_cols[v]).sum()) for v in vars_}}


# ============================================================
# 主流程
# ============================================================
def run(args) -> int:
    match_dir = Path(args.match_dir)
    out_dir = Path(args.out_dir) if args.out_dir else Path(str(match_dir) + "_plus")
    sl_root = Path(args.era5_single_root)
    vars_ = [v.strip() for v in args.vars.split(",") if v.strip()]
    bad = [v for v in vars_ if v not in VAR_DEFS]
    if bad:
        print(f"[ERROR] 未知变量 {bad};可选 {list(VAR_DEFS)}")
        return 2
    files = sorted(match_dir.glob("ACDL_ERA5_MatchV2_*.mat"))
    if not files:
        print(f"[ERROR] {match_dir} 下没有 ACDL_ERA5_MatchV2_*.mat")
        return 1

    print("=" * 78)
    print("为匹配样本追加 ERA5 单层列")
    print(f"  匹配目录: {match_dir}   ({len(files)} 个文件)")
    print(f"  单层根目录: {sl_root}/{args.res}/<YYYYMM>/")
    print(f"  追加变量: {vars_}  →  {[ {'blh':'BLH_m','tcwv':'TCWV_kgm2','zsfc':'Z_sfc_m'}[v] for v in vars_]}")
    print(f"  输出目录: {out_dir}   覆盖: {args.overwrite}")
    print("=" * 78)

    t0 = time.perf_counter()
    reports, skipped = [], 0
    for k, fp in enumerate(files, 1):
        out_file = out_dir / fp.name
        if out_file.exists() and not args.overwrite:
            print(f"[{k}/{len(files)}] {fp.name}: 已存在,跳过")
            skipped += 1
            continue
        try:
            rep = append_columns(fp, out_file, sl_root, args.res, vars_, args.time_tol)
        except FileNotFoundError as exc:
            print(f"[{k}/{len(files)}] {fp.name}: ✗ {exc}")
            continue
        reports.append(rep)
        print(f"[{k}/{len(files)}] {fp.name}: {rep['rows']} 行 → {rep['cols_out']} 列 "
              f"| 节点未命中 {rep['node_miss']} 时次超窗 {rep['time_miss']} "
              f"最大时差 {rep['max_off_min']:.1f} min | NaN {rep['nan']}")

    if reports:
        rows = sum(r["rows"] for r in reports)
        nm = sum(r["node_miss"] for r in reports)
        tm = sum(r["time_miss"] for r in reports)
        nan = {v: sum(r["nan"][v] for r in reports) for v in vars_}
        print("\n" + "=" * 78)
        print(f"[SUMMARY] 处理 {len(reports)} 个文件,共 {rows} 行;跳过 {skipped} 个")
        print(f"  节点未命中 {nm} 行({nm/max(rows,1)*100:.3f}%);时次超窗 {tm} 行;新列 NaN {nan}")
        print(f"  用时 {time.perf_counter()-t0:.0f}s")
        print(f"  输出目录: {out_dir}")
        print("  提示:训练脚本用 --match-dir 指向该目录即可;新列追加在末尾,原有 132 列索引不变")
        print("=" * 78)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="给匹配样本追加 ERA5 单层列(BLH/TCWV/Z_sfc)")
    ap.add_argument("--match-dir", default=CONFIG["MATCH_DIR"],
                    help=f"含 ACDL_ERA5_MatchV2_*.mat 的目录(默认见脚本 CONFIG:{CONFIG['MATCH_DIR']})")
    ap.add_argument("--out-dir", default=CONFIG["OUT_DIR"], help="输出目录(默认见脚本 CONFIG)")
    ap.add_argument("--era5-single-root", default=CONFIG["ERA5_SINGLE_ROOT"],
                    help="ERA5 单层产物根目录(其下为 <res>/<YYYYMM>/)")
    ap.add_argument("--res", default=CONFIG["RES"])
    ap.add_argument("--vars", default=CONFIG["VARS"], help="逗号分隔,可选 blh,tcwv,zsfc")
    ap.add_argument("--time-tol", type=float, default=CONFIG["TIME_TOL_MIN"], help="时间容差(分钟)")
    ap.add_argument("--overwrite", action="store_true", default=CONFIG["OVERWRITE"],
                    help="覆盖已存在的输出文件")
    sys.exit(run(ap.parse_args()))
