#!/usr/bin/env python3
"""
Plot_MatchResult_Map_Profiles.py — 匹配结果可视化(左:地图分布;右:消光廓线)

参照 MatchResult_Map_Profiles.png 的版式,但数据源换成现在的从头匹配产物:
  左图:中国区域 + ACDL 轨道条带 + 全部匹配点(灰) + 抽中的 N 个点(彩色编号 + 时次标注)
  右图:抽中点的 **ACDL 原始 1291 bin 消光廓线**(≈64.5 m 一层,最接近观测本身)

数据来源(两份配合使用)
  <MATCH_DIR>/ACDL_ERA5_MatchV2_*.mat     逐日匹配样本(只用 0-3 列:ERA5_Lon/Lat/Time/Hour)
  <ACDL_DIR>/ACDL11_<YYYYMMDD>*.mat       ACDL 原始廓线(Output/ 下的 Extinction_* 等)

**关键**:匹配产物只存了 32 个 ERA5 层段的均值(Ext_mean_L00–L31),没有 1291 bin 廓线。
所以右图必须按**与匹配完全相同的口径**回原始文件把该 (节点, 时次) 的廓线重新聚合:
    最近 0.25° 节点相同 且 |t_acdl − t_era5| ≤ TIME_TOL_MIN  → 逐 bin 取 nanmean
这样画出来的是"这一行匹配样本的原始分辨率版本",而不是随便找的一条邻近廓线。

内存策略:一天的 Extinction 全读进来约 550 MB(27 文件 × 3936 × 1291 × 4B),
因此**只读轻量索引**(lon/lat/time,约 2.5 MB/天),廓线按需用花式索引只取命中的那几行。

用法
  # 最简(用脚本顶部 CONFIG 的路径,随机抽 8 个点)
  python Plot_MatchResult_Map_Profiles.py

  # 指定日期区间 + 点数 + 种子
  python Plot_MatchResult_Map_Profiles.py --dates 20220601-20220603 \
      --n-points 8 --seed 40 --out figs/Match_202206.png

  # 自己指定要画的样本(全局行号,按过滤后的顺序;行号会打印出来)
  python Plot_MatchResult_Map_Profiles.py --points 120,3400,51000

  # 廓线太毛糙时加滑动中值
  python Plot_MatchResult_Map_Profiles.py --smooth 9

国界依赖(可选,按优先级)
  1) --boundary 指定的 .shp(需要 pyshp)
  2) cartopy 的 Natural Earth 在线数据(需要 cartopy + 首次联网)
  3) 都没有 → 只画散点与轨道,并提示

依赖:numpy, h5py, matplotlib;pyshp 或 cartopy(国界,可选)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# 配置(命令行优先级更高)
# ============================================================
CONFIG = {
    "MATCH_DIR": r"D:/matchdata",
    "MATCH_GLOB": "ACDL_ERA5_MatchV2_*.mat",
    "STRUCT": "MatchV2",
    "ACDL_DIR": r"D:/ACDL/Data/ACDL/ProfileMat",
    "ACDL_PREFIX": "ACDL11_",

    # 匹配口径 —— 必须与 Match_ACDL_ERA5_FromScratch.py 一致
    "GRID_RES": 0.25,        # ERA5 节点间距(度)
    "TIME_TOL_MIN": 30.0,    # 时间窗(分钟)
    "N_BIN": 1291,           # ACDL 原始廓线层数

    # 抽样
    "N_POINTS": 8,
    "SEED": 40,
    "DATES": None,           # None=全部;或 "20220601-20220603" / "20220601,20220602"

    # 廓线显示
    "E_MIN": 0.0,            # ≤ 此值 → NaN(不画)
    "E_MAX": 100.0,          # > 此值 → NaN(剔异常)
    "SMOOTH": 0,             # >1:沿高度做 k 点滑动中值(0/1 = 原始)
    "X_LIM": (1e-4, 100.0),  # 对数横轴

    # 地图
    "MAP_XLIM": (73.0, 136.0),
    "MAP_YLIM": (3.0, 54.0),
    "ORBITS": True,
    "ORBIT_ALPHA": 0.5,
    # 国界:支持 .mat(MATLAB v7.3 的 combined_shp,X/Y 单元数组)/ .shp(pyshp)/ .geojson
    # 置 None → 尝试 cartopy 的 Natural Earth
    "BOUNDARY": r"D:/ACDL/Data/countryProvince.mat",

    # 输出
    "OUT": "MatchResult_Map_Profiles.png",
    "DPI": 300,
}

COL_LON, COL_LAT, COL_TIME, COL_HOUR = 0, 1, 2, 3

# ColorBrewer Set1 风格 10 色(与原 MATLAB 版一致)
CMAP10 = np.array([
    [0.894, 0.102, 0.110],   # 红
    [0.216, 0.494, 0.722],   # 蓝
    [0.302, 0.686, 0.290],   # 绿
    [0.596, 0.306, 0.639],   # 紫
    [1.000, 0.498, 0.000],   # 橙
    [1.000, 1.000, 0.200],   # 黄
    [0.651, 0.337, 0.157],   # 棕
    [0.969, 0.506, 0.749],   # 粉
    [0.400, 0.400, 0.400],   # 灰
    [0.000, 0.749, 0.749],   # 青
])


# ============================================================
# 小工具
# ============================================================
def tpi_to_datenum(tpi: np.ndarray) -> np.ndarray:
    """ACDL Profile_UTC_Time(TAI93, 1e-4 s)→ MATLAB datenum(与匹配脚本一致)。"""
    return np.asarray(tpi, dtype=np.float64) * 1e-4 / 86400.0 + 730486.5


# MATLAB datenum(1970-01-01) = 719529(注意不是 719163,差 366 天)
DN_UNIX_OFFSET = 719529.0


def datenum_to_datestr(dn: float) -> str:
    """MATLAB datenum → 'YYYY-MM-DD HH:MM'(UTC)。"""
    if not np.isfinite(dn):
        return "?"
    base = _dt.datetime(1970, 1, 1) + _dt.timedelta(days=float(dn) - DN_UNIX_OFFSET)
    return base.strftime("%Y-%m-%d %H:%M")


def day_of(src_name: str) -> str:
    """'ACDL_ERA5_MatchV2_20220601.mat' → '20220601'。"""
    return src_name.split("_")[-1].split(".")[0]


def parse_dates(spec: str | None) -> list[str] | None:
    """'20220601-20220603' 或 '20220601,20220602' → 日期列表;None → None。"""
    if not spec:
        return None
    out: list[str] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (x.strip() for x in part.split("-", 1))
            d = _dt.datetime.strptime(a, "%Y%m%d")
            d1 = _dt.datetime.strptime(b, "%Y%m%d")
            while d <= d1:
                out.append(d.strftime("%Y%m%d"))
                d += _dt.timedelta(days=1)
        else:
            out.append(_dt.datetime.strptime(part, "%Y%m%d").strftime("%Y%m%d"))
    return sorted(set(out))


def smooth_profile(y: np.ndarray, k: int) -> np.ndarray:
    """沿高度做 k 点滑动中值(k ≤ 1 原样返回)。"""
    if k is None or k <= 1:
        return y
    n = y.size
    half = k // 2
    out = np.full(n, np.nan)
    for i in range(n):
        w = y[max(0, i - half):min(n, i + half + 1)]
        w = w[np.isfinite(w)]
        if w.size:
            out[i] = np.median(w)
    return out


# ============================================================
# 匹配产物:只取前 4 列标识
# ============================================================
def read_matched(match_dir: Path, glob_pat: str, struct: str, dates: list[str] | None):
    files = sorted(match_dir.glob(glob_pat))
    if dates:
        files = [f for f in files if any(d in f.name for d in dates)]
    if not files:
        raise FileNotFoundError(f"{match_dir} 下没有符合条件的 {glob_pat}"
                                + (f"(dates={dates})" if dates else ""))

    lons, lats, tms, hrs, src = [], [], [], [], []
    for fp in files:
        with h5py.File(fp, "r") as f:
            if struct not in f:
                print(f"  [WARN] {fp.name}: 无 {struct},跳过")
                continue
            g = f[struct]
            D = np.asarray(g["Data"][()], dtype=np.float64)
            # 磁盘列主序 → 逻辑 (N, C);按列数判断(样本数可能少于列数)
            ncol = None
            if "VarNames" in g:
                try:
                    ncol = int(np.asarray(g["VarNames"][()]).size)
                except Exception:
                    ncol = None
            if ncol is not None:
                if D.shape[1] != ncol and D.shape[0] == ncol:
                    D = D.T
            elif D.shape[0] < D.shape[1]:
                D = D.T
            if D.shape[1] < 4:
                print(f"  [WARN] {fp.name}: 列数 {D.shape[1]} < 4,跳过")
                continue
        n = D.shape[0]
        lons.append(D[:, COL_LON]); lats.append(D[:, COL_LAT])
        tms.append(D[:, COL_TIME]); hrs.append(D[:, COL_HOUR])
        src.extend([fp.name] * n)

    lon = np.concatenate(lons); lat = np.concatenate(lats)
    t_dn = np.concatenate(tms); hour = np.concatenate(hrs)
    ok = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(t_dn)
    if not ok.all():
        print(f"[DATA] 剔除 {int((~ok).sum())} 行(经纬度/时间含 NaN)")
    src = [s for s, k in zip(src, ok) if k]
    days = sorted({day_of(s) for s in src})
    print(f"[DATA] 匹配点 {int(ok.sum())} 个,来自 {len(files)} 个文件;日期 {len(days)} 天 {days[0]}~{days[-1]}")
    return {"lon": lon[ok], "lat": lat[ok], "time": t_dn[ok], "hour": hour[ok],
            "src": src, "days": days, "n_files": len(files)}


# ============================================================
# ACDL 原始数据:轻量索引 + 按需取行
# ============================================================
def acdl_day_files(acdl_dir: Path, prefix: str, yyyymmdd: str) -> list[Path]:
    return sorted(acdl_dir.glob(f"{prefix}{yyyymmdd}*.mat"))


def build_day_index(acdl_dir: Path, prefix: str, yyyymmdd: str):
    """一天的轻量索引:每文件的 lon/lat/datenum + 文件清单(不读廓线,约 2.5 MB/天)。"""
    files = acdl_day_files(acdl_dir, prefix, yyyymmdd)
    if not files:
        return None
    lo, la, tm, fpaths, counts, alt = [], [], [], [], [], None
    for fp in files:
        try:
            with h5py.File(fp, "r") as f:
                g = f["Output"]
                la_i = np.asarray(g["Latitude"][()], dtype=np.float64).ravel()
                lo.append(np.asarray(g["Longitude"][()], dtype=np.float64).ravel())
                la.append(la_i)
                tm.append(tpi_to_datenum(np.asarray(g["Profile_UTC_Time"][()]).ravel()))
                if alt is None and "Altitude" in g:
                    alt = np.asarray(g["Altitude"][()], dtype=np.float64).ravel()
        except Exception as exc:
            print(f"  [WARN] 索引失败 {fp.name}: {exc}")
            continue
        fpaths.append(str(fp)); counts.append(la_i.size)
    if not lo:
        return None
    offs = np.concatenate([[0], np.cumsum(counts)[:-1]])      # 各文件在拼接数组里的起始下标
    return {"lon": np.concatenate(lo), "lat": np.concatenate(la),
            "time": np.concatenate(tm), "fpath": fpaths, "alt": alt,
            "file": np.repeat(np.arange(len(fpaths)), counts),
            # base[g] = 全局下标 g 所属文件的首行在拼接数组里的位置 → 文件内行号 = g - base[g]
            "base": np.repeat(offs, counts)}


def recover_profile(idx: dict, node_lon: float, node_lat: float, era5_time: float,
                    cfg: dict):
    """按匹配口径聚合该 (节点, 时次) 的 ACDL 廓线 → (ext_1291, n_prof, alt)。"""
    res = cfg["GRID_RES"]
    tol = cfg["TIME_TOL_MIN"] / 1440.0
    pn_lon = np.round(idx["lon"] / res) * res
    pn_lat = np.round(idx["lat"] / res) * res
    m = (np.abs(pn_lon - node_lon) < 1e-6) & \
        (np.abs(pn_lat - node_lat) < 1e-6) & \
        (np.abs(idx["time"] - era5_time) <= tol)
    if not m.any():
        return None, 0, idx["alt"]

    n_bin = cfg["N_BIN"]
    gsel = np.flatnonzero(m)                           # 命中的全局下标(拼接数组)
    fids = idx["file"][gsel]
    tot = np.zeros(n_bin); cnt = np.zeros(n_bin)
    for fid in np.unique(fids):
        g = gsel[fids == fid]
        rws = np.unique(g - idx["base"][g])            # → 该文件内的行号
        try:
            with h5py.File(idx["fpath"][int(fid)], "r") as f:
                E = np.asarray(f["Output"]["Extinction_Coefficient_532"][rws, :],
                               dtype=np.float64)
        except Exception as exc:
            print(f"  [WARN] 读廓线失败 {Path(idx['fpath'][int(fid)]).name}: {exc}")
            continue
        if E.shape[1] != n_bin:
            if E.shape[0] == n_bin:
                E = E.T
            else:
                continue
        cnt += np.isfinite(E).sum(axis=0)
        tot += np.nansum(E, axis=0)
    if not (cnt > 0).any():
        return None, 0, idx["alt"]
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan), int(m.sum()), idx["alt"]


def read_acdl_lonlat(acdl_dir: Path, prefix: str, days: list[str]):
    """只读经纬度画轨道。返回 (lon, lat, 分段索引),每个 ACDL 文件是一条连续航迹。"""
    lon_parts, lat_parts, seg = [], [], [0]
    n_file = 0
    for d in days:
        for fp in acdl_day_files(acdl_dir, prefix, d):
            try:
                with h5py.File(fp, "r") as f:
                    g = f["Output"]
                    lo = np.asarray(g["Longitude"][()], dtype=np.float64).ravel()
                    la = np.asarray(g["Latitude"][()], dtype=np.float64).ravel()
            except Exception as exc:
                print(f"  [WARN] 轨道读取失败 {fp.name}: {exc}")
                continue
            if lo.size == 0:
                continue
            lon_parts.append(lo); lat_parts.append(la)
            seg.append(seg[-1] + lo.size)
            n_file += 1
    if not lon_parts:
        return None, None, None
    print(f"[ORBIT] {n_file} 个 ACDL 文件,{seg[-1]} 个廓线点")
    return np.concatenate(lon_parts), np.concatenate(lat_parts), np.array(seg)


# ============================================================
# 选点 + 还原廓线(按天分批,索引缓存;廓线按需取行)
# ============================================================
def select_profiles(data: dict, cfg: dict, acdl_dir: Path, explicit: str | None):
    """返回 profs 列表:每项 {row, lon, lat, time, ext, alt, n_prof}。"""
    n_all = len(data["lon"])
    n_want = int(cfg["N_POINTS"])
    by_day: dict[str, list[int]] = {}
    for i, s in enumerate(data["src"]):
        by_day.setdefault(day_of(s), []).append(i)

    idx_cache: dict[str, dict | None] = {}

    def get_idx(d: str):
        if d not in idx_cache:
            print(f"[ACDL] 建立索引 {d} ...")
            idx_cache[d] = build_day_index(acdl_dir, cfg["ACDL_PREFIX"], d)
        return idx_cache[d]

    def take(i: int):
        d = day_of(data["src"][i])
        idx = get_idx(d)
        if idx is None:
            return None
        ext, k, alt = recover_profile(idx, data["lon"][i], data["lat"][i],
                                      data["time"][i], cfg)
        if ext is None:
            return None
        a = np.asarray(alt, dtype=float) if alt is not None else np.array([])
        if a.size != ext.size:
            a = np.arange(ext.size, dtype=float)          # 拿不到高度轴 → 退化用层号
        elif a[0] > a[-1]:
            a, ext = a[::-1], ext[::-1]                   # 统一成升序
        return {"row": int(i), "lon": float(data["lon"][i]), "lat": float(data["lat"][i]),
                "time": float(data["time"][i]), "ext": ext, "alt": a, "n_prof": int(k)}

    profs: list[dict] = []
    if explicit:
        want = [int(x) for x in explicit.split(",") if x.strip()]
        want = [w for w in want if 0 <= w < n_all]
        if not want:
            raise ValueError(f"--points {explicit} 全部越界(共 {n_all} 行)")
        for i in sorted(want):                            # 按行号顺序 → 同一天只建一次索引
            p = take(i)
            if p is None:
                print(f"  [WARN] 行 {i} 找不到对应 ACDL 廓线(时间窗/格点不匹配)")
            else:
                profs.append(p)
        return profs

    rng = np.random.default_rng(cfg["SEED"])
    days = list(by_day.keys())
    rng.shuffle(days)
    quota = max(1, -(-n_want // max(len(days), 1)))       # 每天最多抽几个,保证跨天分散
    for d in days:
        if len(profs) >= n_want:
            break
        rows = by_day[d]
        order = rng.permutation(len(rows))
        got = 0
        for k in order:
            if got >= quota or len(profs) >= n_want:
                break
            p = take(int(rows[k]))
            if p is not None:
                profs.append(p); got += 1
    return profs[:n_want]


# ============================================================
# 国界
# ============================================================
def _iter_lines(geom):
    """从 shapely 几何里迭代出折线坐标(兼容 Line/Polygon 及其 Multi 版本)。"""
    gt = getattr(geom, "geom_type", "")
    if gt == "LineString":
        yield np.asarray(geom.coords)
    elif gt in ("MultiLineString", "GeometryCollection"):
        for g in geom.geoms:
            yield from _iter_lines(g)
    elif gt == "Polygon":
        yield np.asarray(geom.exterior.coords)
        for r in geom.interiors:
            yield np.asarray(r.coords)
    elif gt == "MultiPolygon":
        for g in geom.geoms:
            yield from _iter_lines(g)


def load_mat_boundary(path: Path):
    """读 MATLAB v7.3 的国界文件(combined_shp,X/Y 为变长单元数组)→ [(xs, ys), ...]。"""
    with h5py.File(path, "r") as f:
        root = None
        if "combined_shp" in f and isinstance(f["combined_shp"], h5py.Group):
            root = f["combined_shp"]
        else:                                   # 退化:找第一个同时含 X/Y 的组
            for k in f:
                if isinstance(f[k], h5py.Group) and "X" in f[k] and "Y" in f[k]:
                    root = f[k]
                    break
        if root is None:
            raise ValueError("找不到 combined_shp(也没有同时含 X/Y 的组)")
        rxs = np.asarray(root["X"][()]).ravel()
        rys = np.asarray(root["Y"][()]).ravel()
        segs = []
        for rx, ry in zip(rxs, rys):
            x = np.asarray(f[rx][()], dtype=float).ravel()
            y = np.asarray(f[ry][()], dtype=float).ravel()
            if x.size >= 2 and x.size == y.size:
                segs.append((x, y))
    return segs


def load_shp_boundary(path: Path):
    """读 shapefile(pyshp)→ [(xs, ys), ...]。"""
    import shapefile  # pyshp
    segs = []
    for shp in shapefile.Reader(str(path)).shapes():
        pts = np.asarray(shp.points, dtype=float)
        if pts.shape[0] < 2:
            continue
        parts = list(shp.parts) + [len(pts)]
        for a, b in zip(parts[:-1], parts[1:]):
            if b - a >= 2:
                segs.append((pts[a:b, 0], pts[a:b, 1]))
    return segs


def load_geojson_boundary(path: Path):
    """读 GeoJSON → [(xs, ys), ...]。"""
    with open(path, encoding="utf-8") as fh:
        gj = json.load(fh)
    feats = gj.get("features", [gj])
    segs = []

    def walk(coords, depth):
        if depth == 0:                      # LineString = [[x,y],...]
            arr = np.asarray(coords, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 2:
                segs.append((arr[:, 0], arr[:, 1]))
            return
        for c in coords:                    # Multi*/Polygon 再往下一层
            walk(c, depth - 1)

    for ft in feats:
        geom = ft.get("geometry") or ft
        gtype = (geom.get("type") or "").lower()
        coords = geom.get("coordinates")
        if not coords:
            continue
        walk(coords, 0 if "linestring" in gtype else
             (1 if "multilinestring" in gtype or "polygon" in gtype else 2))
    return segs


def add_boundary(ax, cfg: dict) -> bool:
    """按优先级画海岸线/国界/省界;返回是否成功。

    --boundary 支持 .mat(MATLAB v7.3)/ .shp / .geojson;都没有则尝试 cartopy。
    """
    path = cfg.get("BOUNDARY")
    if path:
        p = Path(str(path))
        if not p.exists():
            print(f"[MAP] --boundary 文件不存在: {p}")
        else:
            try:
                suf = p.suffix.lower()
                if suf == ".mat":
                    segs = load_mat_boundary(p)
                elif suf == ".shp":
                    segs = load_shp_boundary(p)
                elif suf in (".geojson", ".json"):
                    segs = load_geojson_boundary(p)
                else:
                    raise ValueError(f"不支持的后缀 {suf}(用 .mat/.shp/.geojson)")
                for x, y in segs:
                    ax.plot(x, y, "-", color="k", lw=0.65, zorder=2)
                print(f"[MAP] 国界: {p.name}({len(segs)} 段)")
                return True
            except ImportError as exc:
                print(f"[MAP] 读 {p.suffix} 需要额外依赖({exc});试 cartopy ...")
            except Exception as exc:
                print(f"[MAP] {p.name} 读取失败: {exc}")

    try:
        from cartopy.io import shapereader
    except ImportError:
        print("[MAP] 无国界数据 → 只画散点与轨道。"
              "要国界:装 cartopy(pip install cartopy)或 --boundary xxx.shp")
        return False
    try:
        for cat, name, lw in (("physical", "coastline", 0.7),
                              ("cultural", "admin_0_boundary_lines_land", 0.7),
                              ("cultural", "admin_1_states_provinces_lines", 0.5)):
            shp = shapereader.natural_earth(resolution="50m", category=cat, name=name)
            for geom in shapereader.Reader(shp).geometries():
                for arr in _iter_lines(geom):
                    ax.plot(arr[:, 0], arr[:, 1], "-", color="k", lw=lw, zorder=2)
        print("[MAP] 国界: cartopy Natural Earth 50m(海岸线 + 国界 + 省界)")
        return True
    except Exception as exc:
        print(f"[MAP] cartopy 取数失败({exc});只画散点与轨道")
        return False


# ============================================================
# 出图
# ============================================================
def make_figure(data: dict, profs: list[dict], cfg: dict, out_path: Path, orbits=None):
    fig = plt.figure(figsize=(14, 6.4), dpi=cfg["DPI"])
    fig.suptitle("ERA5–ACDL Spatiotemporal Matching Results",
                 fontsize=15, fontweight="bold")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.17,
                          left=0.045, right=0.985, top=0.87, bottom=0.10)

    # ---------------- 左:地图 ----------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(*cfg["MAP_XLIM"]); ax1.set_ylim(*cfg["MAP_YLIM"])
    ax1.set_xlabel("Longitude (°E)", fontsize=11)
    ax1.set_ylabel("Latitude (°N)", fontsize=11)
    ax1.set_title("Spatial Distribution of ERA5–ACDL Matched Points",
                  fontsize=12, fontweight="bold", pad=8)
    ax1.set_xticks(np.arange(70, 141, 10)); ax1.set_yticks(np.arange(0, 58, 5))
    ax1.grid(alpha=0.3, ls=":", lw=0.6)

    if orbits is not None and orbits[0] is not None:
        olo, ola, oseg = orbits
        for a, b in zip(oseg[:-1], oseg[1:]):
            if b - a > 1:
                ax1.plot(olo[a:b], ola[a:b], "-", color="0.80", lw=3.0,
                         alpha=cfg["ORBIT_ALPHA"], zorder=1, solid_capstyle="round")

    add_boundary(ax1, cfg)

    # 匹配点:要比轨道带明显深,否则会"看不见"(浅灰点画在浅灰带上)
    ax1.scatter(data["lon"], data["lat"], s=7, c="0.32", alpha=0.55,
                linewidths=0, zorder=3)
    handles = [Line2D([], [], marker="o", ls="none", ms=6, mfc="0.32", mec="none",
                      label=f"All matched (N={len(data['lon'])})")]
    for j, p in enumerate(profs):
        c = CMAP10[j % len(CMAP10)]
        ax1.scatter(p["lon"], p["lat"], s=42, c=[c], edgecolors="k",
                    linewidths=0.9, zorder=6)
        ax1.annotate(f"P{j+1:02d}", (p["lon"], p["lat"]), xytext=(4, 4),
                     textcoords="offset points", fontsize=8.5,
                     fontweight="bold", color=c, zorder=7)
        handles.append(Line2D([], [], marker="o", ls="none", ms=6, mfc=c, mec="k",
                              label=f"P{j+1:02d} ({p['lon']:.2f}°E, {p['lat']:.2f}°N)"))
    ax1.legend(handles=handles, loc="lower left", fontsize=7.4, ncol=2,
               framealpha=0.92, borderpad=0.5, handletextpad=0.4)

    # ---------------- 右:廓线 ----------------
    ax2 = fig.add_subplot(gs[0, 1])
    for j, p in enumerate(profs):
        c = CMAP10[j % len(CMAP10)]
        y = smooth_profile(p["ext"], cfg["SMOOTH"])
        y = np.where(np.isfinite(y) & (y > cfg["E_MIN"]) & (y <= cfg["E_MAX"]), y, np.nan)
        lab = (f"P{j+1:02d} ({p['lon']:.1f}°E, {p['lat']:.1f}°N) · "
               f"{datenum_to_datestr(p['time'])} · n={p['n_prof']}")
        ax2.plot(y, p["alt"], "-", color=c, lw=0.9, alpha=0.9, label=lab)

    ax2.set_xscale("log")
    ax2.set_xlim(*cfg["X_LIM"])
    a0 = profs[0]["alt"]
    ax2.set_ylim(float(np.nanmin(a0)), float(np.nanmax(a0)))
    ax2.set_xlabel(r"Extinction Coefficient at 532 nm (km$^{-1}$)", fontsize=11)
    ax2.set_ylabel("Altitude (km)", fontsize=11)
    ax2.set_title("Extinction Coefficient Profiles of Selected Points",
                  fontsize=12, fontweight="bold", pad=8)
    ax2.grid(alpha=0.3, ls=":", lw=0.6)
    ax2.legend(loc="upper right", fontsize=7.0, framealpha=0.92)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=cfg["DPI"])
    try:
        fig.savefig(out_path.with_suffix(".pdf"))
    except Exception:
        pass
    plt.close(fig)
    print(f"[OK] 图: {out_path}")
    return True


def summary_table(data: dict, profs: list[dict], cfg: dict):
    print("\n" + "=" * 100)
    print(f"{'点':>4} {'行号':>8} {'经度':>8} {'纬度':>7} {'时次(UTC)':>18} "
          f"{'廓线数':>7} {'有效bin':>8} {'λ中位(km⁻¹)':>13}")
    print("-" * 100)
    for j, p in enumerate(profs):
        e = p["ext"]
        v = np.isfinite(e) & (e > cfg["E_MIN"]) & (e <= cfg["E_MAX"])
        med = float(np.nanmedian(e[v])) if v.any() else float("nan")
        print(f"{'P%02d' % (j+1):>4} {p['row']:8d} {p['lon']:8.2f} {p['lat']:7.2f} "
              f"{datenum_to_datestr(p['time']):>18} {p['n_prof']:7d} "
              f"{int(v.sum()):8d} {med:13.4f}")
    print("=" * 100)

    jp = Path(cfg["OUT"]).with_suffix(".json")
    jp.parent.mkdir(parents=True, exist_ok=True)
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump({"n_matched": int(len(data["lon"])),
                   "match_files": data["n_files"],
                   "dates": data["days"],
                   "config": {k: cfg[k] for k in ("GRID_RES", "TIME_TOL_MIN",
                                                  "N_BIN", "SMOOTH")},
                   "points": [{"id": f"P{j+1:02d}", "row": p["row"],
                               "lon": p["lon"], "lat": p["lat"],
                               "time": datenum_to_datestr(p["time"]),
                               "n_profiles": p["n_prof"]}
                              for j, p in enumerate(profs)]},
                  fh, ensure_ascii=False, indent=2)
    print(f"[OK] 选中点清单: {jp}")


# ============================================================
# 主流程
# ============================================================
def run(args) -> int:
    cfg = dict(CONFIG)
    for k in ("match_dir", "acdl_dir", "out", "seed", "n_points",
              "dates", "smooth", "boundary"):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k.upper()] = v
    if args.no_orbits:
        cfg["ORBITS"] = False
    dates = parse_dates(cfg["DATES"])

    match_dir = Path(cfg["MATCH_DIR"])
    acdl_dir = Path(cfg["ACDL_DIR"])
    print("=" * 78)
    print("匹配结果可视化")
    print(f"  匹配目录: {match_dir}")
    print(f"  ACDL 目录: {acdl_dir}")
    print(f"  日期过滤: {dates if dates else '全部'}")
    print(f"  口径: 节点 {cfg['GRID_RES']}° / 时间窗 ±{cfg['TIME_TOL_MIN']:.0f} min / "
          f"{cfg['N_BIN']} bin")
    print("=" * 78)

    if not acdl_dir.exists():
        print(f"[ABORT] ACDL 目录不存在: {acdl_dir}(画原始廓线必需)")
        return 2
    data = read_matched(match_dir, cfg["MATCH_GLOB"], cfg["STRUCT"], dates)
    if len(data["lon"]) == 0:
        print("[ABORT] 没有可用匹配点")
        return 2

    profs = select_profiles(data, cfg, acdl_dir, args.points)
    if not profs:
        print("[ABORT] 没有一条廓线能还原出来(检查 --acdl-dir 与时间口径)")
        return 2
    print(f"[POINT] 选中 {len(profs)} 个点;全局行号 "
          f"{[p['row'] for p in profs]}")

    orbits = None
    if cfg["ORBITS"]:
        orbits = read_acdl_lonlat(acdl_dir, cfg["ACDL_PREFIX"], data["days"])

    make_figure(data, profs, cfg, Path(cfg["OUT"]), orbits=orbits)
    summary_table(data, profs, cfg)
    print("[DONE]")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="匹配结果可视化(地图 + 原始 1291 bin 廓线)")
    ap.add_argument("--match-dir", default=None, help="匹配产物目录")
    ap.add_argument("--acdl-dir", default=None, help="ACDL 原始 .mat 目录")
    ap.add_argument("--out", default=None, help="输出图路径(.png)")
    ap.add_argument("--dates", default=None,
                    help='日期过滤,如 "20220601-20220603" 或 "20220601,20220602"')
    ap.add_argument("--points", default=None, help="指定全局行号(逗号分隔);不给则随机抽")
    ap.add_argument("--n-points", type=int, default=None, help="随机抽取的点数(默认 8)")
    ap.add_argument("--seed", type=int, default=None, help="随机种子(默认 40)")
    ap.add_argument("--smooth", type=int, default=None,
                    help="廓线滑动中值窗宽(默认 0=原始;太毛糙可试 5~15)")
    ap.add_argument("--boundary", default=None, help="国界 shapefile 路径")
    ap.add_argument("--no-orbits", action="store_true", help="不画 ACDL 轨道条带")
    return ap


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
