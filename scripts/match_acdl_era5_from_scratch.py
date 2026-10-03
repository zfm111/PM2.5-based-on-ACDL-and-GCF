"""
match_acdl_era5_from_scratch.py
============================================================
按《ACDL_ERA5_FromScratch_Matching_Strategy.md》实现"从头匹配"(口径以该文档为准):

  ACDL 原始廓线
    → 水平:归属最近 0.25° ERA5 节点 + 最近整点(±30min),同(节点,时次)廓线先做元素级 nanmean
           **且要求廓线到该节点的球面距离 ≤ `MAX_NODE_DIST_DEG`(默认 0.18°)**;
           这一步是必须的:ACDL 是全球观测,而 ERA5 只覆盖 73-135°E/3-54°N,
           没有距离上限时域外廓线会被吸附到最近**边界节点**上(实测 94.7% 的廓线、
           82% 的匹配行都是这种错配,边界节点还会把半个地球的廓线平均到一起)
    → 垂直:按 ERA5 pressure-level 各层 `H=z/g0` 为界,把 24m bin 聚合成 32 段
           (段0 = 地表→最低层;段1..31 = 相邻层之间)
    → 每(节点,时次)输出:列序 = 标识 → H/T/RH(训练特征) → Ext_mean(训练目标) → 统计/QC

关键约定(见计划书):
  - 位势来自 pressure-level `z`(不用 single-level,不做测高积分);`H = z/g0`
  - 层序:`Level` hPa 降序 1000→10,索引 k=0 为最低层(近地面)
  - 地表下界 = 该廓线 ACDL `DEM_Surface_Elevation`;缺失时兜底 = 最低层高度 H0
  - QC(CAD + 云类型 `Cloud_Subtype_Multi` + 云光学厚度 `Column_Optical_Depth_Cloud_532`):
    `QC_CASE=1` 严格 / `2` 宽松 / `3` 折中(云底缓冲带)/
    `4` 自适应(默认):厚云下方丢弃、薄云下方只丢缓冲带 /
    `5` 极简:只要消光在 [EXT_MIN, EXT_MAX] 且在地表以上就保留(不做云/CAD 过滤);
    一次只产出一种,`--case 1|2|3|4|5`
    相态编码:0=N/A, 1=unknown, 2=ice, 3=water, 4=oriented ice
    默认:水云(3)=厚;冰云(2)/定向冰晶(4)=薄;未知(1)按厚(保守);另受厚度/COD 规则加严
  - **ERA5 单层场(BLH/TCWV/地表位势→Z_sfc)直接并入输出**(0.25°,与气压层同网格同时次;
    `CONFIG["SUPER_LEVELS"]` 可改/置空;`--no-super-levels` 关闭后回到 132 列)
  - 不输出 ERA5-Land 近地面五项
  - 输出 MATLAB v7.3(struct `MatchV2`),列名以 VarNames 为准

依赖:numpy, h5py, scipy, hdf5storage
运行:python match_acdl_era5_from_scratch.py [--start YYYYMMDD] [--end YYYYMMDD] [--dry-run] [--selftest]
============================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import h5py
import hdf5storage
import scipy.io as sio
from scipy.spatial import cKDTree


# ============================================================
# 配置
# ============================================================
CONFIG = {
    # ACDL 原始廓线(.mat,内含 Output 结构体)
    "ACDL_DIR":    r"D:/ACDL/Data/ACDL/ProfileMat",
    "ACDL_PREFIX": "ACDL11_",           # 文件名 ACDL11_YYYYMMDD*.mat

    # 新 ERA5 转换产物(v7.3,ECMWF 结构体)
    "ERA5_MAT_ROOT": r"D:/era5_mat",
    "ERA5_RES":      "0.25deg",

    "OUT_DIR":   r"D:/matchdata",
    "OUT_STRUCT": "MatchV2",

    # ERA5 单层场(0.25°,与气压层同网格同时次):直接并入匹配输出(替代后处理补列)
    #   blh→BLH_m(m)、tcwv→TCWV_kgm2(kg/m²)、zsfc→Z_sfc_m(m,地表位势/g0 = 地形高度)
    #   置 () 可关闭,得到与旧版一致的 132 列。
    "ERA5_SINGLE_ROOT": "",                 # 空 = <ERA5_MAT_ROOT>/era5_single_levels
    "SUPER_LEVELS": ("blh", "tcwv", "zsfc"),

    # 输出列:False=lite(仅训练用 132 列:标识4 + H/T/RH 96 + Ext_mean 32)
    #        True =full(旧版 264 列,额外含 Ext_valid/Cloud_bins/... 等测试统计列)
    "OUTPUT_STATS": False,
    # 日级紧凑缓存(加速重跑;按源文件 size+mtime 校验失效)
    "USE_CACHE": True,
    "CACHE_DIR": "",                    # 空 = OUT_DIR/_cache

    # 日期范围(含首尾)
    "START": "20220601",
    "END":   "20220630",

    # 匹配参数
    "GRID_RES":     0.25,               # ERA5 网格(度)
    "TIME_TOL_MIN": 30.0,               # 时间窗口(分钟)
    # 空间距离上限(度):ACDL 是全球观测,而 ERA5 只覆盖 73-135°E/3-54°N。
    # 不加这道限制时,tree.query 会把域外几千公里外的廓线吸附到最近的**边界节点**上
    # (实测 94.7% 的廓线、82% 的匹配行都是这种错配)。真正属于某节点的廓线到该节点的
    # 距离必然 ≤ 格子半对角线 = 0.25/2×√2 ≈ 0.1768°,所以 0.18 刚好只放行域内。
    # 置 None / 设成很大 → 退回旧行为(不推荐,仅用于复现历史产物)。
    "MAX_NODE_DIST_DEG": 0.18,
    "G0":           9.80665,

    # ACDL 字段名(如与实际产品不同,改这里)
    "F_EXT":  "Extinction_Coefficient_532",
    "F_LON":  "Longitude",
    "F_LAT":  "Latitude",
    "F_TIME": "Profile_UTC_Time",       # TAI93
    "F_ALT":  "Altitude",               # km, 1291 层
    "F_DEM":  "DEM_Surface_Elevation",  # 逐廓线地表海拔(m);没有则兜底
    "F_CAD":   "CAD_Score",             # 逐 bin 云-气溶胶分数
    "F_PHASE": "Cloud_Subtype_Multi",   # 逐 bin 云(子)类型,>0 视为云
    "F_COD":   "Column_Optical_Depth_Cloud_532",   # 廓线级云光学厚度(判厚薄用;可缺)

    # --- QC ---
    "EXT_MIN": 0.0,                     # 消光有效下限
    "EXT_MAX": 1.25,                    # 消光有效上限(产品文档)
    "CAD_AER_THR":  -20.0,              # CAD<=此值 → 气溶胶候选
    "CAD_CLOUD_THR": 20.0,              # CAD>=此值 → 云候选
    "CAD_SPECIAL": (101, 102, 103, 104, 105, 106),   # 特殊编码 → 无效
    "COV_MIN": 0.0,                     # 廓线级最小有效覆盖率(0=不丢弃任何廓线)

    # QC Case(一次只产出一种):
    #   1 = 严格:最低云 bin 以下全部不纳入
    #   2 = 宽松:云下 CAD 合格仍纳入
    #   3 = 折中:仅排除最低云 bin 下方 CLOUD_BUFFER_M 米内的云缘污染带
    #   4 = 自适应(默认):"厚云"下方全部丢弃;"薄云"仅丢云底下方缓冲带
    #   5 = 极简:只要消光在 [EXT_MIN, EXT_MAX] 且在地表以上就保留(不做任何云/CAD 过滤)
    "QC_CASE": 4,
    "CLOUD_BUFFER_M": 1000.0,           # 云底下方缓冲带厚度(米)
    # —— Cloud_Subtype_Multi 编码(0=N/A,1=unknown,2=ice,3=water,4=oriented ice)——
    "CLOUD_THICK_CODES": (3,),          # 水云 → 厚(其下丢弃)
    "CLOUD_UNKNOWN_CODES": (1,),        # 未知类型云编码
    "CLOUD_UNKNOWN_AS_THICK": True,     # 未知类型是否按厚处理(保守;置 False 则按薄)
    # 冰云(2)/定向冰晶(4)默认视为薄(保留下方);但以下任一条命中仍判厚:
    "CLOUD_THICK_M": 1000.0,            # 连通云块厚度 ≥ 此值(米) → 厚云
    "CLOUD_THICK_COD": 0.3,             # 廓线云光学厚度 ≥ 此值 → 该廓线云全部记厚

    "TPI_TO_DATENUM_SCALE": 1.0e-4,     # 复刻旧匹配器: datenum = tai93*1e-4/86400 + base
    "DATENUM_BASE": 730486.5,           # datenum('2000-01-01 12:00:00')

    "N_ALT": 1291,

    # ---- 垂直坐标(--vertical)----
    #   asl = 历史口径:段边界 = 固定气压层位势高度 [min(surf,H0), H0..H31](高原近地面无层可用)
    #   agl = 地形跟随:段边界 = dem_m + AGL_EDGES_KM,每样本"离地高度"分段 —— 高原/盆地近地面
    #         获得 L00;T/RH 段特征由"地表以上气压层"剖面插值,低于最低有效层的薄段用最低两层
    #         局地递减率外推 + 限幅(AGL_EXTRAP_*)。agl 模式同时修复 ACDL DEM 字段 km→m 单位 bug。
    "VERTICAL": "asl",
    # AGL 边界(离地高度,km),33 个边界 = 32 段;低层加密、高层渐疏,顶 = 30 km(ACDL 最高 bin)。
    # 顶部压缩:dem_m > (30km − AGL_TOP) 时,高于 AGL_TOP_SPLIT 的边界线性压入 [TOP_SPLIT, 30km−dem]。
    "AGL_EDGES_KM": (0.0, 0.05, 0.15, 0.35, 0.60, 0.90, 1.25, 1.65, 2.10, 2.60, 3.15,
                     3.75, 4.40, 5.10, 5.85, 6.65, 7.50, 8.40, 9.35, 10.35, 11.40, 12.50,
                     13.65, 14.85, 16.10, 17.40, 18.75, 20.15, 21.60, 23.10, 24.65,
                     27.90, 30.00),
    "AGL_TOP_SPLIT_KM": 12.0,           # 低于此 AGL 的边界永不压缩(保住近地面分辨率)
    "ALT_MAX_M": 30000.0,               # ACDL Altitude 上限(km→m 后),AGL 顶部硬帽
    "AGL_EXTRAP_MAX_DT_M": 0.010,       # 外推温度递减率限幅(K/m,即 10 K/km)
    "AGL_EXTRAP_MAX_DRH_M": 0.02,       # 外推 RH 变化率限幅(%/m,即 2 %/100m)
    "AGL_T_MIN_K": 175.0,               # 外推温度下限(防失控)
}


# ============================================================
# 小工具
# ============================================================
# 叶名 → HDF5 内路径 的缓存(首个 v7.3 文件扫描一次,后续按路径直读)
_FIELD_PATHS: dict[str, str | None] = {}


def _field_names() -> list[str]:
    """当前需要从 ACDL 文件读取的 9 个字段名。"""
    return [CONFIG["F_EXT"], CONFIG["F_LON"], CONFIG["F_LAT"], CONFIG["F_TIME"], CONFIG["F_ALT"],
            CONFIG["F_DEM"], CONFIG["F_CAD"], CONFIG["F_PHASE"], CONFIG["F_COD"]]


def _scan_field_paths(f: h5py.File, names: set[str]) -> dict[str, str]:
    """单次遍历 HDF5,返回 {叶名: 完整路径}(仅收 names 里的)。"""
    found: dict[str, str] = {}

    def visit(n, o):
        leaf = n.rsplit("/", 1)[-1]
        if leaf in names and leaf not in found and isinstance(o, h5py.Dataset):
            found[leaf] = n

    f.visititems(visit)
    return found


def _read_acdl_file(path: Path, names: list[str]) -> dict[str, np.ndarray | None]:
    """一次打开 .mat 取回 names 各字段(缺失为 None);兼容 v7.3 与 v5。"""
    out: dict[str, np.ndarray | None] = {n: None for n in names}
    try:
        f = h5py.File(path, "r")
    except OSError:
        f = None
    if f is not None:                                   # ---- v7.3 ----
        with f:
            unknown = [n for n in names if n not in _FIELD_PATHS]
            if unknown:                                 # 首个文件:一次全树扫描
                for leaf, p in _scan_field_paths(f, set(unknown)).items():
                    _FIELD_PATHS[leaf] = p
                for n in unknown:
                    _FIELD_PATHS.setdefault(n, None)
            for n in names:
                p = _FIELD_PATHS.get(n)
                if not p:
                    continue
                try:
                    out[n] = np.asarray(f[p][()])
                except (KeyError, OSError):              # 结构偶有差异 → 对该字段重扫
                    _FIELD_PATHS.pop(n, None)
                    hit = _scan_field_paths(f, {n})
                    if n in hit:
                        _FIELD_PATHS[n] = hit[n]
                        out[n] = np.asarray(f[hit[n]][()])
        return out
    # ---- v5 ----
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    obj = m.get("Output", m)
    for n in names:
        v = obj.get(n) if isinstance(obj, dict) else getattr(obj, n, None)
        out[n] = None if v is None else np.asarray(v)
    return out


def tpi_to_datenum(tpi: np.ndarray) -> np.ndarray:
    """TAI93 → MATLAB datenum(复刻旧匹配器换算,保持与历史样本可比)。"""
    return np.asarray(tpi, dtype=np.float64) * CONFIG["TPI_TO_DATENUM_SCALE"] / 86400.0 \
        + CONFIG["DATENUM_BASE"]


def chord_to_deg(d):
    return np.degrees(2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0)))


# ============================================================
# ACDL 读取
# ============================================================
def load_acdl_day(acdl_dir: Path, yyyymmdd: str):
    """读取某日全部 ACDL 廓线。

    返回 dict(lon, lat, datenum, ext[N,1291], cad[N,1291]|None, dem[N]|None, alt[1291])。
    """
    files = sorted(acdl_dir.glob(f"{CONFIG['ACDL_PREFIX']}{yyyymmdd}*.mat"))
    if not files:
        return None
    lons, lats, tdis, exts, dems, cads, phases, cods = [], [], [], [], [], [], [], []
    alt = None
    n_cad_missing = 0
    n_phase_missing = 0
    n_cod_missing = 0
    names = _field_names()
    for fp in files:
        raw = _read_acdl_file(fp, names)               # 一次打开;v7.3 走路径缓存
        ext = raw[CONFIG["F_EXT"]]
        if ext is None:
            print(f"    [WARN] {fp.name}: 缺 {CONFIG['F_EXT']},跳过")
            continue
        lon = raw[CONFIG["F_LON"]]
        lat = raw[CONFIG["F_LAT"]]
        tpi = raw[CONFIG["F_TIME"]]
        altv = raw[CONFIG["F_ALT"]]
        dem = raw[CONFIG["F_DEM"]]
        cad = raw[CONFIG["F_CAD"]]
        phase = raw[CONFIG["F_PHASE"]]
        cod = raw[CONFIG["F_COD"]]
        if any(v is None for v in (lon, lat, tpi, altv)):
            print(f"    [WARN] {fp.name}: 关键字段缺失,跳过")
            continue
        ext = np.asarray(ext, dtype=np.float32)
        if ext.ndim != 2:
            print(f"    [WARN] {fp.name}: 消光维度异常 {ext.shape},跳过")
            continue
        # 产品里通常存成 [1291, N](高度 × 廓线)→ 转成 (N, 1291)
        if ext.shape[0] == CONFIG["N_ALT"] and ext.shape[1] != CONFIG["N_ALT"]:
            ext = ext.T
        if ext.shape[1] != CONFIG["N_ALT"]:
            print(f"    [WARN] {fp.name}: 消光第二维 {ext.shape[1]} != {CONFIG['N_ALT']},跳过")
            continue
        n = ext.shape[0]
        alt = np.asarray(altv, dtype=np.float64).ravel()
        if alt.size != CONFIG["N_ALT"]:
            print(f"    [WARN] {fp.name}: Altitude 长度 {alt.size} != {CONFIG['N_ALT']},跳过")
            continue
        if cad is None:
            n_cad_missing += 1
            cad = np.full(ext.shape, np.nan, dtype=np.float32)
        else:
            cad = np.asarray(cad, dtype=np.float32)
            if cad.shape[0] == CONFIG["N_ALT"] and cad.shape[1] != CONFIG["N_ALT"]:
                cad = cad.T
            if cad.shape != ext.shape:
                print(f"    [WARN] {fp.name}: CAD 形状 {cad.shape} 与消光 {ext.shape} 不符,按缺失处理")
                cad = np.full(ext.shape, np.nan, dtype=np.float32)
                n_cad_missing += 1
        if phase is None:
            n_phase_missing += 1
            phase = None
        else:
            phase = np.asarray(phase, dtype=np.float32)
            if phase.shape[0] == CONFIG["N_ALT"] and phase.shape[1] != CONFIG["N_ALT"]:
                phase = phase.T
            if phase.shape != ext.shape:
                print(f"    [WARN] {fp.name}: 云相态形状 {phase.shape} 与消光 {ext.shape} 不符,忽略")
                phase = None
                n_phase_missing += 1
        if cod is None:
            n_cod_missing += 1
            cod = np.full(n, np.nan, dtype=np.float32)
        else:
            cod = np.asarray(cod, dtype=np.float32).ravel()
            if cod.size != n:
                print(f"    [WARN] {fp.name}: 云光学厚度长度 {cod.size} != {n},按缺失处理")
                cod = np.full(n, np.nan, dtype=np.float32)
                n_cod_missing += 1
        lons.append(np.asarray(lon, dtype=np.float64).ravel())
        lats.append(np.asarray(lat, dtype=np.float64).ravel())
        tdis.append(tpi_to_datenum(np.asarray(tpi, dtype=np.float64).ravel()))
        exts.append(ext)
        cads.append(cad)
        phases.append(phase)
        cods.append(cod)
        dems.append(np.asarray(dem, dtype=np.float64).ravel() if dem is not None
                    else np.full(n, np.nan))
    if not exts:
        return None
    if n_cad_missing:
        print(f"    [WARN] {n_cad_missing} 个文件缺 {CONFIG['F_CAD']},这些廓线 CAD QC 失效"
              f"(退化为只用消光范围)")
    if n_phase_missing:
        print(f"    [WARN] {n_phase_missing} 个文件缺 {CONFIG['F_PHASE']},这些廓线不加云类型判定")
    if n_cod_missing:
        print(f"    [WARN] {n_cod_missing} 个文件缺 {CONFIG['F_COD']},厚云判定退化为只用云块厚度")
    phase_out = None
    if any(p is not None for p in phases):
        phase_out = np.vstack([p if p is not None else np.full_like(exts[i], np.nan)
                               for i, p in enumerate(phases)])
    return {
        "lon": np.concatenate(lons),
        "lat": np.concatenate(lats),
        "datenum": np.concatenate(tdis),
        "ext": np.vstack(exts),                 # (N, 1291)
        "cad": np.vstack(cads),                 # (N, 1291)
        "phase": phase_out,                     # (N, 1291) 或 None
        "cod": np.concatenate(cods),            # (N,) 廓线级云光学厚度
        "dem": np.concatenate(dems),            # (N,)
        "alt": alt,                             # (1291,) km
    }


# ============================================================
# 日级紧凑缓存(加速重跑;按源文件 size+mtime 校验失效)
# ============================================================
_CACHE_KEYS = ("lon", "lat", "datenum", "ext", "cad", "phase", "cod", "dem", "alt")


def _cache_dir() -> Path:
    d = CONFIG.get("CACHE_DIR") or (Path(CONFIG["OUT_DIR"]) / "_cache")
    Path(d).mkdir(parents=True, exist_ok=True)
    return Path(d)


def _source_signature(files: list[Path]) -> np.ndarray:
    """源文件指纹:name/size/mtime_ns,拼成 (M,3) 数组(存进缓存用于失效判断)。"""
    rows = []
    for f in files:
        st = f.stat()
        rows.append([len(f.name), int(st.st_size), int(st.st_mtime_ns)])
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3) if rows else np.zeros((0, 3), np.int64)


def _signature_matches(cached: np.ndarray, files: list[Path]) -> bool:
    now = _source_signature(files)
    if cached.shape != now.shape and not (cached.shape[0] == 0 and now.shape[0] == 0):
        return False
    if cached.shape[0] != now.shape[0]:
        return False
    # name 只比较长度以省空间(配合 size+mtime 足够)
    return bool(np.array_equal(cached[:, 1:], now[:, 1:]))


def load_acdl_day_cached(acdl_dir: Path, yyyymmdd: str, use_cache: bool = True):
    """带日级缓存的 ACDL 装载:命中则直接读 .npz,否则解析 .mat 后写缓存。"""
    if not use_cache:
        return load_acdl_day(acdl_dir, yyyymmdd)
    files = sorted(acdl_dir.glob(f"{CONFIG['ACDL_PREFIX']}{yyyymmdd}*.mat"))
    if not files:
        return None
    cache_file = _cache_dir() / f"ACDL_{yyyymmdd}.npz"
    if cache_file.exists():
        try:
            with np.load(cache_file, allow_pickle=False) as z:
                if _signature_matches(z["sig"], files):
                    data = {k: (None if (k == "phase" and int(z["has_phase"]) == 0) else z[k])
                            for k in _CACHE_KEYS}
                    print(f"    [CACHE] 命中 {cache_file.name}"
                          f"(廓线 {data['ext'].shape[0]})")
                    return data
                print(f"    [CACHE] {cache_file.name} 已失效(源文件变化),重建")
        except Exception as exc:
            print(f"    [WARN] 缓存读取失败({exc}),重建")
    data = load_acdl_day(acdl_dir, yyyymmdd)
    if data is None:
        return None
    try:
        payload = {k: (np.zeros((0,), np.float32) if (k == "phase" and data[k] is None) else data[k])
                   for k in _CACHE_KEYS}
        payload["has_phase"] = np.int8(0 if data.get("phase") is None else 1)
        payload["sig"] = _source_signature(files)
        with open(cache_file, "wb") as fh:
            np.savez(fh, **payload)
        print(f"    [CACHE] 写入 {cache_file.name}")
    except Exception as exc:
        print(f"    [WARN] 缓存写入失败({exc}),忽略")
    return data


# ============================================================
# ERA5 单层月度读取(0.25°,直接并入匹配输出)
# ============================================================
# 短名 → (数据文件前缀, 输出列名, 是否需除以 g0)
SUPER_LEVEL_DEFS = {
    "blh":  ("boundary_layer_height",      "BLH_m",     False),
    "tcwv": ("total_column_water_vapour",  "TCWV_kgm2", False),
    "zsfc": ("geopotential",               "Z_sfc_m",   True),
}


class Era5SingleLevel:
    """按月惰性打开单层场;按 (r, c, ti) 取值。Data 磁盘布局 (time, lon, lat)。

    用法:prefetch(所需时次列表) 一次读入这些时次,随后 value(var, ti, r, c) 走内存。
    """

    def __init__(self, yyyymm: str, vars_: tuple[str, ...]):
        root = CONFIG["ERA5_SINGLE_ROOT"] or os.path.join(CONFIG["ERA5_MAT_ROOT"], "era5_single_levels")
        base = Path(root) / CONFIG["ERA5_RES"] / yyyymm
        self.vars = tuple(v for v in vars_ if v in SUPER_LEVEL_DEFS)
        self.h5, self.ds = {}, {}
        self.time = None
        self.nlat = self.nlon = None
        self._node_of: dict[tuple[float, float], tuple[int, int]] = {}
        self._cache: dict[str, tuple[list[int], np.ndarray]] = {}
        self._pos: dict[int, int] = {}
        for v in self.vars:
            prefix = SUPER_LEVEL_DEFS[v][0]
            p = base / f"{prefix}_{yyyymm}.mat"
            if not p.exists():
                print(f"    [WARN] 缺少单层文件,跳过该列: {p}")
                continue
            f = h5py.File(p, "r")
            self.h5[v] = f
            self.ds[v] = f["ECMWF/Data"]
            if self.time is None:
                self.time = np.asarray(f["ECMWF/Time"][()], dtype=np.float64).ravel()
                lat_d = np.asarray(f["ECMWF/Lat"][()], dtype=np.float64)
                lon_d = np.asarray(f["ECMWF/Lon"][()], dtype=np.float64)
                shp = self.ds[v].shape                          # (T, nlon, nlat)
                self.nlon, self.nlat = shp[1], shp[2]
                lat2d = lat_d.T if lat_d.shape == (self.nlon, self.nlat) else lat_d
                lon2d = lon_d.T if lon_d.shape == (self.nlon, self.nlat) else lon_d
                for r in range(self.nlat):
                    for c in range(self.nlon):
                        self._node_of[(round(float(lon2d[r, c]), 4),
                                       round(float(lat2d[r, c]), 4))] = (r, c)
        self.vars = tuple(v for v in self.vars if v in self.h5)
        print(f"    [SINGLE] 单层场 {list(self.vars)};网格 {self.nlat}×{self.nlon};"
              f"时次 {0 if self.time is None else len(self.time)}")

    def node_index(self, lon: float, lat: float):
        return self._node_of.get((round(float(lon), 4), round(float(lat), 4)))

    def time_index(self, datenum: float) -> tuple[int, float]:
        i = int(np.argmin(np.abs(self.time - datenum)))
        return i, (self.time[i] - datenum) * 1440.0

    def prefetch(self, ti_list: list[int]) -> None:
        """把所需时次一次性读入内存(每天调用一次)。"""
        self._cache = {}
        ti_sorted = sorted(set(int(t) for t in ti_list))
        if not ti_sorted:
            return
        pos = {t: k for k, t in enumerate(ti_sorted)}
        for v, ds in self.ds.items():
            arr = np.asarray(ds[ti_sorted, :, :], dtype=np.float64)   # (n_ti, lon, lat)
            self._cache[v] = (ti_sorted, arr)
        self._pos = pos

    def value(self, var: str, ti: int, r: int, c: int) -> float:
        """取 (var, ti, 纬度索引 r, 经度索引 c) 的值;未缓存时直接读。"""
        if var not in self.ds:
            return np.nan
        val = None
        cache = self._cache.get(var)
        if cache is not None and ti in self._pos:
            arr = cache[1][self._pos[ti]]              # (lon, lat)
            val = arr[c, r]
        else:
            val = self.ds[var][ti, c, r]
        val = float(val)
        if SUPER_LEVEL_DEFS[var][2]:
            val = val / CONFIG["G0"]
        return val

    def close(self):
        for f in self.h5.values():
            f.close()
        self.h5, self.ds = {}, {}


# ============================================================
# ERA5 月度读取(v7.3)
# ============================================================
class Era5Month:
    """按月惰性打开 t/r/z,只读需要的 (时次, 节点列) 切片。"""

    def __init__(self, yyyymm: str):
        base = Path(CONFIG["ERA5_MAT_ROOT"]) / "era5_pressure_levels" / CONFIG["ERA5_RES"] / yyyymm
        self.paths = {
            "t": base / f"temperature_{yyyymm}.mat",
            "r": base / f"relative_humidity_{yyyymm}.mat",
            "z": base / f"geopotential_{yyyymm}.mat",
        }
        missing = [str(p) for p in self.paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError("缺少 ERA5 气压层 .mat:" + "; ".join(missing))
        self.h5 = {k: h5py.File(p, "r") for k, p in self.paths.items()}
        self.data = {k: f["ECMWF/Data"] for k, f in self.h5.items()}
        t = self.h5["t"]
        self.time = np.asarray(t["ECMWF/Time"][()], dtype=np.float64).ravel()
        self.level = np.asarray(t["ECMWF/Level"][()], dtype=np.float64).ravel()   # hPa 降序
        shape = self.data["t"].shape                                              # (T, L, C, R)
        assert len(shape) == 4 and shape[1] == len(self.level), f"Data 维度异常 {shape}"
        # 磁盘上 Lat/Lon 形状为 (C, R),逻辑为 (R, C)
        lat_disk = np.asarray(t["ECMWF/Lat"][()], dtype=np.float64)
        lon_disk = np.asarray(t["ECMWF/Lon"][()], dtype=np.float64)
        self.lat2d = lat_disk.T if lat_disk.shape == (shape[2], shape[3]) else lat_disk
        self.lon2d = lon_disk.T if lon_disk.shape == (shape[2], shape[3]) else lon_disk
        assert self.lat2d.shape == (shape[3], shape[2]), \
            f"Lat 形状 {self.lat2d.shape} 与 Data {shape} 不符"
        la = np.radians(self.lat2d.ravel()); lo = np.radians(self.lon2d.ravel())
        self.tree = cKDTree(np.column_stack([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)]))

    def time_index(self, datenum: float) -> tuple[int, float]:
        i = int(np.argmin(np.abs(self.time - datenum)))
        return i, (self.time[i] - datenum) * 1440.0          # 偏移(分钟)

    def time_index_vec(self, datenums: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """向量化最近时次:返回 (索引, 偏移分钟)。依赖 Time 已升序(O(n log m))。"""
        t = np.asarray(datenums, dtype=np.float64)
        i = np.clip(np.searchsorted(self.time, t), 0, len(self.time) - 1)
        left = np.clip(i - 1, 0, len(self.time) - 1)
        dl = np.abs(self.time[left] - t)
        dr = np.abs(self.time[i] - t)
        idx = np.where(dl <= dr, left, i)
        return idx, (self.time[idx] - t) * 1440.0

    def column(self, key: str, t_idx: int, node: int, ncol: int) -> np.ndarray:
        r, c = divmod(int(node), ncol)
        return np.asarray(self.data[key][t_idx, :, c, r], dtype=np.float64)   # (L,)

    def close(self):
        for f in self.h5.values():
            f.close()


# ============================================================
# 垂直聚合:ACDL 24m bin → 32 段
# ============================================================
def level_heights(z_m2s2: np.ndarray) -> np.ndarray:
    """pressseure-level 位势 → 位势高度 H(m),并强制单调递增(索引0=最低层)。"""
    H = np.asarray(z_m2s2, dtype=np.float64) / CONFIG["G0"]
    return np.maximum.accumulate(H)          # 防止偶发逆序


def seg_bounds(surf_h: float, H: np.ndarray) -> np.ndarray:
    """段边界 = [min(surf_h, H0), H0, H1, ..., H_{L-1}] → L 段(段0=底部, L=层数)。"""
    return np.concatenate([[min(surf_h, H[0])], H])


# ---- AGL(离地高度)分段:--vertical agl 时使用 ----
def agl_edges_m(dem_m: float) -> np.ndarray:
    """由地表海拔把 AGL 边界表(km)换算成实际 ASL 边界(m),33 个 → 32 段。

    顶部压缩:dem 较高时,AGL_TOP_SPLIT 以上的边界线性压入 [TOP_SPLIT, ALT_MAX−dem],
    使最高段顶 ≤ ACDL 高度上限;TOP_SPLIT 以下(近地面)边界永不压缩。
    返回严格递增的边界(m),edges[0] = dem_m(段 0 = 地表起)。
    """
    e = np.asarray(CONFIG["AGL_EDGES_KM"], dtype=np.float64) * 1000.0
    assert e.size == 33 and np.all(np.diff(e) > 0), "AGL_EDGES_KM 必须是 33 个严格递增边界(32 段)"
    top_room = CONFIG["ALT_MAX_M"] - dem_m                     # 地表以上还有多少空间
    split = CONFIG["AGL_TOP_SPLIT_KM"] * 1000.0
    out = e.copy()
    if top_room < e[-1]:
        f = max(top_room - split, 100.0) / max(e[-1] - split, 100.0)
        hi = e > split
        out[hi] = split + (e[hi] - split) * f
    out[0] = 0.0
    edges = dem_m + np.maximum.accumulate(out)                 # 强制单调(dem+边界)
    edges[0] = dem_m
    return edges


def profile_at_height(H_lev: np.ndarray, T_lev: np.ndarray, RH_lev: np.ndarray,
                      h_target: float) -> tuple[float, float]:
    """在高度 h_target(m) 处取 (T, RH):地表以上气压层剖面插值;低于最低有效层时
    用最低两层局地递减率外推 + 限幅(《优化路线讨论.md》AGL 设计)。

    H_lev/T_lev/RH_lev: 仅含 z > dem 的有效层(调用方过滤)。返回有限值。
    """
    ok = np.isfinite(H_lev) & np.isfinite(T_lev) & np.isfinite(RH_lev)
    z, t, r = H_lev[ok], T_lev[ok], RH_lev[ok]
    if z.size == 0:                                            # 极端:全无效 → 中性回退
        return 273.15, 50.0
    if z.size == 1 or h_target >= z[0]:
        h_t = float(np.interp(h_target, z, t))
        h_r = float(np.interp(h_target, z, r))
    else:
        # 递减率外推(用最低两个有效层),限幅防失控
        dz = z[1] - z[0]
        dt = (t[1] - t[0]) / dz
        dr = (r[1] - r[0]) / dz
        dt = np.clip(dt, -CONFIG["AGL_EXTRAP_MAX_DT_M"], CONFIG["AGL_EXTRAP_MAX_DT_M"])
        dr = np.clip(dr, -CONFIG["AGL_EXTRAP_MAX_DRH_M"], CONFIG["AGL_EXTRAP_MAX_DRH_M"])
        h_t = float(t[0] + dt * (h_target - z[0]))
        h_r = float(r[0] + dr * (h_target - z[0]))
    return (max(h_t, CONFIG["AGL_T_MIN_K"]), float(np.clip(h_r, 0.0, 100.0)))


def segment_index(alt_m: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """每个 bin 所属段号(L00..);越界 = -1。一次 searchsorted。"""
    sid = (np.searchsorted(bounds, alt_m, side="right") - 1).astype(np.int16)
    sid[(alt_m < bounds[0]) | (alt_m >= bounds[-1])] = -1
    return sid


def aggregate_segments(mean_bin: np.ndarray, sid: np.ndarray, n_seg: int):
    """按段号聚合平均廓线 → (mean, valid_count, total_count);单遍 bincount,替代 32 段循环。"""
    ok = sid >= 0
    tcnt = np.bincount(sid[ok].astype(np.int64), minlength=n_seg).astype(np.int64)
    sel = ok & np.isfinite(mean_bin)
    s = sid[sel].astype(np.int64)
    vcnt = np.bincount(s, minlength=n_seg).astype(np.int64)
    tot = np.bincount(s, weights=np.asarray(mean_bin[sel], dtype=np.float64), minlength=n_seg)
    mean = np.full(n_seg, np.nan)
    nz = vcnt > 0
    mean[nz] = tot[nz] / vcnt[nz]
    return mean, vcnt, tcnt


def count_segments(mask2d: np.ndarray, sid: np.ndarray, n_seg: int) -> np.ndarray:
    """统计布尔矩阵 (n,1291) 每段内 True 总数;单遍 bincount。"""
    cols = np.broadcast_to(sid, mask2d.shape)
    sel = mask2d & (cols >= 0)
    return np.bincount(cols[sel].astype(np.int64), minlength=n_seg).astype(np.int64)


def segment_stats(alt_m: np.ndarray, mean_ext: np.ndarray, bounds: np.ndarray):
    """按 bounds 分段统计 → (mean, valid_count, total_count)。"""
    return aggregate_segments(mean_ext, segment_index(alt_m, bounds), len(bounds) - 1)


def count_in_segments(alt_m: np.ndarray, bounds: np.ndarray, mask2d: np.ndarray) -> np.ndarray:
    """统计布尔矩阵 (n,1291) 在每段内的 True 总数 → (n_seg,)。"""
    return count_segments(mask2d, segment_index(alt_m, bounds), len(bounds) - 1)


def cad_qc_masks(ext, cad, phase, dem, alt_m):
    """逐 bin 掩膜(CAD + 云相态)。

    返回布尔矩阵 (n,1291):
      valid_base  消光在 [EXT_MIN, EXT_MAX] 且有限
      aer         CAD 气溶胶候选(且非特殊编码)
      cloud       CAD 云候选 或 云相态>0
      below_cloud 位于该廓线"最低云 bin"高度以下
      below_surf  低于该廓线 DEM 地表
    """
    valid_base = np.isfinite(ext) & (ext >= CONFIG["EXT_MIN"]) & (ext <= CONFIG["EXT_MAX"])
    special = np.zeros_like(cad, dtype=bool)
    for code in CONFIG["CAD_SPECIAL"]:
        special |= (cad == code)
    cad_ok = np.isfinite(cad) & ~special
    if np.isfinite(cad).any():
        aer = cad_ok & (cad <= CONFIG["CAD_AER_THR"])
        cloud = cad_ok & (cad >= CONFIG["CAD_CLOUD_THR"])
    else:
        # 无 CAD 字段:退化为只用消光范围(不再把 bin 全判为非气溶胶)
        aer = valid_base.copy()
        cloud = np.zeros_like(valid_base)
    if phase is not None:
        cloud = cloud | (np.isfinite(phase) & (phase > 0))
    # 「云底以下」向量化:每行取云 bin 的最低高度(无云 → inf)
    min_cloud = np.where(cloud, alt_m[None, :], np.inf).min(axis=1)      # (n,)
    has_cloud = np.isfinite(min_cloud)
    below_cloud = has_cloud[:, None] & (alt_m[None, :] < min_cloud[:, None])
    below_buffer = below_cloud & (alt_m[None, :] >= min_cloud[:, None] - CONFIG["CLOUD_BUFFER_M"])
    below_surf = alt_m[None, :] < dem[:, None]
    return valid_base, aer, cloud, below_cloud, below_buffer, below_surf


def classify_thick_cloud(cloud, subtype, alt_m, cod):
    """标记"厚云"bin(Case 4 用)。命中任一规则即为厚:

      ① 编码:`Cloud_Subtype_Multi ∈ CLOUD_THICK_CODES`(水云=3)=厚;
         未知类型(1)按 `CLOUD_UNKNOWN_AS_THICK` 决定(默认保守记厚);
         冰云(2)/定向冰晶(4)默认薄(不在此记厚);
      ② 连通云块厚度 ≥ `CLOUD_THICK_M` 米(对冰云同样生效:厚冰云也算厚);
      ③ 廓线云光学厚度(cod) ≥ `CLOUD_THICK_COD` → 该廓线所有云 bin 记厚。
    返回 (n,1291) 布尔矩阵。
    """
    thick = np.zeros_like(cloud, dtype=bool)
    if subtype is not None:
        thick |= cloud & np.isin(subtype, tuple(CONFIG["CLOUD_THICK_CODES"]))
        if CONFIG["CLOUD_UNKNOWN_AS_THICK"]:
            thick |= cloud & np.isin(subtype, tuple(CONFIG["CLOUD_UNKNOWN_CODES"]))
    elif CONFIG["CLOUD_UNKNOWN_AS_THICK"]:
        thick |= cloud                      # 无 subtype 信息 → 全部按未知处理

    # ② 连通云块厚度(对全部云生效)
    dalt = float(np.median(np.abs(np.diff(alt_m)))) if alt_m.size > 1 else 24.0
    n_thick_bins = max(1, int(round(CONFIG["CLOUD_THICK_M"] / max(dalt, 1e-6))))
    for j in range(cloud.shape[0]):
        row = cloud[j]
        if not row.any():
            continue
        idx = np.flatnonzero(row)
        split = np.flatnonzero(np.diff(idx) > 1)
        for seg in np.split(idx, split + 1):
            if seg.size >= n_thick_bins:
                thick[j, seg] = True

    # ③ 廓线级云光学厚度
    if cod is not None:
        hot = np.isfinite(cod) & (cod >= CONFIG["CLOUD_THICK_COD"])
        thick[hot] |= cloud[hot]
    return thick


# ============================================================
# 输出
# ============================================================
def build_column_names(n_level: int, output_stats: bool | None = None,
                       super_levels: tuple[str, ...] | None = None,
                       vertical: str | None = None) -> list[str]:
    """列序:标识 → 训练特征(ERA5 廓线) → 训练目标(ACDL 消光)[→ 统计/QC][→ 单层列][→ DEM_m]。

    output_stats=False(lite,默认);super_levels 为要并入的单层量(blh/tcwv/zsfc)。
    vertical="agl" 时在末尾追加 DEM_m(逐组地表海拔,训练端 dz/GCF 用)。
    """
    if output_stats is None:
        output_stats = bool(CONFIG.get("OUTPUT_STATS", False))
    if vertical is None:
        vertical = str(CONFIG.get("VERTICAL", "asl"))
    n_seg = n_level                         # 32 段(段0=底部)
    # 1) 标识
    names = ["ERA5_Lon", "ERA5_Lat", "ERA5_Time", "Hour"]
    # 2) 训练特征:各 pressure level 的 H/T/RH
    names += [f"H_k{k:02d}" for k in range(n_level)]
    names += [f"T_k{k:02d}" for k in range(n_level)]
    names += [f"RH_k{k:02d}" for k in range(n_level)]
    # 3) 训练目标:各层段的 ACDL 层平均消光
    names += [f"Ext_mean_L{k:02d}" for k in range(n_seg)]
    if output_stats:
        # 4) 统计 / QC(仅测试/诊断用)
        names += [f"Ext_valid_L{k:02d}" for k in range(n_seg)]
        names += [f"Cloud_bins_L{k:02d}" for k in range(n_seg)]
        names += [f"Below_Cloud_bins_L{k:02d}" for k in range(n_seg)]
        names += [f"Ext_bins_L{k:02d}" for k in range(n_seg)]
        names += ["NProfiles_raw", "NProfiles_used", "Coverage_mean", "NodeDist_deg"]
    if super_levels is None:
        super_levels = tuple(CONFIG.get("SUPER_LEVELS", ()))
    for v in super_levels:
        if v in SUPER_LEVEL_DEFS:
            names.append(SUPER_LEVEL_DEFS[v][1])          # BLH_m / TCWV_kgm2 / Z_sfc_m
    if vertical == "agl":
        names.append("DEM_m")
    return names


def write_day(path: Path, rows: np.ndarray, level: np.ndarray, names: list[str],
              struct_name: str, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    m = {struct_name: {
        "Data": rows.astype(np.float64),
        "VarNames": list(names),
        "Level": np.asarray(level, dtype=np.float32).reshape(-1, 1),
        "Meta": meta,
    }}
    hdf5storage.savemat(str(path), m, fmt="7.3", store_python_metadata=False,
                        appendmat=False, matlab_compatible=True, oned_as="column",
                        action_for_matlab_incompatible="error")
    print(f"    [SAVE] {path}  rows={rows.shape[0]} cols={rows.shape[1]}")


# ============================================================
# 主流程
# ============================================================
def run(start: str, end: str, dry_run: bool, skip_existing: bool = False) -> int:
    """按日循环匹配;skip_existing=True 时跳过已有输出文件(便于断点续跑)。返回写出/将写出的总行数。"""
    acdl_dir = Path(CONFIG["ACDL_DIR"])
    out_dir = Path(CONFIG["OUT_DIR"])
    d0 = datetime.strptime(start, "%Y%m%d")
    d1 = datetime.strptime(end, "%Y%m%d")

    month_cache: dict[str, Era5Month] = {}
    single_cache: dict[str, Era5SingleLevel] = {}
    total_rows = 0
    day = d0
    while day <= d1:
        yyyymmdd = day.strftime("%Y%m%d")
        yyyymm = day.strftime("%Y%m")
        out_file = out_dir / f"ACDL_ERA5_MatchV2_{yyyymmdd}.mat"
        if skip_existing and not dry_run and out_file.exists():
            print(f"\n[DAY] {yyyymmdd}  [SKIP] 已存在:{out_file.name}")
            day += timedelta(days=1)
            continue
        print(f"\n[DAY] {yyyymmdd}")
        _t_read0 = time.perf_counter()
        acdl = load_acdl_day_cached(acdl_dir, yyyymmdd, use_cache=CONFIG["USE_CACHE"])
        t_read = time.perf_counter() - _t_read0
        if acdl is None:
            print("    [WARN] 当日无 ACDL 廓线,跳过")
            day += timedelta(days=1)
            continue

        era5 = month_cache.get(yyyymm)
        if era5 is None:
            era5 = Era5Month(yyyymm)
            month_cache[yyyymm] = era5
        sl = single_cache.get(yyyymm)
        if sl is None and CONFIG["SUPER_LEVELS"]:
            sl = Era5SingleLevel(yyyymm, tuple(CONFIG["SUPER_LEVELS"]))
            single_cache[yyyymm] = sl
        ncol = era5.lat2d.shape[1]

        alt_m = acdl["alt"] * 1000.0                       # km → m
        N = acdl["ext"].shape[0]
        n_seg = len(era5.level)
        print(f"    廓线 {N};网格 {era5.lat2d.shape};时次 {len(era5.time)}")

        # 空间:最近 0.25° 节点;时间:最近整点(±tol,向量化)
        la = np.radians(acdl["lat"]); lo = np.radians(acdl["lon"])
        xyz = np.column_stack([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)])
        dist, node = era5.tree.query(xyz)
        node = np.asarray(node).ravel()
        dist_deg = chord_to_deg(np.asarray(dist).ravel())
        ti_all, off_all = era5.time_index_vec(acdl["datenum"])
        in_win = np.abs(off_all) <= CONFIG["TIME_TOL_MIN"]
        n_time_reject = int((~in_win).sum())

        # 距离上限:ACDL 是全球观测,而 ERA5 只覆盖 73-135°E/3-54°N。
        # 不加这道限制,tree.query 会把域外几千公里外的廓线吸附到最近的**边界节点**
        # (实测 94.7% 的廓线、82% 的匹配行都是这种错配,且边界节点会把半个地球的廓线
        #  平均到一起)。真正属于某节点的廓线距离必然 ≤ 半对角线 ≈ 0.1768°。
        max_deg = CONFIG.get("MAX_NODE_DIST_DEG")
        if max_deg is None:
            in_dom = np.ones(in_win.shape, dtype=bool)
        else:
            in_dom = dist_deg <= float(max_deg)
        n_space_reject = int((~in_dom).sum())
        in_win = in_win & in_dom

        groups: dict[tuple[int, int], list[int]] = {}
        for i in np.flatnonzero(in_win):
            groups.setdefault((int(node[i]), int(ti_all[i])), []).append(int(i))
        print(f"    (节点,时次) 组数 {len(groups)}; 时间超窗剔除 {n_time_reject} 廓线;"
              f" 域外剔除 {n_space_reject} 廓线"
              f"(距离上限 {max_deg if max_deg is not None else '∞'}°)"
              f" → 保留 {int(in_win.sum())}/{N} 条")

        out_stats = bool(CONFIG.get("OUTPUT_STATS", False))
        vertical_agl = str(CONFIG.get("VERTICAL", "asl")) == "agl"
        if vertical_agl:
            print(f"    [VERTICAL] agl(地形跟随):{len(CONFIG['AGL_EDGES_KM'])} 边界 → "
                  f"{len(CONFIG['AGL_EDGES_KM'])-1} 段;DEM 字段 km→m 修复启用")
        if sl is not None and sl.vars:
            sl.prefetch([ti for (_, ti) in groups.keys()])
        sl_vars = tuple(sl.vars) if sl is not None else ()
        names = build_column_names(n_seg, out_stats, super_levels=sl_vars,
                                   vertical="agl" if vertical_agl else "asl")
        rows = []
        _t_agg0 = time.perf_counter()
        for (nd, ti), idxs in sorted(groups.items()):
            r, c = divmod(nd, ncol)
            ext = acdl["ext"][idxs]                        # (n, 1291) float32,只读不复制
            cad = acdl["cad"][idxs]
            phase = acdl["phase"][idxs] if acdl.get("phase") is not None else None
            cod = acdl["cod"][idxs]                        # (n,)
            dem = acdl["dem"][idxs]
            if vertical_agl:
                dem = dem * 1000.0                    # ACDL DEM 字段实为 km(实证:天山处读数 1.6);agl 模式修正
            n_raw = len(idxs)

            # ---- 逐 bin QC 掩膜(CAD + 云类型;已向量化)----
            valid_base, aer, cloud, below_cloud, below_buffer, below_surf = cad_qc_masks(
                ext, cad, phase, dem, alt_m)
            above_surf = ~below_surf
            case = int(CONFIG["QC_CASE"])
            m1 = valid_base & aer & ~cloud & ~below_cloud & above_surf          # Case 1 严格
            m2 = valid_base & aer & ~cloud & above_surf                         # Case 2 宽松
            m3 = valid_base & aer & ~cloud & ~below_buffer & above_surf         # Case 3 折中
            m5 = valid_base & above_surf                                        # Case 5 极简(仅范围+地表以上)
            if case == 4:
                # Case 4 自适应:厚云下方全丢;薄云只丢云底缓冲带(向量化,无逐廓线循环)
                thick = classify_thick_cloud(cloud, phase, alt_m, cod)
                min_thick = np.where(thick, alt_m[None, :], np.inf).min(axis=1)
                below_thick = np.isfinite(min_thick)[:, None] & (alt_m[None, :] < min_thick[:, None])
                thin = cloud & ~thick
                min_thin = np.where(thin, alt_m[None, :], np.inf).min(axis=1)
                has_thin = np.isfinite(min_thin)
                below_thin_buffer = (has_thin[:, None]
                                     & (alt_m[None, :] < min_thin[:, None])
                                     & (alt_m[None, :] >= min_thin[:, None] - CONFIG["CLOUD_BUFFER_M"]))
                m4 = valid_base & aer & ~cloud & ~below_thick & ~below_thin_buffer & above_surf
                m_sel_all = m4
            else:
                m_sel_all = {1: m1, 2: m2, 3: m3, 5: m5}[case]

            # ---- 廓线级覆盖率(Case2 口径)与低覆盖剔除 ----
            n_above = above_surf.sum(axis=1)
            cov = np.where(n_above > 0, m2.sum(axis=1) / np.maximum(n_above, 1), 0.0)
            keep = cov >= CONFIG["COV_MIN"]
            if not keep.any():
                continue                                   # 该组无可用廓线
            ext = ext[keep]
            dem = dem[keep]
            above_surf = above_surf[keep]
            m_sel = m_sel_all[keep]
            cov_kept = cov[keep]
            if out_stats:
                cloud_kept = cloud[keep]
                below_cloud_kept = below_cloud[keep]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # 全 NaN 列静默
                mean_sel = np.nanmean(np.where(m_sel, ext, np.nan), axis=0)

            surf_candidates = dem[np.isfinite(dem)]
            surf_h = float(np.mean(surf_candidates)) if surf_candidates.size else np.nan

            z = era5.column("z", ti, nd, ncol)
            t = era5.column("t", ti, nd, ncol)
            rh = era5.column("r", ti, nd, ncol)
            H_lev_raw = level_heights(z)                   # 气压层位势高度(asl=段顶;agl=插值源剖面)
            H = H_lev_raw
            if not np.isfinite(surf_h):
                surf_h = H[0]                              # 兜底:最低层高度(底部段为空)
            if vertical_agl:
                # ---- AGL:边界 = dem_m + AGL 边界表(地形跟随);段特征 = 段顶处剖面插值 ----
                bounds = agl_edges_m(surf_h)               # 段 0 = [dem, dem+e1],恒非空
                H = bounds[1:]                             # 段顶 ASL(训练端 dz 恒 >0)
                ok_lev = (np.isfinite(H_lev_raw) & np.isfinite(np.asarray(t))
                          & np.isfinite(np.asarray(rh)) & (H_lev_raw > surf_h))
                z_v = H_lev_raw[ok_lev]                    # 注意:插值用高度(m),不是原始位势
                t_v = np.asarray(t)[ok_lev]
                r_v = np.asarray(rh)[ok_lev]
                t_seg = np.empty(n_seg); rh_seg = np.empty(n_seg)
                for k in range(n_seg):
                    t_seg[k], rh_seg[k] = profile_at_height(z_v, t_v, r_v, H[k])
                t, rh = t_seg, rh_seg
                if z_v.size == 0:
                    t[:] = np.nan; rh[:] = np.nan          # 该组 ERA5 剖面全无效 → 全 NaN(极端)
            else:
                bounds = seg_bounds(surf_h, H)             # 历史口径(固定气压层),不改动

            sid = segment_index(alt_m, bounds)             # 一次性求段号
            me, vc, tcnt = aggregate_segments(mean_sel, sid, n_seg)

            row_vals = [np.array([era5.lon2d[r, c], era5.lat2d[r, c], era5.time[ti],
                                  int(round((era5.time[ti] - np.floor(era5.time[ti])) * 24))],
                                 dtype=np.float64)]
            row_vals += [H, t, rh, me]
            if sl_vars:
                row_vals.append(np.array([sl.value(v, ti, r, c) for v in sl_vars],
                                         dtype=np.float64))
            if vertical_agl:
                row_vals.append(np.array([surf_h], dtype=np.float64))   # DEM_m:组均值地表海拔(已 km→m)
            if out_stats:
                cloud_bins = count_segments(cloud_kept & above_surf, sid, n_seg)
                below_cloud_bins = count_segments(below_cloud_kept & above_surf, sid, n_seg)
                row_vals += [vc.astype(np.float64), cloud_bins.astype(np.float64),
                             below_cloud_bins.astype(np.float64), tcnt.astype(np.float64),
                             [n_raw, int(keep.sum()), float(np.mean(cov_kept)),
                              float(dist_deg[idxs[0]])]]
            rows.append(np.concatenate(row_vals))

        if not rows:
            print("    [WARN] 当日无匹配样本")
            day += timedelta(days=1)
            continue

        if not rows:
            # 当天全部廓线被时间窗/距离上限剔除(极端情况:轨道整天在 ERA5 域外)
            print("    该日无任何样本通过筛选(时间窗/距离上限),跳过写盘")
            day += timedelta(days=1)
            continue
        data = np.vstack(rows)
        assert data.shape[1] == len(names), (data.shape, len(names))
        total_rows += int(data.shape[0])
        t_agg = time.perf_counter() - _t_agg0
        print(f"    耗时: 读ACDL {t_read:.2f}s | 匹配聚合 {t_agg:.2f}s"
              f" | 样本 {data.shape[0]}×{data.shape[1]}")
        if not dry_run:
            # 单层列(blh/tcwv/zsfc)是**追加**在最后,标签里必须体现,否则列数对不上
            super_cols = [SUPER_LEVEL_DEFS[v][1] for v in CONFIG["SUPER_LEVELS"]
                          if v in SUPER_LEVEL_DEFS]
            n_super = len(super_cols)
            meta = {
                "SchemaVersion": "v2-downsampled-qcCAD",
                "GridRes": CONFIG["GRID_RES"],
                "TimeTolMin": CONFIG["TIME_TOL_MIN"],
                # 廓线到所属 ERA5 节点的最大球面距离(度);超出的直接丢弃。
                # None = 无限制(旧行为:域外廓线会被吸附到边界节点,已证实是错的)。
                "MaxNodeDistDeg": (None if CONFIG.get("MAX_NODE_DIST_DEG") is None
                                   else float(CONFIG["MAX_NODE_DIST_DEG"])),
                "Segments": "32 (L00=surface->H0, L01..L31=between levels)",
                "QC": ("EXT in [%.2f,%.2f]; aer=CAD<=%.0f; cloud=CAD>=%.0f or cloud-type>0; "
                       "QC_CASE=%d (1=drop all below cloud, 2=keep if CAD-valid, 3=buffer %.0f m, "
                       "4=adaptive: thick-cloud-below dropped / thin-cloud-below kept, "
                       "5=minimal: only extinction range + above-surface); "
                       "phase(0=N/A,1=unknown,2=ice,3=water,4=oriented ice): thick=%s, "
                       "unknown_as_thick=%s, thickness>=%.0f m, COD>=%.2f; COV_MIN=%.3f"
                       % (CONFIG["EXT_MIN"], CONFIG["EXT_MAX"],
                          CONFIG["CAD_AER_THR"], CONFIG["CAD_CLOUD_THR"],
                          int(CONFIG["QC_CASE"]), CONFIG["CLOUD_BUFFER_M"],
                          str(tuple(CONFIG["CLOUD_THICK_CODES"])), CONFIG["CLOUD_UNKNOWN_AS_THICK"],
                          CONFIG["CLOUD_THICK_M"], CONFIG["CLOUD_THICK_COD"], CONFIG["COV_MIN"])),
                "Output": ((("lite(%d cols: Lon/Lat/Time/Hour | H_k* | T_k* | RH_k* | Ext_mean_L*"
                             % (132 + n_super))
                            + ((" | " + " | ".join(super_cols)) if n_super else ""))
                           if not out_stats else
                           ("full(%d cols, incl. stats/QC columns" % (264 + n_super))
                           + ((" | " + " | ".join(super_cols)) if n_super else "") + ")"),
                "Cache": "on" if CONFIG["USE_CACHE"] else "off",
                "SuperLevels": ",".join(CONFIG["SUPER_LEVELS"]) if CONFIG["SUPER_LEVELS"] else "(none)",
                "ColumnOrder": ("Lon/Lat/Time/Hour | H_k* | T_k* | RH_k* | Ext_mean_L*"
                                + ("" if not out_stats else
                                   " | Ext_valid_L* | Cloud_bins_L* | Below_Cloud_bins_L* | Ext_bins_L* | NProfiles_*/Coverage_mean/NodeDist_deg")
                                + ((" | " + " | ".join(super_cols)) if n_super else "")),
                "DEMSource": "ACDL DEM_Surface_Elevation (fallback=H0)",
                "ERA5Source": "era5_pressure_levels/0.25deg (z/g0, no integration)",
                "Vertical": ("agl" if vertical_agl else "asl"),
                "VerticalNote": ("terrain-following: bounds = DEM_m + AGL_EDGES_KM, "
                                 "T/RH = profile interp at segment top, below-lowest-level "
                                 "extrapolated with clamped lapse") if vertical_agl else "fixed pressure levels",
                "AGLEdgesKm": (list(CONFIG["AGL_EDGES_KM"]) if vertical_agl else None),
                "DemUnitsFix": ("km->m applied (ACDL DEM field is km; verified Tianshan ~1.6)"
                                if vertical_agl else "not applied (asl keeps legacy behavior)"),
            }
            write_day(out_dir / f"ACDL_ERA5_MatchV2_{yyyymmdd}.mat", data,
                      (np.asarray(CONFIG["AGL_EDGES_KM"][1:], dtype=np.float32)   # agl:Level=段顶离地高度(km)
                       if vertical_agl else era5.level),
                      names, CONFIG["OUT_STRUCT"], meta)
        else:
            print(f"    [DRY] 将写 {data.shape[0]} 行 × {data.shape[1]} 列")

        day += timedelta(days=1)

    for em in month_cache.values():
        em.close()
    for sm in single_cache.values():
        sm.close()
    print(f"\n[DONE] 共写出样本行数 {total_rows}")
    return total_rows


# ============================================================
# 离线自检(不需真实数据)
# ============================================================
def selftest() -> int:
    print("[SELFTEST] 分段统计 + CAD QC 掩膜")
    n = 10
    alt_m = np.arange(n, dtype=np.float64) * 100.0          # 0,100,...,900 m
    H = np.array([200.0, 400.0, 600.0, 800.0])              # 4 层 → 4 段(0=底部)
    bounds = seg_bounds(100.0, H)
    mean_ext = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
    mean_ext[3] = np.nan
    m, v, t = segment_stats(alt_m, mean_ext, bounds)
    print("  mean:", m, "valid:", v, "bins:", t)
    assert t.tolist() == [1, 2, 2, 2], t.tolist()
    assert v.tolist() == [1, 1, 2, 2], v.tolist()
    assert np.allclose(m[0], 2.0) and np.allclose(m[1], 3.0)
    _, _, t2 = segment_stats(alt_m, mean_ext, seg_bounds(H[0], H))   # 兜底:底部段为空
    assert t2[0] == 0, t2

    # CAD QC:1 条廓线,alt=500 处为云;其下(Case1 剔除 / Case2 保留)
    ext = np.full((1, n), 0.5)
    cad = np.full((1, n), -30.0)
    cad[0, 5] = 30.0
    dem = np.array([0.0])
    vb, aer, cloud, bc, bb, bs = cad_qc_masks(ext, cad, None, dem, alt_m)
    assert cloud[0, 5] and not cloud[0, 4]
    assert bc[0, 4] and not bc[0, 5]
    assert bb[0, 4]                                  # alt400 在云底(500)下方 1 km 缓冲内
    # 云类型>0 也应判为云(即使 CAD 为负)
    phase = np.zeros_like(cad); phase[0, 7] = 2.0
    _, _, cloud2, _, _, _ = cad_qc_masks(ext, cad, phase, dem, alt_m)
    assert cloud2[0, 7] and not cloud[0, 7]
    # 厚云判定:云块厚度 / 相态编码 / COD
    zero_code = np.zeros_like(cad)                    # 0=N/A,不参与编码判定
    cloud3 = np.zeros_like(cad, dtype=bool); cloud3[0, 2:10] = True   # 8 bins × 100 m = 800 m
    assert not classify_thick_cloud(cloud3, zero_code, alt_m, None)[0, 5], "800 m 云块不应判厚"
    big = np.zeros_like(cad, dtype=bool); big[0, 0:10] = True         # 10 bins = 1000 m
    assert classify_thick_cloud(big, zero_code, alt_m, None)[0, 5], "1000 m 云块应判厚"
    codv = np.zeros(1); codv[0] = 1.0
    assert classify_thick_cloud(cloud3, zero_code, alt_m, codv)[0, 5], "COD 大应整体判厚"
    # 相态:水云(3)=厚;冰云(2)=薄;未知(1)=保守记厚
    water = np.zeros_like(cad); water[0, 2:10] = 3
    ice = np.zeros_like(cad); ice[0, 2:10] = 2
    unk = np.zeros_like(cad); unk[0, 2:10] = 1
    assert classify_thick_cloud(cloud3, water, alt_m, None)[0, 5], "水云应判厚"
    assert not classify_thick_cloud(cloud3, ice, alt_m, None)[0, 5], "冰云默认应判薄"
    assert classify_thick_cloud(cloud3, unk, alt_m, None)[0, 5], "未知类型默认应判厚"
    # 无 CAD 时退化:不应把 bin 全排除
    _, aer2, _, _, _, _ = cad_qc_masks(ext, np.full_like(cad, np.nan), None, dem, alt_m)
    assert aer2.any(), "无 CAD 应退化为只用消光范围"
    assert bc[0, 4] and not bc[0, 5]
    m1 = vb & aer & ~bc & ~bs
    m2 = vb & aer & ~bs
    assert not m1[0, 4] and m2[0, 4]

    # ---- AGL 分段(--vertical agl 核心)----
    dem_pl, dem_hi = 30.0, 3000.0
    e0, e3 = agl_edges_m(dem_pl), agl_edges_m(dem_hi)
    assert e0.size == 33 and e3.size == 33 and e0[0] == dem_pl and e3[0] == dem_hi
    assert np.all(np.diff(e0) > 0) and np.all(np.diff(e3) > 0)
    assert e3[-1] <= CONFIG["ALT_MAX_M"] + 1e-6, f"顶部必须 ≤ ACDL 上限: {e3[-1]}"
    # 零海拔不压缩:AGL 边界 == 名义表;高原 TOP_SPLIT 以下不压缩
    nom = np.asarray(CONFIG["AGL_EDGES_KM"]) * 1000.0
    e_zero = agl_edges_m(0.0)
    assert np.allclose(e_zero, nom, atol=1e-6), "零海拔 AGL 边界应等于名义表"
    i_split = int(np.flatnonzero(nom > CONFIG["AGL_TOP_SPLIT_KM"] * 1000.0)[0]) - 1
    assert abs((e3[i_split] - dem_hi) - nom[i_split]) < 1e-6, "TOP_SPLIT 以下边界不应被压缩"
    # 3km 高原:近地面 bin 落进 L00(asl 口径下这是 NaN)
    alt_t = np.array([3020.0, 3060.0, 3200.0, 5000.0, 9000.0])   # AGL 20/60/200/2000/6000 m
    sid_t = segment_index(alt_t, e3)
    assert sid_t[0] == 0, "AGL 0-50m 段应有 bin(asl 口径下这是 NaN)"
    assert sid_t[1] == 1 and sid_t[2] == 2 and sid_t[3] > sid_t[2], "bin 归段应正确且单调"
    # 剖面插值 + 限幅外推
    H_lev = np.array([3100.0, 3600.0, 4400.0])
    T_lev = np.array([270.0, 265.0, 260.0]); RH_lev = np.array([30.0, 35.0, 40.0])
    tt, rr = profile_at_height(H_lev, T_lev, RH_lev, 3150.0)     # 层间 → 插值
    assert abs(tt - 269.5) < 1e-9 and abs(rr - 30.5) < 1e-9
    tt2, rr2 = profile_at_height(H_lev, T_lev, RH_lev, 3050.0)   # 低于最低层 50m → 限幅外推
    assert abs(tt2 - 270.5) < 1e-9 and abs(rr2 - 29.5) < 1e-9
    tt3, rr3 = profile_at_height(H_lev[:1], T_lev[:1], RH_lev[:1], 3000.0)  # 单层 → 常值保持
    assert abs(tt3 - 270.0) < 1e-9 and abs(rr3 - 30.0) < 1e-9
    print("[SELFTEST] PASS")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=CONFIG["START"])
    ap.add_argument("--end", default=CONFIG["END"])
    ap.add_argument("--out-dir", default=None, help="输出目录(覆盖 CONFIG['OUT_DIR'])")
    ap.add_argument("--acdl-dir", default=None, help="ACDL 廓线目录(覆盖 CONFIG['ACDL_DIR'])")
    ap.add_argument("--era5-root", default=None, help="ERA5 产物根目录(覆盖 CONFIG['ERA5_MAT_ROOT'])")
    ap.add_argument("--case", type=int, choices=(1, 2, 3, 4, 5), default=CONFIG["QC_CASE"],
                    help="QC Case: 1=严格、2=宽松、3=折中(云底缓冲带)、"
                         "4=自适应(厚云下丢弃/薄云下保留,默认)、5=极简(仅 0–1.25 与地表以上)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="跳过已存在的输出文件(断点续跑)")
    ap.add_argument("--with-stats", action="store_true",
                    help="输出完整 264 列(含测试用统计/QC 列);默认 lite 132 列")
    ap.add_argument("--vertical", choices=("asl", "agl"), default=CONFIG["VERTICAL"],
                    help="垂直坐标: asl=固定气压层段(历史口径,默认) / "
                         "agl=地形跟随离地高度段(近地面全域可用;同时修复 DEM km→m 并追加 DEM_m 列)")
    ap.add_argument("--no-cache", action="store_true", help="不使用日级缓存(强制解析 .mat)")
    ap.add_argument("--super-levels", default=None,
                    help="要并入的单层量(逗号分隔:blh,tcwv,zsfc);默认取 CONFIG;传 \"\" 关闭")
    ap.add_argument("--no-super-levels", action="store_true",
                    help="不并入单层列(等价 --super-levels \"\"),输出 132 列")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.out_dir:    CONFIG["OUT_DIR"] = a.out_dir
    if a.acdl_dir:   CONFIG["ACDL_DIR"] = a.acdl_dir
    if a.era5_root:  CONFIG["ERA5_MAT_ROOT"] = a.era5_root
    CONFIG["QC_CASE"] = a.case
    if a.with_stats:
        CONFIG["OUTPUT_STATS"] = True
    if a.no_cache:
        CONFIG["USE_CACHE"] = False
    CONFIG["VERTICAL"] = a.vertical
    if a.no_super_levels:
        CONFIG["SUPER_LEVELS"] = ()
    elif a.super_levels is not None:
        CONFIG["SUPER_LEVELS"] = tuple(s.strip() for s in a.super_levels.split(",") if s.strip())
    if a.selftest:
        sys.exit(selftest())
    run(a.start, a.end, a.dry_run, skip_existing=a.skip_existing)
    sys.exit(0)
