"""
Train_ACDL_ERA5_MatchV2.py
============================================================
训练脚本:用 MatchV2 匹配样本(ACDL×ERA5 降采样)训练"时空注意力"网络,
由 ERA5 廓线特征预测 ACDL 32 个 ERA5 层段的平均消光。

数据来源:Match_ACDL_ERA5_FromScratch.py 的产物
  <MATCH_DIR>/ACDL_ERA5_MatchV2_YYYYMMDD.mat (struct MatchV2)
  - lite 132 列(默认):0-3 标识 | 4-35 H_k* | 36-67 T_k* | 68-99 RH_k* | 100-131 Ext_mean_L*
  - full 264 列(--with-stats 产物):前 132 列同上,后续为统计列(本脚本不使用)

特征 / 目标
  X = [Lon, Lat, sin(hour), cos(hour), sp_x, sp_y, sp_z, H_k(32), T_k(32), RH_k(32)]  → 103 维
  y = Ext_mean_L00..L31 (32 维,km⁻¹);NaN = 该层无有效观测 → **掩膜,不参与损失/指标**

两种目标模式 --target:
  abs  = 逐层绝对消光 σ_l(km⁻¹),直接回归,重建廓线本身(默认)
  frac = 厚度加权逐层占比 f_l = σ_lΔz_l/Σ(σΔz),和为 1,只学"形状"(把不可学的柱含量除掉)

模型:复用原 SpatioTemporalAttention(时间/空间/气象 三 token + Transformer + 注意力池化),
      仅把输入维度(气象 5→96)与输出维度(1291→32)改成当前任务。

交叉验证:空间块 CV(3×3)与时间块 CV(按 ERA5_Time 排序后切块),两者分别报告。

报告口径(只报两个指标)
  1) **绝对消光逐层 R²** —— 无论哪种目标模式,都先把模型输出还原成 σ̂_l(km⁻¹)再与
     观测消光逐层比较:abs 直接用预测值,frac 借观测 OD(σ̂=f̂·OD_obs/Δz);
  2) **GCF R²** —— 直接由预测廓线算 Σ_{层底<GCF_TOP_M} σ̂Δz / Σσ̂Δz,与观测 GCF 比。
     报两个口径:主口径只用"模型训练所涉"的样本(frac 模式下即 ok_frac ∩ 真值有效),
     全口径(含未参与训练的样本)另存 *_all,偏低、不作结论依据。
  产物:results/per_layer_metrics_<tag>.csv、plots/per_layer_r2_<tag>.png、plots/gcf_scatter_<tag>.png。

损失:掩膜 Huber(逐层标准化后的 y,只对非 NaN 层计损失)。

依赖:numpy, h5py, torch, scikit-learn(绘图可选 matplotlib)

运行(默认:只跑一次 20% 随机留出训练,直接给逐层 R²;CV 与全量模型需显式开启):
  python Train_ACDL_ERA5_MatchV2.py --match-dir /path/Matchoutput --out-dir ./train_out
  python Train_ACDL_ERA5_MatchV2.py ... --holdout 0.3 --epochs 60      # 调留出比例/轮数
  python Train_ACDL_ERA5_MatchV2.py ... --cv                          # 加跑空间块+时间块 CV(慢)
  python Train_ACDL_ERA5_MatchV2.py ... --final                       # 加训全量模型并存 model_final.pt
  python Train_ACDL_ERA5_MatchV2.py --selftest                        # 合成数据自检(不需真实数据)
============================================================
"""

from __future__ import annotations

import argparse
import json
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 绘图统一放在独立模块(与训练解耦;事后可用 Plot_Training_Results.py 重画)
from acdl_plotting import plot_per_layer_r2, plot_gcf_scatter


# ============================================================
# 配置
# ============================================================
CONFIG = {
    # 匹配数据(135 列,含 BLH_m / TCWV_kgm2 / Z_sfc_m)。指向"距离上限修正 + QC case5"的新产物;
    # 想要 case4 的严格 QC 产物就改成 D:/matchdata_fixed_case4
    "MATCH_DIR": r"D:/matchdata_fixed_case5",
    "MATCH_GLOB": "ACDL_ERA5_MatchV2_*.mat",
    # 结果目录:相对路径 → 落在**本工程目录**下(不要写到 D: 盘);所有运行产物统一归 结果汇总/runs/
    "OUT_DIR":   r"./结果汇总/runs/train_out",
    "STRUCT":    "MatchV2",

    # 列(0 基,lite 布局)
    "COL_LON": 0, "COL_LAT": 1, "COL_TIME": 2, "COL_HOUR": 3,
    "COL_H0": 4, "N_LEVEL": 32,               # H/T/RH 各 32 列
    "COL_YM0": 100,                            # Ext_mean 起始列

    # 由 Add_ERA5_SingleLevel_Columns.py 追加的额外列(按列名读取,不写死列号)
    "ADD_COLS": ["BLH_m", "TCWV_kgm2", "Z_sfc_m"],   # 默认启用(135 列产物);置 [] 可关闭
    "ADD_COL_ALIASES": {"blh": "BLH_m", "tcwv": "TCWV_kgm2", "zsfc": "Z_sfc_m"},
    "ZSFC_COL": "Z_sfc_m",                     # 若存在:用于修正 L00 厚度(地表高度)

    # 训练
    "SEED": 42,
    "BATCH": 256,
    "EPOCHS": 200,
    "LR": 3e-4,
    "WEIGHT_DECAY": 1e-4,
    "D_MODEL": 128,
    "N_HEADS": 4,
    "N_LAYERS": 2,
    "DROPOUT": 0.15,
    "PATIENCE": 20,                            # 早停
    "GRAD_CLIP": 1.0,
    "HUBER_DELTA": 1.0,

    # 目标模式(只有两种):
    #   abs  = 逐层绝对消光 σ_l(km⁻¹),重建廓线本身
    #   frac = 厚度加权逐层占比 f_l = σ_lΔz_l/Σ(σΔz),和为 1,只学"形状"
    "TARGET": "abs",

    # ---- 优化路线 Step 2/3 开关(《优化路线讨论.md》§2.5/§3)----
    # 损失高度权重:loss = Σ w_l·mask_l·Huber / Σ w_l·mask_l —— 三种是同一段代码,只换 w_l
    #   none    = 全层等权(基线 B0)
    #   decay   = w_l = exp(-层底高度/τ),低层加权(τ=W_DECAY_TAU_KM)
    #   hard5km = 层底 < HARD_TOP_KM 的层 w=1,其余 0(训练重点截断,输出仍 32 层)
    "W_PROFILE": "none",
    "W_DECAY_TAU_KM": 3.0,
    "HARD_TOP_KM": 5.0,
    # 目标取 log(E2b):y→ln(clip(y,LOG_EPS)) 后再逐层标准化;指标仍在原空间评估
    "LOG_TARGET": False,
    "LOG_EPS": 1e-6,
    # 输出非负(E2c):解码器输出过 softplus;输出恒 ≥0 → 目标 z-score 关闭,原空间回归
    "SOFTPLUS": False,
    "REQ_MIN_LAYERS": 10,     # 形状目标(frac):样本至少要有多少有效层
    "REQ_COV": 0.5,           # 形状目标:有效厚度至少占整柱的比例
    "SURF_M": 0.0,            # 估算 L00 厚度用的地表高度(m;匹配结果未存 surf_h)
    "DROP_L00": False,        # 是否把 L00 排除出形状目标
    "GCF_TOP_M": 500.0,       # GCF 定义的"近地面"厚度(米)
    "REQ_GCF_LAYERS": False,  # 形状目标是否也要求"近地面层完整"(默认否,避免样本进一步缩水)

    # 运行模式(默认"只跑一次留出训练,看逐层 R²",适合快速迭代)
    "HOLDOUT": 0.2,        # 随机留出比例(0 = 不划分,用 --no-holdout 关闭)
    "RUN_CV": False,       # 是否跑空间/时间 CV(慢);用 --cv 开启
    "TRAIN_FINAL": False,  # 是否额外用全量数据训最终模型;用 --final 开启
    # 出图:True=训练后就地出图;False=只落盘 predictions_*.npz,事后用图脚本重画。
    # 绘图代码在独立模块 acdl_plotting.py 里(与训练解耦)。
    "PLOTS": True,

    # 交叉验证(仅在 --cv 时使用)
    "SPATIAL_BLOCKS": 3,                       # 3×3
    "TIME_FOLDS": 4,
    "MIN_TEST": 10, "MIN_TRAIN": 100,

    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}

MISSING = -1.0e30      # 某个 ERA5 层在训练集里全缺失时的占位(标准化前会被掩膜)


# ============================================================
# 数据读取
# ============================================================
def decode_varnames(f, ds) -> list[str] | None:
    """MATLAB v7.3 cell-of-char → list[str];失败返回 None。"""
    try:
        refs = np.asarray(ds[()])
        out = []
        for ref in refs.ravel():
            arr = np.asarray(f[ref][()])
            out.append("".join(chr(int(c)) for c in arr.ravel()))
        return out
    except Exception:
        return None


def read_meta_vertical(g, f) -> str | None:
    """读匹配产物 Meta.Vertical(asl/agl);缺失/解析失败返回 None。"""
    try:
        if "Meta" not in g or "Vertical" not in g["Meta"]:
            return None
        v = np.asarray(g["Meta"]["Vertical"][()])
        if v.dtype == object or v.dtype == h5py.ref_dtype:   # cell-of-char(同 VarNames 编码)
            chars = []
            for ref in v.ravel():
                chars.append("".join(chr(int(c)) for c in np.asarray(f[ref][()]).ravel()))
            return "".join(chars).strip() or None
        if v.dtype.kind in "SU":
            return str(v.ravel()[0]).strip() or None
        if v.dtype.kind in "uif":                            # 字符码数组(uint16 等)
            return "".join(chr(int(c)) for c in v.ravel()).strip() or None
    except Exception:
        return None
    return None


def load_matched_dir(match_dir: Path, pattern: str, struct: str, add_cols=()):
    """读取目录下全部匹配样本,拼成 X(+可选额外列)/ y / 元数据。

    add_cols: 要额外读取的列名(如 BLH_m/TCWV_kgm2/Z_sfc_m);按列名定位,不写死列号。
    """
    files = sorted(match_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"{match_dir} 下没有 {pattern}")
    print(f"[DATA] 找到 {len(files)} 个匹配文件")
    Xs, Xe, ys, times, zsfc_l, src = [], [], [], [], [], []
    dem_m_l = []
    names_ref = None
    level_ref = None
    extra_names_ref = None
    meta_vertical = None
    n_extra_seen = 0
    for fp in files:
        with h5py.File(fp, "r") as f:
            if struct not in f:
                print(f"  [WARN] {fp.name}: 无 {struct},跳过")
                continue
            g = f[struct]
            D = np.asarray(g["Data"][()], dtype=np.float64)
            if D.ndim != 2:
                print(f"  [WARN] {fp.name}: Data 维度异常 {D.shape},跳过")
                continue
            names = decode_varnames(f, g["VarNames"]) if "VarNames" in g else None
            # 磁盘列主序 → 逻辑 (N, C);按"列数"判断(样本数可能少于列数)
            ncol = len(names) if names else None
            if ncol is not None:
                if D.shape[1] != ncol and D.shape[0] == ncol:
                    D = D.T
            elif D.shape[0] >= 132 and D.shape[1] < 132:
                D = D.T
            elif D.shape[0] < D.shape[1]:
                D = D.T
            if D.shape[1] < 132:
                print(f"  [WARN] {fp.name}: 列数 {D.shape[1]} < 132,跳过")
                continue
            if names is not None and names_ref is None:
                names_ref = names
            if level_ref is None and "Level" in g:
                level_ref = np.asarray(g["Level"][()]).ravel()
            if meta_vertical is None:
                meta_vertical = read_meta_vertical(g, f)
        base = D[:, :132]
        extra_idx, extra_names = [], []
        if names is not None:
            for c in add_cols:
                if c in names:
                    extra_idx.append(names.index(c)); extra_names.append(c)
        if extra_names and extra_names_ref is None:
            extra_names_ref = extra_names
        if extra_idx:
            Xe.append(D[:, extra_idx])
            n_extra_seen = len(extra_idx)
        # Z_sfc(地表高度)用于修正 L00 厚度
        if names is not None and CONFIG["ZSFC_COL"] in names:
            zsfc_l.append(D[:, names.index(CONFIG["ZSFC_COL"])])
        # agl 产物:DEM_m(逐组真实地表海拔,已 km→m) — 训练端 dz/GCF 用它替代 Z_sfc
        if names is not None and "DEM_m" in names:
            dem_m_l.append(D[:, names.index("DEM_m")])
        X_f, y_f, t_f = build_features(base, names)
        Xs.append(X_f); ys.append(y_f); times.append(t_f)
        src.append(fp.name)
        print(f"  {fp.name}: {D.shape[0]} 样本" + (f" (+{len(extra_idx)} 列)" if extra_idx else ""))
    if not Xs:
        raise RuntimeError("没有读到任何样本")
    X = np.vstack(Xs); y = np.vstack(ys); t = np.concatenate(times)
    X_extra = np.vstack(Xe).astype(np.float32) if Xe else None
    zsfc = np.concatenate(zsfc_l) if zsfc_l else None
    msg = f"[DATA] 合计 {X.shape[0]} 样本,X={X.shape},y={y.shape};目标有效率 {np.isfinite(y).mean()*100:.1f}%"
    if X_extra is not None:
        msg += f"\n[DATA] 额外列 {n_extra_seen} 个: {extra_names_ref}"
    if zsfc is not None:
        msg += f";Z_sfc 用于修正 L00 厚度(中位 {np.nanmedian(zsfc):.0f} m)"
    print(msg)
    return {"X": X, "X_extra": X_extra, "y": y, "time": t, "names": names_ref,
            "extra_names": extra_names_ref, "zsfc": zsfc, "level": level_ref, "files": src,
            "dem_m": (np.concatenate(dem_m_l) if dem_m_l else None),
            "vertical": meta_vertical}


def build_features(D: np.ndarray, names: list[str] | None):
    """按列构造 X(103 维)与 y(32 维),返回 (X, y, ERA5_Time)。"""
    n_lv = CONFIG["N_LEVEL"]
    lon = D[:, CONFIG["COL_LON"]]
    lat = D[:, CONFIG["COL_LAT"]]
    t_dn = D[:, CONFIG["COL_TIME"]]
    hour = D[:, CONFIG["COL_HOUR"]]
    h0 = CONFIG["COL_H0"]
    H = D[:, h0:h0 + n_lv]
    T = D[:, h0 + n_lv:h0 + 2 * n_lv]
    RH = D[:, h0 + 2 * n_lv:h0 + 3 * n_lv]
    ym0 = CONFIG["COL_YM0"]
    y = D[:, ym0:ym0 + n_lv]

    # 时间周期编码:用 ERA5_Time 的小数天 → 小时(避免把 datenum 当小时用的旧 bug)
    frac = t_dn - np.floor(t_dn)
    hour_f = np.round(frac * 24.0)
    hour_f[~np.isfinite(hour_f)] = hour[~np.isfinite(hour_f)]
    t_sin = np.sin(2 * np.pi * hour_f / 24.0)
    t_cos = np.cos(2 * np.pi * hour_f / 24.0)
    # 空间球面编码
    la = np.radians(lat); lo = np.radians(lon)
    sp_x, sp_y, sp_z = np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)
    X = np.column_stack([lon, lat, t_sin, t_cos, sp_x, sp_y, sp_z, H, T, RH])
    return X.astype(np.float32), y.astype(np.float32), t_dn


# ============================================================
# 目标构造:绝对消光 / 厚度加权逐层占比(形状)/ 柱含量
# ============================================================
def build_targets(y_abs: np.ndarray, X: np.ndarray, cfg: dict, zsfc=None) -> dict:
    """由绝对消光 y_abs(N,32) 与 X 中的 H 列,构造形状/柱含量目标。

    返回 dict:
      y_abs   绝对消光(原样)
      y_frac  (N,32) 厚度加权逐层占比 f_l = σ_lΔz_l / Σ(σΔz);无效层为 NaN
      y_od    (N,)   柱光学厚度 Σ σ_l Δz_l(km)(无量纲)
      gcf     (N,)   近地面占比 Σ_{底层 < GCF_TOP_M} f_l
      ok_frac (N,)   是否满足"最少层数 + 最小厚度覆盖"
      ok_od   (N,)   柱含量是否可用
      dz_km   (N,32) 各层厚度(km;L00 用接近似)
    """
    n_lv = cfg["N_LEVEL"]
    H = np.asarray(X[:, cfg["COL_H0"]:cfg["COL_H0"] + n_lv], dtype=np.float64)   # m
    dz = np.empty_like(H)
    if zsfc is not None:
        surf = np.asarray(zsfc, dtype=np.float64)                  # 地表高度(m,来自单层 geopotential)
        surf = np.where(np.isfinite(surf), surf, float(cfg.get("SURF_M", 0.0)))
    else:
        surf = np.full(H.shape[0], float(cfg.get("SURF_M", 0.0)))
    dz[:, 0] = H[:, 0] - surf                                   # L00:地表→H_k00
    dz[:, 1:] = np.diff(H, axis=1)
    dz = np.clip(dz, 0.0, None) / 1000.0                          # → km
    if cfg.get("DROP_L00"):
        dz[:, 0] = np.nan

    valid = np.isfinite(y_abs)
    with np.errstate(invalid="ignore"):
        contrib = np.where(valid, y_abs * dz, np.nan)             # 各层对柱光学厚度的贡献
        od = np.nansum(contrib, axis=1)                           # 柱光学厚度
        frac = contrib / od[:, None]                              # 逐层占比(和为 1)
        covered = np.nansum(np.where(valid, dz, np.nan), axis=1)  # 有效厚度
        total = np.nansum(dz, axis=1)                             # 整柱厚度
        cov = covered / total

    # 近地面占比 GCF:层底高度 < GCF_TOP_M 的层求和
    bottom = np.empty_like(H)
    bottom[:, 0] = surf if zsfc is not None else float(cfg.get("SURF_M", 0.0))
    bottom[:, 1:] = H[:, :-1]
    if str(cfg.get("VERTICAL", "asl")) == "agl" and zsfc is not None:
        # agl 口径:H 列 = dem+AGL 段顶 → 层底离地高 = bottom − dem;L00 底恒为 0
        low = (bottom - zsfc[:, None]) < float(cfg["GCF_TOP_M"])
    else:
        low = bottom < float(cfg["GCF_TOP_M"])
    # 近地面窗口内的层必须**全部有效**,否则 GCF 无法定义(不能把"缺数据"当成 0)
    low_complete = np.all(valid | ~low, axis=1)
    with np.errstate(invalid="ignore"):
        gcf_raw = np.nansum(np.where(low & valid, frac, np.nan), axis=1)
    gcf = np.where(low_complete & np.isfinite(od) & (od > 0), gcf_raw, np.nan)

    n_valid = valid.sum(axis=1)
    ok_frac = (n_valid >= int(cfg["REQ_MIN_LAYERS"])) & (cov >= float(cfg["REQ_COV"])) \
        & np.isfinite(od) & (od > 0)
    ok_od = np.isfinite(od) & (od > 0) & (n_valid >= 3)
    ok_gcf = low_complete & np.isfinite(gcf)
    if cfg.get("REQ_GCF_LAYERS"):        # 可选:形状目标也要求近地面完整(默认关闭)
        ok_frac = ok_frac & ok_gcf
    return {"y_abs": y_abs, "y_frac": frac.astype(np.float32), "y_od": od,
            "gcf": gcf, "ok_frac": ok_frac, "ok_od": ok_od, "ok_gcf": ok_gcf,
            "dz_km": dz.astype(np.float32), "low_mask": low}


def report_target_stats(T: dict, y_abs: np.ndarray):
    """打印目标可行性统计(决定这条路能不能走)。"""
    n = y_abs.shape[0]
    vp = np.isfinite(y_abs)
    print("\n[TARGET] 目标可行性统计")
    print(f"  样本 {n};每层有效数(前 8 层): " +
          " ".join(f"L{k:02d}={int(vp[:, k].sum())}" for k in range(min(8, vp.shape[1]))))
    print(f"  每层有效数(后 8 层): " +
          " ".join(f"L{k:02d}={int(vp[:, k].sum())}" for k in range(max(0, vp.shape[1]-8), vp.shape[1])))
    print(f"  形状目标可用样本: {int(T['ok_frac'].sum())}/{n} "
          f"({T['ok_frac'].mean()*100:.1f}%)   [需 ≥{CONFIG['REQ_MIN_LAYERS']} 层 且 厚度覆盖 ≥{CONFIG['REQ_COV']:.0%}]")
    print(f"  柱含量可用样本  : {int(T['ok_od'].sum())}/{n} ({T['ok_od'].mean()*100:.1f}%)")
    low_idx = np.flatnonzero(T["low_mask"][0]) if T["low_mask"].size else np.array([], dtype=int)
    print(f"  近地面窗口(层底 < {CONFIG['GCF_TOP_M']:.0f} m)包含层: "
          + ", ".join(f"L{i:02d}" for i in low_idx))
    print(f"  近地面完整样本  : {int(T['ok_gcf'].sum())}/{n} ({T['ok_gcf'].mean()*100:.1f}%)"
          f"   ← 这些才可用于 GCF 评估")
    ok = T["ok_gcf"] & np.isfinite(T["gcf"])
    if ok.any():
        q = np.nanpercentile(T["gcf"][ok], [5, 25, 50, 75, 95])
        print(f"  GCF(近地面 {CONFIG['GCF_TOP_M']:.0f} m 占比)分位数: "
              f"p5={q[0]:.3f} p25={q[1]:.3f} p50={q[2]:.3f} p75={q[3]:.3f} p95={q[4]:.3f}")
        print(f"  GCF 均值 {np.nanmean(T['gcf'][ok]):.3f} ± {np.nanstd(T['gcf'][ok]):.3f}"
              f"(仅统计近地面完整的样本)")
    if T["ok_od"].any():
        q = np.nanpercentile(T["y_od"][T["ok_od"]], [5, 50, 95])
        print(f"  柱光学厚度分位数: p5={q[0]:.3f} p50={q[1]:.3f} p95={q[2]:.3f}")


# ============================================================
# 模型(复用原 SpatioTemporalAttention,仅改输入/输出维度)
# ============================================================
class SpatioTemporalAttention32(nn.Module):
    """时间(2) / 空间(3) / 气象廓线(3×32=96) 三 token → Transformer → 注意力池化 → MLP → 32 层。"""

    def __init__(self, n_level: int = 32, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.15, n_extra: int = 0,
                 softplus: bool = False):
        super().__init__()
        self.n_level = n_level
        self.n_extra = int(n_extra)      # >0: 额外标量列(BLH/TCWV/...)单独作为一个 token
        self.softplus = bool(softplus)   # 非负输出(E2c):σ̂ 恒 ≥0;配合"关闭目标 z-score"使用
        n_meteo = 3 * n_level
        self.time_embed = nn.Sequential(nn.Linear(2, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.space_embed = nn.Sequential(nn.Linear(3, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.meteo_embed = nn.Sequential(nn.Linear(n_meteo, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.extra_embed = (nn.Sequential(nn.Linear(self.n_extra, d_model),
                                             nn.LayerNorm(d_model), nn.GELU())
                            if self.n_extra > 0 else None)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                         dim_feedforward=d_model * 4, dropout=dropout,
                                         activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.attn_pool = nn.Linear(d_model, 1)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, n_level),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        t_tok = self.time_embed(x[:, 2:4]).unsqueeze(1)
        s_tok = self.space_embed(x[:, 4:7]).unsqueeze(1)
        m_tok = self.meteo_embed(x[:, 7:103]).unsqueeze(1)   # 固定 96 维(H/T/RH),不含末尾额外列
        toks = [t_tok, s_tok, m_tok]
        if self.extra_embed is not None:
            e_tok = self.extra_embed(x[:, 103:103 + self.n_extra]).unsqueeze(1)
            toks.append(e_tok)
        tokens = torch.cat(toks, dim=1)
        enc = self.transformer(tokens)
        w = F.softmax(self.attn_pool(enc), dim=1)   # 注意力池化(含额外 token)
        pooled = (w * enc).sum(dim=1)
        out = self.decoder(pooled)                       # [B, 32] 占比 f_l 或绝对消光 σ_l
        return F.softplus(out) if self.softplus else out


# ============================================================
# 标准化 / 损失 / 指标
# ============================================================
def fit_target_scaler(y_tr: np.ndarray):
    """逐层标准化统计量:忽略 NaN;某层全缺 → mean=0,std=1(该层后面会被掩膜)。"""
    mu = np.nanmean(np.where(np.isfinite(y_tr), y_tr, np.nan), axis=0)
    sd = np.nanstd(np.where(np.isfinite(y_tr), y_tr, np.nan), axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
    return mu.astype(np.float32), sd.astype(np.float32)


def masked_huber(pred, target_z, mask, delta: float, w=None):
    """只对 mask=1 的层计 Huber;若整批无有效层 → 返回 0(不产生梯度)。

    w: 可选逐层高度权重 (32,) — loss = Σ w·mask·Huber / Σ w·mask(《优化路线讨论.md》§2.5)。
       w=None → 全层等权(基线);w 含 0 层(硬截断)时该层零贡献,输出仍 32 层。
    """
    loss = F.huber_loss(pred, target_z, reduction="none", delta=delta)
    if w is not None:
        mask = mask * w                              # (…,32) 逐层权重广播
    m = mask.sum()
    if float(m) <= 0:
        return loss.sum() * 0.0
    return (loss * mask).sum() / m


def build_w_profile(cfg: dict):
    """由层底高度构造逐层权重 w_l(32,)或 None(none=等权)。

    层底高度:bot_0 = 0(近似地表),bot_k = 层顶 k-1;层顶取训练样本 H 列的有限均值。
    """
    kind = (cfg.get("W_PROFILE") or "none").lower()
    if kind == "none":
        return None
    h_top = np.asarray(cfg.get("LAYER_TOP_KM") or [], dtype=np.float64)
    if h_top.size != cfg["N_LEVEL"]:
        print(f"[WARN] W_PROFILE={kind}: 层高不可用({h_top.size} 层) → 回退等权")
        return None
    bot = np.empty_like(h_top)
    bot[0] = 0.0
    bot[1:] = h_top[:-1]
    if kind == "decay":
        tau = float(cfg.get("W_DECAY_TAU_KM", 3.0))
        w = np.exp(-np.maximum(bot, 0.0) / tau)
    elif kind.startswith("hard"):
        top_km = float(cfg.get("HARD_TOP_KM", 5.0))
        w = (bot < top_km).astype(np.float64)
        n_in = int((w > 0).sum())
        print(f"[W] 硬截断:层底 < {top_km:.0f} km 的 {n_in}/{cfg['N_LEVEL']} 层 w=1,其余 0"
              f"(仅训练加权,输出/评估仍 32 层)")
    else:
        raise ValueError(f"未知 W_PROFILE: {kind}")
    return w.astype(np.float32)


def reconstruct_abs(pred, od_true, dz_km, mode):
    """把模型输出还原成**绝对消光廓线**(km⁻¹),供逐层 R² 评估。

    abs  : σ̂_l = 预测值本身(已是 km⁻¹)
    frac : σ̂_l = f̂_l·OD_obs/Δz_l —— 模型只给形状,柱含量借自观测
           (所以 frac 的绝对消光 R² 是"形状能力"的度量,含了观测总谱的信息)

    Δz ≤ 0 或无效的层 → NaN(评估时自动被掩膜掉)。
    """
    dz = np.asarray(dz_km, dtype=np.float64)
    dz_ok = np.isfinite(dz) & (dz > 1e-6)
    if mode == "abs":
        return np.asarray(pred, dtype=np.float64)
    f = np.asarray(pred, dtype=np.float64)
    s = f.sum(axis=1, keepdims=True)
    good = np.isfinite(s) & (s > 1e-12)
    f = np.where(good, f / np.where(good, s, 1.0), np.nan)      # 归一到和为 1
    od_hat = np.asarray(od_true, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        sigma = f * od_hat[:, None] / np.where(dz_ok, dz, np.nan)
    return np.where(dz_ok, sigma, np.nan)


def per_layer_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """逐层 R²/RMSE/MAE(仅有限值)+ 全局指标。"""
    n_lv = y_true.shape[1]
    r2 = np.full(n_lv, np.nan); rmse = np.full(n_lv, np.nan); mae = np.full(n_lv, np.nan)
    for k in range(n_lv):
        m = np.isfinite(y_true[:, k]) & np.isfinite(y_pred[:, k])
        if m.sum() < 2 or np.nanstd(y_true[m, k]) < 1e-12:
            continue
        r2[k] = r2_score(y_true[m, k], y_pred[m, k])
        rmse[k] = np.sqrt(mean_squared_error(y_true[m, k], y_pred[m, k]))
        mae[k] = mean_absolute_error(y_true[m, k], y_pred[m, k])
    m_all = np.isfinite(y_true) & np.isfinite(y_pred)
    out = {"R2_per_layer": r2, "RMSE_per_layer": rmse, "MAE_per_layer": mae}
    if m_all.sum() > 1:
        out.update({
            "R2_global": float(r2_score(y_true[m_all], y_pred[m_all])),
            "RMSE_global": float(np.sqrt(mean_squared_error(y_true[m_all], y_pred[m_all]))),
            "MAE_global": float(mean_absolute_error(y_true[m_all], y_pred[m_all])),
            "N_valid": int(m_all.sum()), "N_test": int(y_true.shape[0]),
        })
    else:
        out.update({"R2_global": float("nan"), "RMSE_global": float("nan"),
                    "MAE_global": float("nan"), "N_valid": 0, "N_test": int(y_true.shape[0])})
    return out


# ============================================================
# 单折训练 + 评估
# ============================================================
def run_fold(X, T: dict, tr_idx, te_idx, fold_name: str, cfg: dict, verbose: bool = True):
    """单折训练+评估。T 为 build_targets() 的产物;按 cfg['TARGET'] 选择目标。"""
    torch.manual_seed(cfg["SEED"]); np.random.seed(cfg["SEED"])
    dev = torch.device(cfg["DEVICE"])
    mode = cfg["TARGET"]

    # --- 特征标准化(只用训练集) ---
    sx = StandardScaler().fit(X[tr_idx])
    Xtr = sx.transform(X[tr_idx]).astype(np.float32)
    Xte = sx.transform(X[te_idx]).astype(np.float32)

    # --- 形状 / 绝对目标 ---
    Y = T["y_frac"] if mode == "frac" else T["y_abs"]
    log_t = bool(cfg.get("LOG_TARGET")) and mode == "abs"   # log 只对 abs 目标(frac 本身在 (0,1))
    soft = bool(cfg.get("SOFTPLUS"))
    if log_t:
        with np.errstate(invalid="ignore"):
            Y = np.log(np.clip(Y, float(cfg.get("LOG_EPS", 1e-6)), None))
        print(f"[TARGET] log 空间训练: y→ln(clip(y,{cfg.get('LOG_EPS', 1e-6)})),评估仍还原原空间")
    if mode == "abs":
        mtr = np.isfinite(Y[tr_idx]); mte = np.isfinite(Y[te_idx])
    else:                      # 形状:仅用"满足完整性"的样本,且逐层掩膜
        mtr = np.isfinite(Y[tr_idx]) & T["ok_frac"][tr_idx][:, None]
        mte = np.isfinite(Y[te_idx]) & T["ok_frac"][te_idx][:, None]
    # 逐层标准化统计量(形状模式:仅统计掩膜内的有效值)
    # softplus(E2c):输出恒 ≥0 → 不能拟合可为负的 z-score → 关闭标准化,原空间回归
    if soft:
        mu = np.zeros(cfg["N_LEVEL"], dtype=np.float32); sd = np.ones(cfg["N_LEVEL"], dtype=np.float32)
    else:
        mu, sd = fit_target_scaler(np.where(mtr, Y[tr_idx], np.nan) if mode != "abs" else Y[tr_idx])
    Ytr_z = np.where(mtr, (Y[tr_idx] - mu) / sd, 0.0).astype(np.float32)
    Yte_z = np.where(mte, (Y[te_idx] - mu) / sd, 0.0).astype(np.float32)

    w_np = build_w_profile(cfg)
    w_t = torch.from_numpy(w_np).to(dev) if w_np is not None else None
    if w_np is not None and (cfg.get("W_PROFILE") or "").startswith("decay"):
        print(f"[W] 低层加权 decay: w=exp(-层底/τ), τ={cfg.get('W_DECAY_TAU_KM', 3.0)} km;"
              f" w(L00)={w_np[0]:.2f} w(≈5km)={w_np[min(17, cfg['N_LEVEL']-1)]:.2f} w(≈10km)={w_np[min(23, cfg['N_LEVEL']-1)]:.2f}")

    model = SpatioTemporalAttention32(
        n_level=cfg["N_LEVEL"], d_model=cfg["D_MODEL"], n_heads=cfg["N_HEADS"],
        n_layers=cfg["N_LAYERS"], dropout=cfg["DROPOUT"],
        n_extra=cfg.get("N_EXTRA", 0), softplus=soft).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["EPOCHS"])

    Xtr_t = torch.from_numpy(Xtr)
    Ytr_t = torch.from_numpy(Ytr_z); Mtr_t = torch.from_numpy(mtr.astype(np.float32))
    Xte_t = torch.from_numpy(Xte); Yte_t = torch.from_numpy(Yte_z)
    Mte_t = torch.from_numpy(mte.astype(np.float32))

    n = Xtr_t.shape[0]
    best = {"loss": float("inf"), "state": None, "epoch": -1}
    wait = 0
    for ep in range(1, cfg["EPOCHS"] + 1):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0; nb = 0
        for i in range(0, n, cfg["BATCH"]):
            idx = perm[i:i + cfg["BATCH"]]
            opt.zero_grad()
            loss = masked_huber(model(Xtr_t[idx]), Ytr_t[idx], Mtr_t[idx], cfg["HUBER_DELTA"], w=w_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["GRAD_CLIP"])
            opt.step()
            tot += float(loss.detach()); nb += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            vloss = float(masked_huber(model(Xte_t), Yte_t, Mte_t, cfg["HUBER_DELTA"], w=w_t))
        if vloss < best["loss"] - 1e-6:
            best = {"loss": vloss, "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                    "epoch": ep}
            wait = 0
        else:
            wait += 1
        if verbose and (ep == 1 or ep % 10 == 0 or wait == 0):
            print(f"    [{fold_name}] ep{ep:3d} train={tot/max(nb,1):.4f} val={vloss:.4f}"
                  f"{'  *best' if wait == 0 else ''}")
        if wait >= cfg["PATIENCE"]:
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        pred_z = model(Xte_t).cpu().numpy()
    if soft:                                                  # softplus:输出已是原空间 σ̂ ≥ 0
        pred = pred_z
    else:
        pred = pred_z * sd + mu                               # 目标单位:占比 f_l 或 km⁻¹ / ln(km⁻¹)
        if log_t:                                             # log 目标:还原回 σ̂(km⁻¹)
            with np.errstate(over="ignore", invalid="ignore"):
                pred = np.exp(np.clip(pred, None, 20.0))      # clip 防溢出(exp(20)≈4.9e8)
    dz_te = T["dz_km"][te_idx]

    # --- 还原成绝对消光廓线 → 这是唯一对外报告的逐层指标 ---
    sigma_hat = reconstruct_abs(pred, T["y_od"][te_idx], dz_te, mode)
    met = per_layer_metrics(T["y_abs"][te_idx], sigma_hat)
    met.update({"fold": fold_name, "best_epoch": best["epoch"], "val_loss": best["loss"],
                "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
                "target": mode, "METRIC": "abs_extinction (km^-1)",
                "recon": ("obs_od" if mode == "frac" else "identity")})
    met["TARGET_KIND"] = "frac" if mode == "frac" else "abs"
    # 实验配置来源(Step 2/3 消融需要逐项追溯)
    met["log_target"] = bool(log_t)
    met["softplus"] = bool(soft)
    met["w_profile"] = cfg.get("W_PROFILE", "none")

    # --- GCF:直接由预测廓线算(比值,柱含量的量纲自动抵消)---
    #   真值端筛选:`T["gcf"]` 只在"近地面窗口各层全有效"时才存在(缺测不能当 0)。
    #   模型端还要再对齐一次:frac 模式训练时只用了 ok_frac 的样本,其余样本对梯度零贡献;
    #   拿没训过的样本评 GCF 会混进另一个分布(实测其 GCF 真值均值是合格样本的 3 倍),
    #   所以主口径限定在 ok_frac 内,全口径另存一份供对照。
    low = T["low_mask"][te_idx]
    with np.errstate(invalid="ignore"):
        contrib_hat = np.where(np.isfinite(sigma_hat), sigma_hat * dz_te, np.nan)
        tot_hat = np.nansum(contrib_hat, axis=1)
        gcf_pred = np.nansum(np.where(low, contrib_hat / tot_hat[:, None], np.nan), axis=1)
    gcf_true = T["gcf"][te_idx]
    ok_all = np.isfinite(gcf_pred) & np.isfinite(gcf_true)           # 全部可评样本
    trained = (T["ok_frac"][te_idx] if mode == "frac"
               else np.ones(ok_all.shape, dtype=bool))               # abs 模式全部样本都参与了训练
    ok_g = ok_all & trained                                          # ← 主口径

    met["n_gcf"] = int(ok_g.sum())
    met["n_gcf_all"] = int(ok_all.sum())
    for suffix, m in (("", ok_g), ("_all", ok_all)):
        if m.sum() >= 2 and np.nanstd(gcf_true[m]) > 1e-12:
            met[f"GCF_R2{suffix}"] = float(r2_score(gcf_true[m], gcf_pred[m]))
            met[f"GCF_MAE{suffix}"] = float(mean_absolute_error(gcf_true[m], gcf_pred[m]))

    # 下划线开头 → 只在本进程内传递(落盘给绘图 / 存模型),不写入 JSON
    met["_gcf_true"] = gcf_true[ok_g]
    met["_gcf_pred"] = gcf_pred[ok_g]
    met["_gcf_true_out"] = gcf_true[ok_all & ~trained]      # 可评但没训过的那批(散点图画灰)
    met["_gcf_pred_out"] = gcf_pred[ok_all & ~trained]
    # 完整预测/真值廓线 + 测试样本坐标 → 落盘后可用 Plot_Training_Results.py 任意重画
    met["_y_true"] = np.asarray(T["y_abs"][te_idx], dtype=np.float32)
    met["_y_pred"] = np.asarray(sigma_hat, dtype=np.float32)
    met["_lon"] = np.asarray(X[te_idx, cfg["COL_LON"]], dtype=np.float64)
    met["_lat"] = np.asarray(X[te_idx, cfg["COL_LAT"]], dtype=np.float64)
    met["_time"] = np.asarray(X[te_idx, cfg["COL_TIME"]], dtype=np.float64)
    # 层厚/近地面掩膜/柱有效标记 → 评估台统一算 OD 相对误差 + Gfrac(EE) 用
    met["_dz"] = np.asarray(dz_te, dtype=np.float32)
    met["_low"] = np.asarray(low, dtype=bool)
    met["_ok_od"] = np.asarray(T["ok_od"][te_idx], dtype=bool)

    # --- 柱含量(仅存档,不打印)---
    #   abs 模式:由预测廓线积分得到;frac 模式没有含量信息 → 不报
    od_true = T["y_od"][te_idx]
    if met["TARGET_KIND"] == "abs":
        od_pred = np.nansum(np.where(np.isfinite(sigma_hat), sigma_hat * dz_te, np.nan), axis=1)
    else:
        od_pred = None
    if od_pred is not None:
        ok_o = np.isfinite(od_pred) & T["ok_od"][te_idx]
        if ok_o.sum() >= 2 and np.nanstd(od_true[ok_o]) > 1e-12:
            met["OD_R2"] = float(r2_score(od_true[ok_o], od_pred[ok_o]))
            met["OD_MAE"] = float(mean_absolute_error(od_true[ok_o], od_pred[ok_o]))
    if verbose:
        extra = ""
        if "GCF_R2" in met:
            extra += f"  GCF R²={met['GCF_R2']:.3f}(MAE={met['GCF_MAE']:.3f}, n={met['n_gcf']})"
            if met.get("n_gcf_all", 0) > met["n_gcf"]:
                allr = met.get("GCF_R2_all")
                extra += (f" [全口径 n={met['n_gcf_all']}"
                          + (f", R²={allr:.3f}" if allr is not None else "") + "]")
        print(f"    [{fold_name}] 绝对消光: R²={met['R2_global']:.4f} "
              f"RMSE={met['RMSE_global']:.4f} MAE={met['MAE_global']:.4f}{extra} (best ep{best['epoch']})")
    return met, model, sx, (mu, sd)


# ============================================================
# 两种交叉验证
# ============================================================
def spatial_block_cv(X, T, cfg):
    """3×3 经纬度块留一。块边界按 ERA5 格点经纬度划分 → 同格点样本不跨折。"""
    print(f"\n[CV-空间块] {cfg['SPATIAL_BLOCKS']}×{cfg['SPATIAL_BLOCKS']}")
    B = cfg["SPATIAL_BLOCKS"]
    lon, lat = X[:, 0], X[:, 1]
    lon_edges = np.linspace(lon.min() - 1e-6, lon.max() + 1e-6, B + 1)
    lat_edges = np.linspace(lat.min() - 1e-6, lat.max() + 1e-6, B + 1)
    bi = (np.digitize(lon, lon_edges) - 1) * B + (np.digitize(lat, lat_edges) - 1)
    res = []
    for blk in np.unique(bi):
        te = np.where(bi == blk)[0]
        tr = np.where(bi != blk)[0]
        if len(te) < cfg["MIN_TEST"] or len(tr) < cfg["MIN_TRAIN"]:
            print(f"  块 {blk}: 样本不足(te={len(te)},tr={len(tr)}),跳过")
            continue
        i_lon, i_lat = min(blk // B, B - 1), min(blk % B, B - 1)   # 兜底:防非法块号越界
        met, *_ = run_fold(X, T, tr, te, f"Spatial({i_lon+1},{i_lat+1})", cfg)
        met["lon_range"] = (float(lon_edges[i_lon]), float(lon_edges[i_lon + 1]))
        met["lat_range"] = (float(lat_edges[i_lat]), float(lat_edges[i_lat + 1]))
        res.append(met)
    return res


def temporal_block_cv(X, T, time_dn, cfg):
    """按 ERA5_Time 排序后切成连续块(时间分块),避免同(格点,时次)跨折。"""
    print(f"\n[CV-时间块] {cfg['TIME_FOLDS']} 折(按 ERA5_Time 排序切块)")
    order = np.argsort(time_dn, kind="stable")
    n = len(order)
    size = n // cfg["TIME_FOLDS"]
    res = []
    for k in range(cfg["TIME_FOLDS"]):
        s = k * size
        e = (k + 1) * size if k < cfg["TIME_FOLDS"] - 1 else n
        te = order[s:e]
        tr = np.concatenate([order[:s], order[e:]])
        if len(te) < cfg["MIN_TEST"] or len(tr) < cfg["MIN_TRAIN"]:
            print(f"  时间折 {k+1}: 样本不足,跳过")
            continue
        met, *_ = run_fold(X, T, tr, te, f"Temporal-Fold{k+1}", cfg)
        met["time_range_datenum"] = (float(np.min(time_dn[te])), float(np.max(time_dn[te])))
        res.append(met)
    return res


# ============================================================
# 层几何(高度/气压)+ 逐层 R² 出图
# ============================================================
def layer_geometry(X: np.ndarray, level_hpa: np.ndarray | None,
                   cfg: dict | None = None, zsfc: np.ndarray | None = None):
    """返回 (每段平均高度 km, 代表气压 hPa, 每段平均离地高度 km 或 None)。

    高度取样本 H_k* 的均值;气压取该段上下层的均值;
    agl 口径下 h_agl_km = mean(H_k − dem)/1000(评估台 AGL 分带用)。
    """
    n_lv = CONFIG["N_LEVEL"]
    h_m = X[:, CONFIG["COL_H0"]:CONFIG["COL_H0"] + n_lv].astype(np.float64)
    with np.errstate(invalid="ignore"):
        h_km = np.nanmean(h_m, axis=0) / 1000.0
    h_agl_km = None
    if (cfg is not None and str(cfg.get("VERTICAL", "asl")) == "agl"
            and zsfc is not None and np.isfinite(zsfc).all()):
        with np.errstate(invalid="ignore"):
            h_agl_km = (np.nanmean(h_m - zsfc[:, None], axis=0)) / 1000.0
    p_seg = None
    if level_hpa is not None and len(level_hpa) == n_lv:
        lv = np.asarray(level_hpa, dtype=np.float64)      # hPa 降序 1000→10
        p_seg = np.empty(n_lv)
        p_seg[0] = lv[0]
        p_seg[1:] = 0.5 * (lv[:-1] + lv[1:])
    return h_km, p_seg, h_agl_km


# ============================================================
# 汇总输出
# ============================================================
def save_predictions(results: list[dict], tag: str, out_dir: Path, geom=None) -> Path | None:
    """把绘图所需的原始数组落盘(一个自包含的 npz),供事后重画而不必重训。

    内容:逐折的逐层 R²/RMSE/MAE、GCF 观测/预测(含被排除的那批)、完整预测/真值廓线、
          测试样本经纬度与时间、层几何(高度/气压)。
    """
    def cat(key):
        return np.concatenate([np.asarray(r[key], dtype=np.float32) for r in results
                               if key in r and np.size(r[key])]) \
            if any(key in r and np.size(r[key]) for r in results) else np.array([], dtype=np.float32)

    payload = {
        "fold_names": np.array([str(r.get("fold", f"Fold{i+1}")) for i, r in enumerate(results)]),
        "r2_per_layer": np.vstack([np.asarray(r["R2_per_layer"], dtype=np.float32) for r in results]),
        "rmse_per_layer": np.vstack([np.asarray(r["RMSE_per_layer"], dtype=np.float32) for r in results]),
        "mae_per_layer": np.vstack([np.asarray(r["MAE_per_layer"], dtype=np.float32) for r in results]),
        "gcf_true": cat("_gcf_true"), "gcf_pred": cat("_gcf_pred"),
        "gcf_true_out": cat("_gcf_true_out"), "gcf_pred_out": cat("_gcf_pred_out"),
        "y_true": cat("_y_true"), "y_pred": cat("_y_pred"),
        "dz_km": cat("_dz"), "ok_od": cat("_ok_od"),
        "lon": cat("_lon"), "lat": cat("_lat"), "time": cat("_time"),
        "h_km": (np.asarray(geom[0], dtype=np.float32) if geom and geom[0] is not None else np.array([])),
        "p_hpa": (np.asarray(geom[1], dtype=np.float32) if geom and len(geom) > 1 and geom[1] is not None else np.array([])),
        "h_agl_km": (np.asarray(geom[2], dtype=np.float32) if geom and len(geom) > 2 and geom[2] is not None else np.array([])),
    }
    p = Path(out_dir) / "results" / f"predictions_{tag}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **payload)
    print(f"  预测值已落盘: {p}  ({p.stat().st_size/1e6:.1f} MB)"
          f"  → 可用 Plot_Training_Results.py 重画")
    return p


def summarize(results: list[dict], tag: str, out_dir: Path, geom=None, plots: bool = True):
    if not results:
        print(f"[WARN] {tag}: 没有有效折"); return None
    r2g = np.array([r["R2_global"] for r in results], dtype=float)
    rmg = np.array([r["RMSE_global"] for r in results], dtype=float)
    mag = np.array([r["MAE_global"] for r in results], dtype=float)
    per_layer = np.vstack([r["R2_per_layer"] for r in results])
    print(f"\n[{tag}] 折数 {len(results)}   —— 逐层指标 = 绝对消光 σ(km⁻¹)")
    print(f"  全局 R²   : {np.nanmean(r2g):.4f} ± {np.nanstd(r2g):.4f}")
    print(f"  全局 RMSE : {np.nanmean(rmg):.4f} ± {np.nanstd(rmg):.4f}")
    print(f"  全局 MAE  : {np.nanmean(mag):.4f} ± {np.nanstd(mag):.4f}")
    gcf_r2 = np.array([r["GCF_R2"] for r in results if "GCF_R2" in r], dtype=float)
    gcf_mae = np.array([r["GCF_MAE"] for r in results if "GCF_MAE" in r], dtype=float)
    gcf_r2_all = np.array([r["GCF_R2_all"] for r in results if "GCF_R2_all" in r], dtype=float)
    if gcf_r2.size:
        print(f"  GCF  R²   : {np.nanmean(gcf_r2):.4f} ± {np.nanstd(gcf_r2):.4f}"
              f"   MAE {np.nanmean(gcf_mae):.4f}   (n={sum(int(r.get('n_gcf', 0)) for r in results)})"
              "   ← 只用训练所涉样本")
        n_all = sum(int(r.get("n_gcf_all", 0)) for r in results)
        if gcf_r2_all.size and n_all > sum(int(r.get("n_gcf", 0)) for r in results):
            print(f"  GCF  R² 全口径: {np.nanmean(gcf_r2_all):.4f} ± {np.nanstd(gcf_r2_all):.4f}"
                  f"   (n={n_all})   ← 含未参与训练的样本,偏低且不可解释")
    else:
        print("  GCF  R²   : nan(近地面完整样本不足)")
    print("  逐层 R²(绝对消光;L00=最低层 → L31=最高层):")
    mean_r2 = np.nanmean(per_layer, axis=0)
    for i in range(0, per_layer.shape[1], 8):
        row = mean_r2[i:i + 8]
        print("    " + "  ".join(
            f"L{i+j:02d}={v:6.3f}" if np.isfinite(v) else f"L{i+j:02d}=   nan"
            for j, v in enumerate(row)))
    # 逐层指标 CSV(绝对消光)
    res_dir = out_dir / "results"; res_dir.mkdir(parents=True, exist_ok=True)
    nl = per_layer.shape[1]
    with open(res_dir / f"per_layer_metrics_{tag}.csv", "w", encoding="utf-8") as fh:
        fh.write("Layer,LayerName,R2_mean,R2_std,RMSE_mean,MAE_mean,Target\n")
        for k in range(nl):
            r2k = per_layer[:, k]
            rmk = np.array([r["RMSE_per_layer"][k] for r in results]); mak = np.array([r["MAE_per_layer"][k] for r in results])
            fh.write(f"{k},L{k:02d},{np.nanmean(r2k):.6f},{np.nanstd(r2k):.6f},"
                     f"{np.nanmean(rmk):.6f},{np.nanmean(mak):.6f},abs_extinction\n")
    # 逐折摘要(下划线开头的键是绘图用的大数组,不入档)
    with open(res_dir / f"folds_{tag}.json", "w", encoding="utf-8") as fh:
        json.dump([{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in r.items() if not k.startswith("_")} for r in results],
                  fh, ensure_ascii=False, indent=2)
    # 原始预测数组落盘 → 图可以事后重画,不必重训
    save_predictions(results, tag, out_dir, geom=geom)

    if not plots:
        print(f"  (--no-plots:跳过出图;以后可运行 "
              f"python Plot_Training_Results.py --run-dir {out_dir} --tags {tag})")
        return {"R2_global_mean": float(np.nanmean(r2g)), "R2_global_std": float(np.nanstd(r2g)),
                "RMSE_global_mean": float(np.nanmean(rmg)), "MAE_global_mean": float(np.nanmean(mag)),
                "GCF_R2_mean": float(np.nanmean(gcf_r2)) if gcf_r2.size else float("nan"),
                "GCF_MAE_mean": float(np.nanmean(gcf_mae)) if gcf_mae.size else float("nan"),
                "n_folds": len(results), "r2_mean": mean_r2}

    # 图 1:y=高度(km), x=R²
    plot_dir = out_dir / "plots"
    curves = {}
    if 1 < len(results) <= 4:                     # 折数少时把每折也画出来
        for i, r in enumerate(results):
            curves[r.get("fold", f"Fold{i+1}")] = np.asarray(r["R2_per_layer"], dtype=float)
    curves[f"{tag} (mean{' over folds' if len(results) > 1 else ''})"] = mean_r2
    plot_per_layer_r2(curves, plot_dir / f"per_layer_r2_{tag}", geom=geom,
                      title=f"Per-layer R² of absolute extinction — {tag}")
    # 图 2:GCF 观测 vs 预测(各折样本合并;灰点 = 可评但未参与训练)
    def _cat(key):
        return np.concatenate([np.asarray(r.get(key, []), dtype=float) for r in results]) \
            if any(key in r for r in results) else np.array([])
    gt, gp = _cat("_gcf_true"), _cat("_gcf_pred")
    ot, op = _cat("_gcf_true_out"), _cat("_gcf_pred_out")
    plot_gcf_scatter(gt, gp, plot_dir / f"gcf_scatter_{tag}", title=f"GCF — {tag}",
                     met={"GCF_R2": float(np.nanmean(gcf_r2)) if gcf_r2.size else None,
                          "GCF_MAE": float(np.nanmean(gcf_mae)) if gcf_mae.size else None,
                          "GCF_R2_all": float(np.nanmean(gcf_r2_all)) if gcf_r2_all.size else None,
                          "n_gcf": int(gt.size),
                          "n_gcf_all": int(gt.size + ot.size)},
                     out_true=ot, out_pred=op)
    return {"R2_global_mean": float(np.nanmean(r2g)), "R2_global_std": float(np.nanstd(r2g)),
            "RMSE_global_mean": float(np.nanmean(rmg)), "MAE_global_mean": float(np.nanmean(mag)),
            "GCF_R2_mean": float(np.nanmean(gcf_r2)) if gcf_r2.size else float("nan"),
            "GCF_MAE_mean": float(np.nanmean(gcf_mae)) if gcf_mae.size else float("nan"),
            "n_folds": len(results), "r2_mean": mean_r2}


def train_final_model(X, T, cfg, out_dir: Path, geom=None):
    """用全部数据训练并保存(含 scaler),供后续推理。注意:其指标是训练集自评。"""
    print("\n[FINAL] 用全部数据训练最终模型 ...")
    tr = np.arange(X.shape[0])
    met, model, sx, (mu, sd) = run_fold(X, T, tr, tr, "Full-data", cfg, verbose=True)
    met["fold"] = "FullData (train-set self-eval)"
    summarize([met], "FullData", out_dir, geom=geom)      # 打印 32 层 R² 并存 CSV/图
    torch.save({"state_dict": model.state_dict(),
                "scaler_x_mean": sx.mean_.astype(np.float32), "scaler_x_scale": sx.scale_.astype(np.float32),
                "target_mu": mu, "target_sd": sd,
                "config": cfg, "n_level": cfg["N_LEVEL"]},
               out_dir / "model_final.pt")
    print(f"[FINAL] 已保存 {out_dir / 'model_final.pt'}(注意:该指标是在训练集自身上算的,不代表泛化)")
    return met


# ============================================================
# 主流程
# ============================================================
def run(args) -> int:
    cfg = dict(CONFIG)
    cfg.update({"MATCH_DIR": args.match_dir, "OUT_DIR": args.out_dir})
    if args.epochs: cfg["EPOCHS"] = args.epochs
    if args.batch: cfg["BATCH"] = args.batch
    if args.seed is not None: cfg["SEED"] = args.seed
    if getattr(args, "list_cols", False):
        fp = sorted(Path(args.match_dir).glob(cfg["MATCH_GLOB"]))[0]
        with h5py.File(fp, "r") as f:
            nm = decode_varnames(f, f[cfg["STRUCT"]]["VarNames"])
        print(f"{fp.name}: {len(nm)} 列")
        print(", ".join(nm))
        return 0
    if getattr(args, "add_cols", None):
        want = [c.strip() for c in args.add_cols.split(",") if c.strip()]
        cfg["ADD_COLS"] = [cfg["ADD_COL_ALIASES"].get(c, c) for c in want]
    if getattr(args, "target", None): cfg["TARGET"] = args.target
    if getattr(args, "req_cov", None) is not None: cfg["REQ_COV"] = args.req_cov
    if getattr(args, "req_min_layers", None) is not None: cfg["REQ_MIN_LAYERS"] = args.req_min_layers
    if getattr(args, "gcf_top_m", None) is not None: cfg["GCF_TOP_M"] = args.gcf_top_m
    if getattr(args, "drop_l00", False): cfg["DROP_L00"] = True
    if getattr(args, "req_gcf_layers", False): cfg["REQ_GCF_LAYERS"] = True
    # ---- 优化路线 Step 2/3 开关 ----
    cfg["W_PROFILE"] = getattr(args, "w_profile", None) or cfg.get("W_PROFILE", "none")
    cfg["LOG_TARGET"] = bool(getattr(args, "log_target", False))
    cfg["SOFTPLUS"] = bool(getattr(args, "softplus", False))
    if getattr(args, "w_decay_tau", None): cfg["W_DECAY_TAU_KM"] = args.w_decay_tau
    if getattr(args, "hard_top_km", None): cfg["HARD_TOP_KM"] = args.hard_top_km
    exp_tag = getattr(args, "tag", None)     # 实验后缀:输出文件名 → *_Holdout_<tag>.*
    out_dir = Path(cfg["OUT_DIR"]); out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("ACDL×ERA5 MatchV2 训练")
    print(f"  样本目录: {cfg['MATCH_DIR']}")
    print(f"  输出目录: {out_dir}")
    print(f"  设备: {cfg['DEVICE']}  epochs={cfg['EPOCHS']} batch={cfg['BATCH']} "
          f"lr={cfg['LR']} d_model={cfg['D_MODEL']}")
    print(f"  开关: target={cfg['TARGET']}  w_profile={cfg['W_PROFILE']}  "
          f"log_target={cfg['LOG_TARGET']}  softplus={cfg['SOFTPLUS']}  tag={exp_tag}")
    print("=" * 78)

    data = load_matched_dir(Path(cfg["MATCH_DIR"]), cfg["MATCH_GLOB"], cfg["STRUCT"],
                            add_cols=cfg["ADD_COLS"])
    X, y, t_dn = data["X"], data["y"], data["time"]
    X_extra, zsfc = data.get("X_extra"), data.get("zsfc")
    if args.limit_days and len(data["files"]) > args.limit_days:
        # 仅用于快速试跑:按文件顺序截断
        keep = int(len(X) * args.limit_days / len(data["files"]))
        X, y, t_dn = X[:keep], y[:keep], t_dn[:keep]
        if X_extra is not None: X_extra = X_extra[:keep]
        if zsfc is not None: zsfc = zsfc[:keep]
        print(f"[LIMIT] 仅用前 {args.limit_days} 天(约 {keep} 样本)")

    # --- 剔除 lon/lat/time 非有限的样本(否则空间分块/标准化会出 NaN) ---
    ok = np.isfinite(X[:, 0]) & np.isfinite(X[:, 1]) & np.isfinite(t_dn)
    if not ok.all():
        print(f"[WARN] 剔除 {int((~ok).sum())} 行(ERA5_Lon/Lat/Time 含 NaN)")
        X, y, t_dn = X[ok], y[ok], t_dn[ok]
        if X_extra is not None: X_extra = X_extra[ok]
        if zsfc is not None: zsfc = zsfc[ok]
    if X.shape[0] < 50:
        print("[ABORT] 有效样本过少")
        return 2
    print(f"[DATA] 参与训练 {X.shape[0]} 样本;目标有效率 {np.isfinite(y).mean()*100:.1f}%")

    # --- 目标构造(绝对消光 / 形状占比)---
    print(f"[TARGET] 模式 = {cfg['TARGET']}"
          + ("(绝对消光 σ_l, km⁻¹)" if cfg["TARGET"] == "abs"
             else "(厚度加权逐层占比 f_l,只学形状)"))
    # 垂直口径:agl 产物(H=dem+AGL 段顶) → dz/GCF 用 DEM_m 列;asl 沿用 Z_sfc(历史行为不变)
    vertical = str(data.get("vertical") or "asl")
    cfg["VERTICAL"] = vertical
    zsfc_eff = zsfc
    if vertical == "agl":
        dem_m = data.get("dem_m")
        if dem_m is not None:
            zsfc_eff = dem_m
            print(f"[VERTICAL] agl 产物:dz/GCF 使用 DEM_m(真实地表) 替代 Z_sfc;"
                  f"DEM_m 中位 {np.nanmedian(dem_m):.0f} m")
        else:
            print("[VERTICAL] agl 产物但无 DEM_m 列 → 退回 Z_sfc(结果口径注明)")
    if X_extra is not None:
        X = np.hstack([X, X_extra]).astype(np.float32)
        cfg["N_EXTRA"] = int(X_extra.shape[1])
        print(f"[COLS] 额外 token 维度 = {cfg['N_EXTRA']} ({data.get('extra_names')})")
    else:
        cfg["N_EXTRA"] = 0
    T = build_targets(y, X, cfg, zsfc=zsfc_eff)
    report_target_stats(T, y)

    # --- 模式解析:默认 = 一次随机留出训练(快);CV 与全量模型需显式开启 ---
    holdout = args.holdout if args.holdout is not None else cfg["HOLDOUT"]
    run_cv = (getattr(args, "cv", False) or cfg["RUN_CV"]) and not getattr(args, "no_cv", False)
    do_final = (getattr(args, "final", False) or cfg["TRAIN_FINAL"]) and not getattr(args, "no_final", False)
    suffix = f"_{exp_tag}" if exp_tag else ""    # 实验后缀 → 标签 Holdout<E|W…>
    # 出图开关:绘图代码在 acdl_plotting.py;关掉后仍会落盘 predictions_*.npz,可事后重画
    plots = bool(cfg.get("PLOTS", True)) and not getattr(args, "no_plots", False)
    if holdout <= 0 and not run_cv and not do_final:
        print("[INFO] 未指定 CV/全量模型 且留出比例=0 → 自动执行一次 20% 随机留出")
        holdout = 0.2

    # --- 层几何(平均高度/气压),用于出图与存档 ---
    geom = layer_geometry(X, data.get("level"), cfg=cfg, zsfc=zsfc_eff)
    cfg["LAYER_TOP_KM"] = (np.asarray(geom[0], dtype=float).tolist()
                           if geom[0] is not None else None)   # 高度加权/硬截断按层底高度算 w_l
    res_dir0 = out_dir / "results"; res_dir0.mkdir(parents=True, exist_ok=True)
    with open(res_dir0 / "layer_geometry.json", "w", encoding="utf-8") as fh:
        json.dump({"height_km": np.asarray(geom[0]).tolist(),
                   "pressure_hpa": None if geom[1] is None else np.asarray(geom[1]).tolist(),
                   "height_agl_km": (None if geom[2] is None else np.asarray(geom[2]).tolist()),
                   "vertical": vertical,
                   "note": "height = mean of H_k* over samples; agl = mean(H_k - DEM); "
                           "pressure = segment-mean of Level"},
                  fh, ensure_ascii=False, indent=2)

    summary = {}
    curves_all = {}
    if holdout > 0:
        rng = np.random.default_rng(cfg["SEED"])
        n = X.shape[0]
        idx = rng.permutation(n)
        n_te = max(int(n * holdout), 1)
        te, tr = np.sort(idx[:n_te]), np.sort(idx[n_te:])
        print(f"\n[HOLDOUT] 随机留出 {holdout:.0%}(train={len(tr)}, test={len(te)})"
              f" —— 注意:随机划分,同格点/同时次样本可能跨集,结果偏乐观")
        met, *_ = run_fold(X, T, tr, te, "Holdout", cfg)
        s = summarize([met], "Holdout" + suffix, out_dir, geom=geom, plots=plots)
        if s: curves_all["Holdout"] = s["r2_mean"]
        summary["holdout"] = {k: v for k, v in met.items() if not isinstance(v, np.ndarray)}
    if not run_cv and not do_final:
        print("\n[提示] 默认不跑空间/时间 CV 与全量模型;"
              "需要时加 --cv(空间+时间 CV)或 --final(全量模型,并存 model_final.pt)")

    if run_cv:
        t0 = time.perf_counter()
        sp = spatial_block_cv(X, T, cfg)
        s = summarize(sp, "Spatial" + suffix, out_dir, geom=geom, plots=plots)
        if s:
            curves_all["Spatial"] = s["r2_mean"]
            summary["spatial"] = {k: v for k, v in s.items() if k != "r2_mean"}
        tp = temporal_block_cv(X, T, t_dn, cfg)
        s = summarize(tp, "Temporal" + suffix, out_dir, geom=geom, plots=plots)
        if s:
            curves_all["Temporal"] = s["r2_mean"]
            summary["temporal"] = {k: v for k, v in s.items() if k != "r2_mean"}
        print(f"\n[CV] 用时 {time.perf_counter()-t0:.0f}s")

    # 多口径对比图(y=高度, x=R²)
    if plots and len(curves_all) >= 2:
        plot_per_layer_r2({k: np.asarray(v, dtype=float) for k, v in curves_all.items()},
                          out_dir / "plots" / "per_layer_r2_all",
                          geom=geom,
                          title="Per-layer R² of absolute extinction — all protocols")
    if do_final:
        train_final_model(X, T, cfg, out_dir, geom=geom)
    # 实验配置快照(追溯:每次运行把全部生效配置存入 cv_summary.json)
    summary["config"] = {k: (v if isinstance(v, (int, float, str, bool, list)) else str(v))
                         for k, v in cfg.items() if not k.startswith("_")}
    summary["config"]["exp_tag"] = exp_tag
    with open(out_dir / "results" / "cv_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print("\n[DONE]")
    return 0


# ============================================================
# 自检:合成小数据跑通全流程(不需真实数据)
# ============================================================
def selftest() -> int:
    import tempfile
    import hdf5storage
    print("[SELFTEST] 合成 MatchV2 数据 + 跑通 空间/时间 CV 与最终模型")
    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)
    n_level = 32
    names = (["ERA5_Lon", "ERA5_Lat", "ERA5_Time", "Hour"] +
             [f"H_k{k:02d}" for k in range(n_level)] +
             [f"T_k{k:02d}" for k in range(n_level)] +
             [f"RH_k{k:02d}" for k in range(n_level)] +
             [f"Ext_mean_L{k:02d}" for k in range(n_level)] +
             ["BLH_m", "TCWV_kgm2", "Z_sfc_m"])
    for d in ("20220601", "20220602", "20220603"):
        N = 120
        lon = rng.uniform(73, 135, N); lat = rng.uniform(20, 50, N)
        hour = rng.integers(0, 24, N)
        H = np.tile(np.linspace(100, 30000, n_level), (N, 1))
        T = 290 - 70 * (H / 30000) + rng.normal(0, 2, (N, n_level))
        RH = np.clip(60 + rng.normal(0, 10, (N, n_level)), 0, 100)
        y = np.abs(rng.normal(0.1, 0.05, (N, n_level)))
        y[rng.random((N, n_level)) < 0.15] = np.nan                # 掩膜
        BLH = 200 + 800 * rng.random(N)
        TCWV = 5 + 4 * rng.random(N)
        ZSFC = np.clip(50 + 900 * rng.random(N), 0, None)
        Data = np.column_stack([lon, lat, 738673 + hour / 24, hour, H, T, RH, y,
                                BLH, TCWV, ZSFC])
        hdf5storage.savemat(str(tmp / f"ACDL_ERA5_MatchV2_{d}.mat"),
                            {"MatchV2": {"Data": Data, "VarNames": names,
                                         "Level": np.arange(n_level, dtype=np.float32).reshape(-1, 1),
                                         "Meta": {"Output": "lite(132)"}}},
                            fmt="7.3", store_python_metadata=False, appendmat=False,
                            matlab_compatible=True, oned_as="column",
                            action_for_matlab_incompatible="error")
    class A:  # 简易参数对象(自检要覆盖全流程:留出 + CV + 全量模型)
        match_dir = str(tmp); out_dir = str(tmp / "out"); epochs = 3; batch = 64
        seed = 0; limit_days = None; holdout = 0.2
        cv = True; final = True; no_cv = False; no_final = False
        add_cols = "blh,tcwv,zsfc"; list_cols = False
    rc = run(A())
    ok = (tmp / "out" / "model_final.pt").exists() and \
         (tmp / "out" / "results" / "cv_summary.json").exists() and \
         (tmp / "out" / "results" / "per_layer_metrics_Spatial.csv").exists() and \
         (tmp / "out" / "results" / "per_layer_metrics_Temporal.csv").exists()
    print(f"[SELFTEST] 产物齐全: {ok}  (输出目录 {tmp/'out'})")
    print("[SELFTEST]", "PASS" if (rc == 0 and ok) else "FAIL")
    return 0 if (rc == 0 and ok) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="用 MatchV2 匹配样本训练 ERA5→ACDL 消光模型")
    ap.add_argument("--match-dir", default=CONFIG["MATCH_DIR"], help="匹配样本目录")
    ap.add_argument("--out-dir", default=CONFIG["OUT_DIR"], help="输出目录(模型/指标/图)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--limit-days", type=int, default=None, help="仅用前 N 天数据(快速试跑)")
    ap.add_argument("--add-cols", default=None,
                    help="额外列(逗号分隔;短名 blh,tcwv,zsfc 或精确列名),需先用 Add_ERA5_SingleLevel_Columns.py 追加")
    ap.add_argument("--list-cols", action="store_true", help="打印匹配文件的列名后退出")
    ap.add_argument("--target", choices=("abs", "frac"), default=None,
                    help="目标:abs=逐层绝对消光(默认)/ frac=厚度加权逐层占比(只学形状)")
    ap.add_argument("--req-cov", type=float, default=None, help="形状目标:最小有效厚度占比(默认 0.5)")
    ap.add_argument("--req-min-layers", type=int, default=None, help="形状目标:最少有效层数(默认 10)")
    ap.add_argument("--gcf-top-m", type=float, default=None, help="GCF 的近地面厚度(默认 500 m)")
    ap.add_argument("--drop-l00", action="store_true", help="形状目标排除 L00(地表层厚度是近似值)")
    ap.add_argument("--req-gcf-layers", action="store_true",
                    help="形状目标也要求近地面层完整(样本会进一步减少,但 GCF 语义更干净)")
    ap.add_argument("--holdout", type=float, default=None,
                    help="随机留出比例,默认 0.2(只跑一次训练就能看到逐层 R²;随机划分,结论偏乐观)")
    ap.add_argument("--cv", action="store_true",
                    help="跑空间块 + 时间块 CV(慢,默认关闭)")
    ap.add_argument("--final", action="store_true",
                    help="额外用全量数据训练并保存 model_final.pt(默认关闭)")
    ap.add_argument("--no-plots", action="store_true",
                    help="不出图(只落盘 predictions_*.npz);事后用 Plot_Training_Results.py 重画")
    # ---- 优化路线 Step 2/3 开关 ----
    ap.add_argument("--w-profile", choices=("none", "decay", "hard5km"), default=None,
                    help="损失高度权重: none=全层等权(基线) / decay=低层加权 exp(-层底/τ) / hard5km=仅层底<5km 计损失(输出仍32层)")
    ap.add_argument("--w-decay-tau", type=float, default=None, help="decay 权重的 τ(km),默认 3.0")
    ap.add_argument("--hard-top-km", type=float, default=None, help="hard 截断厚度(km),默认 5.0")
    ap.add_argument("--log-target", action="store_true",
                    help="E2b: 目标取 ln(clip(y,1e-6)) 后再标准化;评估自动还原原空间(仅 abs)")
    ap.add_argument("--softplus", action="store_true",
                    help="E2c: 解码器输出过 softplus(σ̂ 恒≥0);目标 z-score 自动关闭,原空间回归")
    ap.add_argument("--tag", default=None,
                    help="实验标签后缀:输出文件命名 <tag>_<后缀>(如 Holdout_E2b)")
    ap.add_argument("--no-cv", action="store_true", help="(兼容)显式关闭 CV")
    ap.add_argument("--no-final", action="store_true", help="(兼容)显式关闭全量模型")
    ap.add_argument("--selftest", action="store_true", help="合成数据自检")
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else run(a))
