"""
plot_cv_case_profiles.py — 交叉验证个例廓线对比(空间块 CV / 时间块 CV)
============================================================
与《agl_case_profiles.png》(留出版)同款的六地形带个例图,但预测来自 CV:
  - 空间块 CV:预测来自**从未训练过该经纬度块**的模型(12 块轮流留出,全域覆盖);
  - 时间块 CV:预测来自**从未见过该日期**的模型(4 折轮换)。
逐样本通过 (格点, 时次) 把 CV 池化预测接回匹配样本(y_true 全等校验),不重训。

用法:python plot_cv_case_profiles.py
产物:结果汇总/figures/cv_case_profiles_{spatial,temporal}.png(300dpi)
============================================================
"""
from __future__ import annotations

import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

import train_acdl_era5_match_v2 as T2

AGL_DIR = "D:/matchdata_agl_case5"
RUN_CV = Path("结果汇总/runs/train_agl_b0_cv")
OUT = Path(__file__).resolve().parent.parent / "结果汇总" / "figures"
C_OBS, C_PRD = "#0C5DA5", "#C44E52"


def load_all():
    """匹配数据全量 + (格点,时次) 键。"""
    cfg = dict(T2.CONFIG)
    d = T2.load_matched_dir(Path(AGL_DIR), cfg["MATCH_GLOB"], cfg["STRUCT"], add_cols=cfg["ADD_COLS"])
    X, y, t = d["X"], d["y"], d["time"]
    ok = np.isfinite(X[:, 0]) & np.isfinite(X[:, 1]) & np.isfinite(t)
    X, y, t = X[ok], y[ok], t[ok]
    dem = (d["dem_m"] if d.get("dem_m") is not None else d["zsfc"])[ok]
    lon, lat = X[:, 0], X[:, 1]
    key = [f"{a:.3f}|{b:.3f}" for a, b in zip(lon, lat)]   # 两段式;同时次用 y_true 指纹消歧
    return dict(X=X, y=y, dem=dem, lon=lon, lat=lat, key=key)


def join_cv(npz, data):
    """按 (格点, 真值指纹) 把 CV 池化预测接回样本;返回对齐后的 (y_true, y_pred, dem, H, idx)。

    npz 的 time 被存成 float32(datenum≈7.4e5 时误差达 ±0.03 天),时次键不可靠;
    改用 (lon,lat) 定位格点 + y_true 逐行 float32 全等(真值指纹)唯一配对。
    """
    from collections import defaultdict
    lut = defaultdict(list)
    for i, k in enumerate(data["key"]):
        lut[k].append(i)
    y32 = data["y"].astype(np.float32)
    idx = np.empty(len(npz["y_true"]), dtype=np.int64)
    miss = 0
    for i in range(len(npz["y_true"])):
        k = f"{npz['lon'][i]:.3f}|{npz['lat'][i]:.3f}"
        hit = -1
        for j in lut.get(k, ()):
            if np.array_equal(y32[j], npz["y_true"][i], equal_nan=True):
                hit = j
                break
        idx[i] = hit
        miss += (hit < 0)
    assert miss == 0, f"{miss} 行未能接回匹配样本"
    assert np.allclose(data["y"][idx], npz["y_true"], equal_nan=True, atol=1e-5), "y_true 校验失败"
    H = data["X"][idx][:, 7:39]
    return npz["y_true"], npz["y_pred"], data["dem"][idx], H, idx


def main() -> int:
    from acdl_plotting import setup_cjk
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_cjk()
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "grid.alpha": 0.25, "grid.linestyle": "--", "grid.linewidth": 0.5,
        "legend.frameon": False, "lines.solid_capstyle": "round",
    })
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_all()
    H = data["X"][:, 7:39]
    agl = (H - data["dem"][:, None]) / 1000.0
    od = np.nansum(np.where(np.isfinite(data["y"]),
                            data["y"] * (np.diff(H, axis=1, prepend=data["dem"][:, None]) / 1000.0),
                            np.nan), axis=1)
    dem = data["dem"]

    cases = [
        ("平原清洁", (dem < 120), "min"),
        ("平原高消光(沙尘型)", (dem < 120), "max"),
        ("低山 0.3–0.9km", ((dem >= 300) & (dem < 900)), "max"),
        ("盆地 0.9–1.8km", ((dem >= 900) & (dem < 1800)), "max"),
        ("高原 1.8–2.6km", ((dem >= 1800) & (dem < 2600)), "max"),
        ("高原 >2.6km", (dem >= 2600), "max"),
    ]

    for tag, note in [("Spatial_AGLB0CV", "空间块 CV:预测来自从未训练过该区域的模型(12 块轮流留出)"),
                      ("Temporal_AGLB0CV", "时间块 CV:预测来自从未见过该日期的模型(4 折轮换)")]:
        npz = np.load(RUN_CV / "results" / f"predictions_{tag}.npz")
        y_true, y_pred, dem_i, H_i, idx = join_cv(npz, data)
        z_agl = (H_i - dem_i[:, None]) / 1000.0

        fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.6), sharey=True)
        for ax, (label, m, how) in zip(axes.ravel(), cases):
            mm = m[idx]                                   # 该带内、且在 CV 测试覆盖中的样本
            pool = np.flatnonzero(mm & np.isfinite(od[idx]))
            if pool.size == 0:
                ax.set_title(f"{label}(无样本)"); ax.grid(True); continue
            pick = pool[np.argmax(od[idx][pool])] if how == "max" else pool[np.argmin(od[idx][pool])]
            o, p_, zz = y_true[pick], y_pred[pick], z_agl[pick]
            mm2 = np.isfinite(o) & np.isfinite(p_)
            r = float(np.corrcoef(o[mm2], p_[mm2])[0, 1]) if mm2.sum() >= 3 and np.std(o[mm2]) > 1e-12 else None
            hi = max(np.nanmax(o[np.isfinite(o)]), np.nanmax(p_[np.isfinite(p_)]))
            ax.plot(o, zz, "-o", ms=3.5, lw=1.9, color=C_OBS, label="ACDL 观测", zorder=3)
            ax.plot(p_, zz, "-s", ms=3.2, lw=1.7, color=C_PRD, mfc="none", label="B0-AGL 预测", zorder=2)
            ax.set_xlim(0, hi * 1.12)
            ax.set_xlabel(r"消光系数 $\sigma$ (km$^{-1}$)")
            ax.set_title(label, fontsize=11)
            ax.text(0.97, 0.97, f"DEM≈{dem_i[pick]:.0f} m\nOD≈{od[idx][pick]:.2f}\n"
                                + (f"r = {r:.2f}(n={mm2.sum()}层)" if r is not None else ""),
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.85))
            ax.grid(True)
            ax.legend(loc="lower right", fontsize=8.5)
            ax.set_ylim(0, 12)
        axes[0, 0].set_ylabel("离地高度 (km)"); axes[1, 0].set_ylabel("离地高度 (km)")
        for axx in axes.ravel():
            axx.set_xlabel(r"消光系数 $\sigma$ (km$^{-1}$)")
        fig.suptitle(f"交叉验证个例:ACDL 观测 vs B0-AGL 预测(y=离地高度;每带挑柱含量最大的沙尘型个例)\n{note}",
                     fontsize=12)
        fig.tight_layout()
        out = OUT / f"cv_case_profiles_{tag.split('_')[0].lower()}"
        fig.savefig(out.with_suffix(".png"), dpi=300)
        plt.close(fig)
        print(f"  图: {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
