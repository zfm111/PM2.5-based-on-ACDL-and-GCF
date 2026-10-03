"""
plot_case_profiles.py — 个例廓线对比图:模型预测 vs ACDL 观测消光系数
============================================================
图 1 (agl_case_profiles.png):B0-AGL 按地形选 6 个留出集个例,
    每例画 ACDL 观测(实心) vs 预测(空心) 消光廓线,y 轴 = 离地高度(AGL, km)。
图 2 (agl_vs_asl_plateau.png):高原与盆地个例上,B0-ASL vs B0-AGL 预测对比
    (y 轴 = 海拔 ASL),直观展示 ASL 口径的近地面缺口在 AGL 下被补齐。

用同 seed 重放留出索引取个例(y_true 全等校验),不重训。
用法:python plot_case_profiles.py
产物:结果汇总/figures/{agl_case_profiles,agl_vs_asl_plateau}.png
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

import Train_ACDL_ERA5_MatchV2 as T2

ASL_DIR = "D:/matchdata_fixed_case5"
AGL_DIR = "D:/matchdata_agl_case5"
OUT = Path(__file__).resolve().parent.parent / "结果汇总" / "figures"
SEED, HOLDOUT = 42, 0.2


def test_rows(match_dir: str):
    """重放留出索引 → (y_true, y_pred, dem_m, H列, lon, lat);以 y_true 全等校验兜底。"""
    cfg = dict(T2.CONFIG)
    cfg["MATCH_DIR"] = match_dir
    d = T2.load_matched_dir(Path(match_dir), cfg["MATCH_GLOB"], cfg["STRUCT"],
                            add_cols=cfg["ADD_COLS"])
    X, y, t = d["X"], d["y"], d["time"]
    ok = np.isfinite(X[:, 0]) & np.isfinite(X[:, 1]) & np.isfinite(t)
    X, y = X[ok], y[ok]
    dem = (d["dem_m"] if d.get("dem_m") is not None else d["zsfc"])[ok]
    n = X.shape[0]
    n_te = max(int(n * HOLDOUT), 1)
    te = np.sort(np.random.default_rng(SEED).permutation(n)[:n_te])
    return X[te], y[te], np.asarray(dem)[te], n_te


def main() -> int:
    from acdl_plotting import setup_cjk
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_cjk()
    # ---- 出版级样式(纯呈现,不改数据) ----
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "grid.alpha": 0.25, "grid.linestyle": "--", "grid.linewidth": 0.5,
        "legend.frameon": False, "lines.solid_capstyle": "round",
    })
    C_OBS, C_PRD = "#0C5DA5", "#C44E52"                        # 色盲友好深蓝/砖红
    OUT.mkdir(parents=True, exist_ok=True)

    def _style(ax):
        ax.grid(True, which="major")
        ax.tick_params(direction="out", length=3)

    def _pair_stats(o, p, z):
        m = np.isfinite(o) & np.isfinite(p)
        if m.sum() < 3 or np.std(o[m]) < 1e-12:
            return None, int(m.sum())
        r = float(np.corrcoef(o[m], p[m])[0, 1])
        return r, int(m.sum())

    Xa, ya, dema, n_te = test_rows(AGL_DIR)
    npz_a = np.load(Path("结果汇总/runs/train_agl_b0/results/predictions_Holdout_AGLB0.npz"))
    npz_s = np.load(Path("结果汇总/runs/train_fixed_case5_abs/results/predictions_Holdout.npz"))
    assert np.allclose(ya, npz_a["y_true"], equal_nan=True, atol=1e-5), "AGL y_true 校验失败"
    Xs, ys, dems, n_te2 = test_rows(ASL_DIR)
    assert n_te2 == n_te and np.allclose(ys, npz_s["y_true"], equal_nan=True, atol=1e-5), "ASL y_true 校验失败"

    H_a = Xa[:, 7:39]   # X=标识7列+H32+T32+RH32;COL_H0=4 是产物列布局,勿用于 X
    agl_a = (H_a - dema[:, None]) / 1000.0                          # 每样本各层离地高度 km
    H_s = Xs[:, 7:39]
    asl_s = H_s / 1000.0
    od_a = np.nansum(np.where(np.isfinite(ya), ya * (np.diff(H_a, axis=1, prepend=dema[:, None]) / 1000.0), np.nan), axis=1)

    # ---- 图 1:AGL 六个地形个例 ----
    dem_band_cases = [
        ("平原清洁", (dema < 120), "min"),
        ("平原高消光(沙尘型)", (dema < 120), "max"),
        ("低山 0.3–0.9km", ((dema >= 300) & (dema < 900)), "max"),
        ("盆地 0.9–1.8km", ((dema >= 900) & (dema < 1800)), "max"),
        ("高原 1.8–2.6km", ((dema >= 1800) & (dema < 2600)), "max"),
        ("高原 >2.6km", (dema >= 2600), "max"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.6), sharey=True)
    for ax, (label, m, how) in zip(axes.ravel(), dem_band_cases):
        idx_pool = np.flatnonzero(m & np.isfinite(od_a))
        if idx_pool.size == 0:
            ax.set_title(f"{label}(无样本)"); _style(ax); continue
        pick = idx_pool[np.argmax(od_a[idx_pool])] if how == "max" else idx_pool[np.argmin(od_a[idx_pool])]
        z = agl_a[pick]
        o, p = ya[pick], npz_a["y_pred"][pick]
        r, npair = _pair_stats(o, p, z)
        hi = max(np.nanmax(o[np.isfinite(o)]), np.nanmax(p[np.isfinite(p)]))
        ax.plot(o, z, "-o", ms=3.5, lw=1.9, color=C_OBS, label="ACDL 观测", zorder=3)
        ax.plot(p, z, "-s", ms=3.2, lw=1.7, color=C_PRD, mfc="none", label="B0-AGL 预测", zorder=2)
        ax.set_xlim(0, hi * 1.12)
        ax.set_xlabel(r"消光系数 $\sigma$ (km$^{-1}$)")
        ax.set_title(f"{label}", fontsize=11)
        ax.text(0.97, 0.97, f"DEM≈{dema[pick]:.0f} m\nOD≈{od_a[pick]:.2f}\n"
                            + (f"r = {r:.2f}(n={npair}层)" if r is not None else ""),
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.85))
        _style(ax)
        ax.legend(loc="lower right", fontsize=8.5)
        ax.set_ylim(0, 12)
    axes[0, 0].set_ylabel("离地高度 (km)"); axes[1, 0].set_ylabel("离地高度 (km)")
    fig.suptitle("留出集个例:ACDL 观测 vs B0-AGL 预测消光廓线(y=离地高度;每带挑柱含量最大的沙尘型个例)", fontsize=12)
    fig.tight_layout()
    p1 = OUT / "agl_case_profiles"
    fig.savefig(p1.with_suffix(".png"), dpi=300); plt.close(fig)
    print(f"  图: {p1.with_suffix('.png')}")

    # ---- 图 2:高原/盆地个例 ASL vs AGL 预测(y=海拔) ----
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 6.4), sharey=True)
    for ax, (label, m, zmax) in zip(axes, [("高原 >2.6km", (dema >= 2600), 12.0),
                                           ("盆地 0.9–1.8km", ((dema >= 900) & (dema < 1800)), 12.0)]):
        pool = np.flatnonzero(m & np.isfinite(od_a))
        pick = pool[np.argmax(od_a[pool])]
        z_asl = asl_s[pick]; z_agl = agl_a[pick]
        o = ya[pick]
        o_s = npz_s["y_true"][pick]
        pa, ps = npz_a["y_pred"][pick], npz_s["y_pred"][pick]
        r_a, _ = _pair_stats(o, pa, z_agl)
        hi = np.nanmax(np.concatenate([o_s[np.isfinite(o_s)], ps[np.isfinite(ps)],
                                       pa[np.isfinite(pa)]]))
        ax.plot(o_s, z_asl, "-o", ms=3.5, lw=1.9, color=C_OBS, label="ACDL 观测", zorder=3)
        ax.plot(ps, z_asl, "-s", ms=3.0, lw=1.5, color="0.45", mfc="none",
                label="B0-ASL 预测(固定气压层)", zorder=2)
        ax.plot(pa, dema[pick] / 1000.0 + z_agl, "-s", ms=3.2, lw=1.7,
                color=C_PRD, mfc="none", label="B0-AGL 预测(地形跟随)", zorder=3)
        ax.axhline(dema[pick] / 1000.0, color="k", ls=":", lw=1.1)
        ax.text(0.02, dema[pick] / 1000.0 + 0.15, f"地表 DEM≈{dema[pick]:.0f} m",
                fontsize=8.5, transform=ax.get_yaxis_transform())
        ax.set_xlim(0, hi * 1.12)
        ax.set_xlabel(r"消光系数 $\sigma$ (km$^{-1}$)")
        ax.set_title(label, fontsize=11)
        ax.text(0.97, 0.97, f"OD≈{od_a[pick]:.2f}" + (f"\nAGL 廓线 r = {r_a:.2f}" if r_a is not None else ""),
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.85))
        _style(ax)
        ax.legend(loc="lower right", fontsize=8.5)
        ax.set_ylim(0, zmax)
    axes[0].set_ylabel("海拔 (km)")
    fig.suptitle("同一沙尘型个例(y=海拔):ASL 口径高原近地面无观测层(蓝线自 DEM 起,地下为无监督外推) vs AGL 补齐贴地层", fontsize=11)
    fig.tight_layout()
    p2 = OUT / "agl_vs_asl_plateau"
    fig.savefig(p2.with_suffix(".png"), dpi=300); plt.close(fig)
    print(f"  图: {p2.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
