"""
plot_node_validation.py — 单格点(0.25°)全时间廓线精度验证(空间块 CV 口径)
============================================================
选 3 个时次最多的 0.25° 格点(平原/盆地/低山各一,全月 n=3——轨道重访稀疏),
每个格点画:
  细线 = 该格点全部时次的逐次 ACDL 观测(蓝)/B0-AGL 预测(红)
  粗实线 = 逐层均值对比;虚线 = 逐层中位数对比
标注 n、DEM、均值廓线 r 与中位数廓线 r、均值 MAE。
预测来自空间块 CV(该格点所在的经纬度块从未参与训练)。

用法:python plot_node_validation.py
产物:结果汇总/figures/cv_node_validation.png(300dpi)
============================================================
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

from plot_cv_case_profiles import load_all, join_cv

RUN_CV = Path("结果汇总/runs/train_agl_b0_cv")
OUT = Path(__file__).resolve().parent.parent / "结果汇总" / "figures"
C_OBS, C_PRD = "#0C5DA5", "#C44E52"


def main() -> int:
    from acdl_plotting import setup_cjk
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_cjk()
    plt.rcParams.update({
        "font.size": 10.5, "axes.labelsize": 10.5, "axes.titlesize": 10.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "grid.alpha": 0.25, "grid.linestyle": "--", "legend.frameon": False,
    })
    OUT.mkdir(parents=True, exist_ok=True)

    data = load_all()
    npz = np.load(RUN_CV / "results" / "predictions_Spatial_AGLB0CV.npz")
    y, p, dem, H, idx = join_cv(npz, data)
    lon, lat = data["lon"][idx], data["lat"][idx]
    z_agl = (H - dem[:, None]) / 1000.0
    key = [(round(a, 3), round(b, 3)) for a, b in zip(lon, lat)]

    # 三个不同地形带的最密格点(全月 n=3 已是轨道重访上限)
    picks = [("平原 (125.25°E, 29.25°N)", (125.25, 29.25)),
             ("盆地 (112.00°E, 38.50°N)", (112.00, 38.50)),
             ("低山 (78.25°E, 18.50°N)", (78.25, 18.50))]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 6.2), sharey=True)
    for ax, (label, node) in zip(axes, picks):
        ids = np.flatnonzero([k == node for k in key])
        n = len(ids)
        o_t, p_t, z_t = y[ids], p[ids], z_agl[ids]
        # 逐时次细线
        for i in range(n):
            ax.plot(o_t[i], z_t[i], "-", lw=0.9, color=C_OBS, alpha=0.45, zorder=1)
            ax.plot(p_t[i], z_t[i], "-", lw=0.9, color=C_PRD, alpha=0.45, zorder=1)
        # 均值 / 中位数(逐层)
        mo, mp = np.nanmean(o_t, 0), np.nanmean(p_t, 0)
        meo, mep = np.nanmedian(o_t, 0), np.nanmedian(p_t, 0)
        zm = np.nanmean(z_t, 0)
        m_f = np.isfinite(mo) & np.isfinite(mp)
        r_mean = float(np.corrcoef(mo[m_f], mp[m_f])[0, 1]) if m_f.sum() >= 3 else np.nan
        m_f2 = np.isfinite(meo) & np.isfinite(mep)
        r_med = float(np.corrcoef(meo[m_f2], mep[m_f2])[0, 1]) if m_f2.sum() >= 3 else np.nan
        mae = float(np.nanmean(np.abs(mp - mo)))
        ax.plot(mo, zm, "-o", ms=4.5, lw=2.6, color=C_OBS, label="观测 均值", zorder=4)
        ax.plot(mp, zm, "-s", ms=4.0, lw=2.3, color=C_PRD, mfc="none", label="预测 均值", zorder=4)
        ax.plot(meo, zm, "--", lw=1.4, color=C_OBS, alpha=0.85, label="观测 中位数", zorder=3)
        ax.plot(mep, zm, "--", lw=1.4, color=C_PRD, alpha=0.85, label="预测 中位数", zorder=3)
        ax.set_xlim(0, max(np.nanmax(o_t), np.nanmax(p_t)) * 1.12)
        ax.set_xlabel(r"消光系数 $\sigma$ (km$^{-1}$)")
        ax.set_title(f"{label}\nn = {n} 个时次 · DEM ≈ {np.nanmean(dem[ids]):.0f} m", fontsize=10.5)
        ax.text(0.97, 0.97,
                (f"均值廓线 r = {r_mean:.2f}\n中位数廓线 r = {r_med:.2f}"
                 f"\n均值 MAE = {mae:.4f} km$^{{-1}}$"),
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.9))
        ax.grid(True)
        ax.legend(loc="lower right")
    axes[0].set_ylabel("离地高度 (km)")
    fig.suptitle("单格点(0.25°)全时间廓线精度验证 —— 空间块 CV 口径(该格点所在经纬度块从未参与训练;"
                 "细线=逐时次,粗线=均值,虚线=中位数;全月轨道重访稀疏,单格点至多 3 时次)", fontsize=11.5)
    fig.tight_layout()
    out = OUT / "cv_node_validation"
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  图: {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
