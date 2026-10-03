"""
plot_node_profile_validation.py — 单格点廓线精度验证(平均 / 中位数两种口径)
============================================================
轨道限制:同一 0.25° 格点全月最多过境 3 次(28,866 格点几乎不重访)。
挑过境次数最多的两个格点,对每个格点画:
  平均口径   = 该格点全部过境时次的逐层 mean(观测 vs 预测)
  中位数口径 = 逐层 median(观测 vs 预测)
并叠上全部原始过境散点(灰点)。预测取空间块 CV 池化结果(区域外,最严格)。
用法:python plot_node_profile_validation.py
产物:结果汇总/figures/node_profile_validation.png(300dpi)
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

from plot_cv_case_profiles import join_cv, load_all

OUT = Path(__file__).resolve().parent.parent / "结果汇总" / "figures"
C_OBS, C_PRD = "#0C5DA5", "#C44E52"


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
        "grid.alpha": 0.25, "grid.linestyle": "--", "legend.frameon": False,
    })
    OUT.mkdir(parents=True, exist_ok=True)

    data = load_all()
    npz = np.load("结果汇总/runs/train_agl_b0_cv/results/predictions_Spatial_AGLB0CV.npz")
    y_true, y_pred, dem, H, idx = join_cv(npz, data)          # 全部 34,860 样本的区域外预测
    z_agl = (H - dem[:, None]) / 1000.0

    # 过境次数最多的两个格点
    key = [(round(a, 3), round(b, 3)) for a, b in zip(data["lon"][idx], data["lat"][idx])]
    cnt = Counter(key)
    top = [k for k, _ in cnt.most_common(2)]
    print("选中格点:", [(k, cnt[k]) for k in top])

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.0))
    for col, node in enumerate(top):
        ids = np.flatnonzero([k == node for k in key])
        lon0, lat0 = node
        dem0 = float(np.nanmean(dem[ids]))
        zm = np.nanmean(z_agl[ids], axis=0)                   # 该格点各层平均离地高度
        mo = np.nanmean(y_true[ids], 0); mp = np.nanmean(y_pred[ids], 0)
        do = np.nanmedian(y_true[ids], 0); dp = np.nanmedian(y_pred[ids], 0)
        mae_mean = float(np.nanmean(np.abs(mp - mo)))
        mae_med = float(np.nanmean(np.abs(dp - do)))
        for row, (stat, oo, pp, mae) in enumerate([
                ("平均", mo, mp, mae_mean), ("中位数", do, dp, mae_med)]):
            ax = axes[row, col]
            for i in ids:                                     # 原始过境散点(灰)
                v = np.isfinite(y_true[i]) & np.isfinite(y_pred[i])
                ax.scatter(y_true[i][v], z_agl[i][v], s=14, color="0.6", alpha=0.55,
                           edgecolors="none", zorder=2)
            ax.plot(oo, zm, "-o", ms=4, lw=2.0, color=C_OBS, label="ACDL 观测", zorder=4)
            ax.plot(pp, zm, "-s", ms=3.5, lw=1.8, color=C_PRD, mfc="none",
                    label="B0-AGL 预测", zorder=3)
            hi = np.nanmax([np.nanmax(oo[np.isfinite(oo)]), np.nanmax(pp[np.isfinite(pp)])])
            ax.set_xlim(0, hi * 1.15)
            ax.set_xlabel(r"消光系数 $\sigma$ (km$^{-1}$)")
            ax.set_title(f"格点 {lon0:.2f}°E,{lat0:.2f}°N · {stat}口径"
                         f"(MAE = {mae:.4f} km$^{{-1}}$)", fontsize=10.5)
            ax.text(0.97, 0.97, f"过境 n = {len(ids)}\nDEM ≈ {dem0:.0f} m\n灰点 = 原始过境",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.9))
            ax.grid(True)
            ax.legend(loc="lower right", fontsize=8.5)
            ax.set_ylim(0, 12)
        axes[0, col].set_title(f"格点 {lon0:.2f}°E,{lat0:.2f}°N · 平均口径"
                               f"(MAE = {mae_mean:.4f} km$^{{-1}}$)", fontsize=10.5)
    axes[0, 0].set_ylabel("离地高度 (km)"); axes[1, 0].set_ylabel("离地高度 (km)")
    fig.suptitle("单格点廓线精度验证(平均/中位数)——全月该格点仅 3 次过境(轨道限制);\n"
                 "预测为空间块 CV 区域外结果,灰点为全部原始过境", fontsize=12)
    fig.tight_layout()
    out = OUT / "node_profile_validation"
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  图: {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
