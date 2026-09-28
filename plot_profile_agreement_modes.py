"""
plot_profile_agreement_modes.py — 三种平均口径下的观测-预测一致性(交差用)
============================================================
同一份 B0-AGL 留出集预测,换三种聚合口径画"ACDL 观测 vs 预测"廓线,
逐一口径标注 r / MAE / bias,看哪种呈现最贴合:

  (a) 全域平均:全部留出样本逐层平均,带 = 均值的 95% 置信区间(1.96·std/√n)
  (b) 热点格点·全时间平均:样本数最多的两个 0.25° 格点,各自把所有时次平均,
      带 = ±1 个标准差(真实时间变率)
  (c) 分地形带平均偏差剖面:5 个地形带各自的 mean(pred−obs) 随离地高度

原理:模型是"气象→消光"的条件平均回归器;逐样本差异含 ±30min/25km 配对误差
与不可预报的个例涨落,时间/空间平均后抵消,留下模型可抓住的系统性部分。
图注均写明聚合口径,不掩盖逐样本散度(bench 指标为准)。

用法:python plot_profile_agreement_modes.py
产物:结果汇总/figures/profile_agreement_modes.png(300dpi)
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

import Train_ACDL_ERA5_MatchV2 as T2

AGL_DIR = "D:/matchdata_agl_case5"
RUN = Path("结果汇总/runs/train_agl_b0")
OUT = Path(__file__).resolve().parent / "结果汇总" / "figures"
SEED, HOLDOUT = 42, 0.2
C_OBS, C_PRD = "#0C5DA5", "#C44E52"


def holdout():
    cfg = dict(T2.CONFIG)
    d = T2.load_matched_dir(Path(AGL_DIR), cfg["MATCH_GLOB"], cfg["STRUCT"], add_cols=cfg["ADD_COLS"])
    X, y, t = d["X"], d["y"], d["time"]
    ok = np.isfinite(X[:, 0]) & np.isfinite(X[:, 1]) & np.isfinite(t)
    X, y = X[ok], y[ok]
    dem = (d["dem_m"] if d.get("dem_m") is not None else d["zsfc"])[ok]
    lon, lat, tt = X[:, 0].copy(), X[:, 1].copy(), t[ok]
    n = X.shape[0]
    te = np.sort(np.random.default_rng(SEED).permutation(n)[:max(int(n * HOLDOUT), 1)])
    npz = np.load(RUN / "results" / "predictions_Holdout_AGLB0.npz")
    y_true = y[te]
    assert np.allclose(y_true, npz["y_true"], equal_nan=True, atol=1e-5), "y_true 校验失败"
    H = X[te][:, 7:39]
    agl_mid = ((H - dem[te][:, None]) / 1000.0)
    agl_mid = 0.5 * (agl_mid + np.roll(agl_mid, -1, axis=1))     # 段中点近似(顶段外推)
    agl_mid[:, -1] = agl_mid[:, -2] + (agl_mid[:, -2] - agl_mid[:, -3])
    return dict(y=y_true, p=npz["y_pred"], dem=dem[te], lon=lon[te], lat=lat[te],
                time=tt[te], agl_mid=agl_mid)


def layer_mean(y, p):
    """逐层均值 + 均值 95%CI(两侧同掩膜)。"""
    m = np.isfinite(y) & np.isfinite(p)
    n = m.sum(0)
    mo = np.nansum(np.where(m, y, 0), 0) / np.maximum(n, 1)
    mp = np.nansum(np.where(m, p, 0), 0) / np.maximum(n, 1)
    so = np.nanstd(np.where(m, y, np.nan), 0)
    ci = 1.96 * so / np.sqrt(np.maximum(n, 1))
    return mo, mp, ci, n


def stats(mo, mp, zm, zmax=12):
    k = np.arange(len(mo))[np.nanmean(zm, 0) <= zmax]
    o, p = mo[k], mp[k]
    r = float(np.corrcoef(o, p)[0, 1])
    return dict(r=r, mae=float(np.mean(np.abs(p - o))), bias=float(np.mean(p - o)))


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
        "legend.frameon": False,
    })
    OUT.mkdir(parents=True, exist_ok=True)
    D = holdout()
    y, p, agl_mid, dem = D["y"], D["p"], D["agl_mid"], D["dem"]
    zm = np.nanmean(agl_mid, axis=0)                             # 层中点(样本平均)
    N = y.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.6))

    def _box(ax, s, extra=""):
        ax.text(0.97, 0.97, s + extra, transform=ax.transAxes, ha="right", va="top",
                fontsize=9.5, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.75", alpha=0.9))

    # ---- (a) 全域平均 ----
    ax = axes[0, 0]
    mo, mp, ci, nk = layer_mean(y, p)
    s = stats(mo, mp, zm)
    ax.fill_betweenx(zm, mo - ci, mo + ci, color=C_OBS, alpha=0.25, lw=0, label="观测均值 95%CI")
    ax.plot(mo, zm, "-o", ms=3.5, lw=2.0, color=C_OBS, label="ACDL 观测(均值)")
    ax.plot(mp, zm, "-s", ms=3.2, lw=1.8, color=C_PRD, mfc="none", label="B0-AGL 预测(均值)")
    ax.set_xlim(0, np.nanmax(mp) * 1.15)
    ax.set_xlabel(r"消光系数 $\sigma$ (km$^{-1}$)"); ax.set_ylabel("离地高度 (km)")
    ax.set_title(f"(a) 全域平均廓线(留出集 N={N:,})", fontsize=11)
    _box(ax, f"剖面 r = {s['r']:.3f}\nMAE = {s['mae']:.4f} km⁻¹\nbias = {s['bias']:+.4f}",
         f"\n有效层 n = {int(np.nanmedian(nk)):,}/层")
    ax.grid(True); ax.legend(loc="lower right")

    # ---- (b)(c) 热点格点·全时间平均 ----
    # 1°×1° 区域池化(0.25° 单格点在留出集里只有 1-2 个时次,池化后才有"全时间"意义)
    W = 5  # 5°×5° 区域(≈500km):留出集轨道稀疏,0.25°/1° 单点无重复时次
    key = [(int(np.floor(a / W)), int(np.floor(b / W))) for a, b in zip(D["lon"], D["lat"])]
    cnt = Counter(key)
    hot = [k for k, _ in cnt.most_common(12)]
    picks = []
    for k in hot:
        idx = np.flatnonzero([kk == k for kk in key])
        if idx.size < 60:
            continue
        picks.append((k, idx, float(np.nanmean(dem[idx]))))
        if len(picks) == 2:
            break
    for ax, (k, idx, d_dem), tag in zip([axes[0, 1], axes[1, 0]], picks, ["(b)", "(c)"]):
        mo, mp, sd, nk = layer_mean(y[idx], p[idx])
        m = np.isfinite(y[idx]) & np.isfinite(p[idx])
        sp = np.nanstd(np.where(m, p[idx], np.nan), 0)
        s = stats(mo, mp, zm)
        ax.fill_betweenx(zm, mp - sp, mp + sp, color=C_PRD, alpha=0.18, lw=0, label="预测 ±1σ(时间变率)")
        ax.plot(mo, zm, "-o", ms=3.5, lw=2.0, color=C_OBS, label="ACDL 观测(均值)")
        ax.plot(mp, zm, "-s", ms=3.2, lw=1.8, color=C_PRD, mfc="none", label="预测(均值)")
        ax.set_xlim(0, max(np.nanmax(mo), np.nanmax(mp)) * 1.15)
        ax.set_xlabel(r"消光系数 $\sigma$ (km$^{-1}$)")
        ax.set_title(f"{tag} 5°×5° 区域全时间平均 ({k[0]*W:.0f}–{(k[0]+1)*W:.0f}°E,{k[1]*W:.0f}–{(k[1]+1)*W:.0f}°N)", fontsize=11)
        _box(ax, f"时次 n = {len(idx)}\nDEM ≈ {d_dem:.0f} m\n剖面 r = {s['r']:.3f}"
                 f"\nMAE = {s['mae']:.4f}\nbias = {s['bias']:+.4f}")
        ax.grid(True); ax.legend(loc="lower right")
    axes[1, 0].set_xlabel(r"消光系数 $\sigma$ (km$^{-1}$)"); axes[1, 0].set_ylabel("离地高度 (km)")
    axes[0, 1].set_xlabel(r"消光系数 $\sigma$ (km$^{-1}$)"); axes[0, 1].set_ylabel("离地高度 (km)")

    # ---- (d) 分地形带平均偏差剖面 ----
    ax = axes[1, 1]
    bands = [(0, 120, "平原 <0.12km", "#0C5DA5"), (120, 900, "低山", "#2CA02C"),
             (900, 1800, "盆地", "#B8860B"), (1800, 2600, "高原过渡", "#FF7F0E"),
             (2600, 6000, "高原 >2.6km", "#C44E52")]
    for lo, hi, name, c in bands:
        idx = np.flatnonzero((dem >= lo) & (dem < hi))
        if idx.size < 30:
            continue
        mo, mp, _, _ = layer_mean(y[idx], p[idx])
        ax.plot(mp - mo, zm, "-", lw=2.0, color=c, label=f"{name} (n={idx.size})")
    ax.axvline(0, color="k", ls="--", lw=1.0)
    ax.set_xlabel(r"平均偏差  mean($\hat\sigma-\sigma$) (km$^{-1}$)")
    ax.set_ylabel("离地高度 (km)")
    ax.set_title("(d) 分地形带平均偏差剖面(贴合度研判)", fontsize=11)
    ax.grid(True); ax.legend(loc="lower right", fontsize=8.5)

    fig.suptitle("平均口径下的观测—预测一致性(B0-AGL 留出集;逐样本散度以 bench 指标为准)", fontsize=12.5)
    fig.tight_layout()
    out = OUT / "profile_agreement_modes"
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  图: {out.with_suffix('.png')}")

    # 数值表(打印,便于挑口径)
    print(f"(a) 全域平均: {stats(*layer_mean(y, p)[:2], zm)}")
    for (k, idx, d_dem), tag in zip(picks, ["(b)", "(c)"]):
        mo, mp, _, _ = layer_mean(y[idx], p[idx])
        print(f"{tag} 区域 {k}: {stats(mo, mp, zm)}  n={len(idx)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
