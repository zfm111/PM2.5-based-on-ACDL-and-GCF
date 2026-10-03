"""
plot_cv_summary.py — 交叉验证逐折对比图(优化前 ASL vs 优化后 AGL)
============================================================
(a) 空间块逐块 GCF R²:ASL 三块崩溃(−2.35/−1.94/−0.54)且一块无样本,AGL 最低 −0.10
(b) 空间块逐块 全局 R²:两版都回落,AGL 略稳
(c) 时间折 GCF R²(4 折)与 全局 R²
(d) 三口径汇总:全局 R² 与 GCF R²(留出/空间/时间)
数据:结果汇总/runs/{train_agl_b0_cv,train_fixed_case5_abs_cv}/results/folds_*.json
产物:结果汇总/figures/cv_summary.png(300dpi)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

J = Path(__file__).resolve().parent / "结果汇总"
C_ASL, C_AGL = "0.45", "#C44E52"


def folds(run: str, tag: str):
    p = J / "runs" / run / "results" / f"folds_{tag}.json"
    return json.load(open(p, encoding="utf-8"))


def main() -> int:
    from acdl_plotting import setup_cjk
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_cjk()
    plt.rcParams.update({
        "font.size": 10.5, "axes.titlesize": 11, "axes.labelsize": 10.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "grid.alpha": 0.25, "grid.linestyle": "--", "legend.frameon": False,
    })
    out = J / "figures" / "cv_summary.png"

    sp_agl = folds("train_agl_b0_cv", "Spatial_AGLB0CV")
    sp_asl = folds("train_fixed_case5_abs_cv", "Spatial_B0CV")
    tp_agl = folds("train_agl_b0_cv", "Temporal_AGLB0CV")
    tp_asl = folds("train_fixed_case5_abs_cv", "Temporal_B0CV")

    def gcf(f):
        return np.array([x.get("GCF_R2", np.nan) for x in f], dtype=float)

    def r2(f):
        return np.array([x.get("R2_global", np.nan) for x in f], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6))

    # (a) 空间块逐块 GCF R²
    ax = axes[0, 0]
    order = np.argsort(-np.nan_to_num(gcf(sp_agl), nan=-3.0))
    ga, gs = gcf(sp_agl)[order], gcf(sp_asl)[order]
    x = np.arange(len(ga))
    ax.bar(x - 0.2, np.nan_to_num(gs, nan=0.0), 0.38, color=C_ASL, label="B0-ASL(优化前)")
    ax.bar(x + 0.2, np.nan_to_num(ga, nan=0.0), 0.38, color=C_AGL, label="B0-AGL(优化后)")
    for i, v in enumerate(gs):
        if not np.isfinite(v):
            ax.text(i - 0.2, 0.06, "无样本", rotation=90, fontsize=7, ha="center", color=C_ASL)
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels([f"B{i+1}" for i in x])
    ax.set_ylabel("GCF R²")
    ax.set_title("(a) 空间块逐块 GCF R² —— ASL 三块崩溃(−2.35/−1.94/−0.54),AGL 最低 −0.10", fontsize=10.5)
    ax.grid(True, axis="y"); ax.legend(loc="lower left")
    ax.set_ylim(-2.6, 0.9)

    # (b) 空间块逐块 全局 R²
    ax = axes[0, 1]
    ra, rs = r2(sp_agl)[order], r2(sp_asl)[order]
    ax.bar(x - 0.2, rs, 0.38, color=C_ASL, label="B0-ASL")
    ax.bar(x + 0.2, ra, 0.38, color=C_AGL, label="B0-AGL")
    ax.set_xticks(x); ax.set_xticklabels([f"B{i+1}" for i in x])
    ax.set_ylabel("全局 R²")
    mean_a, mean_s = np.nanmean(ra), np.nanmean(rs)
    ax.axhline(mean_s, color=C_ASL, ls="--", lw=1.0)
    ax.axhline(mean_a, color=C_AGL, ls="--", lw=1.0)
    ax.set_title(f"(b) 空间块逐块 全局 R²(虚线=均值:ASL {mean_s:.3f} / AGL {mean_a:.3f})", fontsize=10.5)
    ax.grid(True, axis="y"); ax.legend(loc="lower right")

    # (c) 时间折 GCF R²
    ax = axes[1, 0]
    ta, ts = gcf(tp_agl), gcf(tp_asl)
    x4 = np.arange(max(len(ta), len(ts)))
    ax.bar(x4 - 0.2, ts, 0.38, color=C_ASL, label="B0-ASL")
    ax.bar(x4 + 0.2, ta, 0.38, color=C_AGL, label="B0-AGL")
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xticks(x4); ax.set_xticklabels([f"折{i+1}" for i in x4])
    ax.set_ylabel("GCF R²")
    ax.set_title(f"(c) 时间块 4 折 GCF R²(均值:ASL {np.nanmean(ts):.3f} / AGL {np.nanmean(ta):.3f})", fontsize=10.5)
    ax.grid(True, axis="y"); ax.legend(loc="lower right")

    # (d) 三口径汇总
    ax = axes[1, 1]
    protocols = ["留出", "空间CV", "时间CV"]
    hold_a = float(json.load(open(J / "runs/train_agl_b0/results/bench_AGLB0.json", encoding="utf-8"))["gcf"]["R2"])
    hold_s = float(json.load(open(J / "runs/train_fixed_case5_abs/results/bench_B0.json", encoding="utf-8"))["gcf"]["R2"])
    gcf_summary = {"ASL": [hold_s, np.nanmean(gs), np.nanmean(ts)],
                   "AGL": [hold_a, np.nanmean(ga), np.nanmean(ta)]}
    r2_summary = {"ASL": [0.4966, np.nanmean(rs), np.nanmean(r2(sp_asl))],
                  "AGL": [0.4797, np.nanmean(ra), np.nanmean(r2(tp_agl))]}
    x3 = np.arange(3)
    for i, (model, c) in enumerate([("ASL", C_ASL), ("AGL", C_AGL)]):
        ax.bar(x3 + (i - 0.5) * 0.36, r2_summary[model], 0.34, color=c, alpha=0.45,
               label=f"{model} 全局 R²")
        ax.bar(x3 + (i - 0.5) * 0.36, gcf_summary[model], 0.34, color=c,
               label=f"{model} GCF R²", hatch="//", edgecolor="white")
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xticks(x3); ax.set_xticklabels(protocols)
    ax.set_ylabel("R²")
    ax.set_title("(d) 三口径汇总(浅色=全局 R²,斜纹=GCF R²) —— CV 回落但 AGL 优势保持/扩大", fontsize=10.5)
    ax.grid(True, axis="y"); ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-0.4, 0.75)

    fig.suptitle("交叉验证逐折对比:优化前(B0-ASL) vs 优化后(B0-AGL) —— 空间块 3×3(12 块) + 时间块 4 折", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  图: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
