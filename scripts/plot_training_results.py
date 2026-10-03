#!/usr/bin/env python3
"""
plot_training_results.py — 从训练产物重画图(不需要重训)

与训练脚本完全解耦:只读 <RUN_DIR>/results/ 下的落盘文件,调用 acdl_plotting.py 出图。
改配色、改标题、改采样点数、加图都不用重跑 20 分钟的训练。

输入(训练脚本自动产出)
  <RUN_DIR>/results/predictions_<tag>.npz   原始数组:逐层 R²/RMSE/MAE、GCF 观测/预测、
                                            完整预测/真值廓线、测试样本经纬度与时间、层几何
  <RUN_DIR>/results/layer_geometry.json     层高(km)与代表气压(hPa);npz 里也有,作为兜底

能出的图
  per_layer  逐层 R²(绝对消光),y=高度(km),次轴气压
  gcf        GCF 观测 vs 预测散点(1:1 线;灰点=可评但未参与训练)
  profile    逐层"预测 vs 观测"密度散点,看偏差随高度的变化

用法
  # 全部图(所有 tag)
  python plot_training_results.py --run-dir ./train_fixed_case5_abs

  # 只画某几张 / 某个 tag
  python plot_training_results.py --run-dir ./train_fixed_case5_frac --plots gcf --tags Holdout

  # 出一张把多口径放一起的对比图(读同一 run-dir 下的所有 tag)
  python plot_training_results.py --run-dir ./train_fixed_case5_abs --plots per_layer --compare

依赖:numpy, matplotlib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

from acdl_plotting import (load_layer_geometry, plot_gcf_scatter,
                           plot_per_layer_r2, plot_profile_compare)

ALL_PLOTS = ("per_layer", "gcf", "profile")


# ============================================================
# 读取
# ============================================================
def find_tags(results_dir: Path, want: list[str] | None) -> list[str]:
    tags = sorted(p.stem.replace("predictions_", "")
                  for p in results_dir.glob("predictions_*.npz"))
    if want:
        missing = [t for t in want if t not in tags]
        if missing:
            print(f"[WARN] 这些 tag 没有 predictions 文件: {missing}")
        tags = [t for t in tags if t in want]
    return tags


def load_npz(results_dir: Path, tag: str) -> dict:
    p = results_dir / f"predictions_{tag}.npz"
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def geom_of(data: dict, results_dir: Path):
    """优先用 npz 里存的层几何;没有再读 layer_geometry.json。"""
    h = data.get("h_km"); p = data.get("p_hpa")
    if h is not None and h.size:
        return h, (p if (p is not None and p.size) else None)
    return load_layer_geometry(results_dir)


# ============================================================
# 各张图
# ============================================================
def draw_per_layer(results_dir: Path, tag: str, data: dict, out_dir: Path) -> None:
    geom = geom_of(data, results_dir)
    r2 = np.asarray(data["r2_per_layer"], dtype=float)          # (n_folds, 32)
    names = [str(s) for s in data.get("fold_names", [])]
    curves = {}
    if 1 < r2.shape[0] <= 4:                                    # 折数少时每折都画
        for i in range(r2.shape[0]):
            curves[names[i] if i < len(names) else f"Fold{i+1}"] = r2[i]
    mean = np.nanmean(r2, axis=0)
    curves[f"{tag} (mean{' over folds' if r2.shape[0] > 1 else ''})"] = mean
    plot_per_layer_r2(curves, out_dir / f"per_layer_r2_{tag}", geom=geom,
                      title=f"Per-layer R² of absolute extinction — {tag}")


def draw_gcf(results_dir: Path, tag: str, data: dict, out_dir: Path) -> None:
    gt = np.asarray(data.get("gcf_true", []), dtype=float)
    gp = np.asarray(data.get("gcf_pred", []), dtype=float)
    ot = np.asarray(data.get("gcf_true_out", []), dtype=float)
    op = np.asarray(data.get("gcf_pred_out", []), dtype=float)

    # 指标直接从数组重算(不依赖 JSON,保证图和点一致)
    m = np.isfinite(gt) & np.isfinite(gp)
    met = {"n_gcf": int(m.sum()), "n_gcf_all": int(m.sum() + (np.isfinite(ot) & np.isfinite(op)).sum())}
    if m.sum() >= 2 and np.nanstd(gt[m]) > 1e-12:
        from sklearn.metrics import mean_absolute_error, r2_score
        met["GCF_R2"] = float(r2_score(gt[m], gp[m]))
        met["GCF_MAE"] = float(mean_absolute_error(gt[m], gp[m]))
    ma = np.isfinite(ot) & np.isfinite(op)
    if ma.sum() >= 2 and np.nanstd(ot[ma]) > 1e-12:
        from sklearn.metrics import r2_score
        met["GCF_R2_all"] = float(r2_score(ot[ma], op[ma]))

    plot_gcf_scatter(gt, gp, out_dir / f"gcf_scatter_{tag}", title=f"GCF — {tag}",
                     met=met, out_true=ot, out_pred=op)


def draw_profile(results_dir: Path, tag: str, data: dict, out_dir: Path) -> None:
    y_true = np.asarray(data.get("y_true", []), dtype=float)
    y_pred = np.asarray(data.get("y_pred", []), dtype=float)
    if y_true.ndim != 2 or y_true.shape != y_pred.shape:
        print(f"  [SKIP] profile: {tag} 的 npz 里没有完整廓线(旧版产物?)")
        return
    geom = geom_of(data, results_dir)
    plot_profile_compare(y_true, y_pred, out_dir / f"profile_compare_{tag}",
                         h_km=geom[0], p_hpa=geom[1],
                         title=f"Observed vs predicted profile — {tag}")


DRAW = {"per_layer": draw_per_layer, "gcf": draw_gcf, "profile": draw_profile}


# ============================================================
# 主流程
# ============================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="从训练产物重画图(不重训)")
    ap.add_argument("--run-dir", required=True, help="训练输出目录(含 results/)")
    ap.add_argument("--plots", default=",".join(ALL_PLOTS),
                    help=f"要画的图,逗号分隔:{','.join(ALL_PLOTS)}(默认全部)")
    ap.add_argument("--tags", default=None, help="只画这些标签(逗号分隔,如 Holdout,Spatial)")
    ap.add_argument("--out-dir", default=None, help="图输出目录;默认 <run-dir>/plots")
    ap.add_argument("--compare", action="store_true",
                    help="额外画一张把所有 tag 放一起的逐层 R² 对比图")
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    results_dir = run_dir / "results"
    if not results_dir.exists():
        print(f"[ERROR] 找不到 {results_dir}(--run-dir 要指向训练输出目录)")
        return 1
    out_dir = Path(a.out_dir) if a.out_dir else (run_dir / "plots")

    want_plots = [s.strip() for s in a.plots.split(",") if s.strip()]
    bad = [s for s in want_plots if s not in DRAW]
    if bad:
        print(f"[ERROR] 未知的图类型 {bad};可选 {list(DRAW)}")
        return 2
    want_tags = [s.strip() for s in a.tags.split(",")] if a.tags else None

    tags = find_tags(results_dir, want_tags)
    if not tags:
        print(f"[ERROR] {results_dir} 下没有 predictions_*.npz"
              "  → 这批训练可能是旧版本跑的(没有落盘预测值),需要重跑一次")
        return 1
    print(f"[INFO] run-dir = {run_dir}")
    print(f"[INFO] tags = {tags};图 = {want_plots}")

    curves_cmp = {}
    for tag in tags:
        data = load_npz(results_dir, tag)
        print(f"\n[{tag}]  折数 {np.asarray(data['r2_per_layer']).shape[0]}")
        for kind in want_plots:
            DRAW[kind](results_dir, tag, data, out_dir)
        r2 = np.asarray(data["r2_per_layer"], dtype=float)
        curves_cmp[tag] = np.nanmean(r2, axis=0)

    if a.compare and len(curves_cmp) >= 2:
        plot_per_layer_r2({k: v for k, v in curves_cmp.items()},
                          out_dir / "per_layer_r2_ALL_together",
                          geom=geom_of(load_npz(results_dir, tags[0]), results_dir),
                          title="Per-layer R² — all tags")
    print("\n[DONE]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
