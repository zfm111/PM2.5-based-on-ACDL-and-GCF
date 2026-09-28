"""
evaluate_bench.py — 优化路线 Step 1 评估台(《优化路线讨论.md》§1.2 四指标)
============================================================
从训练落盘的 results/predictions_<tag>.npz 计算主报告指标并出图,不重训:

  1. 分层带 MAE/nMAE 剖面(0-1 / 1-3 / 3-5 / 5-10 / >10 km,按层中点高度分带,标注有效样本数)
  2. 整柱 OD 相对误差 + Gfrac(EE 包络内比例,EE = ±(0.05 + 0.15×AOD),≥66% 为满意;
     532 nm 沿用 550 nm(Levy 2010, MODIS 陆地)的 EE 形式)
     —— OD 一律用"观测廓线积分 Σ Ext_mean·Δz + 同一套有效层掩膜",不用 ACDL 官方 AOD
  3. 逐层 MB(带符号),与 #2 配对区分"重分布型 / 总量型"误差
  4. 回归 slope / intercept(逐层 + 整柱 OD),暴露固定偏移
  附图保留:逐层 R²(降为附图)、逐层 log10(pred/obs) 中位数曲线

用法:
  python evaluate_bench.py --run-dir train_fixed_case5_abs --tag Holdout --name B0
  # 旧 npz(无 dz_km)自动走"重放留出索引 + 重载数据"路径,需 --match-dir(--seed/--holdout 可调)

产物(写入 <run-dir>/):
  results/bench_<name>.json / bench_<name>.csv
  plots/bench_od_scatter_<name>.png / bench_profile_<name>.png / bench_per_layer_<name>.png
============================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

BAND_EDGES = [0.0, 1.0, 3.0, 5.0, 10.0, np.inf]      # km,按层中点高度
BAND_NAMES = ["0-1km", "1-3km", "3-5km", "5-10km", ">10km"]
EE_FOOTNOTE = "EE = ±(0.05 + 0.15×AOD),沿用 550nm(MODIS 陆地, Levy 2010)形式于 532nm;≥66% 为满意"


# ============================================================
# 层几何与分层带
# ============================================================
def layer_mid_heights(npz) -> np.ndarray:
    """层中点高度(km):bot_0=0(近似地表),bot_k=top_{k-1};mid=(bot+top)/2。"""
    top = np.asarray(npz["h_km"], dtype=np.float64)
    bot = np.empty_like(top)
    bot[0] = 0.0
    bot[1:] = top[:-1]
    return 0.5 * (bot + np.maximum(top, bot)), bot


def band_of(mid_km: np.ndarray) -> np.ndarray:
    idx = np.digitize(mid_km, BAND_EDGES) - 1
    return np.clip(idx, 0, len(BAND_NAMES) - 1)


# ============================================================
# 旧 npz 补 dz:重放留出索引 + 重载数据(以 y_true 全等校验兜底)
# ============================================================
def reconstruct_dz(run_dir: Path, tag: str, match_dir: str, seed: int, holdout: float,
                   y_true: np.ndarray):
    import Train_ACDL_ERA5_MatchV2 as T2
    cfg = dict(T2.CONFIG)
    cfg["MATCH_DIR"] = match_dir
    data = T2.load_matched_dir(Path(match_dir), cfg["MATCH_GLOB"], cfg["STRUCT"],
                               add_cols=cfg["ADD_COLS"])
    X, y, t_dn = data["X"], data["y"], data["time"]
    X_extra, zsfc = data.get("X_extra"), data.get("zsfc")
    ok = np.isfinite(X[:, 0]) & np.isfinite(X[:, 1]) & np.isfinite(t_dn)
    X, y, t_dn = X[ok], y[ok], t_dn[ok]
    if X_extra is not None:
        X = np.hstack([X, X_extra]).astype(np.float32)
    T = T2.build_targets(y, X, cfg, zsfc=zsfc)
    n = X.shape[0]
    if y_true.shape[0] != max(int(n * holdout), 1):
        raise RuntimeError(f"留出样本数不符:重放得 {max(int(n*holdout),1)} vs npz {y_true.shape[0]}"
                           f"(检查 --seed/--holdout/数据是否同一批)")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    te = np.sort(idx[: max(int(n * holdout), 1)])
    y_ref = T["y_abs"][te]
    if not np.allclose(y_ref, y_true, equal_nan=True, atol=1e-5):
        n_bad = int((~np.isclose(y_ref, y_true, equal_nan=True, atol=1e-5)).sum())
        raise RuntimeError(f"y_true 校验失败({n_bad} 个元素不符)→ npz 与当前数据/索引不一致,拒绝补算")
    print(f"[DZ] 旧 npz 补算成功:重放留出索引(seed={seed}),y_true 校验通过")
    return T["dz_km"][te].astype(np.float32), T["ok_od"][te].astype(bool)


# ============================================================
# 指标
# ============================================================
def _ols(x: np.ndarray, yv: np.ndarray):
    if x.size < 2 or np.std(x) < 1e-12:
        return float("nan"), float("nan")
    a, b = np.polyfit(x, yv, 1)
    return float(a), float(b)


def per_layer_table(y_true, y_pred, mid_km, bands):
    n_lv = y_true.shape[1]
    rows = []
    for k in range(n_lv):
        m = np.isfinite(y_true[:, k]) & np.isfinite(y_pred[:, k])
        yt, yp = y_true[m, k], y_pred[m, k]
        n = int(m.sum())
        if n < 2 or np.std(yt) < 1e-12:
            rows.append({"layer": k, "band": BAND_NAMES[bands[k]], "h_mid_km": round(float(mid_km[k]), 3),
                         "n": n, "MAE": np.nan, "nMAE": np.nan, "MB": np.nan, "RMSE": np.nan,
                         "slope": np.nan, "intercept": np.nan, "R2": np.nan,
                         "med_log10_ratio": np.nan, "iqr_log10_ratio": np.nan})
            continue
        mae = float(np.mean(np.abs(yp - yt)))
        obs_mean = float(np.mean(yt))
        lr = np.log10(yp[np.isfinite(yp) & (yp > 0) & (yt > 0)] /
                      yt[np.isfinite(yp) & (yp > 0) & (yt > 0)]) if ((yp > 0) & (yt > 0)).any() else np.array([])
        slope, intercept = _ols(yt, yp)
        ss = float(np.sum((yt - yt.mean()) ** 2))
        r2 = float(1 - np.sum((yt - yp) ** 2) / ss) if ss > 1e-12 else np.nan
        rows.append({
            "layer": k, "band": BAND_NAMES[bands[k]], "h_mid_km": round(float(mid_km[k]), 3),
            "n": n, "MAE": mae, "nMAE": (mae / obs_mean if obs_mean > 1e-12 else np.nan),
            "MB": float(np.mean(yp - yt)), "RMSE": float(np.sqrt(np.mean((yp - yt) ** 2))),
            "slope": slope, "intercept": intercept, "R2": r2,
            "med_log10_ratio": (float(np.median(lr)) if lr.size else np.nan),
            "iqr_log10_ratio": (float(np.percentile(lr, 75) - np.percentile(lr, 25)) if lr.size >= 4 else np.nan),
        })
    return rows


def band_table(rows):
    out = []
    for b, name in enumerate(BAND_NAMES):
        rs = [r for r in rows if r["band"] == name and np.isfinite(r["MAE"])]
        n = sum(r["n"] for r in rs)
        if not rs:
            out.append({"band": name, "n_layers": 0, "n_obs": 0}); continue
        mae = float(np.mean([r["MAE"] for r in rs]))
        obs_vals = []   # 带内观测均值要用原值,不能由 nMAE 反推 → 用权重近似:MAE 加权
        w = np.array([r["n"] for r in rs], dtype=float)
        nmae = float(np.sum(w * np.array([r["nMAE"] for r in rs])) / np.sum(w))
        mb = float(np.sum(w * np.array([r["MB"] for r in rs])) / np.sum(w))
        slope = float(np.nanmean([r["slope"] for r in rs]))
        out.append({"band": name, "n_layers": len(rs), "n_obs": int(n),
                    "MAE": mae, "nMAE": nmae, "MB": mb, "slope": slope})
    return out


def od_metrics(y_true, y_pred, dz, ok_od=None):
    """整柱 OD:观测廓线积分 + 同一套有效层掩膜(两侧同掩膜),Gfrac 用 EE 包络。"""
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(dz) & (dz > 1e-6)
        od_true = np.nansum(np.where(valid, y_true * dz, np.nan), axis=1)
        od_pred = np.nansum(np.where(valid, y_pred * dz, np.nan), axis=1)
    m = np.isfinite(od_true) & np.isfinite(od_pred) & (od_true > 1e-6)
    if ok_od is not None:
        m &= np.asarray(ok_od, dtype=bool)
    ot, op = od_true[m], od_pred[m]
    rel = (op - ot) / ot
    ee = 0.05 + 0.15 * ot
    within = np.abs(op - ot) <= ee
    slope, intercept = _ols(ot, op)
    ss = float(np.sum((ot - ot.mean()) ** 2))
    return {
        "n": int(m.sum()),
        "bias_rel_mean": float(np.mean(rel)), "bias_rel_median": float(np.median(rel)),
        "mae_rel": float(np.mean(np.abs(rel))),
        "R2": (float(1 - np.sum((ot - op) ** 2) / ss) if ss > 1e-12 else np.nan),
        "slope": slope, "intercept": intercept,
        "within_25pct": float(np.mean(np.abs(rel) <= 0.25)),
        "Gfrac_EE": float(np.mean(within)), "Gfrac_EE_pass": bool(np.mean(within) >= 0.66),
        "od_true_mean": float(np.mean(ot)), "od_pred_mean": float(np.mean(op)),
    }, ot, op


def gcf_metrics(npz):
    gt, gp = npz["gcf_true"], npz["gcf_pred"]
    m = np.isfinite(gt) & np.isfinite(gp)
    gt, gp = gt[m], gp[m]
    if gt.size < 2:
        return {"n": int(gt.size)}
    slope, intercept = _ols(gt, gp)
    ss = float(np.sum((gt - gt.mean()) ** 2))
    return {"n": int(gt.size),
            "R2": (float(1 - np.sum((gt - gp) ** 2) / ss) if ss > 1e-12 else np.nan),
            "MAE": float(np.mean(np.abs(gp - gt))),
            "MB": float(np.mean(gp - gt)), "slope": slope, "intercept": intercept}


# ============================================================
# 绘图
# ============================================================
def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from acdl_plotting import setup_cjk
        setup_cjk()
        return plt
    except Exception as exc:
        print(f"  (跳过绘图: {exc})")
        return None


def plot_od_scatter(ot, op, out_base, od_m):
    plt = _plt()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.scatter(ot, op, s=12, alpha=0.35, edgecolors="none", c="#1f77b4")
    hi = max(float(ot.max()), float(op.max())) * 1.06
    xs = np.linspace(0, hi, 200)
    ax.plot(xs, xs, "k--", lw=1.1, label="1:1")
    ax.plot(xs, xs + (0.05 + 0.15 * xs), "r-", lw=1.0, label="EE envelope")
    ax.plot(xs, xs - (0.05 + 0.15 * xs), "r-", lw=1.0)
    txt = (f"n = {od_m['n']}\nGfrac(EE) = {od_m['Gfrac_EE']*100:.1f}%"
           f"({'PASS' if od_m['Gfrac_EE_pass'] else '<66%'} )\n"
           f"bias(rel) = {od_m['bias_rel_mean']*100:+.1f}%\n"
           f"MAE(rel) = {od_m['mae_rel']*100:.1f}%\n"
           f"slope = {od_m['slope']:.3f}, intercept = {od_m['intercept']:.4f}")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.set_xlabel("Observed OD (from profile integral)")
    ax.set_ylabel("Predicted OD")
    ax.set_title(f"Column OD — {out_base.name}", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    fig.text(0.99, 0.01, EE_FOOTNOTE, ha="right", va="bottom", fontsize=6, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_base.with_suffix(".png"), dpi=200); plt.close(fig)
    print(f"  图: {out_base.with_suffix('.png')}")


def plot_profile(y_true, y_pred, mid_km, out_base, n_lv=32):
    plt = _plt()
    if plt is None:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 7.0), sharey=True,
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})
    z = mid_km

    def _q(arr, qs):
        out = np.full((len(qs), n_lv), np.nan)
        for k in range(n_lv):
            m = np.isfinite(y_true[:, k]) & np.isfinite(y_pred[:, k])
            if m.sum() >= 5:
                out[:, k] = np.percentile(arr[m, k], qs)
        return out

    qs = [25, 50, 75]
    tq, pq = _q(y_true, qs), _q(y_pred, qs)
    ax1.fill_betweenx(z, tq[0], tq[2], color="#1f77b4", alpha=0.22, lw=0)
    ax1.plot(tq[1], z, "-o", color="#1f77b4", ms=3.2, lw=1.8, label="Observed (median, IQR)")
    ax1.fill_betweenx(z, pq[0], pq[2], color="#d62728", alpha=0.22, lw=0)
    ax1.plot(pq[1], z, "-s", color="#d62728", ms=3.0, lw=1.8, label="Predicted (median, IQR)")
    ax1.set_xlabel(r"Extinction $\sigma$ (km$^{-1}$)"); ax1.set_ylabel("Layer mid-height (km)")
    ax1.set_title("Profile quartile bands (paired, same mask)", fontsize=9.5)
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8, loc="lower right")

    # 差值曲线:逐层 log10(pred/obs) 的中位数与 25-75 分位(同批样本同掩膜配对)
    lr = np.full((y_true.shape[0], n_lv), np.nan)
    ok = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 1e-8) & (y_pred > 1e-8)
    lr[ok] = np.log10(y_pred[ok] / y_true[ok])
    lq = np.full((3, n_lv), np.nan)
    for k in range(n_lv):
        v = lr[np.isfinite(lr[:, k]), k]
        if v.size >= 5:
            lq[:, k] = np.percentile(v, qs)
    ax2.fill_betweenx(z, lq[0], lq[2], color="#2ca02c", alpha=0.22, lw=0)
    ax2.plot(lq[1], z, "-o", color="#2ca02c", ms=3.2, lw=1.8, label="median log$_{10}$(pred/obs)")
    ax2.axvline(0.0, color="k", ls="--", lw=1.0)
    ax2.set_xlabel(r"$\log_{10}(\hat\sigma/\sigma)$")
    ax2.set_title("Log-space difference curve", fontsize=9.5)
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8, loc="lower right")
    fig.suptitle(out_base.name, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=200); plt.close(fig)
    print(f"  图: {out_base.with_suffix('.png')}")


def plot_per_layer(rows, out_base):
    plt = _plt()
    if plt is None:
        return
    z = np.array([r["h_mid_km"] for r in rows])
    panels = [("nMAE", "nMAE (MAE/mean obs)"), ("MB", "MB (signed, km$^{-1}$)"),
              ("slope", "OLS slope (pred vs obs)"), ("R2", "R$^2$ (annex)")]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 6.4), sharey=True)
    for ax, (key, label) in zip(axes, panels):
        v = np.array([r[key] for r in rows], dtype=float)
        n = np.array([r["n"] for r in rows])
        ax.plot(v, z, "-o", ms=3.2, lw=1.6, color="#1f77b4")
        if key == "MB":
            ax.axvline(0.0, color="k", ls="--", lw=0.9)
        if key == "slope":
            ax.axvline(1.0, color="k", ls="--", lw=0.9)
        ax.set_xlabel(label, fontsize=9)
        ax.grid(alpha=0.3)
        # 低样本层标灰(逐层 R²/指标在 n 低的层不报/慎读)
        ax.scatter(v[n < 500], z[n < 500], s=26, facecolors="none", edgecolors="r", lw=1.0)
    axes[0].set_ylabel("Layer mid-height (km)")
    fig.suptitle(f"{out_base.name}  (红圈 = 有效样本 <500 的层,慎读)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=200); plt.close(fig)
    print(f"  图: {out_base.with_suffix('.png')}")


# ============================================================
# 主流程
# ============================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Step 1 评估台:四指标 + EE 包络 + 分位带图(不重训)")
    ap.add_argument("--run-dir", required=True, help="训练输出目录(含 results/predictions_<tag>.npz)")
    ap.add_argument("--tag", default="Holdout", help="预测文件标签,如 Holdout / Holdout_E2b")
    ap.add_argument("--name", default=None, help="产物命名(默认 = tag)")
    ap.add_argument("--match-dir", default=None, help="旧 npz 无 dz 时重载数据补算用")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout", type=float, default=0.2)
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    name = a.name or a.tag
    npz_path = run_dir / "results" / f"predictions_{a.tag}.npz"
    d = np.load(npz_path)
    y_true = np.asarray(d["y_true"], dtype=np.float64)
    y_pred = np.asarray(d["y_pred"], dtype=np.float64)
    print(f"[BENCH] {npz_path}  (N_test={y_true.shape[0]})")

    if "dz_km" in d and d["dz_km"].size == y_true.size:
        dz = np.asarray(d["dz_km"], dtype=np.float64)
        ok_od = (np.asarray(d["ok_od"], dtype=bool) if "ok_od" in d and d["ok_od"].size == y_true.shape[0]
                 else None)
        print("[DZ] npz 自带 dz_km,直接使用")
    else:
        if not a.match_dir:
            raise SystemExit("旧 npz 无 dz_km:需 --match-dir 以重放留出索引补算")
        dz, ok_od = reconstruct_dz(run_dir, a.tag, a.match_dir, a.seed, a.holdout, y_true)

    mid_km, _ = layer_mid_heights(d)
    bands = band_of(mid_km)
    rows = per_layer_table(y_true, y_pred, mid_km, bands)
    bands_stat = band_table(rows)
    od_m, ot, op = od_metrics(y_true, y_pred, dz, ok_od=ok_od)
    gcf_m = gcf_metrics(d)
    m_all = np.isfinite(y_true) & np.isfinite(y_pred)
    glob = {"R2": float(1 - np.sum((y_true[m_all] - y_pred[m_all]) ** 2) /
                        np.sum((y_true[m_all] - y_true[m_all].mean()) ** 2)),
            "RMSE": float(np.sqrt(np.mean((y_true[m_all] - y_pred[m_all]) ** 2))),
            "MAE": float(np.mean(np.abs(y_true[m_all] - y_pred[m_all]))), "N_valid": int(m_all.sum())}

    cfg_path = run_dir / "results" / "cv_summary.json"
    cfg_snap = None
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as fh:
            cfg_snap = json.load(fh).get("config")

    out = {"name": name, "tag": a.tag, "run_dir": str(run_dir), "n_test": int(y_true.shape[0]),
           "global": glob, "bands": bands_stat, "per_layer": rows, "od": od_m, "gcf": gcf_m,
           "config": cfg_snap, "ee_footnote": EE_FOOTNOTE,
           "note_dz": "legacy npz: dz 由同 seed 重放留出索引重算,y_true 校验通过" if "dz_km" not in d else None}
    res_dir = run_dir / "results"; res_dir.mkdir(exist_ok=True)
    with open(res_dir / f"bench_{name}.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    with open(res_dir / f"bench_{name}.csv", "w", encoding="utf-8") as fh:
        fh.write("Layer,Band,H_mid_km,N,MAE,nMAE,MB,RMSE,Slope,Intercept,R2,MedLog10Ratio,IQRLog10Ratio\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k])
                              for k in ["layer", "band", "h_mid_km", "n", "MAE", "nMAE", "MB",
                                        "RMSE", "slope", "intercept", "R2",
                                        "med_log10_ratio", "iqr_log10_ratio"]) + "\n")

    plot_dir = run_dir / "plots"; plot_dir.mkdir(exist_ok=True)
    plot_od_scatter(ot, op, plot_dir / f"bench_od_scatter_{name}", od_m)
    plot_profile(y_true, y_pred, mid_km, plot_dir / f"bench_profile_{name}")
    plot_per_layer(rows, plot_dir / f"bench_per_layer_{name}")

    # ---- 一句话判据(Step 1 验收)----
    low = next(b for b in bands_stat if b["band"] == "0-1km")
    low_all = [b for b in bands_stat if b["band"] in ("0-1km", "1-3km", "3-5km")]
    lo5_mae = float(np.mean([b["MAE"] for b in low_all]))
    print("\n" + "=" * 78)
    print(f"[{name}] 一句话基线:")
    print(f"  低层(0-5km, 三带平均) MAE = {lo5_mae:.4f} km⁻¹ | 近地面带(0-1km) MAE = {low['MAE']:.4f}"
          f" (nMAE {low['nMAE']*100:.1f}%, n={low['n_obs']})")
    print(f"  柱含量 OD 相对误差 bias = {od_m['bias_rel_mean']*100:+.1f}% (|rel| MAE {od_m['mae_rel']*100:.1f}%,"
          f" n={od_m['n']}), Gfrac(EE) = {od_m['Gfrac_EE']*100:.1f}%"
          f" ({'达标≥66%' if od_m['Gfrac_EE_pass'] else '未达 66%'})")
    print(f"  OD slope = {od_m['slope']:.3f}, intercept = {od_m['intercept']:.4f};"
          f" GCF(n={gcf_m.get('n')}) R²={gcf_m.get('R2', float('nan')):.3f}, MAE={gcf_m.get('MAE', float('nan')):.4f},"
          f" MB={gcf_m.get('MB', float('nan')):+.4f}")
    print(f"  产物: bench_{name}.json / .csv + 3 图")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
