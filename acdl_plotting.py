"""
acdl_plotting.py — 训练结果的绘图函数(与训练解耦)

这里只做"读数据 → 出图",不含任何训练逻辑、不依赖 torch。
被两边共用:
  - Train_ACDL_ERA5_MatchV2.py 训练结束时直接调用(可选,`--no-plots` 关闭)
  - Plot_Training_Results.py   事后从落盘产物重画(**不用重训**)

坐标约定
  - 逐层 R² 图:y 轴 = 高度(km),右次轴 = 近似气压(hPa);x 轴 = R²
  - GCF 散点图:x = 观测 GCF,y = 预测 GCF + 1:1 虚线

依赖:numpy, matplotlib(绘图必备)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ============================================================
# 公共小工具
# ============================================================
def setup_cjk():
    """中文字体设置(防豆腐框):Windows 首选微软雅黑,再试黑体/Noto/思源;负号用 ASCII。"""
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
        "PingFang SC", "WenQuanYi Micro Hei", "DejaVu Sans",
    ]
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False


def _plt():
    """惰性导入 matplotlib(无头模式);失败返回 None。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        setup_cjk()
        return plt
    except Exception as exc:
        print(f"  (跳过绘图: {exc})")
        return None


def _save(fig, out_base: Path, plt) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=200)
    try:
        fig.savefig(out_base.with_suffix(".pdf"))
    except Exception:
        pass
    plt.close(fig)
    print(f"  图: {out_base.with_suffix('.png')}")


def load_layer_geometry(results_dir: Path):
    """读 results/layer_geometry.json → (h_km, p_hpa);缺失返回 (None, None)。"""
    p = Path(results_dir) / "layer_geometry.json"
    if not p.exists():
        return None, None
    with open(p, encoding="utf-8") as fh:
        g = json.load(fh)
    h = g.get("height_km")
    pr = g.get("pressure_hpa")
    return (None if h is None else np.asarray(h, dtype=float),
            None if pr is None else np.asarray(pr, dtype=float))


# ============================================================
# 图 1:逐层 R²(绝对消光)
# ============================================================
def plot_per_layer_r2(curves: dict, out_base: Path, geom=None, title: str = "") -> bool:
    """y 轴 = 高度(km,次轴标气压),x 轴 = R²;curves = {标签: R²数组(32,)}。"""
    plt = _plt()
    if plt is None:
        return False
    h_km, p_hpa = geom if geom is not None else (None, None)
    n_lv = len(next(iter(curves.values())))
    y = h_km if (h_km is not None and len(h_km) == n_lv) else np.arange(n_lv, dtype=float)
    ylabel = "Altitude (km)" if h_km is not None else "Layer index (L00=lowest)"

    fig, ax = plt.subplots(figsize=(5.2, 7.0))
    for label, r2 in curves.items():
        r2 = np.asarray(r2, dtype=float)
        m = np.isfinite(r2)
        if m.any():
            ax.plot(r2[m], y[m], "-o", ms=3.5, lw=1.6, label=label)
    ax.axvline(0.0, color="gray", lw=0.9, ls="--")
    ax.set_xlabel(r"$R^2$"); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # 次轴:气压(hPa),与高度一一对应 → 仅作刻度参考
    if p_hpa is not None and h_km is not None and len(p_hpa) == len(h_km):
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ticks = np.linspace(0, len(y) - 1, min(9, len(y))).round().astype(int)
        ax2.set_yticks(y[ticks])
        ax2.set_yticklabels([f"{p_hpa[i]:.0f}" for i in ticks], fontsize=8)
        ax2.set_ylabel("Approx. pressure (hPa)", fontsize=9)
    fig.tight_layout()
    _save(fig, out_base, plt)
    return True


# ============================================================
# 图 2:GCF 观测 vs 预测
# ============================================================
def plot_gcf_scatter(gcf_true, gcf_pred, out_base: Path, title: str = "",
                     met: dict | None = None, out_true=None, out_pred=None) -> bool:
    """GCF 散点图:观测 vs 预测,带 1:1 线与 R²/MAE/n 标注。

    out_true/out_pred:可评但**未被训练使用**的样本 → 画成灰点,直观展示选择偏差。
    """
    plt = _plt()
    if plt is None:
        return False

    def _clean(a, b=None):
        a = np.asarray(a, dtype=float)
        if b is None or a.size == 0:
            return a, (None if b is None else np.asarray(b, dtype=float))
        b = np.asarray(b, dtype=float)
        m = np.isfinite(a) & np.isfinite(b)
        return a[m], b[m]

    gt, gp = _clean(gcf_true, gcf_pred)
    ot, op = _clean(out_true, out_pred)

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    pool = [a for a in (gt, gp, ot, op) if a is not None and a.size]
    if pool:
        lo = min(0.0, float(min(a.min() for a in pool)))
        hi = max(float(max(a.max() for a in pool)), 1e-3) * 1.05
        if ot.size:
            ax.scatter(ot, op, s=14, alpha=0.45, edgecolors="none", c="0.65",
                       label=f"evaluable, not used in training (n={ot.size})")
        ax.scatter(gt, gp, s=18, alpha=0.6, edgecolors="none", c="#1f77b4",
                   label=f"used in training (n={gt.size})")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    txt = []
    if met:
        if met.get("GCF_R2") is not None:
            txt.append(f"$R^2$ = {met['GCF_R2']:.3f}")
        if met.get("GCF_MAE") is not None:
            txt.append(f"MAE = {met['GCF_MAE']:.3f}")
        if met.get("GCF_R2_all") is not None and met.get("n_gcf_all", 0) > met.get("n_gcf", 0):
            txt.append(f"(all-sample $R^2$ = {met['GCF_R2_all']:.3f}, n={met['n_gcf_all']})")
    txt.append(f"n = {int(gt.size)}")
    ax.text(0.03, 0.97, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.set_xlabel("Observed GCF")
    ax.set_ylabel("Predicted GCF")
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    if ot.size or gt.size:
        ax.legend(loc="lower right", fontsize=7.5, framealpha=0.92)
    fig.tight_layout()
    _save(fig, out_base, plt)
    return True


# ============================================================
# 图 3:逐层"预测 vs 观测"廓线对比(中位数 + IQR 带)
# ============================================================
def plot_profile_compare(y_true, y_pred, out_base: Path, h_km=None, p_hpa=None,
                         title: str = "", min_n: int = 2) -> bool:
    """逐层对比观测与预测的消光廓线:每层画中位数曲线 + IQR 带。

    为什么不用散点:逐层散点只能看到"观测的分布",看不出预测偏在哪。
    中位数曲线能直接读出"模型把廓线画高/画低",IQR 带给出离散程度。

    y_true/y_pred: (N, 32),无效层可为 NaN。返回 (n_layers_valid, 每层样本数)。
    """
    plt = _plt()
    if plt is None:
        return False
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim != 2 or y_true.shape != y_pred.shape or y_true.shape[1] == 0:
        print("  (跳过绘图: 廓线维度不匹配)")
        return False

    n_lv = y_true.shape[1]
    z = (np.asarray(h_km, dtype=float) if (h_km is not None and len(h_km) == n_lv)
         else np.arange(n_lv, dtype=float))

    def _q(arr, qs):
        out = np.full((len(qs), n_lv), np.nan)
        for k in range(n_lv):
            m = np.isfinite(y_true[:, k]) & np.isfinite(y_pred[:, k])
            if m.sum() >= min_n:
                out[:, k] = np.percentile(arr[m, k], qs)
        return out

    qs = [25, 50, 75]
    tq = _q(y_true, qs)
    pq = _q(y_pred, qs)
    if not np.isfinite(tq[1]).any():
        print("  (跳过绘图: 没有足够的有效层)")
        return False

    fig, ax = plt.subplots(figsize=(5.6, 7.0))
    ax.fill_betweenx(z, tq[0], tq[2], color="#1f77b4", alpha=0.22, lw=0)
    ax.plot(tq[1], z, "-o", color="#1f77b4", ms=3.5, lw=1.9, label="Observed (median, IQR)")
    ax.fill_betweenx(z, pq[0], pq[2], color="#d62728", alpha=0.22, lw=0)
    ax.plot(pq[1], z, "-s", color="#d62728", ms=3.2, lw=1.9, label="Predicted (median, IQR)")

    hi = np.nanmax([np.nanpercentile(tq[2][np.isfinite(tq[2])], 98) if np.isfinite(tq[2]).any() else 0.1,
                    np.nanpercentile(pq[2][np.isfinite(pq[2])], 98) if np.isfinite(pq[2]).any() else 0.1])
    ax.set_xlim(0.0, max(hi * 1.15, 1e-3))
    ax.set_xlabel(r"Extinction coefficient $\sigma$ (km$^{-1}$)")
    ax.set_ylabel("Altitude (km)" if h_km is not None else "Layer index (L00=lowest)")
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    if p_hpa is not None and len(p_hpa) == n_lv:
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ticks = np.linspace(0, n_lv - 1, min(9, n_lv)).round().astype(int)
        ax2.set_yticks(z[ticks])
        ax2.set_yticklabels([f"{p_hpa[i]:.0f}" for i in ticks], fontsize=8)
        ax2.set_ylabel("Approx. pressure (hPa)", fontsize=9)

    fig.tight_layout()
    _save(fig, out_base, plt)
    return True

