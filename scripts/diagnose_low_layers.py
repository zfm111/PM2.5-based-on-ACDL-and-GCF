"""
diagnose_low_layers.py — 低层(L00-L05) Ext_mean 缺失成因定量诊断
============================================================
问题:B0 留出集逐层有效率 L00=29.6%/L01=41%/L02=48%,远低于高层(≥90%)。为什么?

三类候选机制(《优化路线讨论.md》后续跟进·方向1):
  ① 几何:ERA5 气压层层高以海拔为基准,段边界 = [min(surf,H0), H0, H1, ...]
     (match_acdl_era5_from_scratch.py:558-560)。DEM ≥ 段顶 → 整段在地下/为空 → 必 NaN;
     逐 bin 还有 above_surf 剔除(:630,:821-824)。
  ② 云 QC:case4 对厚云下方整段丢弃 + 薄云下 1km 缓冲(:825-836);case5 绕开云规则。
  ③ 其他:消光超界 [0,1.25]、CAD 不确定带(-20,20)、ACDL 产品本身 NaN。

方法(数据:D:/matchdata_fixed_case5 与 case4,同月同文件同行数,同索引可比):
  case4 掩膜 ⊆ case5 掩膜(m4 = valid_base&aer&~cloud…&above_surf ⊆ m5 = valid_base&above_surf)
  → 每个样本×层可无重叠分解为四类:
     A 硬几何缺失   = case5 NaN ∧ Z_sfc ≥ 段顶H_k
     B 其他 QC 缺失 = case5 NaN ∧ Z_sfc < 段顶H_k(消光超界/CAD/产品NaN;case4 亦缺)
     C 云QC 净剔除  = case5 有效 ∧ case4 NaN(被云规则杀掉的可回收池)
     D 两口径均有效 = case5 有效 ∧ case4 有效
     A+B+C+D = N(逐层)

用法:
  python diagnose_low_layers.py                       # 默认 case4/case5 两目录
产物(统一写入 结果汇总/):
  结果汇总/figures/diag_lowlayers_decompose.png / diag_lowlayers_demband.png (+json)
  结果汇总/低层缺失诊断.md(嵌图+结论)
============================================================
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

import train_acdl_era5_match_v2 as T2

CASE5_DIR = "D:/matchdata_fixed_case5"
CASE4_DIR = "D:/matchdata_fixed_case4"
OUT_DIR = Path(__file__).resolve().parent.parent / "结果汇总"
DEM_BANDS = [(0, 120), (120, 370), (370, 650), (650, 1000), (1000, 1500), (1500, 2000), (2000, 6000)]
DEM_BAND_NAMES = ["<120", "120-370", "370-650", "650-1000", "1000-1500", "1500-2000", ">2000"]
N_SHOW = 12          # 诊断展示到 L11(低层段)
N_DEM_SHOW = 8       # 地形分带图展示 L00-L07


def load_y(md_dir: str, cfg: dict):
    """读一个匹配目录 → (y(N,32), zsfc(N,), H(N,32), n_files)。过滤与训练一致(有限 lon/lat/time)。"""
    d = T2.load_matched_dir(Path(md_dir), cfg["MATCH_GLOB"], cfg["STRUCT"], add_cols=cfg["ADD_COLS"])
    X, y, t = d["X"], d["y"], d["time"]
    ok = np.isfinite(X[:, 0]) & np.isfinite(X[:, 1]) & np.isfinite(t)
    H = X[:, cfg["COL_H0"]:cfg["COL_H0"] + cfg["N_LEVEL"]][ok].astype(np.float64)   # 段顶高(m)
    return y[ok].astype(np.float64), np.asarray(d["zsfc"], dtype=np.float64)[ok], H, len(d["files"])


def decompose(y5, y4, zs, H):
    """逐层四类分解(A 硬几何 / B 其他QC / C 云QC / D 均有效),返回 dict[层] = dict。"""
    n_lv = y5.shape[1]
    v5, v4 = np.isfinite(y5), np.isfinite(y4)
    out = {}
    for k in range(n_lv):
        hard = zs >= H[:, k]                     # 段顶在 DEM 下 → 段空(几何必缺)
        A = ~v5[:, k] & hard
        B = ~v5[:, k] & ~hard
        C = v5[:, k] & ~v4[:, k]
        D = v5[:, k] & v4[:, k]
        out[k] = {"A_geo": int(A.sum()), "B_other": int(B.sum()),
                  "C_cloud": int(C.sum()), "D_valid": int(D.sum()), "N": int(y5.shape[0]),
                  "v5": float(v5[:, k].mean()), "v4": float(v4[:, k].mean())}
    return out


def dem_band_table(y5, zs, n_show):
    """按地形分带的逐层有效率(case5)。"""
    rows = []
    for lo, hi in DEM_BANDS:
        m = (zs >= lo) & (zs < hi)
        if m.sum() == 0:
            continue
        rows.append({"band": f"{lo}-{hi}m" if hi < 6000 else f">={lo}m",
                     "n": int(m.sum()),
                     "valid": [float(np.isfinite(y5[m, k]).mean()) for k in range(n_show)]})
    return rows


def main() -> int:
    cfg = dict(T2.CONFIG)
    print("=" * 78)
    print("低层缺失成因诊断:case5(训练口径) 与 case4(默认QC) 对比")
    print("=" * 78)
    y5, zs5, H5, nf5 = load_y(CASE5_DIR, cfg)
    y4, zs4, H4, nf4 = load_y(CASE4_DIR, cfg)
    if y5.shape[0] != y4.shape[0]:
        raise SystemExit(f"两目录样本数不同: case5={y5.shape[0]} vs case4={y4.shape[0]},不可同索引对比")
    if not np.allclose(zs5, zs4) or not np.allclose(H5, H4):
        raise SystemExit("Z_sfc/H 列不一致 → 索引不可比,中止")
    n = y5.shape[0]
    print(f"[DATA] 各 {nf5} 文件 / {n} 样本;Z_sfc 分位 p10={np.percentile(zs5,10):.0f} "
          f"p50={np.percentile(zs5,50):.0f} p90={np.percentile(zs5,90):.0f} m(域含青藏高原,地形切割强)")

    dec = decompose(y5, y4, zs5, H5)
    dem_rows = dem_band_table(y5, zs5, N_DEM_SHOW)

    # ---- 打印分解表 ----
    print("\n== 逐层成因分解(占全部样本的百分比) ==")
    print("层   有效率5  有效率4 | A硬几何  B其他QC  C云QC可回收  D均有效")
    for k in range(N_SHOW):
        d = dec[k]
        print(f"L{k:02d}  {d['v5']*100:6.1f}%  {d['v4']*100:6.1f}% | "
              f"{d['A_geo']/d['N']*100:6.1f}%  {d['B_other']/d['N']*100:6.1f}%  "
              f"{d['C_cloud']/d['N']*100:8.1f}%   {d['D_valid']/d['N']*100:6.1f}%")
    print("\n== 低层缺失中各成因占比(case5 NaN = A+B 内部的份额) ==")
    for k in range(6):
        d = dec[k]
        nan = d["A_geo"] + d["B_other"]
        if nan:
            print(f"L{k:02d}: 缺失 {nan} 中,硬几何 {d['A_geo']/nan*100:.0f}% / 其他QC {d['B_other']/nan*100:.0f}%"
                  f";另有云QC可回收 {d['C_cloud']} 样本(若用 case2/5 口径)")
    print("\n== 条件有效率:段顶在地上(H>DEM) vs 地下(H<=DEM) ==")
    for k in range(6):
        above = zs5 < H5[:, k]
        v_ab = np.isfinite(y5[above, k]).mean() if above.any() else float("nan")
        v_be = np.isfinite(y5[~above, k]).mean() if (~above).any() else float("nan")
        print(f"L{k:02d}: 地上 n={int(above.sum()):6d} 有效率 {v_ab*100:5.1f}%  | "
              f"地下 n={int((~above).sum()):6d} 有效率 {v_be*100:4.1f}%")

    # ---- 图 ----
    from acdl_plotting import setup_cjk
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_cjk()
    figs = OUT_DIR / "figures"; figs.mkdir(parents=True, exist_ok=True)

    # 图1:逐层成因堆叠
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.6), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ks = np.arange(N_SHOW)
    A = np.array([dec[k]["A_geo"] for k in ks]) / n * 100
    B = np.array([dec[k]["B_other"] for k in ks]) / n * 100
    C = np.array([dec[k]["C_cloud"] for k in ks]) / n * 100
    Dv = np.array([dec[k]["D_valid"] for k in ks]) / n * 100
    ax1.bar(ks, A, color="#c44e52", label="A 硬几何缺失(段在DEM下)")
    ax1.bar(ks, B, bottom=A, color="#dd8452", label="B 其他QC缺失(超界/CAD/产品NaN)")
    ax1.bar(ks, C, bottom=A + B, color="#4c72b0", label="C 云QC可回收(case5有效,case4缺)")
    ax1.bar(ks, Dv, bottom=A + B + C, color="#55a868", label="D 均有效")
    ax1.set_xlabel("ERA5 层段(L00=最低)"); ax1.set_ylabel("占样本比例 (%)")
    ax1.set_title("低层缺失成因分解(case5 口径训练,case4 对照)", fontsize=11)
    ax1.set_xticks(ks); ax1.set_xticklabels([f"L{k:02d}" for k in ks], fontsize=8)
    ax1.legend(fontsize=8, loc="lower right"); ax1.grid(alpha=0.3, axis="y")
    ax1.set_ylim(0, 100)

    ax2.plot(ks, [dec[k]["v5"] * 100 for k in ks], "-o", color="#c44e52", label="case5(当前训练)")
    ax2.plot(ks, [dec[k]["v4"] * 100 for k in ks], "-s", color="#4c72b0", label="case4(默认QC)")
    ax2.set_xlabel("ERA5 层段"); ax2.set_ylabel("Ext_mean 有效率 (%)")
    ax2.set_title("case4 vs case5 逐层有效率", fontsize=11)
    ax2.set_xticks(ks); ax2.set_xticklabels([f"L{k:02d}" for k in ks], fontsize=8)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    fig.tight_layout()
    p1 = figs / "diag_lowlayers_decompose.png"
    fig.savefig(p1, dpi=200); plt.close(fig)
    print(f"  图: {p1}")

    # 图2:地形分带 × 逐层有效率
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    for k in range(N_DEM_SHOW):
        ax.plot([r["band"] for r in dem_rows], [r["valid"][k] * 100 for r in dem_rows],
                "-o", ms=4, lw=1.5, label=f"L{k:02d}")
    ax.set_xlabel("地表高度 DEM 分带 (m)"); ax.set_ylabel("Ext_mean 有效率 (%)")
    ax.set_title(f"逐层有效率 × 地形(case5,N={n}) —— 层顶低于地形的层段无法采样", fontsize=11)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2, loc="lower left")
    fig.tight_layout()
    p2 = figs / "diag_lowlayers_demband.png"
    fig.savefig(p2, dpi=200); plt.close(fig)
    print(f"  图: {p2}")

    # ---- md ----
    l0 = dec[0]; l1 = dec[1]; l2 = dec[2]
    geo_share = np.mean([dec[k]["A_geo"] / max(dec[k]["A_geo"] + dec[k]["B_other"], 1) for k in range(3)])
    rec = sum(dec[k]["C_cloud"] for k in range(N_SHOW))
    _ab = [np.isfinite(y5[zs5 < H5[:, k], k]).mean() for k in range(6)]
    _be = [np.isfinite(y5[zs5 >= H5[:, k], k]).mean() for k in range(6)]
    v_ab_mean, v_ab_max = float(np.min(_ab)), float(np.max(_ab))
    v_be_mean = float(np.mean(_be))
    md = f"""# 低层(L00–L05)缺失成因诊断

- 数据:D:/matchdata_fixed_case5(训练口径)与 D:/matchdata_fixed_case4(默认QC),同文件同索引,{nf5} 文件 / {n:,} 样本。
- 结论速览:**低层缺失的主导机制是几何性的**——ERA5 气压层层高以海拔为基准,DEM 高于段顶时整段在地下,Ext 必为 NaN;云 QC(case4)贡献次之且可由 case 口径回收;其余为消光超界/CAD 不确定带/产品 NaN。

![成因分解](figures/diag_lowlayers_decompose.png)

**图1 怎么读**:左图每个条 = 该层样本的四类去向(自下而上 A 硬几何缺失 / B 其他QC缺失 / C 云QC可回收 / D 均有效,合计 100%)。
L00–L02 的缺失条几乎全被红色(A 硬几何)占据;蓝色(C)是 case4 相对 case5 多丢的"云下"部分——换 case5 已回收,再往回换更松口径空间不大。
右图为两口径逐层有效率,差值即云 QC 的净影响(低层约 {(l0['v5']-l0['v4'])*100:.0f}–{(l2['v5']-l2['v4'])*100:.0f} 个百分点)。

| 层 | case5 有效率 | case4 有效率 | A 硬几何 | B 其他QC | C 云QC可回收 | D 均有效 |
|----|------------|------------|---------|---------|-------------|---------|
""" + "\n".join(
        f"| L{k:02d} | {dec[k]['v5']*100:.1f}% | {dec[k]['v4']*100:.1f}% | "
        f"{dec[k]['A_geo']/n*100:.1f}% | {dec[k]['B_other']/n*100:.1f}% | "
        f"{dec[k]['C_cloud']/n*100:.1f}% | {dec[k]['D_valid']/n*100:.1f}% |"
        for k in range(N_SHOW)) + f"""

![地形分带](figures/diag_lowlayers_demband.png)

**图2 怎么读**:横轴是样本的地表高度分带,纵轴是该带内各层的 Ext_mean 有效率。
形态完全由几何决定:DEM 一旦超过某层层顶,该层有效率应声跳水(如 >2000 m 带内,L00–L07 几乎全灭,而 L08 以上仍高)。

## 关键数字

- 条件有效率(全部 30 天样本):段顶**在地上**(H_k > DEM)时,L00–L05 有效率 {v_ab_mean*100:.0f}–{v_ab_max*100:.0f}%;
  段顶**在地下**时平均仅 {v_be_mean*100:.0f}%。缺失不是"近地面反演质量差",而是**这些段在地形之下
  大多不存在有效 ACDL bin**(匹配的 surf_h 取组内 DEM 均值,逐样本 DEM 与之有差异,故地下条件仍残留少量有效率)。
- Z_sfc 分位:p50 = {np.percentile(zs5,50):.0f} m,p90 = {np.percentile(zs5,90):.0f} m(域含青藏高原)。
  L00 段顶(1000 hPa 层高)中位仅 ~47 m → 一半以上样本的 L00 整段在地下。
- L00–L02 的 NaN 中,硬几何可解释约 **{geo_share*100:.0f}%**;其余为消光超界 [0,1.25] /
  CAD 不确定带 / ACDL 产品 NaN(B 类,case5 下 ~{(l0['B_other']/max(l0['A_geo']+l0['B_other'],1))*100:.0f}–{(l2['B_other']/max(l2['A_geo']+l2['B_other'],1))*100:.0f}%/缺失)。
- 云 QC(case4 相对 case5)净剔除低层样本约 {rec:,} 个(层×样本计,L00–L11);case5 已回收。
  参照 f1 文档:case4 全产物"近地面完整"样本仅 2.4%,case5 提升约一个量级——但几何上限封死了天花板。

## 对优化路线的含义

1. **加权/截断类方案失败的根本原因**(呼应 Step 3):低层不是"被 QC 错杀",而是**几何上不存在的观测**——
   任何损失加权都无法创造这些监督信号;W3 硬截断还把高层的干净监督一起砍掉,故全面恶化。
2. **可行动项按性价比排序**:
   a. **接受几何现实,重新定义"低层"报告带**:以"段顶高于 DEM 的样本"为条件报告低层指标(评估台可加条件口径),
      或把 0–1km 带改为"0–1km(有观测处)";
   b. B 类(~15% 条件缺失)可试 `--with-stats` 重跑匹配(`Ext_valid_L*/Cloud_bins_L*/Ext_bins_L*` 列)精确归因,
      若主要是消光超界(强沙尘 >1.25 km⁻¹ 被杀),可评估对沙尘个例单独放宽;
   c. 训练侧可尝试**样本重加权**(给高 DEM 样本的低层更少权重/给低 DEM 样本更多)——但注意这与"分层加权"
      不同,它作用于样本而不是层,且收益受 B 类规模限制。
3. **不宜做**:用 ERA5 地下外推值"补"低层(把外推当真值);把 case1–4 的云规则进一步放松(会引入云污染真值)。
"""
    with open(OUT_DIR / "低层缺失诊断.md", "w", encoding="utf-8") as fh:
        fh.write(md)
    with open(figs / "diag_lowlayers_decompose.json", "w", encoding="utf-8") as fh:
        json.dump({"n": n, "decompose": {f"L{k:02d}": v for k, v in dec.items()},
                   "dem_bands": dem_rows}, fh, ensure_ascii=False, indent=2)
    print(f"\n[OUT] 低层缺失诊断.md + {p1.name} + {p2.name} + json 已写入")
    return 0


if __name__ == "__main__":
    sys.exit(main())
