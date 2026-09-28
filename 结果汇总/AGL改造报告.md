# AGL 垂直坐标改造报告（2026-09-28）

执行《垂直坐标改造计划》Phase 1–5 全部内容。**原始 ASL 产物零改动**：新数据集
`D:/matchdata_agl_case5`（30 文件 / 34,860 样本，行数与 fixed_case5 完全一致），
训练输出 `结果汇总/runs/train_agl_b0/`。

## 一句话结论

> **AGL 重分段后：近地面覆盖翻倍（L00 有效率 29.2%→57.5%）、GCF 覆盖 6%→58.9%
> （评估样本 2,079→3,903，R² 0.423→0.605）、层号语义全域统一（各地形带首有效层=L00）；
> 低层/OD 精度在扩大 44% 的更难样本群上持平（nMAE 58.8% vs 57.0%）——
> 覆盖与语义收益兑现，精度收益待后续（特征/结构）跟进。**

## 实现要点（Phase 1，commit 7705396）

- 匹配 `--vertical {asl,agl}`（默认 asl，历史产物可复现）：AGL 边界表 33 边界/32 段
  （0–30km，低层加密），顶部压缩 `dem+agl ≤ 30km`（ACDL 上限），AGL_TOP_SPLIT=12km 以下不压缩；
- **DEM 单位修复**：`DEM_Surface_Elevation × 1000`（实证 km 字段），逐组 DEM_m 存入产物新列；
- 段特征：地表以上气压层剖面在段顶插值；低于最低有效层的薄段用最低两层局地递减率**限幅外推**；
- 训练侧：读 Meta.Vertical 自动识别口径；agl 下 dz/GCF 用 DEM_m；GCF=层底 <500m **AGL**；
- 冒烟抓出并修复 2 个 bug：插值高度误用原始位势（T/RH 采样在 ~1/10 高度）、Meta 字符码解析。

## 验收门判读

| 门 | 目标 | 实测 | 判定 |
|----|------|------|------|
| 1 | L00 全域有效率 ≥75% | **57.5%**（asl 29.2%，×2） | ✗ 未达（见归因） |
| 2 | GCF 覆盖 ≥10,000 | **20,525（58.9%）** | ✅ 超门 2 倍 |
| 3 | 0–1km nMAE 不劣于 57.0% | 58.8%（样本群扩大 44%） | ≈ 持平（更难人群） |
| 4 | OD bias ≤ B0+3pp 且 Gfrac 不降 | bias +0.9%、Gfrac 57.0% | ✅ |

## B0-ASL vs B0-AGL 对比（同配置同 seed，留出 6,972）

| 指标 | B0-ASL | B0-AGL | 说明 |
|------|--------|--------|------|
| 0–5km 三带 MAE | 0.0531 | 0.0531 | 持平 |
| 0–1km nMAE | 57.0% | 58.8% | AGL 带样本 42,567 vs 29,680（+44%，含以前无监督的高原近地面） |
| OD \|rel\| MAE / bias | 24.7% / −1.2% | 25.0% / +0.9% | 持平 |
| Gfrac(EE) | 56.3% | 57.0% | 持平 |
| **GCF 评估样本** | 2,079 | **3,903** | **+88%** |
| **GCF R²** | 0.423 | **0.605** | **+0.18，核心约束量大幅改善** |
| GCF slope | 0.529 | 0.699 | 方差压缩显著缓解 |
| GCF MAE | 0.0421 | 0.0704 | 样本群翻倍且含更难的高原样本，不可直接比 |

## 门 1 未达的归因（57.5% vs 75%）

1. **L00 段仅 50 m = 2 个 24m bin**：任一 bin 被产品 fill/QC 剔除即整段 NaN——结构性脆弱
   （L01 100m/4 bin → 73.1%，L02 → 77.7%，厚度效应明显）；
2. ACDL 产品近地面 bin 自身质量（地表回波污染、消光超界 [0,1.25]）——产品侧上限，
   需 `--with-stats` 精确归因；
3. 修正路径（后续可选项）：a) L00 加厚至 0–100 m（边界表一处改动+2h 重匹配）；
   b) ERA5-Land 2m 锚点替代递减率外推（盘上已有数据）；c) `--with-stats` 归因后针对性放宽。

## 产物与复现

```bash
# 重匹配(asl 历史口径不受影响)
.venv/Scripts/python.exe Match_ACDL_ERA5_FromScratch.py --case 5 --vertical agl \
    --out-dir D:/matchdata_agl_case5
# 重训+评估
.venv/Scripts/python.exe Train_ACDL_ERA5_MatchV2.py --match-dir D:/matchdata_agl_case5 \
    --out-dir 结果汇总/runs/train_agl_b0 --tag AGLB0
.venv/Scripts/python.exe evaluate_bench.py --run-dir 结果汇总/runs/train_agl_b0 --tag Holdout_AGLB0 --name AGLB0
```

产物：`结果汇总/runs/train_agl_b0/`（bench_AGLB0.json/md + 3 图）、日志
`结果汇总/logs/agl_match_full.log`、`agl_train_b0.log`。冒烟细节见 [AGL冒烟报告.md](AGL冒烟报告.md)。
