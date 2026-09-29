# 评估台报告 — AGLB0

- 运行目录:`结果汇总\runs\train_agl_b0` | 标签:`Holdout_AGLB0` | N_test = 6,972 | 生成于 2026-09-28 23:02
- 配置:target=abs  w_profile=none  log_target=False  softplus=False  tag=AGLB0
- 指标定义与计算方式:[指标说明](../../../指标说明.md)(MAE/nMAE/MB/slope/OD/Gfrac EE/log10 比值/分带规则)

## 一句话结论

> **低层(0–5km 三带平均) MAE = 0.0531 km⁻¹(0–1km 带 0.0713,nMAE 58.8%,
> n=42,567);柱含量 OD 相对误差 bias = +0.9%,|rel| MAE = 25.0%,
> Gfrac(EE) = 57.0%(未达 66% 满意线);
> OD slope = 0.733 / intercept = 0.1948;GCF(n=3903) R² = 0.605,
> MAE = 0.0704。全局 R² = 0.4797(附图口径)。**

## 图 1|柱含量 OD 散点 + EE 包络

![OD scatter](C:/Users/admin/Desktop/Main_ACDL_ERA5_SpatiotemporalAttention_DL/结果汇总/runs/train_agl_b0/plots/bench_od_scatter_AGLB0.png)

**看什么**:点云贴 1:1 虚线的程度 = 柱含量保真能力;红线为 EE 包络 ±(0.05+0.15×AOD)(550nm 形式用于 532nm),
包络内比例 Gfrac = **57.0%**(低于 66% 满意线)。
slope = 0.733 < 1 且 intercept = 0.1948 > 0 → 干净样本略抬、污染样本压扁(动态范围压缩);
bias = +0.9% 说明总量基本无偏。

## 图 2|廓线分位带 + 对数差值曲线

![profile](C:/Users/admin/Desktop/Main_ACDL_ERA5_SpatiotemporalAttention_DL/结果汇总/runs/train_agl_b0/plots/bench_profile_AGLB0.png)

**看什么**:左栏蓝/红 = 观测/预测的逐层中位数与 25–75 分位带(同批样本同掩膜配对)——
红带比蓝带"瘦"即动态范围压缩;右栏 log10(σ̂/σ) 中位线在 0 上下 = 典型样本无系统偏差,
若中位线偏正而均值偏置为负,则是**高消光尾部被低估**的形态(基线 B0 低层即如此)。

## 图 3|逐层 nMAE / MB / slope / R² 四联

![per layer](C:/Users/admin/Desktop/Main_ACDL_ERA5_SpatiotemporalAttention_DL/结果汇总/runs/train_agl_b0/plots/bench_per_layer_AGLB0.png)

**看什么**(自左至右):nMAE 跨层可比的主指标;MB 带符号偏置(偏离 0 的方向 = 总量型误差);
slope 相对 1 的偏离 = 压缩程度(此栏是基线病灶最直观的面板);R² 仅附图(分母为观测方差,跨层不可比)。
红圈层有效样本 <500,数字慎读。

## 分层带表

| 带 | 层均样本 n_obs | MAE (km⁻¹) | nMAE | MB | slope |
|----|--------------|-----------|------|-----|-------|
| 0-1km | 42,567 | 0.0713 | 58.8% | -0.0146 | 0.351 |
| 1-3km | 30,432 | 0.0468 | 60.7% | -0.0063 | 0.400 |
| 3-5km | 19,609 | 0.0411 | 65.1% | -0.0048 | 0.494 |
| 5-10km | 41,097 | 0.0427 | 59.8% | -0.0038 | 0.609 |
| >10km | 69,720 | 0.0071 | 26.0% | -0.0011 | 0.643 |

> 机器可读明细:`bench_AGLB0.json`(含逐层 32 行)/ `bench_AGLB0.csv`。
