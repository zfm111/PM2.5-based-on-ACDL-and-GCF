# 评估台报告 — W2

- 运行目录:`结果汇总\runs\train_exp_W2` | 标签:`Holdout_W2` | N_test = 6,972 | 生成于 2026-09-28 19:11
- 配置:target=abs  w_profile=decay  log_target=False  softplus=False  tag=W2
- 指标定义与计算方式:[指标说明](../../../指标说明.md)(MAE/nMAE/MB/slope/OD/Gfrac EE/log10 比值/分带规则)

## 一句话结论

> **低层(0–5km 三带平均) MAE = 0.0545 km⁻¹(0–1km 带 0.0674,nMAE 57.0%,
> n=29,680);柱含量 OD 相对误差 bias = +3.2%,|rel| MAE = 34.3%,
> Gfrac(EE) = 42.8%(未达 66% 满意线);
> OD slope = 0.546 / intercept = 0.3335;GCF(n=2079) R² = 0.298,
> MAE = 0.0447。全局 R² = 0.3940(附图口径)。**

## 图 1|柱含量 OD 散点 + EE 包络

![OD scatter](C:/Users/admin/Desktop/Main_ACDL_ERA5_SpatiotemporalAttention_DL/结果汇总/runs/train_exp_W2/plots/bench_od_scatter_W2.png)

**看什么**:点云贴 1:1 虚线的程度 = 柱含量保真能力;红线为 EE 包络 ±(0.05+0.15×AOD)(550nm 形式用于 532nm),
包络内比例 Gfrac = **42.8%**(低于 66% 满意线)。
slope = 0.546 < 1 且 intercept = 0.3335 > 0 → 干净样本略抬、污染样本压扁(动态范围压缩);
bias = +3.2% 说明总量存在系统性偏移。

## 图 2|廓线分位带 + 对数差值曲线

![profile](C:/Users/admin/Desktop/Main_ACDL_ERA5_SpatiotemporalAttention_DL/结果汇总/runs/train_exp_W2/plots/bench_profile_W2.png)

**看什么**:左栏蓝/红 = 观测/预测的逐层中位数与 25–75 分位带(同批样本同掩膜配对)——
红带比蓝带"瘦"即动态范围压缩;右栏 log10(σ̂/σ) 中位线在 0 上下 = 典型样本无系统偏差,
若中位线偏正而均值偏置为负,则是**高消光尾部被低估**的形态(基线 B0 低层即如此)。

## 图 3|逐层 nMAE / MB / slope / R² 四联

![per layer](C:/Users/admin/Desktop/Main_ACDL_ERA5_SpatiotemporalAttention_DL/结果汇总/runs/train_exp_W2/plots/bench_per_layer_W2.png)

**看什么**(自左至右):nMAE 跨层可比的主指标;MB 带符号偏置(偏离 0 的方向 = 总量型误差);
slope 相对 1 的偏离 = 压缩程度(此栏是基线病灶最直观的面板);R² 仅附图(分母为观测方差,跨层不可比)。
红圈层有效样本 <500,数字慎读。

## 分层带表

| 带 | 层均样本 n_obs | MAE (km⁻¹) | nMAE | MB | slope |
|----|--------------|-----------|------|-----|-------|
| 0-1km | 29,680 | 0.0674 | 57.0% | -0.0098 | 0.386 |
| 1-3km | 37,719 | 0.0469 | 57.8% | -0.0121 | 0.349 |
| 3-5km | 19,437 | 0.0492 | 68.6% | -0.0106 | 0.360 |
| 5-10km | 34,164 | 0.0602 | 75.3% | -0.0079 | 0.368 |
| >10km | 62,730 | 0.0231 | 45.5% | -0.0055 | 0.356 |

> 机器可读明细:`bench_W2.json`(含逐层 32 行)/ `bench_W2.csv`。
