# 评估台报告 — E2c

- 运行目录:`结果汇总\runs\train_exp_E2c` | 标签:`Holdout_E2c` | N_test = 6,972 | 生成于 2026-09-28 19:11
- 配置:target=abs  w_profile=none  log_target=False  softplus=True  tag=E2c
- 指标定义与计算方式:[指标说明](../../../指标说明.md)(MAE/nMAE/MB/slope/OD/Gfrac EE/log10 比值/分带规则)

## 一句话结论

> **低层(0–5km 三带平均) MAE = 0.0567 km⁻¹(0–1km 带 0.0708,nMAE 60.2%,
> n=29,680);柱含量 OD 相对误差 bias = +6.5%,|rel| MAE = 30.0%,
> Gfrac(EE) = 50.8%(未达 66% 满意线);
> OD slope = 0.806 / intercept = 0.1709;GCF(n=2079) R² = 0.316,
> MAE = 0.0457。全局 R² = 0.4552(附图口径)。**

## 图 1|柱含量 OD 散点 + EE 包络

![OD scatter](C:/Users/admin/Desktop/Main_ACDL_ERA5_SpatiotemporalAttention_DL/结果汇总/runs/train_exp_E2c/plots/bench_od_scatter_E2c.png)

**看什么**:点云贴 1:1 虚线的程度 = 柱含量保真能力;红线为 EE 包络 ±(0.05+0.15×AOD)(550nm 形式用于 532nm),
包络内比例 Gfrac = **50.8%**(低于 66% 满意线)。
slope = 0.806 < 1 且 intercept = 0.1709 > 0 → 干净样本略抬、污染样本压扁(动态范围压缩);
bias = +6.5% 说明总量存在系统性偏移。

## 图 2|廓线分位带 + 对数差值曲线

![profile](C:/Users/admin/Desktop/Main_ACDL_ERA5_SpatiotemporalAttention_DL/结果汇总/runs/train_exp_E2c/plots/bench_profile_E2c.png)

**看什么**:左栏蓝/红 = 观测/预测的逐层中位数与 25–75 分位带(同批样本同掩膜配对)——
红带比蓝带"瘦"即动态范围压缩;右栏 log10(σ̂/σ) 中位线在 0 上下 = 典型样本无系统偏差,
若中位线偏正而均值偏置为负,则是**高消光尾部被低估**的形态(基线 B0 低层即如此)。

## 图 3|逐层 nMAE / MB / slope / R² 四联

![per layer](C:/Users/admin/Desktop/Main_ACDL_ERA5_SpatiotemporalAttention_DL/结果汇总/runs/train_exp_E2c/plots/bench_per_layer_E2c.png)

**看什么**(自左至右):nMAE 跨层可比的主指标;MB 带符号偏置(偏离 0 的方向 = 总量型误差);
slope 相对 1 的偏离 = 压缩程度(此栏是基线病灶最直观的面板);R² 仅附图(分母为观测方差,跨层不可比)。
红圈层有效样本 <500,数字慎读。

## 分层带表

| 带 | 层均样本 n_obs | MAE (km⁻¹) | nMAE | MB | slope |
|----|--------------|-----------|------|-----|-------|
| 0-1km | 29,680 | 0.0708 | 60.2% | -0.0044 | 0.446 |
| 1-3km | 37,719 | 0.0506 | 62.1% | -0.0020 | 0.435 |
| 3-5km | 19,437 | 0.0487 | 67.9% | -0.0051 | 0.453 |
| 5-10km | 34,164 | 0.0527 | 66.1% | +0.0015 | 0.605 |
| >10km | 62,730 | 0.0178 | 41.1% | -0.0017 | 0.530 |

> 机器可读明细:`bench_E2c.json`(含逐层 32 行)/ `bench_E2c.csv`。
