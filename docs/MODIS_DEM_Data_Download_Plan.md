# MODIS / SRTM 数据下载与匹配方案

> **状态更新(2026-09-21)**
> - **SRTM 已推迟**:当前匹配的"最低层地表下界"改用 **ACDL 自带 `DEM_Surface_Elevation`**(缺失兜底 `H0`),**不再依赖 SRTM**;
> - **地形高度另有来源**:ERA5 单层 `geopotential/g0` 已作为 `Z_sfc_m` 追加进匹配样本(用于修正形状/GCF 的 `L00` 厚度),见 [ACDL_ERA5_FromScratch_Matching_Strategy.md](ACDL_ERA5_FromScratch_Matching_Strategy.md) §5.0.1;
> - 因此 **SRTM 的用途改为**:后续 PM2.5 阶段的**地形辅助/校验**(DEM、坡度等),或与 ACDL DEM 做一致性检查;
> - **MODIS(MCD12Q1 土地覆盖、MOD13C1 NDVI)部分仍然有效**,属 PM2.5 阶段的输入特征,与本版匹配无冲突。

## 1. 数据清单

本项目使用以下三类外部地表数据：

| 数据 | 最终产品 | 主要用途 |
|---|---|---|
| 土地覆盖 | `MCD12Q1.061` | 2022 年土地覆盖类别 |
| NDVI | `MOD13C1.061` | 0.05°、16 天植被指数 |
| 地形高程 | `SRTMGL1.003` 或 `NASADEM_HGT.001` | ACDL 垂直匹配的地表下界 |

说明：`MOD12Q1` 是旧版 Terra-only 归档产品，不作为 2022 年数据源；本项目使用当前的 Terra+Aqua 联合产品 `MCD12Q1.061`。

## 2. 土地覆盖：MCD12Q1.061

### 产品特征

- Terra + Aqua 联合产品；
- 全球年度土地覆盖产品；
- 时间范围：2001 年至今；
- 原始空间分辨率：500 m；
- MODIS Sinusoidal 投影；
- HDF-EOS5 格式；
- 分类变量，不是连续数值。

### 使用字段

主要下载并使用：

```text
LC_Type1
```

`LC_Type1` 为 IGBP 土地覆盖分类。建议同时保留：

```text
QC
LW
```

其中 `QC` 用于质量控制，`LW` 为陆地/水体掩膜。

本项目选择：

```text
产品：MCD12Q1.061
年份：2022
主要字段：LC_Type1
辅助字段：QC、LW
```

土地覆盖重采样到 ERA5 网格时使用多数类别（mode/majority vote），禁止双线性或三次插值。

## 3. NDVI：MOD13C1.061

截图中的“0.05°、16 days”对应：

```text
MOD13C1.061
MODIS/Terra Vegetation Indices 16-Day L3 Global 0.05Deg CMG
```

### 产品特征

- Terra MODIS；
- 0.05° 全球经纬度网格；
- 16 天合成产品；
- 提供 NDVI、EVI、质量标志和像元可靠性等字段；
- NDVI 通常以整数存储。

NDVI 应使用比例因子转换：

```text
NDVI_real = NDVI_raw × 0.0001
```

必须利用质量字段过滤无效值、云污染和低可靠性像元，不能直接把填充值作为 NDVI。

本项目选择：

```text
产品：MOD13C1.061
年份：2022
主要字段：NDVI
辅助字段：VI_Quality、pixel reliability
```

如果后续需要更高空间分辨率，可改用 `MOD13Q1.061`（250 m、16 天），但当前方案固定使用 `MOD13C1.061`。

## 4. SRTM DEM

SRTM 不是 MODIS 产品，而是独立的地形高程数据。

### 推荐产品

```text
SRTMGL1.003
```

或：

```text
NASADEM_HGT.001
```

### 产品特征

- 静态地形高程；
- 原始分辨率约 1 arc-second，约 30 m；
- 覆盖约 60°S–60°N；
- 高程单位通常为米；
- 一般以 HGT 或 GeoTIFF 栅格形式提供。

### 项目用途

**当前状态(2026-09-21 更正)**：

```text
不再作为匹配的最低层地表下界
（该下界现由 ACDL DEM_Surface_Elevation 提供，缺失时兜底 H0）
用途改为：PM2.5 阶段的 DEM/坡度等地形辅助，以及与 ACDL DEM 的一致性校验
```

不作为 ERA5 训练特征；地形高度信息现由 **ERA5 单层 `geopotential/g0`** 以 `Z_sfc_m` 列提供(见匹配策略文档 §5.0.1)。

处理范围：

```text
73–135°E
3–54°N
```

推荐流程：

```text
原始 SRTM/NASADEM DEM
    ↓
裁剪研究区域
    ↓
重投影或聚合到 ERA5 0.25°节点
    ↓
得到每个 ERA5 节点的地表高程
```

山区建议同时保留：

```text
mean elevation
median elevation
minimum elevation
maximum elevation
```

## 5. 下载方式

### MODIS 数据

使用 NASA Earthdata + LP DAAC AppEEARS API：

```text
1. 注册 NASA Earthdata 账号
2. 获取 Earthdata 登录凭据
3. 创建 AppEEARS area request
4. 设置研究区域、产品、年份和字段
5. 查询任务状态
6. 下载任务结果
7. 解压并检查 QA
```

研究区域：

```text
North = 54
West  = 73
South = 3
East  = 135
```

建议分别建立：

```text
MCD12Q1.061 / 2022 / LC_Type1 + QC + LW
MOD13C1.061 / 2022 / NDVI + QA
```

AppEEARS 支持按空间范围、时间和变量裁剪，并返回处理后的结果。

### SRTM 数据

通过 NASA Earthdata/LP DAAC 下载原始 SRTM 或 NASADEM 瓦片，再使用 Python、GDAL、rasterio 或 xarray 完成裁剪和聚合。

## 6. 与 ERA5 网格匹配

| 数据 | 原始分辨率 | 匹配处理 |
|---|---:|---|
| MCD12Q1 | 500 m | 多数类别聚合到 ERA5 0.25°节点 |
| MOD13C1 | 0.05° | QA 过滤后平均或按时间匹配到 ERA5 节点 |
| SRTM/NASADEM | 约 30 m | 聚合为 ERA5 节点地表高程 |
| ERA5 | 0.25° | 作为匹配目标网格 |

变量类型对应的重采样方法：

```text
土地覆盖：mode / majority vote
NDVI：质量控制后平均或时间匹配
DEM：mean / median 聚合
```

## 7. 官方参考

- [MCD12Q1 Collection 6.1 User Guide](https://lpdaac.usgs.gov/documents/1409/MCD12_User_Guide_V61.pdf)
- [MODIS Land Cover 产品说明](https://modis.gsfc.nasa.gov/data/dataprod/mod12.php)
- [MODIS Vegetation Index 产品说明](https://modis.gsfc.nasa.gov/data/dataprod/mod13.php)
- [AppEEARS](https://appeears.earthdatacloud.nasa.gov/)
- [NASA LP DAAC](https://www.earthdata.nasa.gov/centers/lp-daac)
