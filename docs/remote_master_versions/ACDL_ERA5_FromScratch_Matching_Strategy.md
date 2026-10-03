# ACDL × ERA5 从头匹配策略(当前实现版,2026-09-09)

> 本文件描述**当前代码实际执行**的匹配策略(脚本 [Match_ACDL_ERA5_FromScratch.py](Match_ACDL_ERA5_FromScratch.py))。
> 取代早期的"追加式增强"口径(相关旧文档已删除)与设计稿 [ACDL_ERA5_Matching_FromScratch_Plan.md](ACDL_ERA5_Matching_FromScratch_Plan.md)(设计依据,仍可参考)。
> 训练侧说明见 [Train_ACDL_ERA5_MatchV2_README.md](Train_ACDL_ERA5_MatchV2_README.md);产品级清洗规则见 [ACDL_Data_Cleaning_and_QC_Guidelines.md](ACDL_Data_Cleaning_and_QC_Guidelines.md);**研究动机(f1/GCF 的定位)见 [ACDL_f1_Motivation_and_Scope.md](ACDL_f1_Motivation_and_Scope.md)**。

---

## 1. 一句话概括

**把 ACDL 原始廓线(ACPro)降采样到 ERA5 的尺度**:空间上归到最近的 **0.25° ERA5 格点 + 整点(±30 min)**,同格点同时次的廓线先平均;垂直上按 ERA5 各等压面的位势高度 `H=z/g0` 分成 **32 段**,得到每段的层平均消光;同时用 **CAD + 云相态子类型 + 云光学厚度** 做逐 bin 云质控(**5 种口径** `--case 1..5`,默认 4 自适应"厚云下方丢弃、薄云下方保留");输出 MATLAB v7.3 长表(每行一个"格点-小时"样本;列数 lite 132 / 补列后 135 / `--with-stats` 264)。

---

## 2. 输入数据

| 数据 | 说明 | 用到的字段 |
|---|---|---|
| ACDL(ACPro) | `D:/ACDL/Data/ACDL/ProfileMat/ACDL11_YYYYMMDD*.mat`,内含 `Output` 结构体(v5/v7.3 均可) | `Extinction_Coefficient_532`(1291×N,km⁻¹)、`CAD_Score`(逐 bin)、`Cloud_Subtype_Multi`(逐 bin 相态子类型)、`Column_Optical_Depth_Cloud_532`(廓线级)、`Latitude`、`Longitude`、`Profile_UTC_Time`(TAI93)、`Altitude`(km,1291)、`DEM_Surface_Elevation`(逐廓线,m) |
| ERA5 气压层 | `D:/era5_mat/era5_pressure_levels/0.25deg/YYYYMM/{temperature,relative_humidity,geopotential}_YYYYMM.mat`(v7.3) | `.Data`(MATLAB 逻辑 `[lat,lon,level,time]`)、`.Level`(hPa **降序 1000→10**,32 层)、`.Lat/.Lon`、`.Time`(datenum) |

**要点**
- 位势来自 **pressure level 的 `geopotential`**(`H = z/g0`),**不用 single-level、不做测高积分**;
- ERA5 文件用 h5py 读,**逻辑形状 = 磁盘形状反序**,需 `.T`;
- ACDL 时间:TAI93 → datenum,换算 `datenum = TAI93 × 1e-4 / 86400 + 730486.5`(复刻旧匹配器,保持与历史样本可比,可在 `CONFIG` 调);
- **ERA5-Land 近地面五项不参与本匹配**;ERA5 single-level 的 **BLH / TCWV / 地表位势(→`Z_sfc_m`)** 由匹配脚本**直接并入输出**(见 §5.0.1,可用 `--no-super-levels` 关闭)。

---

## 3. 匹配流程

### 3.1 空间与时间(决定"行")
- 用 ERA5 0.25° 网格建 KDTree(球面 xyz),每条 ACDL 廓线取**最近格点**,记录 `NodeDist_deg`;
- **空间距离上限**:若廓线到该最近格点的球面距离 > `MAX_NODE_DIST_DEG`(默认 **0.18°**)→ **整条廓线丢弃**。
  - **为什么必须加**(2026-09-22 修正):ACDL 是**全球观测**(实测 lon[-180,180]、lat[-82,82]),而 ERA5 域只有 `73–135°E / 3–54°N`。不加这道限制时,`tree.query` 会把域外几千公里外的廓线**吸附到最近的边界节点**上 —— 实测 **94.7% 的廓线、82% 的匹配行**都是这种错配,而且边界节点会把半个地球的廓线平均到一起(例:被记成 `(73.0°E, 30.0°N)` 的那一行,实际平均了**真实经度 −177.9°~+58.6°** 的 112 条廓线)。这些样本的 ERA5 输入与 ACDL 目标之间**没有任何物理关系**,是此前 R² 上不去的直接原因之一。
  - 阈值取 **0.18°**:0.25° 格子的半对角线 = `0.25/2 × √2 ≈ 0.1768°`,所以真正属于某节点的廓线必然 ≤ 此值,超过的一定是域外。
  - 设 `MAX_NODE_DIST_DEG = None`(或极大值)可退回旧行为,仅供复现历史产物。
- 时间:**吸收到最近的 ERA5 整点**;若 |时差| > `TIME_TOL_MIN`(默认 **30 min**)→ **整条廓线丢弃**;
- 按 **(格点编号, 时次索引) 分组**,每个分组产出**一行**。
  - 日志示例:`(节点,时次) 组数 1685; 时间超窗剔除 0 廓线; 域外剔除 108081 廓线(距离上限 0.18°) → 保留 6053/114144 条`

### 3.2 垂直(决定"目标列")
1. 取该格点该小时的 ERA5 列:`z/t/r`(32 层)→ `H = z/g0`(强制单调递增,防偶发逆序);
2. **地表下界** = 该组廓线 `DEM_Surface_Elevation` 的均值;缺失/无效 → 兜底为最低层高度 `H0`;
3. **段边界** = `[min(surf, H0), H0, H1, …, H31]` → **32 段**:
   - `L00` = 地表 → `H0`;`L01` = `H0→H01`;…;`L31` = `H30→H31`;
   - 与 ERA5 **层点**(`k00..k31`)相差一位:`Lk` 夹在 `k-1` 与 `k` 两个等压面之间。
4. 每段统计:落段 ACDL bin 的**算术平均**(仅通过 QC 的 bin)+ **有效 bin 数** + 总 bin 数。

---

## 4. 质控(仅用 ACPro 字段)

### 4.1 逐 bin 基础掩膜
```
valid_base = isfinite(ext) & 0 ≤ ext ≤ 1.25            # 产品有效范围
aer        = isfinite(CAD) & CAD ≤ -20                 # 气溶胶候选(排除特殊编码 101–106)
cloud      = (CAD ≥ 20) | (Cloud_Subtype_Multi > 0)    # 云
below_surf = 高度 < 该廓线 DEM_Surface_Elevation
```
- **无 `CAD_Score` 时退化**:不做 CAD 过滤,仅用消光范围(避免把 bin 全判为非气溶胶 → 全 NaN)。

### 4.2 `Cloud_Subtype_Multi` 相态编码与默认厚薄
| 编码 | 含义 | 下方数据 | 默认处理 |
|---|---|---|---|
| 3 | water 水云 | ❌ 不可用 | **厚云** → 下方丢弃 |
| 2 | ice 冰云 | ✅ 可用 | **薄云** → 下方保留(留缓冲带) |
| 4 | oriented ice 定向冰晶 | ✅ 可用 | 同上 |
| 1 | unknown 未知 | ⚠️ 保守 | **按厚处理**(`CLOUD_UNKNOWN_AS_THICK=True`,可关) |
| 0 | N/A 无分类 | — | 不当作云(回落 CAD) |
| NaN | 无效 | — | 同上 |

### 4.3 "厚云"判定(三条规则,任一命中即为厚)
1. 相态编码命中 `CLOUD_THICK_CODES`(默认 `(3,)` 水云)/未知类型;
2. **连通云块厚度 ≥ `CLOUD_THICK_M`**(默认 1000 m)——厚冰云也会被判厚;
3. **廓线 COD ≥ `CLOUD_THICK_COD`**(默认 0.3,占位值)→ 该廓线所有云 bin 记厚。

### 4.4 五种 QC Case(一次只跑一种,`--case 1|2|3|4|5`)
| Case | 规则 | 云下数据 |
|---|---|---|
| 1 严格 | 有效气溶胶 且 非云下 且 非地下 | 全丢 |
| 2 宽松 | 有效气溶胶 且 非地下 | 全留 |
| 3 折中 | 仅丢云底下方 `CLOUD_BUFFER_M`(1000 m)缓冲带 | 留 |
| **4 自适应(默认)** | 厚云下方全丢;薄云下方仅丢缓冲带 | 视云而定 |
| 5 极简 | **只要消光 ∈ [0,1.25] km⁻¹ 且在地表以上**(不做任何云/CAD 过滤) | 全留(含云内) |

> Case 5 用途:**检验"是不是 QC 把数据杀太狠"**,以及给模型提供最大样本量;但它会把**云污染/云下衰减**的 bin 带进来,作为正式训练目标需谨慎(建议与 Case 4 对比着用)。
> 各判据的字段级定义见 §4.1–4.3;逐 bin 六步展开见 [ACDL_Data_Cleaning_and_QC_Guidelines.md](ACDL_Data_Cleaning_and_QC_Guidelines.md) §14。

### 4.5 廓线级与防泄漏
- `Coverage = 有效气溶胶 bin / 地表以上 bin`;`COV_MIN` 以下的廓线整条丢弃(默认 0=不丢);
- `CAD / 相态 / COD` 等 QC 字段**只作掩膜与计数,绝不作为模型输入**(防目标泄漏)。

---

## 5. 输出

**文件**:`D:/matchdata/ACDL_ERA5_MatchV2_YYYYMMDD.mat`(每天一个),MATLAB v7.3,结构体 `MatchV2`:
`Data`(**N 行 × 132/135/264 列,取决于模式**)、`VarNames`、`Level`(32,hPa 降序)、`Meta`(口径记录)。

> **列布局以本节(§5.1)为唯一定义**;训练侧文档只写"用哪些列",不再重复布局。

### 5.0 输出模式(2026-09-13 起)
- **默认 lite:132 列** = 标识 4 + 特征 96 + 目标 32(**测试用统计列已去掉**);
- `--with-stats` 可临时恢复 **264 列**(含 `Ext_valid/Cloud_bins/Below_Cloud_bins/Ext_bins/NProfiles_*/Coverage_mean/NodeDist_deg`),仅诊断用;
- **日级缓存**:`OUT_DIR/_cache/ACDL_YYYYMMDD.npz`(按源文件 size+mtime 校验失效),重跑/调 QC 参数时跳过 .mat 解析;`--no-cache` 强制重建;
- 每天日志打印 `读ACDL / 匹配聚合` 两段耗时,便于观察瓶颈。

### 5.0.1 ERA5 单层列(2026-09-21 起**已并入匹配本体**,135 列)
匹配时直接读取 ERA5 单层场(0.25°,与气压层**同网格同时次**),在每行末尾写出三列:

```bash
python Match_ACDL_ERA5_FromScratch.py --start ... --end ...        # 默认写出 135 列
python Match_ACDL_ERA5_FromScratch.py --no-super-levels ...        # 关闭 → 132 列
python Match_ACDL_ERA5_FromScratch.py --super-levels blh,tcwv ...  # 只并其中几列
```
> 旧产物(132 列)可用 [Add_ERA5_SingleLevel_Columns.py](Add_ERA5_SingleLevel_Columns.py) 后补;**新匹配无需该脚本**。
| 追加列 | 来源(0.25° 单层,逐小时) | 单位 | 用途 |
|---|---|---|---|
| `BLH_m` | `boundary_layer_height` | m | **边界层高度**:近地面气溶胶稀释/堆积,形状与 GCF 的关键驱动 |
| `TCWV_kgm2` | `total_column_water_vapour` | kg m⁻² | 柱水汽含量(湿度背景) |
| `Z_sfc_m` | `geopotential / g0` | m | **地形高度**:①训练时修正 `L00` 厚度(Δz₀ = H_k00 − 地形)②近地面窗口按真实地形起算 |

- 对齐方式:**按 (0.25° 格点, 整点) 精确查表**(匹配产物存的 `ERA5_Lon/Lat` 就是格点值、`ERA5_Time` 就是整点),实测 205,349 行**零未命中、零 NaN**;
- 训练时用 `--add-cols blh,tcwv,zsfc` 启用(已默认写在训练脚本 `CONFIG["ADD_COLS"]`),三者作为**独立 token** 进入网络;**与匹配内联写出完全等价**(已回归验证:135 列逐值一致);
- 详见 [Train_ACDL_ERA5_MatchV2_README.md](Train_ACDL_ERA5_MatchV2_README.md) §2/§3。

> **v3(计划中)**:这一步将被**并入匹配合体** —— 新增脚本在匹配时直接读 ERA5 单层(0.25°,同网格同时次)写出这三列,v3 与 v2 并存、互为校验;详见 [ACDL_ERA5_Matching_FromScratch_Plan.md](ACDL_ERA5_Matching_FromScratch_Plan.md) §11。

### 5.1 列布局(顺序 = 训练相关在前,统计在后)
| 列区间(0 基) | 名称 | 块 |
|---|---|---|
| 0–3 | `ERA5_Lon, ERA5_Lat, ERA5_Time(datenum), Hour` | 标识 |
| 4–35 | `H_k00..H_k31`(m,位势高度) | **训练特征** |
| 36–67 | `T_k00..T_k31`(K) | 训练特征 |
| 68–99 | `RH_k00..RH_k31`(%) | 训练特征 |
| 100–131 | `Ext_mean_L00..L31`(km⁻¹,层平均消光) | **训练目标** |
| 132–263 | `Ext_valid_L*` / `Cloud_bins_L*` / `Below_Cloud_bins_L*` / `Ext_bins_L*` / `NProfiles_*/Coverage_mean/NodeDist_deg` | **仅 `--with-stats` 时存在**(测试用) |
| 132–134 | `BLH_m, TCWV_kgm2, Z_sfc_m` | **匹配时直接写出**(`SUPER_LEVELS`/`--super-levels`,`--no-super-levels` 关闭);见 §5.0.1 |

> 训练用时:**前 132 列**(4 标识 + 96 特征 + 32 目标)即为 X/y;默认输出没有统计列,补列后末尾多 3 列 ERA5 单层量。

### 5.2 行/列语义
- **一行 = 一个 (0.25° 格点 × 整点时次) 样本**:该格点该小时内的 ACDL 廓线(经 QC)平均并垂直降采样后的结果;
- 同一经纬度一天可有多行(每个有观测的整点一行);
- (**仅 `--with-stats` 时存在**)`NProfiles_raw` = 该格点小时内的原始廓线数;`NProfiles_used` = 过覆盖率筛后保留数;`Coverage_mean` = 保留廓线的平均有效覆盖率;
- **NaN 的常见来源**:该层段内无通过 QC 的 bin、低于地表、高于顶层、ACDL 本身缺测、云掩膜排除。

---

## 6. 读取与对接训练(Python 示例)

```python
import h5py, numpy as np
with h5py.File("ACDL_ERA5_MatchV2_20220601.mat") as f:
    g = f["MatchV2"]
    Data  = g["Data"][()].T                      # → (行, 132 / 135 / 264,见 §5.1)
    names = [''.join(chr(int(c)) for c in np.asarray(f[r][()]).ravel())
             for r in g["VarNames"][()].ravel()]
    Level = g["Level"][()].ravel()               # 32 个 hPa(降序)

def cols(pref): return [i for i, n in enumerate(names) if n.startswith(pref)]
X = Data[:, 4:100]                               # H/T/RH 特征
y = Data[:, cols("Ext_mean_L")]                  # 目标(32 段)
valid = Data[:, cols("Ext_valid_L")]             # 掩膜/加权用(仅 --with-stats 产物有)
```

---

## 7. 运行

```bash
python Match_ACDL_ERA5_FromScratch.py --start 20220601 --end 20220630      # 默认 Case 4
python Match_ACDL_ERA5_FromScratch.py --case 1 ...                        # 严格
python Match_ACDL_ERA5_FromScratch.py --dry-run ...                       # 只统计不写盘
python Match_ACDL_ERA5_FromScratch.py --selftest                          # 离线自检
```
依赖:`numpy, scipy, h5py, hdf5storage`(建议在 `era5_work` 环境)。
关键配置集中在文件顶部 `CONFIG`(路径、`TIME_TOL_MIN`、`EXT_MIN/MAX`、`CAD_*`、`COV_MIN`、`QC_CASE`、`CLOUD_*`)。

---

## 8. 与其它脚本/文档的关系

| 文件 | 角色 |
|---|---|
| [Match_ACDL_ERA5_FromScratch.py](Match_ACDL_ERA5_FromScratch.py) | **本策略的实现** |
| [Check_ACPro_ACLay_Alignment.py](Check_ACPro_ACLay_Alignment.py) | 字段盘点/ACPro↔ACLay 对齐检查(`--dump` 列全部字段与形状) |
| [ACDL_ERA5_Matching_FromScratch_Plan.md](ACDL_ERA5_Matching_FromScratch_Plan.md) | 设计依据(含需求、里程碑) |
| [ACDL_Data_Cleaning_and_QC_Guidelines.md](ACDL_Data_Cleaning_and_QC_Guidelines.md) | QC 规则来源(其中 ACLay 相关条目当前暂不可用) |
| `Enrich_ACDL_ERA5_Vertical.py` / `*_enhanced.mat` | 已废弃的"追加式"过渡方案(该脚本已删除) |
| [era5_grib_to_mat.py](era5_grib_to_mat.py) / [download_era5.py](download_era5.py) | ERA5 GRIB→MAT 流式转换 与 下载 |

---

## 9. 已知限制与待办

1. **只有 ACPro,无 ACLay**:云底只能用"最低云 bin"近似,而非 ACLay 的真实 `Layer_Base`;`Layer_*`/`Number_Layers_Found`/`Column_Feature_Fraction` 等暂不可用。
2. **阈值待按数据统计**:`CLOUD_THICK_COD=0.3`、`CLOUD_THICK_M=1000`、`CLOUD_BUFFER_M=1000` 目前是经验值;建议先统计"相态×云厚×下方消光可用性"再定。
3. **相态 0/NaN**:按"非云"处理,靠 CAD 兜底;若发现漏判可改判据。
4. **下游口径要统一**:清洗指南的消光有效上限为 **1.25 km⁻¹**,而现有 DL 训练脚本仍 clip 到 50,需在训练侧同步。
5. **DL 侧迁移**:本输出已改为"按 ERA5 层段的 32 维目标 + 层点特征",旧 loader(`Data[:, 8:]` 假设 1291 层)需重写。
6. **单日一文件**:跨月/跨年请用 `--start/--end` 批量跑,目前不做合并文件(需要可加 `--merge`)。
