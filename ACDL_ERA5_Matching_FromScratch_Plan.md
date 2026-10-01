# ACDL×ERA5 从头匹配计划书(设计定稿,2026-09-06;2026-09-09 更新地表下界方案)

> 状态:**设计定稿**。取代早期"旧匹配产物 + 追加列"的描述;**与本文冲突之处以此为准**。
> 背景变更:位势改用 pressure-level 各层 `z/g0`(弃测高积分);匹配方式改为 **ACDL 降采样匹配 ERA5(水平+垂直都降)**;ERA5-Land 近地面五项**剔除、不参与本匹配**;**地表下界改用 ACDL 自带 `DEM_Surface_Elevation`(不再依赖 SRTM,SRTM 推迟)**。
> 关联:垂直换算已改为 **pressure-level `z/g0` 直取层高**(不再做测高积分);相关垂直换算旧文档已删除,现行口径见 [ACDL_ERA5_FromScratch_Matching_Strategy.md](ACDL_ERA5_FromScratch_Matching_Strategy.md) §3.2。

## 1. 定位与目标

> **研究动机(为什么要有 f1、GCF 干什么用)见 [ACDL_f1_Motivation_and_Scope.md](ACDL_f1_Motivation_and_Scope.md)。**

用**单一流程**从原始数据产出干净匹配样本,服务 f1(ERA5 → ACDL 532nm 消光廓线)训练:

> ACDL 原始足印 →(水平:归属 0.25° ERA5 节点 + 整点 ±30min)→ (节点,时次) 平均 →(垂直:把 ACDL 24m bin 按 ERA5 相邻层 `z/g0` 高度聚合为层段)→ 每(节点,时次,ERA5层)输出层平均消光 + 覆盖数 + ERA5 层特征(t/r、层高)。

本计划书只覆盖**匹配设计与数据链**;DL 网络/loader 的迁移在其后单独处理(见 §7)。

## 2. 需求(2026-09-06 与导师讨论确定)

### 2.1 位势:改用 pressure level 各层 geopotential,弃 single-level,弃积分
- hPa→m **采用 pressure level 每层 `geopotential`**:层位势高度 `H_k = z_k / g0`,一层一个值,无需积分。
- **不再采用 single-level 的 geopotential**;相应地 **hypsometric 自地表逐层积分不再需要**(它依赖 single-level `z` 与地表 `sp` 作起点)。
- 数据链路已就绪:[download_era5.py](download_era5.py) 气压层请求含 `geopotential`;[era5_grib_to_mat.py](era5_grib_to_mat.py) 气压层映射含 `z→geopotential`(产出 4D `[lat,lon,level,time]` + `.Level`)。

### 2.2 ACDL 降采样匹配 ERA5:水平 + 垂直都降
- ACDL 分辨率高(24m 垂直、足印逐点),ERA5 低(0.25°、32 层)。按导师意见把 ACDL **降采样到 ERA5 的尺度**,而非把 ERA5 插到 1291 层。
- **水平**:以 ERA5 **0.25° 节点为中心**、±30 min 整点窗口,把落入的 ACDL 足印平均成一行(半格容差,≈0.176°);无足印的 (节点,时次) 不产出。
- **垂直**:ACDL 24 m bin 按相邻 pressure-level 的 `H=z/g0` 为界聚合成"每 ERA5 层一段";每层聚合统计 = **层内算术平均消光 + 覆盖 bin 数**(供掩码/加权)。
- **最底层下界 = 地表高度,来源用 ACDL 自带的 `DEM_Surface_Elevation`**(逐廓线,零额外下载、与 ACDL 同源):把最低 ERA5 层到地表之间的近地面 ACDL bin 也纳入,以免丢掉近地面信号。地表高度仅作**边界/掩码**,不进入训练特征。
  - 该字段缺失/异常(如 `Surface_Detection_Flags=0`)时,兜底取**该柱最低有效气压层高度**(完全基于 ERA5);并在输出记录实际采用的地表高度来源。
  - 好处:避免"ACDL DEM / SRTM / ERA5 地形"三源不一致(清洗指南 §6 的顾虑);SRTM 推迟到后续阶段再定。

### 2.3 剔除 ERA5-Land 近地面五项
- 2m 温度、2m 露点温度(RH)、10m u/v 风、地表气压(ERA5-Land 0.1°):**训练不需要、不参与本匹配**,属后续 PM2.5 阶段;**从头匹配时剔除**,不进匹配样本/输出。
- 这些文件仍保留在盘,供后续阶段使用。

### 2.4 ERA5 single level 0.25° 下载不动
- geopotential / BLH / TCWV 的下载与转换**暂不修改**;是否作为 f1 输入(如 BLH)后续再定。本匹配**不使用** single-level 数据(地表下界改由 ACDL 自带 `DEM_Surface_Elevation` 提供,见 §3/§4.3)。
- **2026-09-16 更新(后处理接入,不改匹配本体)**:已经用 [Add_ERA5_SingleLevel_Columns.py](Add_ERA5_SingleLevel_Columns.py) 把 **`BLH_m` / `TCWV_kgm2` / `Z_sfc_m`(地形高度)** 按 (0.25° 格点, 整点)**精确追加**到匹配产物的末尾(132 → **135 列**),供训练使用。实测 205,349 行**零未命中、零 NaN**。
  - `Z_sfc_m` 还用于**修正形状/GCF 里的 `L00` 厚度**(Δz₀ = H_k00 − 地形),取代原先 `SURF_M=0` 的近似;
  - 细节见 [ACDL_ERA5_FromScratch_Matching_Strategy.md](ACDL_ERA5_FromScratch_Matching_Strategy.md) §5.0.1 与 [Train_ACDL_ERA5_MatchV2_README.md](Train_ACDL_ERA5_MatchV2_README.md);
  - **下一步候选**:把 **10 m 风 u/v、垂直速度**也接进来(形状/输送的关键驱动),那一步很可能需要动匹配本体或另取 ERA5-Land 0.1° 数据。

## 3. 数据输入

| 数据 | 说明 | 在本匹配中的角色 |
|---|---|---|
| ACDL 原始廓线 | 逐足印;1291 层 × 24 m;几何高度 30→−0.96 km;`Extinction_Coefficient_532`、经纬度、TAI93 时间 | 目标(降采样到 ERA5 层) |
| ERA5 pressure levels(t/r/z) | 0.25°,32 层(1000→10 hPa),逐小时;`D:/era5_mat/era5_pressure_levels/0.25deg/...` | 层高(每个 `z/g0`)+ 层特征(t/r) |
| **ACDL `DEM_Surface_Elevation`** | ACDL 产品内自带的逐廓线地表海拔(与消光同文件) | **最低段地表下界**(仅边界/掩码,非特征);缺失时兜底=最低有效 ERA5 层高度 |
| ~~SRTM DEM~~ | 推迟:后续阶段再统一地表源 | 本期不使用 |
| ERA5-Land 0.1° 五项 / ERA5 single-level 0.25° | 保留不删 | **不参与匹配本体**;其中 single-level 的 `BLH`/`TCWV`/地表位势已由补列脚本**追加到匹配产物末尾**(见 §2.4,135 列);Land 五项仍留后续阶段 |

## 4. 匹配流程(定稿口径)

### 4.1 位势层高
每个 (格点,时次):`H_k = z_k / g0`(k=0..31,按 `.Level` 降序 1000→10 hPa 即自下而上)。层间差即各层厚度;H 随高度单调增、随列/时次变。

### 4.2 水平匹配(降采样到 0.25° 节点)
- ACDL 足印归属最近 0.25° 节点(半格容差;距离记录供 QC)。
- 时间:归属最近整点小时,**±30 min** 内。
- 同一 (节点,整点时次) 的多条足印先做廓线平均(或先逐层平均再聚合,见 §4.3;实现时定先后,两者统计一致除非足印内层覆盖不同)。

### 4.3 垂直匹配(降采样到 ERA5 层段)
- 层边界:相邻 level 高度 `[H_k, H_{k+1})`;**最底层下界 = 该廓线 ACDL `DEM_Surface_Elevation`**(逐廓线;参与水平平均的多条廓线各用自身值,或取聚合后的代表值,见 §7);顶层上界 = 最高有限层高度。DEM 缺失/`Surface_Detection_Flags=0` 时兜底为最低有效层高度。
- ACDL 24 m bin 按其几何高度落入所在层段 → **层内算术平均消光** + **覆盖 bin 数**。
- 低于地表或高于顶层、以及缺测 bin:掩码不计入(输出覆盖数=0 的层保留为缺测标记)。
- 结果:每 (节点,时次) 一条**按 ERA5 层序**排列的降采样廓线。

### 4.4 特征(ERA5 侧)
- 每 ERA5 **level**:`t`、`r`(层内均值 or level 点取值见 §7)、`H=z/g0`。
- **不输出** Land 五项;**层高**仍只用 pressure-level `z/g0`(不用 single-level `z` 做积分);但 single-level 的 `BLH`/`TCWV`/地表位势已**直接并入输出**(`Z_sfc_m` 亦用于修正 `L00` 厚度,见 §2.4 与 §11)。

## 5. 输出 schema

> **列布局以 [ACDL_ERA5_FromScratch_Matching_Strategy.md](ACDL_ERA5_FromScratch_Matching_Strategy.md) §5.1 为唯一定义**(lite 132 / 补列后 135 / `--with-stats` 264;标识 → H/T/RH → `Ext_mean` → 统计)。
> 本节仅记录**设计原则**:以 `VarNames` 为唯一索引、每行 = 一个 (0.25° 节点 × 整点时次)、层序与 ERA5 `Level`(hPa 降序)对齐、不使用"前 8 列 + 1291 层"的旧布局。

## 6. 与现有代码/产物关系

- **待替换基线**:`Main_ACDL_ERA5_Match.m`(其 ACDL 读取、时间换算、足印聚合逻辑可移植)。
- **过渡产物退役**:`Enrich_ACDL_ERA5_Vertical.py` 与 `ACDL_ERA5_Matched_*_enhanced.mat`(其"插到 1291 层 + Land sp/z"路线被本文取代;仅可作对照)。
- **可复用件**:ACDL .mat 解码、`#refs#`/VarNames 与 hdf5storage v7.3 写盘、球面 KDTree、ERA5 pressure-level .mat 读取(4D + Level)。
- **文档现状**:匹配/输出/QC 的现行口径统一在 [ACDL_ERA5_FromScratch_Matching_Strategy.md](ACDL_ERA5_FromScratch_Matching_Strategy.md);训练侧在 [Train_ACDL_ERA5_MatchV2_README.md](Train_ACDL_ERA5_MatchV2_README.md);产品级规则在 [ACDL_Data_Cleaning_and_QC_Guidelines.md](ACDL_Data_Cleaning_and_QC_Guidelines.md)。旧文档已删除。

## 7. 剩余待定项(实现前需定)

1. 层/level 对齐与段数:31 段(相邻 level 对)还是含半层/扩展层;每层特征取 **level 点值**还是**层内均值**。
2. 顶层(>最高有限层)与地表以下 ACDL bin 的处置(掩码/丢)的最终口径。
3. **地表下界实现细节**:`DEM_Surface_Elevation` 的有效性判据与兜底阈值;同一 (节点,时次) 多条廓线的 DEM 取哪个值(逐条用各自值聚合,还是先取代表值);`Surface_Detection_Flags=0` 廓线近地面 bin 的处置。SRTM 推迟,不在本期。
4. DL 目标迁移时间表:1291 层 → 每 ERA5 层段;loader/网络改动(单独任务)。

## 8. 里程碑与验收

1. 本文定稿(§7 待定项敲定);
2. 地表下界规则(ACDL `DEM_Surface_Elevation` → 兜底最低有效层)在原型中验证;
3. 原型单日(20220601)从头匹配,QC(节点距离、时次偏移、每层覆盖数、层段物理区间)并与旧增强产物抽检对照;
4. 全月运行 → 退役 Enrich/enhanced → 同步修订策略/垂直 memo 与 DL loader。

## 9. 一句话结论

> 从头匹配 = ACDL 足印按 **0.25° 节点 ±30min** 水平聚合,再按 **pressure-level 各层 `z/g0`(下界用 ACDL 自带 `DEM_Surface_Elevation`,缺失时兜底最低有效层高度)** 垂直聚合成 ERA5 层段平均消光与覆盖数;位势全部来自 pressure level,不做积分;**匹配本体不用 ERA5-Land 五项与 single-level**,产出干净、以 VarNames 索引的样本;后续由补列脚本可选追加 `BLH_m/TCWV_kgm2/Z_sfc_m`(132 → 135 列,见 §2.4)。

---

## 10. QC 实现(决策记录;实现细节见策略文档)

脚本:[Match_ACDL_ERA5_FromScratch.py](Match_ACDL_ERA5_FromScratch.py);对齐检查:[Check_ACPro_ACLay_Alignment.py](Check_ACPro_ACLay_Alignment.py)。

> **本节只记录"为什么这样定";逐 bin 判据、五种 Case 的完整定义与输出列布局,统一以
> [ACDL_ERA5_FromScratch_Matching_Strategy.md](ACDL_ERA5_FromScratch_Matching_Strategy.md) §4 与 §5 为准。**

**决策要点(2026-09-09 定,2026-09-21 增补)**
1. **只用 ACPro 字段**(CAD + `Cloud_Subtype_Multi` + COD)做云质控 —— 因为当前**只有 ACPro**,ACLy 的层边界/层数暂不可用;云底只能用"最低云 bin"近似。
2. **不再"云下一律屏蔽"**:`Case 4(默认)` 厚云下方丢弃、薄云下方仅丢缓冲带 —— 兼顾"防云下衰减污染"与"保住近地面样本";另提供 `Case 1/2/3` 供敏感性对比。
3. **新增 `Case 5(极简)`**:只保留"消光在 0–1.25 且在地表以上"的 bin(不做云/CAD 过滤)—— 用于**诊断 QC 是否杀数据过狠**,以及给模型提供最大样本量(代价:云污染进入训练集)。
4. **保留最低有效层兜底**:`DEM_Surface_Elevation` 缺失时,最低段下界退化为最低层高度 `H0`。
5. **防泄漏**:CAD/相态/COD 只作掩膜与计数,**不作模型输入**。
6. **暂未启用**:ACLy 层边界(`Layer_Top/Base`、`Number_Layers_Found`)—— 需先通过对齐检查确认索引一致后再决定是否替换"最低云 bin"。

---

## 11. v3 已实现(2026-09-21):单层特征并入匹配本体

**结果**:已在 [Match_ACDL_ERA5_FromScratch.py](Match_ACDL_ERA5_FromScratch.py) 内实现 —— 匹配时新增单层月度读取(`Era5SingleLevel`),按天预取所需时次,在每行末尾直接写出三列。**不再需要单独脚本**([Add_ERA5_SingleLevel_Columns.py](Add_ERA5_SingleLevel_Columns.py) 保留,仅供旧的 132 列产物后补)。

**已定范围(用户拍板)**
1. **只并入现有三项**:`BLH`(边界层高度)、`TCWV`(整层水汽)、`Z_sfc`(地表位势/g₀ = 地形高度);
2. **全部用 ERA5 单层 0.25°** —— 与气压层**同网格、同时次** → 匹配就是一次查表,**无双重网格问题**;
3. **v3 与 v2 并存**:新文件名/新目录(`ACDL_ERA5_MatchV3_*.mat`,struct `MatchV3`),v2 产物保留用于对比。

**设计要点(实现时照此)**
- **读数**:新增"单层月度读取"(与 `Era5Month` 并列),惰性打开 `boundary_layer_height_*.mat` / `total_column_water_vapour_*.mat` / `geopotential_*.mat`(0.25°,逐小时),暴露 `Time/Lat/Lon`,并支持按 (r, c, ti) 取值;`Z_sfc = z/g0`。
- **列布局**:建议 **前 132 列与 v2 完全一致**,新三列**追加在末尾**(→135 列)→ **训练脚本零改动**(它按列名读 `BLH_m/TCWV_kgm2/Z_sfc_m`)。
- **复用件**:QC 逻辑(`cad_qc_masks` / `classify_thick_cloud` / 五种 Case)、空间/时间匹配、段聚合、v7.3 写盘、**日级 ACDL 缓存**(缓存内容只含 ACDL 侧字段,与 v3 兼容,可直接复用)。
- **可选项**:`--super-levels {blh,tcwv,zsfc}` 选择要并入哪些;不选则退化为 v2 行为。

**验收结果(2026-09-21,已通过)**
> 同一天、同一 QC 口径下,「匹配内联单层」与「v2(132 列)+ 补列脚本」产物:**列名完全一致、前 132 列逐值一致、单层三列逐值一致(rtol≤1e-6)**。

**原验收标准**
> 对**同一天、同一 QC 口径**运行 v3 与"v2 + 补列脚本",要求:
> ① 前 132 列 **逐值一致**;② 末尾三列(BLH/TCWV/Z_sfc)**逐值一致(rtol≤1e-6)**;
> ③ 行数、`Level`、`Meta` 结构一致。三者都过 → v3 可放心取代"v2+补列"两步流程。

**后续候选(本次不做)**
- **10 m 风 u/v、垂直速度** → 形状/输送的关键驱动;需先确认数据来源(ERA5 单层 0.25° 已有 10m 风?或需 ERA5-Land 0.1°,后者会引入双网格匹配);
- 地表 2 m 温度/露点、地表气压、降水(湿清除)。
