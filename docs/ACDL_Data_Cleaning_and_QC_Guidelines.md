# ACDL 数据清洗与质量控制指南

本文档根据《ACDL HDF 格式气溶胶反演与层次分类产品数据格式说明》以及当前 ACDL–ERA5 匹配方案整理。目标是从 ACDL 532 nm 消光系数中筛选可靠的气溶胶消光标签，并为后续 ERA5 垂直特征匹配和深度学习训练提供统一的质量控制规则。

本文档中的阈值是当前建议值。正式处理全部数据前，应根据实际 ACDL 样本统计结果进行复核。

## 1. 数据产品和目标

ACDL 产品包含 ACPro 廓线反演产品和 ACLay 层次分类产品。本项目的目标变量为：

- `Extinction_Coefficient_532`
- 单位：`km^-1`
- 波长：532 nm
- 原始高度分辨率：约 24 m
- 高度范围：约 -0.96–30 km

项目目标是生成经过云污染筛选、空间时间匹配和垂直聚合的气溶胶消光廓线，用于 ERA5 气象场到 ACDL 消光廓线的建模。

## 2. 字段分类

### 2.1 必须读取的字段

| 字段 | 作用 |
|---|---|
| `Extinction_Coefficient_532` | 532 nm 消光系数，模型训练目标 |
| `CAD_Score` | 逐高度层云–气溶胶分类和置信度判断 |
| `Profile_UTC_Time` | 与 ERA5 时次匹配 |
| `Latitude` | 空间匹配 |
| `Longitude` | 空间匹配 |
| `Altitude` | 垂直层定位 |
| `DEM_Surface_Elevation` | 地表高度和近地面层判断 |
| `Surface_Detection_Flags` | 判断地表是否可靠检测 |

### 2.2 重要的质量控制字段

| 字段 | 用途 |
|---|---|
| `Cloud_Phase_Subtype` | 识别水云、冰云和定向冰晶云 |
| `Column_Feature_Fraction` | 廓线级特征覆盖比例 |
| `Column_Optical_Depth_Cloud_532` | 廓线级云光学厚度 |
| `Column_Optical_Depth_Tropospheric_Aerosols_532` | 对流层气溶胶光学厚度 |
| `Number_Layers_Found` | 廓线中检测到的云/气溶胶层数 |
| `Layer_Top_Altitude` | 检测层顶部高度 |
| `Layer_Base_Altitude` | 检测层底部高度 |
| `Particulate_Depolarization_Ratio_Profile_532` | 辅助识别冰云、沙尘等非球形粒子 |
| `Total_Backscatter_Coefficient_532` | 辅助识别强回波、异常峰值和饱和 |

### 2.3 可用于分层分析的字段

| 字段 | 用途 |
|---|---|
| `Aerosol_Subtype_Multi` | 海洋、沙尘、烟雾、污染大陆等气溶胶子类型 |
| `Day_Night_Flag` | 白天/夜间分层统计 |
| `IGBP_Surface_Type` | 地表类型和地理分层分析 |
| `Tropopause_Height` | 对流层/平流层边界参考 |

## 3. 消光系数有效范围

文档给出的 `Extinction_Coefficient_532` 有效范围为：

```text
0.0–1.25 km^-1
```

无效填充值为 `NaN`。因此，当前项目不应允许负消光值，也不应使用过大的上限，例如 `50 km^-1`。

建议的基础有效性判断为：

```matlab
valid_ext = isfinite(ext) ...
          & ext >= 0 ...
          & ext <= 1.25;

ext(~valid_ext) = NaN;
```

这里的范围是产品文档规定的物理/产品有效范围，不代表所有落在范围内的值都一定可靠；仍必须结合 CAD 和质量字段筛选。

## 4. 云–气溶胶筛选

### 4.1 CAD_Score 的解释

`CAD_Score` 是逐高度层的云–气溶胶鉴别分数：

- 正值：倾向于云；
- 负值：倾向于气溶胶；
- 绝对值越大：分类置信度越高；
- 接近 0：云和气溶胶难以区分；
- `NaN`：无效；
- `101–106` 等特殊值：不能按普通 CAD 分数直接解释，应依据产品文档单独处理。

### 4.2 当前建议阈值

| CAD 范围 | 判定 | 处理 |
|---|---|---|
| `CAD >= 20` | 云候选 | 屏蔽 |
| `-20 < CAD < 20` | 分类不确定 | 屏蔽 |
| `CAD <= -20` | 气溶胶候选 | 可保留 |
| `NaN` 或特殊编码 | 无效/未知 | 屏蔽 |

主数据集建议使用：

```matlab
valid_cad = isfinite(cad) & cad <= -20;
```

高置信度敏感性数据集可以使用：

```matlab
cad <= -70
```

`CAD <= -20` 是当前主规则，`CAD <= -70` 用于评估阈值敏感性。正式阈值仍应结合样本数量、空间分布和消光统计结果复核。

### 4.3 Cloud_Phase_Subtype

当 `Cloud_Phase_Subtype > 0` 时，说明对应高度层存在云相态分类，应屏蔽该高度层。其编码包括未知云、冰云、水云和定向冰晶云。

该字段是 CAD 的辅助判断，不能单独代替 CAD；`0` 或 `NaN` 也不能证明该层一定无云。

### 4.4 逐高度层最终有效条件

推荐的初始规则为：

```matlab
valid_bin = isfinite(ext) ...
         & ext >= 0 ...
         & ext <= 1.25 ...
         & isfinite(cad) ...
         & cad <= -20 ...
         & (cloud_phase <= 0 | isnan(cloud_phase));
```

此外还应屏蔽地表以下的高度层，以及产品质量标识明确判定为无效的 bin。

## 5. 廓线级质量控制

云污染应优先按高度 bin 屏蔽，而不是只要一条廓线出现云就整条删除。廓线中仍然存在足够晴空气溶胶层时，可以保留有效部分。

### 5.1 Number_Layers_Found

`Number_Layers_Found` 表示一条廓线中发现的云和气溶胶层总数，范围为 `0–15`。

- `0`：没有识别到云/气溶胶层，不等于一定是晴空，也不自动删除；
- `1–15`：存在一个或多个云/气溶胶层，不能据此判断是否为云；
- `NaN` 或超出范围：字段异常。

它适合作为廓线级辅助 QC，不适合作为逐层云掩膜，也不应作为 ERA5 → ACDL 模型输入。

### 5.2 Column_Feature_Fraction

表示 30 km 到地表之间被识别为云或气溶胶特征的信号点比例。它可以用于发现：

- 几乎没有有效特征的低信息廓线；
- 大范围云污染或复杂多层结构。

该字段不区分云和气溶胶，因此不能替代 `CAD_Score`。

### 5.3 建议删除整条廓线的情况

满足以下任一情况时，可删除整条廓线：

- 没有任何有效的气溶胶 bin；
- 有效 bin 比例过低；
- 边界层目标高度全部被云覆盖；
- 地表检测失败且近地面高度无法可靠确定；
- 时间、经纬度或高度字段异常；
- 质量字段显示检索失败、信号饱和或明显异常。

有效 bin 比例的具体阈值暂不固定，建议先统计数据分布后再确定。

## 6. 地表和高度处理

### `Surface_Detection_Flags`

- `1`：成功检测到地表；
- `0`：地表检测失败。

当该字段为 `0` 时，不一定需要删除整条廓线，但应谨慎处理近地面部分，必要时只保留可靠的高空有效层。

### `DEM_Surface_Elevation`

用于判断 ACDL 高度层是否低于地表。低于地表的 bin 不应作为气溶胶消光训练标签。

当前匹配计划同时涉及 SRTM/外部 DEM。正式处理时必须统一地表高度来源，避免 ACDL DEM、SRTM DEM 和 ERA5 地形之间产生不一致。建议：

- 用 ACDL `DEM_Surface_Elevation` 做逐廓线质量检查；
- 按项目最终确定的 DEM 作为垂直聚合下界；
- 在输出中保存实际采用的地表高度来源。

## 7. 层边界字段

`Layer_Top_Altitude` 和 `Layer_Base_Altitude` 给出检测到的云/气溶胶层边界，适合用于：

- 检查 CAD 逐 bin 分类是否合理；
- 判断边界层是否被云覆盖；
- 辅助计算 ERA5 层的有效覆盖情况。

但这些层是云和气溶胶的混合结果，不能单独把所有层都当作气溶胶层。最终仍以逐 bin `CAD_Score` 为主。

## 8. 气溶胶类型和辅助物理量

### `Aerosol_Subtype_Multi`

该字段可用于沙尘、烟雾、海洋气溶胶和污染大陆气溶胶等分层分析。

文档中有效范围写为 `1–10`，但定义又包含 `0 = N/A`。因此建议：

- `1–10`：有有效子类型分类；
- `0` 或 `NaN`：未分类，不直接判为消光无效；
- 仅在专门的气溶胶类型分析中使用该字段做更严格筛选。

### `Particulate_Depolarization_Ratio_Profile_532`

可辅助识别沙尘、冰云和其他非球形粒子，但暂不作为第一层硬筛选条件。

### `Total_Backscatter_Coefficient_532`

可用于发现强回波、异常峰值和可能的信号饱和，但不能单独判断云或气溶胶。

## 9. 防止目标泄漏

以下字段是 ACDL 观测反演或目标质量信息，不能直接作为 ERA5 → ACDL 消光模型输入：

- `CAD_Score`
- `Number_Layers_Found`
- `Cloud_Phase_Subtype`
- `Aerosol_Subtype_Multi`
- `Column_Optical_Depth_*`
- `Layer_Top_Altitude`
- `Layer_Base_Altitude`

它们应仅用于数据清洗、掩膜、样本分层和误差评估。

可考虑作为外部辅助输入或分层变量的字段包括：

- `Day_Night_Flag`
- `IGBP_Surface_Type`
- `DEM_Surface_Elevation`
- `Tropopause_Height`

是否加入模型输入，需要结合最终预测场景决定。

## 10. 推荐的数据清洗顺序

```text
读取 HDF5/转换后的字段
    ↓
检查字段存在性、形状和数据类型
    ↓
检查时间、经纬度、高度和地表高度
    ↓
按 0–1.25 km^-1 清理消光系数
    ↓
按 CAD_Score 做逐 bin 云–气溶胶筛选
    ↓
结合 Cloud_Phase_Subtype 和质量字段进一步屏蔽
    ↓
屏蔽地表以下高度层
    ↓
计算每条廓线有效 bin 数和覆盖率
    ↓
删除有效覆盖不足的廓线
    ↓
进行 ERA5 空间、时间和垂直匹配
    ↓
输出消光平均值、有效覆盖数和质量掩膜
```

## 11. 输出中建议保留的 QC 信息

每个匹配样本或 ERA5 垂直层建议同时保存：

```text
Extinction_Mean
Valid_Aerosol_Bin_Count
Valid_Aerosol_Coverage
Cloud_Bin_Count
Number_Layers_Found
Column_Feature_Fraction
Column_Optical_Depth_Cloud_532
Surface_Detection_Flags
Profile_QC_Flag
```

其中 `Valid_Aerosol_Bin_Count` 和 `Valid_Aerosol_Coverage` 对后续掩膜损失、加权训练和高度层指标计算尤其重要。

## 12. 当前待确认事项

正式修改清洗代码前，需要使用实际 ACDL 文件统计以下内容：

1. `CAD_Score` 中特殊值 `101–106` 的具体含义；
2. `CAD <= -20` 和 `CAD <= -70` 下的样本数量与空间分布；
3. 消光值落在 `0–1.25 km^-1` 范围外的比例；
4. `Cloud_Phase_Subtype` 与 CAD 云分类的一致性；
5. `Number_Layers_Found = 0` 廓线中的有效气溶胶 bin 比例；
6. `Surface_Detection_Flags = 0` 对近地面层的影响；
7. 不同有效覆盖率阈值对最终样本数的影响。

在完成这些统计前，本文档中的阈值均视为可调整的初始方案。

## 13. 云下信号衰减与缺少反演质量字段

当前 ACDL HDF5 文件中未发现以下标准化反演质量字段：

```text
Extinction_QC_532
Overlying_Integrated_Attenuated_Backscatter_532
Layer_IAB_QA
Opacity_Flag
Layer_Base_Extended
Extinction_Uncertainty_532
```

因此，目前无法直接判断某个消光值是否收敛、是否完全衰减、消光不确定度有多大，或激光雷达比是否在反演中被调整。云下消光的质量只能通过现有字段建立保守的风险代理，不能等价于完整的反演质量控制。

向下观测的激光雷达传播路径为：

```text
卫星 → 高空 → 云层 → 云下大气 → 地表
```

云层内部的消光主要包含云粒子贡献；云层下方的回波受到上方云层衰减，可能出现信号不足、消光低估或反演不稳定。因此，有限数值不等于可靠数值。

### 13.1 三级质量标记

建议将每个高度 bin 分为以下类别：

```text
0 = 无效
1 = 高置信度气溶胶
2 = 低置信度气溶胶或云邻近风险
3 = 云内
4 = 云下衰减风险
5 = 地表或地形污染
6 = 其他质量异常
```

#### 高置信度气溶胶

满足：

```text
0 <= Extinction_Coefficient_532 <= 1.25
CAD_Score <= -20
CAD_Score 不是特殊值
不在云层内部
不在最低云底以下
不低于地表
```

这部分作为主训练集候选。

#### 云下衰减风险

如果某个 bin 满足 `CAD_Score <= -20`，但其高度低于某个云层的 `Layer_Base_Altitude`，则标记为云下风险：

```text
CAD_Score <= -20
且 Altitude < Cloud_Layer_Base
```

这类数据不建议直接放入主训练集，可以单独保存并用于敏感性分析。

#### 无效数据

以下情况直接屏蔽：

- `CAD_Score >= 20`；
- `-20 < CAD_Score < 20`；
- CAD 特殊值；
- `Cloud_Phase_Subtype > 0`；
- 位于云层内部；
- 位于地表以下；
- 消光为 `NaN`、小于 0 或大于 `1.25 km^-1`；
- 近地面且 `Surface_Detection_Flags = 0`；
- 有效 bin 数过少。

### 13.2 云层边界的使用

对每条廓线，先用 `CAD_Score` 和 `Cloud_Phase_Subtype` 找到云候选，再利用 `Layer_Top_Altitude` 与 `Layer_Base_Altitude` 构造云层区间：

```text
云层内部：屏蔽
最低云底以下：默认标记为云下衰减风险
云层以上：根据 CAD 和其他 QC 条件保留
```

如果边界层被云遮挡，则该廓线不应作为完整边界层训练样本。云上方的气溶胶仍可以保留，不应因为整条廓线存在云就全部删除。

### 13.3 廓线级云风险字段

`Column_Optical_Depth_Cloud_532` 是整条廓线的云光学厚度，只能作为廓线级风险指标：

- 大于 0 表示存在云影响风险，但不代表整条廓线全部无效；
- 值较大时，云下区域应采用更严格的屏蔽；
- 不在缺少样本统计的情况下人为固定一个通用阈值。

`Column_Feature_Fraction` 和 `Number_Layers_Found` 同样用于廓线级风险分层，不能替代逐 bin 的 `CAD_Score`。

### 13.4 ACPro 和 ACLay 的对齐要求

`Extinction_Coefficient_532`、`CAD_Score` 等主要来自 ACPro 廓线反演产品，而 `Layer_Top_Altitude`、`Layer_Base_Altitude`、`Number_Layers_Found` 等来自 ACLay 层次分类产品。使用层边界屏蔽消光前，必须确认两个产品的：

- 廓线数量；
- 时间索引；
- 空间索引；
- 水平分辨率；
- 高度坐标定义。

如果分辨率或索引不一致，不能直接按列合并，应先建立明确的空间、时间和廓线对应关系。

### 13.5 建议的输出质量字段

除消光平均值外，匹配结果建议保存：

```text
ACDL_QC_Class
Valid_Aerosol_Bin_Count
Cloud_Bin_Count
Below_Cloud_Risk_Bin_Count
Valid_Aerosol_Coverage
Column_Optical_Depth_Cloud_532
Number_Layers_Found
Surface_Detection_Flags
```

这样可以分别构造高置信度主数据集、包含云下风险的数据集和严格质量敏感性数据集。

### 13.6 相关算法依据

CALIOP 相关研究表明，云下和卷云边缘附近的特征即使具有较高的负 CAD 分数，也可能是云边缘、噪声或上方衰减造成的伪气溶胶。因此，CAD 不能单独解决云下信号衰减问题。

可参考：

- Liu et al.，CALIOP 云–气溶胶鉴别算法：<https://journals.ametsoc.org/view/journals/atot/26/7/2009jtecha1229_1.xml>
- CALIOP Level 3 aerosol profile algorithm：<https://pmc.ncbi.nlm.nih.gov/articles/PMC7840064/>
- CALIOP Level 2 Layer Products Quality Summary：<https://asdc.larc.nasa.gov/documents/calipso/quality_summaries/CALIOP_L2LayerProducts_3.01.pdf>

这些资料可作为 ACDL 代理质量筛选的参考，但其中的 CALIOP 专用 QC 位标志不能未经确认直接套用到 ACDL。

---

## 14. 与当前匹配实现的对照（实现细节已外置）

> **逐 bin 六步判据、五种 QC Case（1 严格 / 2 宽松 / 3 折中 / 4 自适应【默认】 / 5 极简）
> 与输出列布局，统一以 [ACDL_ERA5_FromScratch_Matching_Strategy.md](ACDL_ERA5_FromScratch_Matching_Strategy.md) §4、§5 为准。**
> 本节只保留"产品规则 → 实现"的对照表与差异说明，不再重复实现细节。

### 14.1 判据 → 字段 → 实现对照

| 判据（本文档条款） | 字段 | 实现要点 |
|---|---|---|
| 消光有效范围 0–1.25 km⁻¹（§3） | `Extinction_Coefficient_532` | 有限且落在区间内；NaN/负值/>1.25 一律不参与 |
| 云 / 气溶胶判别（§4.1、§4.2） | `CAD_Score` | 气溶胶候选 `CAD ≤ -20`；云候选 `CAD ≥ 20`；特殊编码 101–106 记为无效；**缺该字段时退化为"只用消光范围"** |
| 云（相态补充）（§4.3） | `Cloud_Subtype_Multi` | 逐 bin，编码 0=N/A、1=unknown、2=ice、3=water、4=oriented ice；`>0` 视为云 |
| 云下风险（§13.1–13.3） | 同上 + `Column_Optical_Depth_Cloud_532` | "厚云"判定：相态为水云/未知，或连通云块厚度 ≥1000 m，或廓线 COD ≥0.3 |
| 地下屏蔽 / 最低段下界（§6） | `DEM_Surface_Elevation` | 逐廓线屏蔽地下 bin；最低段下界取其均值，缺失时兜底 `H0` |

### 14.2 与本文档其它条款的差异（当前实现口径）

1. **只有 ACPro，没有 ACLay**：§5.1、§7、§13.2、§13.4 依赖的 `Number_Layers_Found`、`Layer_Top/Base_Altitude`、`Column_Feature_Fraction` 等**当前不可用**；"云底"只能用**最低云 bin 高度**近似，`ACDL_QC_Class` 的 0–6 全类别尚未实现。
2. **相态字段命名**：本文档旧称 `Cloud_Phase_Subtype`，本产品实际为 **`Cloud_Subtype_Multi`**（逐 bin）。
3. **"云下"不再一律屏蔽**：默认 `Case 4` 厚云下方丢弃、薄云下方保留（仅丢 1000 m 缓冲带）；需严格全集用 `Case 1`，需最大样本量用 `Case 5`（仅消光范围+地表以上，不做云过滤）。
4. **消光上限口径**：匹配与训练侧统一用 **1.25 km⁻¹**；如有历史脚本仍按 50 裁剪，属于旧口径，勿混用。
5. **防泄漏仍然适用**（§9）：`CAD_Score`、`Cloud_Subtype_Multi`、`Column_Optical_Depth_Cloud_532` 等只做掩膜/计数，**不得作为模型输入**。
