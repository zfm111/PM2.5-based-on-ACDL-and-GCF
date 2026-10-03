# 训练脚本说明:ERA5 → ACDL 消光廓线(SpatioTemporalAttention)

> 脚本:[scripts/train_acdl_era5_match_v2.py](scripts/train_acdl_era5_match_v2.py)
> 数据来源:[scripts/match_acdl_era5_from_scratch.py](scripts/match_acdl_era5_from_scratch.py) 产出的 `ACDL_ERA5_MatchV2_YYYYMMDD.mat`
> 相关:[ACDL_ERA5_FromScratch_Matching_Strategy.md](ACDL_ERA5_FromScratch_Matching_Strategy.md)(匹配口径与列布局)、[ACDL_f1_Motivation_and_Scope.md](ACDL_f1_Motivation_and_Scope.md)(为什么要做 f1)

---

## 1. 任务定义

用 ERA5 廓线特征预测 ACDL 消光的**垂直结构**。两种目标模式(`--target`):

| 模式 | 目标 | 维度 | 说明 |
|---|---|---|---|
| **`abs`(默认)** | 逐层绝对消光 σ_l | 32 | 直接重建廓线本身;**受"气溶胶有多少"制约,天花板低** |
| `frac` | **厚度加权逐层占比** f_l = σ_lΔz_l / Σ(σΔz) | 32(和为 1) | **垂直"形状"**;把不可学的"柱含量"除掉,只学相对分布 |

> 两种模式的**网络结构完全相同**,只有损失目标不同 → 可以直接横向对比。

- 一行 = 一个 (0.25° ERA5 格点 × 整点时次) 匹配样本;输入 X(103 维)= `[Lon, Lat, sin/cos(hour), sp_x/y/z, H_k(32), T_k(32), RH_k(32)]`;
- 层厚 `Δz_l` 由 `H_k` 列差分得到;`L00` 的地表高度取 **`Z_sfc_m`**(匹配已直接写出;若产物无该列则退回 `--surf-m`,默认 0),或用 `--drop-l00` 排除该层;
- **GCF(近地面占比)** = 层底 < `GCF_TOP_M`(默认 500 m)的层占比之和 → 正是 Stage-2 需要的量;每次评估都会单独报告其 R²/MAE;
- 目标里的 **NaN = 该层无有效观测** → 掩膜,不参与损失与指标;
- 形状模式要求样本"足够完整"(`--req-min-layers` 默认 10 层、`--req-cov` 默认有效厚度 ≥50%),以保证不同样本的占比可比。

---

## 2. 数据与字段

| 项 | 说明 |
|---|---|
| 文件 | `<MATCH_DIR>/ACDL_ERA5_MatchV2_*.mat`(逐日一个) |
| 结构体 | `MatchV2`:`Data`、`VarNames`、`Level`(32,hPa 降序)、`Meta` |
| **列布局** | **见** [ACDL_ERA5_FromScratch_Matching_Strategy.md](ACDL_ERA5_FromScratch_Matching_Strategy.md) **§5.1(唯一定义)**;本脚本只用**前 132 列**做基础特征 |
| 兼容 | `--with-stats` 的 264 列、追加了额外列的 133/135 列产物均可 |

**额外列(ERA5 单层)**:`BLH_m` / `TCWV_kgm2` / `Z_sfc_m` 由**匹配脚本直接写出**(2026-09-21 起,见策略文档 §5.0.1;旧 132 列产物可用 [add_era5_single_level_columns.py](add_era5_single_level_columns.py) 后补)。训练脚本**按列名读取**、不写死列号,`CONFIG["ADD_COLS"]` 已默认启用三者。启用后:
- `BLH`/`TCWV` 作为**独立 token** 进入网络(与 时间/空间/气象廓线 并列,由注意力决定权重);
- **`Z_sfc_m` 自动用于修正 `L00` 厚度**(`Δz₀ = H_k00 − 地形高度`),取代 `SURF_M=0` 的近似 → 形状与 GCF 更准。

```bash
# 新匹配已自带这三列;仅当使用旧 132 列产物时才需要后补:
python add_era5_single_level_columns.py          # 路径写在脚本顶部 CONFIG;输出 <MATCH_DIR>_plus/
```

**特征构造**(保持原网络的 `[Lon, Lat, T_sin, T_cos, Sp_X, Sp_Y, Sp_Z, meteo…]` 布局):

```
X = [ Lon, Lat, sin(hour), cos(hour), sp_x, sp_y, sp_z, H_k00..31, T_k00..31, RH_k00..31 ]
     └── 标识 ──┘ └─ 时间编码 ─┘ └── 空间球面编码 ──┘ └────── ERA5 廓线 96 维 ──────┘
     = 7 + 96 = 103 维
```

- 时间编码由 `ERA5_Time` 的**小数天换算小时**再取 sin/cos(**不再把 datenum 当小时**,修掉旧脚本的 bug);
- 空间编码用球面坐标(`cos(lat)cos(lon), cos(lat)sin(lon), sin(lat)`);
- 不引入任何 QC 字段(CAD/云类型/COD 等),避免目标泄漏。

---

## 3. 模型

复用原 `SpatioTemporalAttention`,仅改输入/输出维度:

| 组成部分 | 原脚本 | 本脚本 |
|---|---|---|
| 时间 token | `Linear(2→d)` | 同 |
| 空间 token | `Linear(3→d)` | 同 |
| 气象 token | `Linear(5→d)` | **`Linear(96→d)`**(H/T/RH 三层 32 层;固定切片,不含末尾额外列) |
| **额外 token(可选)** | — | `Linear(k→d)`,k = `--add-cols` 的列数(BLH/TCWV/… 各 1 维),与前三者并列 |
| Transformer | Pre-LN,`d_model=128`,`n_heads=4`,`n_layers=2`,GELU,dropout 0.15 | 同 |
| 池化 | 注意力池化(`Linear(d→1)` + softmax) | 同 |
| 解码器 | MLP `128→512→1024→512→1291` | **`128→512→1024→512→32`** |

参数量约为原网络的 1/40(输出头从 1291 降到 32),对当前样本量更友好。

---

## 4. 训练设置

| 项 | 值(可在 `CONFIG` 改) |
|---|---|
| 优化器 | AdamW,`lr=3e-4`,`weight_decay=1e-4`,余弦退火 |
| 批大小 / 轮数 | 256 / 200(早停 `patience=20`) |
| 梯度裁剪 | `max_norm=1.0` |
| 特征标准化 | `StandardScaler`(**仅用训练折统计量**) |
| 目标标准化 | **逐层** 标准化(忽略 NaN;某层训练集全缺 → 跳过该层) |
| 损失 | **掩膜 Huber**(`delta=1.0`):只对非 NaN 层计损失;整批无有效层 → 不产生梯度 |
| 设备 | 自动 `cuda`(否则 CPU) |

---

## 5. 两种交叉验证(都要)

| CV | 划分方式 | 防的是什么 |
|---|---|---|
| **空间块 CV** | 经纬度 **3×3** 分块,逐块留一 | 同格点样本跨折 → 空间泄漏 |
| **时间块 CV** | 按 `ERA5_Time` **排序**后切成 4 个连续块 | 同时次/相邻时刻样本跨折 → 时间泄漏 |

> 注意:匹配输出是按"格点→时次"排的行序,**不能直接按行号切时间折**;脚本先按时间排序再切,这是必须的。

两者结果**分别报告**,不混在一起解释。

---

## 6. 输出产物

```
<OUT_DIR>/
├── model_final.pt                       # 全量模型 + scaler_x + 目标 mu/sd + config(需 --final)
├── results/
│   ├── cv_summary.json                  # 各口径的均值±标准差(含 GCF_R2_mean)
│   ├── layer_geometry.json              # 每段平均高度(km)与代表气压(hPa),供出图
│   ├── folds_Spatial.json               # 逐折明细(含经纬度范围、最佳 epoch)
│   ├── folds_Temporal.json
│   ├── per_layer_metrics_Spatial.csv    # 逐层**绝对消光** R²/RMSE/MAE(均值±std)
│   ├── per_layer_metrics_Temporal.csv
│   └── predictions_Spatial.npz          # ★ 绘图数据(自包含):逐层指标 / GCF 观测·预测 /
│       predictions_Temporal.npz         #   完整预测·真值廓线 / 测试样本 lon·lat·time / 层几何
└── plots/
    ├── per_layer_r2_Spatial.png         # 逐层 R²(y=高度 km,次轴=气压)
    ├── per_layer_r2_Temporal.png
    ├── gcf_scatter_Spatial.png          # GCF 观测 vs 预测(1:1 线 + R²/MAE/n)
    └── gcf_scatter_Temporal.png
```

> ★ `predictions_*.npz` 是**绘图与训练解耦的关键**:有了它,`scripts/plot_training_results.py`
> 可以在不重训的情况下重画/新增图。旧版本跑出的目录没有这个文件,对应脚本会明确报错提示重跑。

---

## 7. 运行方式

```bash
# 本地(项目 .venv 已含 torch/sklearn);默认路径 D:/matchdata_fixed_case5(135 列,自带三列额外特征)
./.venv/Scripts/python.exe scripts/train_acdl_era5_match_v2.py --out-dir "./结果汇总/runs/train_out"

# 服务器(后台 + 日志)
nohup python scripts/train_acdl_era5_match_v2.py \
    --match-dir /media/data61/ZhangYi/FangMingZhao/Matchoutput \
    --out-dir  /media/data61/ZhangYi/FangMingZhao/train_out \
    > train_202206.log 2>&1 &

# 匹配产物已自带 BLH/TCWV/地形三列(无需再补);先看有哪些列
python scripts/train_acdl_era5_match_v2.py --list-cols
# 若用旧的 132 列产物,先补列再训:
python add_era5_single_level_columns.py --match-dir "D:/matchdata"      # → D:/matchdata_plus

# 目标模式(默认 abs = 逐层绝对消光)
python scripts/train_acdl_era5_match_v2.py --target abs        # 逐层绝对消光(默认)
python scripts/train_acdl_era5_match_v2.py --target frac       # 只学垂直形状
python scripts/train_acdl_era5_match_v2.py --req-min-layers 15 --req-cov 0.6   # 收紧样本完整性
python scripts/train_acdl_era5_match_v2.py --drop-l00 --gcf-top-m 300          # 排除 L00;GCF 定义改 300 m

# 快速试跑 / 分步
python scripts/train_acdl_era5_match_v2.py --limit-days 3 --epochs 20
python scripts/train_acdl_era5_match_v2.py --cv               # 加跑空间块 + 时间块 CV(慢)
python scripts/train_acdl_era5_match_v2.py --final            # 加训全量模型并存 model_final.pt
python scripts/train_acdl_era5_match_v2.py --selftest         # 合成数据自检(不需真实数据)
```

常用参数:`--match-dir`、`--out-dir`、`--add-cols`、`--list-cols`、`--target`、`--req-min-layers`、`--req-cov`、`--drop-l00`、`--gcf-top-m`、`--epochs`、`--batch`、`--seed`、`--limit-days`、`--holdout`、`--cv`、`--final`、`--selftest`。

**配套脚本**:
| 脚本 | 作用 |
|---|---|
| [add_era5_single_level_columns.py](add_era5_single_level_columns.py) | 【仅旧产物】把 ERA5 单层场追加到 **132 列**的匹配样本(新匹配已自带,无需此步) |
| [scripts/acdl_plotting.py](scripts/acdl_plotting.py) | **绘图模块**(与训练解耦,不含 torch):逐层 R²、GCF 散点、廓线密度 |
| [scripts/plot_training_results.py](scripts/plot_training_results.py) | **从落盘产物重画所有图,不用重训**:`--run-dir <OUT_DIR> [--plots ...] [--compare]` |

> 训练与绘图已解耦(2026-09-22):绘图实现全部在 `scripts/acdl_plotting.py`;训练结束时会把
> `results/predictions_<tag>.npz`(**逐层指标 + GCF 观测/预测 + 完整预测/真值廓线 + 测试样本坐标 + 层几何**)
> 落盘,因此改配色/改标题/加图都不用重跑训练。训练默认仍会调用绘图(方便),
> 加 `--no-plots` 则只落盘不出图。

**依赖**:`numpy, h5py, torch, scikit-learn`(绘图可选 `matplotlib`)。

---

## 8. 指标怎么读 / 已知限制

**只对外报告两个口径**(2026-09-22 起):

1. **逐层 R²(绝对消光 σ,km⁻¹)** —— 无论 `--target` 是什么模式,都先把模型输出**还原成绝对消光**再与观测逐层比较:
   | 模式 | 还原方式 |
   |---|---|
   | `abs` | `σ̂_l` 就是预测值本身(已是 km⁻¹) |
   | `frac` | `σ̂_l = f̂_l·OD_obs/Δz_l`(`f̂` 先归一到和为 1;没有含量信息,借观测柱含量 → 衡量的是"形状"能力) |
2. **GCF R² / MAE** —— 直接由**预测廓线**算 `Σ_{层底<GCF_TOP_M} σ̂Δz / Σσ̂Δz`,与观测 GCF 比;GCF 是比值,柱含量的量纲自动抵消。

   **有两层筛选,两个口径都会报:**

   | 口径 | 条件 | 含义 |
   |---|---|---|
   | **主口径**(`GCF_R2`, `n_gcf`) | 真值有效 **且** `ok_frac`(`frac` 模式) | 只用**模型真正训练过**的样本 —— 这个才是模型的 GCF 能力 |
   | 全口径(`GCF_R2_all`, `n_gcf_all`) | 只要真值有效 | 含"可评但未参与训练"的样本,**偏低且不可解释**,仅供参考 |

   - **真值端筛选**:`T["gcf"]` 只在"近地面窗口各层全有效"时才存在 —— 缺测不能当 0,否则 GCF 被系统性压低;
   - **模型端对齐**:`frac` 模式训练时只用了 `ok_frac`(≥10 层且厚度覆盖 ≥50%)的样本,其余样本对梯度零贡献。实测这批"没训过"的样本 GCF 真值均值是合格样本的 **3 倍**(0.427 vs 0.132),不排掉会把 R² 拉成大幅负值;
   - `abs` 模式所有样本都参与训练(逐层掩膜),所以两个口径完全一致,不重复打印。

要点与坑:
- **全局 R²/RMSE/MAE** 是把 32 层展平后算的(与逐层表同源,都是绝对消光),会被近地面大值层主导;
- `frac` 的绝对消光 R² 里**混进了观测柱含量**(借 `OD_obs` 还原),所以它衡量的是"形状学得怎么样",**不能**当作"能否重建绝对廓线"的证据;要那个结论只能看 `abs`;
- **⚠️ GCF 的样本极少**(2022-06 实测约 1.3k/41k,≈3%),且是"晴空少云"的选择性子集 → 该 R² 的置信区间很宽,不要拿小数点后两位说事;
- `OD_R²` 仍会写进 `cv_summary.json` / `folds_*.json`,但不再打印(它只是中间量);
- **L00 在高地形样本上会被排除**:`Δz₀ = H_k00 − Z_sfc ≤ 0` 时该层厚度为 0,还原无意义(σ̂ 记为 NaN)→ `frac` 的 L00 R² 只统计低地形样本,而 `abs` 模式不受此限。比较两种模式的 L00 时要留意这一点;
- 逐层 R² 对"该层测试集方差≈0"的层会留空(无法定义);
- **单月(6 月)样本时空相关性高**,CV 数字偏乐观;要稳健结论需多个月数据;
- `model_final.pt` 上的指标是**在全量数据自身上**算的(过拟合于该数据),只能说明拟合能力,不代表泛化;
- 目标量纲是 **消光(km⁻¹)**,不是 PM2.5;PM2.5 属于后续阶段。

---

## 9. 后续可加(需要就说)

1. **推理脚本**:加载 `model_final.pt` 对新样本预测并写回 `.mat`;
2. **预测 vs 真值廓线图**(分层抽样可视化);
3. **分高度带汇总**(0–1/1–3/3–6/6–10/10–30 km 的 R²/RMSE);
4. **按层加权 / 分高度带建模**;
5. 多月份合并训练与"按月留出"CV。
