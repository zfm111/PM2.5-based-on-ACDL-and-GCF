# PM2.5-based-on-ACDL-and-GCF

基于 **ACDL 星载激光雷达 532nm 消光廓线** 与 **ERA5 再分析气象场**的匹配样本，训练时空注意力网络
（`SpatioTemporalAttention32`）预测 32 层**离地高度（AGL）分层**的平均消光系数，并导出近地面占比
**GCF** 等约束量，服务后续近地面 PM2.5 估算。

核心结论与全部实验记录见 **[结果汇总/优化策略总结.md](结果汇总/优化策略总结.md)**。

## 目录结构

```
├─ scripts/                 全部可执行代码(从仓库根目录运行)
│   ├─ Match_ACDL_ERA5_FromScratch.py   ACDL×ERA5 匹配(--vertical asl|agl, QC case1–5)
│   ├─ Train_ACDL_ERA5_MatchV2.py       训练(--target abs, --cv, --w-profile, --log-target…)
│   ├─ evaluate_bench.py                评估台:四指标+EE 散点+分位带图(bench_*.md/json)
│   ├─ acdl_plotting.py / Plot_Training_Results.py   绘图(与训练解耦)
│   ├─ diagnose_low_layers.py           低层缺失成因定量诊断
│   ├─ plot_case_profiles.py            留出集个例廓线对比
│   ├─ plot_cv_case_profiles.py         交叉验证个例廓线(空间/时间块)
│   ├─ plot_cv_summary.py / plot_node_validation.py / plot_profile_agreement_modes.py
│   ├─ Add_ERA5_SingleLevel_Columns.py / download_era5.py / era5_grib_to_mat.py / server_run_matching.py
│   ├─ matlab/                          历史 MATLAB 工具
│   └─ legacy/                          旧版脚本(归档)
├─ docs/                    方法与背景文档
│   ├─ ACDL_ERA5_FromScratch_Matching_Strategy.md   匹配策略
│   ├─ ACDL_Data_Cleaning_and_QC_Guidelines.md      数据清洗与 QC 规范
│   ├─ Train_ACDL_ERA5_MatchV2_README.md            训练脚本详细说明
│   ├─ 优化路线讨论.md                               优化路线讨论稿(Step 0–5)
│   └─ legacy_figures/ + 历史资料(pptx/docx/html)
├─ 结果汇总/                 全部实验结果与总结(交差从这里拿)
│   ├─ 优化策略总结.md / AGL改造报告.md / 交叉验证报告.md / baseline_report.md
│   ├─ experiments.md(实验登记表) / facts.md(代码事实) / 指标说明.md / 术语表.md
│   ├─ runs/   各次训练:results(bench json/md·csv·npz)+plots
│   ├─ figures/  诊断与论文图
│   └─ logs/   训练与匹配日志
├─ ACDL_ERA5_Matched_20220601.mat / ACDL_SpatioTempAttn_final.pth   本地数据/权重样例(不入库)
└─ .venv/                   Python 环境(numpy/torch/h5py/matplotlib/sklearn)
```

## 快速开始

```bash
# 1) 匹配(垂直坐标二选一: asl=固定气压层(历史) / agl=离地高度(推荐))
.venv/Scripts/python.exe scripts/Match_ACDL_ERA5_FromScratch.py \
    --case 5 --vertical agl --out-dir D:/matchdata_agl_case5

# 2) 训练(abs 目标, 20% 留出;加 --cv 跑空间/时间块交叉验证)
.venv/Scripts/python.exe scripts/Train_ACDL_ERA5_MatchV2.py \
    --match-dir D:/matchdata_agl_case5 --out-dir 结果汇总/runs/<实验名> --tag <TAG>

# 3) 评估(四指标 + 图, 不重训)
.venv/Scripts/python.exe scripts/evaluate_bench.py \
    --run-dir 结果汇总/runs/<实验名> --tag Holdout_<TAG> --name <NAME>
```

数据目录（本机）：`D:/matchdata_agl_case5`（AGL）、`D:/matchdata_fixed_case5`（ASL 基线）、
`D:/era5_mat`、`D:/ACDL/Data/ACDL/ProfileMat`。

## 关键结果速览（详见《优化策略总结》）

| | 随机留出 | 空间块 CV | 时间块 CV |
|---|---|---|---|
| GCF R²（前→后） | 0.423→**0.605** | 0.140→**0.472** | 0.142→**0.430** |
| GCF 可评估样本 | 2,079→**3,903** | 10,198→**19,573** | 同左 |
| L00 有效率（产品层） | 29.2%→**57.5%** | — | — |

（"前/后" = ASL 固定气压层分段 → AGL 地形跟随分段；逐样本散度与系统偏差见《指标说明》决策表。）

## 分支说明

- `demo`：本仓库全部工作分支（优化 + AGL 改造 + CV）；
- `main`：早期版本快照。
