# 交接文档（HANDOFF）——新对话从这里恢复上下文

> 用途：本文件是跨对话交接锚点。新对话第一句话可以说：
> **"读取 结果汇总/HANDOFF.md，继续 ACDL 项目的后续工作"**，
> 并可引用原会话 `sess_ac437759-4a07-4e20-ac9b-942815c7abe9`（ZCode 支持跨会话读取上下文）。
> 本文件最后更新：B0-AGL CV 刚挂起时。

## 0. ✅ CV 已完成（2026-09-29 18:34，两并行运行结果确定性一致）

空间块CV 与 时间块CV 均已完成并评估（bench_CV-Spatial / bench_CV-Temporal）。
**核心数字（随机holdout → 空间CV → 时间CV）**：全局R² 0.48→0.27→0.27；0-1km nMAE 58.8→69.2→69.1；
OD |rel| 25.0→41.3→40.1；Gfrac 57.0→35.4→35.6；ODslope 0.733→0.348→0.348；GCF R² 0.605→0.472→0.430（n≈19.6k）。
判读：随机holdout偏乐观；两套CV高度一致=结论稳健；**对外报数必须用CV口径**；Gfrac 35% vs 66% 目标
→ 信息量升级（u/v、AOD）必要性进一步坐实。详见 runs/train_agl_b0_cv/results/bench_CV-*.md。

## 1. ~~正在后台运行的任务~~（已完成，见 §0）


- **任务**：B0-AGL 交叉验证（空间 3×3 留一 9 折 + 时间块 4 折，共 13 折完整训练）
- **命令**：
  `.venv/Scripts/python.exe Train_ACDL_ERA5_MatchV2.py --match-dir D:/matchdata_agl_case5 --out-dir 结果汇总/runs/train_agl_b0_cv --tag CV --cv --holdout 0`
- **日志**：`结果汇总/logs/agl_train_cv.log`（PYTHONUNBUFFERED=1，实时可查；结尾出现 `[EXIT 0]` = 正常完成）
- **预计时长**：5–7.5 小时（CPU）；每折日志形如 `[Spatial(1,1)] ep... `
- **完成后待办**：用 `evaluate_bench.py --run-dir 结果汇总/runs/train_agl_b0_cv --tag Spatial_CV / Temporal_CV` 分别评估，与 holdout 数字并排对比（重点：低层 nMAE 方向是否保持 ~58% 量级、GCF 改善是否保持），写进 experiments.md
- 注意：`--holdout 0` 跳过了 holdout（结果已有），本任务只产出 CV；**不会覆盖任何已有产物**

## 2. 项目当前状态（截至交接）

- 已完成：《优化路线讨论.md》Step 0–5 中的 Step 0–3（事实核实/评估台/训练侧消融/加权消融——全部失败，
  证明瓶颈在任务定义）→ **AGL 垂直坐标改造**（Phase 1–5 全部完成）→ B0-AGL 重训与评估
- 核心结论：L00 有效率 29.2%→57.5%、GCF 覆盖 6%→58.9%（R² 0.423→0.605）、层语义全域统一；
  验收门 2/4 过，门 1 未达（L00=2bin 结构性脆弱+产品侧 QC 上限）
- 详细依据全部在 `结果汇总/` 下（见下节索引），git log 每步一提交

## 3. 恢复上下文的最小阅读集（按序）

1. `结果汇总/术语表.md` —— 全部专有名词与实验编号（B0/AGLB0/ASL/AGL/case5/GCF…）
2. `结果汇总/优化策略总结.md` —— 优化全貌（缺陷→方法→前后对比，含 5 图）
3. `结果汇总/experiments.md` —— 实验登记表（B0/E0/E2b/E2c/W2/W3/AGLB0 数字）
4. `结果汇总/AGL改造报告.md` —— AGL 改造细节与验收门判定
5. `结果汇总/指标说明.md` —— 各指标计算方式与判读标尺
6. `结果汇总/facts.md`、`结果汇总/低层缺失诊断.md` —— 缺失成因证据链
7. `git log --oneline` —— 每步提交信息即工作日志

## 4. 已知待办与候选方向（按性价比）

1. CV 完成后的评估对比（见上）
2. L00 加厚至 0–100 m（改 `Match_ACDL_ERA5_FromScratch.py` 的 `AGL_EDGES_KM` 一处 + 2h 重匹配，预计 L00→~73%）
3. ERA5-Land 2t/2d/sp 锚点替换递减率外推（数据已在 D:/era5_data/era5_land，schema 不变）
4. `--with-stats` 重匹配归因 L00 残余缺失（产品侧 fill/超界占比）
5. 信息量升级：接 ERA5 风场 u/v 或 AOD（治 slope<1 尾部压缩的根）
6. 多种子（≥3）显著性后再对外报数

## 5. 环境速记

- Python：`结果汇总` 工程根的 `.venv/Scripts/python.exe`（torch 2.13 CPU、12 核）
- 数据：`D:/matchdata_fixed_case5`（ASL，只读）、`D:/matchdata_agl_case5`（AGL）、`D:/era5_mat`、`D:/ACDL/Data/ACDL/ProfileMat`
- 训练输出统一进 `结果汇总/runs/`；ASL 历史产物从未被覆盖
