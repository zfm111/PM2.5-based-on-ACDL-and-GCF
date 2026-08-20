"""
============================================================================
文件名：ACDL_ERA5_SpatiotemporalAttention_DL.py
功能：
  1. 读取 ACDL_ERA5_Matched_20220601.mat 中的 MatchResult
  2. 对时间/经纬度特征进行注意力机制特征变换
  3. 构建 SpatioTemporal Attention + MLP 深度学习网络
  4. 多输出回归：预测 1291 个高度层的 532nm 消光系数廓线
  5. 时间交叉验证 + 空间块交叉验证 双重精度评价
============================================================================
依赖：torch, scipy, numpy, sklearn, matplotlib, pandas
安装：pip install torch scipy scikit-learn matplotlib pandas
============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ============================================================
# 0. 全局配置
# ============================================================
CFG = {
    'mat_file'      : r'C:/Users/admin/Desktop/Main_ACDL_ERA5_SpatiotemporalAttention_DL/ACDL_ERA5_Matched_20220601.mat',
    'device'        : 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed'          : 42,
    'batch_size'    : 256,
    'epochs'        : 100,
    'lr'            : 3e-4,
    'weight_decay'  : 1e-4,
    'd_model'       : 128,       # Attention 嵌入维度
    'n_heads'       : 4,         # 多头注意力头数
    'n_layers'      : 2,         # Transformer 层数
    'dropout'       : 0.15,
    'n_time_folds'  : 4,         # 时间交叉验证折数
    'n_spatial_folds': 9,        # 空间块数（3×3 网格）
    'lon_bins'      : 3,         # 空间分块：经度方向
    'lat_bins'      : 3,         # 空间分块：纬度方向
    'ext_clip_min'  : -0.01,     # 消光系数有效下限 km⁻¹
    'ext_clip_max'  : 50.0,      # 消光系数有效上限 km⁻¹
    'sg_window': 11,             # 对消光系数廓线进行Savitzky-Golay平滑，建议奇数，越大越平滑
    'sg_order': 2,               # 二次多项式，保留峰值形状
}

torch.manual_seed(CFG['seed'])
np.random.seed(CFG['seed'])
print(f"Device: {CFG['device']}")

# ============================================================
# 1. 数据加载（h5py 读取 MATLAB v7.3 / HDF5 格式）
# ============================================================
def load_data(mat_file: str):
    print(f"\n[1/6] Loading (HDF5/v7.3): {mat_file} ...")

    with h5py.File(mat_file, 'r') as f:
        print(f"  Top-level keys    : {list(f.keys())}")
        mr = f['MatchResult']
        print(f"  MatchResult fields: {list(mr.keys())}")

        # ------ Data：MATLAB 列主序，需转置 ------
        Data = mr['Data'][()]
        print(f"  Data raw shape    : {Data.shape}  (will transpose if needed)")
        if Data.ndim == 2 and Data.shape[0] < Data.shape[1]:
            Data = Data.T                              # → (N_samples, N_cols)
        Data = Data.astype(np.float64)
        print(f"  Data final        : {Data.shape}")

        # ------ ProfileAlti ------
        ProfileAlti = mr['ProfileAlti'][()].ravel().astype(np.float32)
        print(f"  ProfileAlti       : {ProfileAlti.shape}, "
              f"[{ProfileAlti.min():.2f}, {ProfileAlti.max():.2f}] km")

        # ------ VarNames（可选）------
        try:
            vn_refs  = mr['VarNames'][()].ravel()
            VarNames = [''.join(chr(c) for c in f[ref][()].ravel())
                        for ref in vn_refs]
            print(f"  VarNames[:10]     : {VarNames[:10]}")
        except Exception as e:
            print(f"  VarNames skipped  ({e})")

    # ------ 列分割 ------
    X_raw = Data[:, :8].astype(np.float32)    # (N, 8)    气象特征
    y_raw = Data[:, 8:].astype(np.float32)    # (N, 1291) 消光系数廓线

    # ==============================================================
    # Step A：统计原始 NaN 情况（基线）
    # ==============================================================
    print(f"\n  ── Step A: 原始 NaN 统计 ──")
    print(f"    X_raw : {np.isnan(X_raw).sum():>8d} 个 NaN")
    print(f"    y_raw : {np.isnan(y_raw).sum():>8d} 个 NaN  "
          f"(含 NaN 的行: {np.isnan(y_raw).any(axis=1).sum()})")

    # 诊断：打印非 NaN 值的百分位分布
    y_valid_vals = y_raw[~np.isnan(y_raw)]
    if len(y_valid_vals) > 0:
        pcts = np.nanpercentile(y_valid_vals, [0.1, 1, 5, 50, 95, 99, 99.9])
        print(f"    非NaN值百分位 [0.1,1,5,50,95,99,99.9]%:")
        print(f"      {pcts.round(6)}")

    # ==============================================================
    # Step B：超出物理范围的值 → clip 到边界值（压边，不丢弃）
    # ==============================================================
    print(f"\n  ── Step B: 物理范围 clip ──")
    print(f"    设定范围: [{CFG['ext_clip_min']}, {CFG['ext_clip_max']}] km⁻¹")

    ext_min = CFG['ext_clip_min']
    ext_max = CFG['ext_clip_max']

    # 仅对非 NaN 值做 clip，NaN 保持不变
    valid_pos = ~np.isnan(y_raw)
    n_below   = ((y_raw < ext_min) & valid_pos).sum()
    n_above   = ((y_raw > ext_max) & valid_pos).sum()
    y_raw     = np.where(valid_pos, np.clip(y_raw, ext_min, ext_max), y_raw)
    print(f"    低于下限 clip : {n_below} 个值 → {ext_min}")
    print(f"    高于上限 clip : {n_above} 个值 → {ext_max}")

    # X 特征：MATLAB 填充值（>1e30）→ NaN（后续列均值填充）
    x_fill_mask = np.abs(X_raw) > 1e30
    if x_fill_mask.any():
        X_raw[x_fill_mask] = np.nan
        print(f"    X_raw 填充值(>1e30) → NaN : {x_fill_mask.sum()} 个")

    # ==============================================================
    # Step C：NaN 填充
    #   C-1: X 特征  → 逐列均值
    #   C-2: y 廓线  → 逐列均值；若整列全为 NaN → 相邻高度层二次样条插值
    # ==============================================================
    print(f"\n  ── Step C: NaN 填充 ──")

    # ---- C-1: X 特征 ----
    nan_in_X = np.isnan(X_raw).sum()
    if nan_in_X > 0:
        for col in range(X_raw.shape[1]):
            mask = np.isnan(X_raw[:, col])
            if mask.any():
                fill = np.nanmean(X_raw[:, col])
                fill = fill if not np.isnan(fill) else 0.0
                X_raw[mask, col] = fill
                print(f"    X col {col:2d}: 填充 {mask.sum():4d} 个 NaN → 均值 {fill:.4f}")
    else:
        print(f"    X_raw 无 NaN，跳过")

    # ---- C-2: y 廓线 ----
    from scipy.interpolate import CubicSpline

    nan_in_y      = np.isnan(y_raw).sum()
    n_col_mean    = 0    # 用列均值填充的列数
    n_col_interp  = 0    # 用样条插值填充的列数（全列 NaN）
    allnan_cols   = []   # 记录全列 NaN 的列索引，最后统一插值

    if nan_in_y > 0:
        # 第一遍：逐列均值填充（非全列 NaN）
        for col in range(y_raw.shape[1]):
            mask = np.isnan(y_raw[:, col])
            if not mask.any():
                continue
            col_mean = np.nanmean(y_raw[:, col])
            if not np.isnan(col_mean):              # 该列有非 NaN 值
                y_raw[mask, col] = col_mean
                n_col_mean += 1
            else:                                   # 整列全为 NaN
                allnan_cols.append(col)

        print(f"    y_raw 列均值填充 : {n_col_mean} 列")
        print(f"    全列 NaN 待插值  : {len(allnan_cols)} 列")

        # 第二遍：对全列 NaN 的高度层 → 用相邻高度层二次样条插值
        if allnan_cols:
            # 以高度层索引为自变量，用已知列的均值廓线做插值参考
            # 构建"列均值廓线"（已填充部分），作为插值的已知节点
            col_means_profile = np.nanmean(y_raw, axis=0)  # (1291,)
            all_col_idx       = np.arange(y_raw.shape[1])

            # 已知列（非全 NaN 列）的索引和均值
            known_mask   = ~np.isin(all_col_idx, allnan_cols)
            known_idx    = all_col_idx[known_mask]
            known_vals   = col_means_profile[known_mask]

            if len(known_idx) >= 4:
                # 二次样条（k=2）插值
                cs = CubicSpline(known_idx, known_vals,
                                 bc_type='not-a-knot',
                                 extrapolate=True)
                for col in allnan_cols:
                    interp_val = float(np.clip(cs(col), ext_min, ext_max))
                    y_raw[:, col] = interp_val
                    n_col_interp += 1
                print(f"    样条插值填充     : {n_col_interp} 列 ✓")
            else:
                # 已知节点太少，退化为全局均值填充
                global_mean = float(np.nanmean(col_means_profile))
                global_mean = global_mean if not np.isnan(global_mean) else 0.0
                for col in allnan_cols:
                    y_raw[:, col] = global_mean
                    n_col_interp += 1
                print(f"    已知节点不足4个，退化为全局均值({global_mean:.4f})填充: "
                      f"{n_col_interp} 列")
    else:
        print(f"    y_raw 无 NaN，跳过")


    # ==============================================================
    # Step D：断言确认无残余 NaN
    # ==============================================================
    assert not np.isnan(X_raw).any(), "X_raw 仍含 NaN！"
    assert not np.isnan(y_raw).any(), "y_raw 仍含 NaN！"
    print(f"\n  ── Step D: NaN 检查通过 ✓ ──")

    # ==============================================================
    # Step E：兜底过滤——clip + 填充后仍超范围的整行删除（安全保障）
    # ==============================================================
    valid_mask = np.all(
        (y_raw >= ext_min) & (y_raw <= ext_max), axis=1
    )
    n_before = X_raw.shape[0]
    X_raw    = X_raw[valid_mask]
    y_raw    = y_raw[valid_mask]
    print(f"\n  ── Step E: 兜底行过滤 ──")
    print(f"    保留 {X_raw.shape[0]} / {n_before} 行 "
          f"({valid_mask.mean()*100:.1f}%)")

    if X_raw.shape[0] == 0:
        raise ValueError(
            "过滤后样本数为 0！请检查 ext_clip_min / ext_clip_max 设置。"
        )
    # ==============================================================
    # Step F：逐行廓线平滑（Savitzky-Golay）
    #   放在最后：此时所有行均无 NaN，可安全对每行做平滑
    # ==============================================================
    from scipy.signal import savgol_filter

    sg_window = CFG.get('sg_window', 11)  # 平滑窗口（奇数，单位：高度层数）
    sg_order = CFG.get('sg_order', 2)  # 多项式阶数

    print(f"\n  ── Step F: 逐行廓线平滑（Savitzky-Golay）──")
    print(f"    window={sg_window}, polyorder={sg_order}, 共 {y_raw.shape[0]} 行")

    for i in range(y_raw.shape[0]):
        y_raw[i] = savgol_filter(y_raw[i], window_length=sg_window,
                                 polyorder=sg_order).astype(np.float32)
        # 平滑后再次 clip，防止 SG 滤波在边界产生轻微振铃越界
        y_raw[i] = np.clip(y_raw[i], ext_min, ext_max)

    print(f"    平滑完成 ✓")
    return X_raw, y_raw, ProfileAlti

# ============================================================
# 2. 特征工程：时间周期编码 + 空间极坐标编码
# ============================================================
def feature_engineering(X_raw: np.ndarray) -> np.ndarray:
    """
    输入列顺序：[Lon, Lat, Time, DewT, T, U, V, SP]
    输出：扩展后的特征矩阵（增加周期编码）

    时间编码（Time 假设单位为小时 0~23）：
      sin(2π·t/24), cos(2π·t/24)
    空间编码（经纬度 → 球面坐标）：
      sin(lat)·cos(lon), sin(lat)·sin(lon), cos(lat)
    """
    print("\n[2/6] Feature engineering ...")

    lon  = np.radians(X_raw[:, 0])   # 经度 → 弧度
    lat  = np.radians(X_raw[:, 1])   # 纬度 → 弧度
    time = X_raw[:, 2]               # 时间（小时）

    # 时间周期编码
    t_sin = np.sin(2 * np.pi * time / 24.0)
    t_cos = np.cos(2 * np.pi * time / 24.0)

    # 空间球面编码
    sp_x = np.cos(lat) * np.cos(lon)
    sp_y = np.cos(lat) * np.sin(lon)
    sp_z = np.sin(lat)

    # 气象特征（原始 DewT, T, U, V, SP）
    meteo = X_raw[:, 3:]             # [N, 5]

    # 拼接：[Lon, Lat, t_sin, t_cos, sp_x, sp_y, sp_z, DewT, T, U, V, SP]
    X_eng = np.column_stack([
        X_raw[:, 0],   # Lon（保留原始用于空间CV）
        X_raw[:, 1],   # Lat（保留原始用于空间CV）
        t_sin, t_cos,  # 时间周期编码
        sp_x, sp_y, sp_z,  # 球面空间编码
        meteo          # 气象参量
    ]).astype(np.float32)

    print(f"  Engineered feature dim: {X_eng.shape[1]}")
    # 特征名称
    feat_names = ['Lon','Lat','T_sin','T_cos','Sp_X','Sp_Y','Sp_Z',
                  'DewT_K','T_K','U_ms','V_ms','SP_Pa']
    print(f"  Features: {feat_names}")
    return X_eng, feat_names

# ============================================================
# 3. 网络结构：SpatioTemporal Attention + MLP
# ============================================================
class SpatioTemporalAttention(nn.Module):
    """
    时空注意力模块：
      - 将时间特征 [T_sin, T_cos] 和空间特征 [Sp_X, Sp_Y, Sp_Z]
        分别嵌入到 d_model 维空间，进行多头自注意力
      - 输出与气象特征拼接后送入 MLP 解码器
    """
    def __init__(self, n_feat: int, d_model: int, n_heads: int,
                 n_layers: int, dropout: float):
        super().__init__()
        self.d_model = d_model

        # --- 时间特征嵌入 (2维 → d_model) ---
        self.time_embed = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # --- 空间特征嵌入 (3维 → d_model) ---
        self.space_embed = nn.Sequential(
            nn.Linear(3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # --- 气象特征嵌入 (5维 → d_model) ---
        self.meteo_embed = nn.Sequential(
            nn.Linear(5, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # --- 跨模态 Transformer 编码器 ---
        # 输入序列长度 = 3 (time_token, space_token, meteo_token)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model   = d_model,
            nhead     = n_heads,
            dim_feedforward = d_model * 4,
            dropout   = dropout,
            activation= 'gelu',
            batch_first = True,
            norm_first  = True    # Pre-LN，训练更稳定
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # --- 注意力池化（加权聚合3个token）---
        self.attn_pool = nn.Linear(d_model, 1)

        # --- MLP 解码器 ---
        # 输入：d_model（注意力池化后）
        # 输出：1291 个高度层消光系数
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 1291)   # 输出 1291 个高度层,注意这里是写死的。
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 12]
        列顺序：[Lon, Lat, T_sin, T_cos, Sp_X, Sp_Y, Sp_Z,
                 DewT_K, T_K, U_ms, V_ms, SP_Pa]
        """
        # 特征分组
        time_feat  = x[:, 2:4]    # [B, 2]  T_sin, T_cos
        space_feat = x[:, 4:7]    # [B, 3]  Sp_X, Sp_Y, Sp_Z
        meteo_feat = x[:, 7:]     # [B, 5]  DewT, T, U, V, SP

        # 嵌入 → [B, d_model]
        t_tok = self.time_embed(time_feat).unsqueeze(1)    # [B, 1, d]
        s_tok = self.space_embed(space_feat).unsqueeze(1)  # [B, 1, d]
        m_tok = self.meteo_embed(meteo_feat).unsqueeze(1)  # [B, 1, d]

        # 拼接为序列 [B, 3, d_model]
        tokens = torch.cat([t_tok, s_tok, m_tok], dim=1)

        # Transformer 编码
        encoded = self.transformer(tokens)   # [B, 3, d_model]

        # 注意力池化
        attn_w = F.softmax(self.attn_pool(encoded), dim=1)  # [B, 3, 1]
        pooled = (attn_w * encoded).sum(dim=1)              # [B, d_model]

        # MLP 解码 → 消光系数廓线
        out = self.decoder(pooled)   # [B, 1291]
        return out


# ============================================================
# 4. 训练与验证工具函数
# 深度学习一般步骤：
# ——定义网络结构及parameters
# ——定义optimizer
# ——optimizer梯度清零
# —— 计算损失Loss，然后Loss.backward()计算损失函数对参数的梯度
# ——执行optimizer.step()根据损失梯度更新参数。
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, trues = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        total_loss += criterion(pred, yb).item() * xb.size(0)
        preds.append(pred.cpu().numpy())
        trues.append(yb.cpu().numpy())
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    return total_loss / len(loader.dataset), preds, trues


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算逐高度层和全局精度指标"""
    # 全局（展平）
    r2_global  = r2_score(y_true.ravel(), y_pred.ravel())
    rmse_global = np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel()))
    mae_global  = mean_absolute_error(y_true.ravel(), y_pred.ravel())

    # 逐高度层 R²
    r2_per_alt = np.array([
        r2_score(y_true[:, k], y_pred[:, k])
        for k in range(y_true.shape[1])
    ])

    return {
        'R2_global'  : r2_global,
        'RMSE_global': rmse_global,
        'MAE_global' : mae_global,
        'R2_per_alt' : r2_per_alt,
    }


# ============================================================
# 5. 时间交叉验证
# ============================================================
def temporal_cv(X: np.ndarray, y: np.ndarray,
                time_col_idx: int = 2,
                n_folds: int = 4) -> list:
    """
    按时间顺序划分 n_folds 折（不打乱顺序）
    模拟真实预报场景：用过去预测未来
    """
    print(f"\n[4/6] Temporal Cross-Validation ({n_folds} folds) ...")
    N = X.shape[0]
    fold_size = N // n_folds
    results = []

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end   = (fold + 1) * fold_size if fold < n_folds - 1 else N
        test_idx   = np.arange(test_start, test_end)
        train_idx  = np.concatenate([
            np.arange(0, test_start),
            np.arange(test_end, N)
        ])

        if len(train_idx) < 100:
            continue

        metrics = _run_fold(X, y, train_idx, test_idx,
                            fold_name=f'Temporal-Fold{fold+1}')
        results.append(metrics)
        print(f"  Fold {fold+1}: R²={metrics['R2_global']:.4f}  "
              f"RMSE={metrics['RMSE_global']:.6f}  "
              f"MAE={metrics['MAE_global']:.6f}")

    return results


# ============================================================
# 6. 空间块交叉验证
# ============================================================
def spatial_cv(X: np.ndarray, y: np.ndarray,
               lon_bins: int = 3, lat_bins: int = 3) -> list:
    """
    将研究区域按经纬度划分为 lon_bins × lat_bins 个空间块
    依次以每个块为测试集，其余为训练集
    """
    print(f"\n[5/6] Spatial Block Cross-Validation "
          f"({lon_bins}×{lat_bins} blocks) ...")

    lon = X[:, 0]
    lat = X[:, 1]

    lon_edges = np.linspace(lon.min() - 1e-6, lon.max() + 1e-6, lon_bins + 1)
    lat_edges = np.linspace(lat.min() - 1e-6, lat.max() + 1e-6, lat_bins + 1)

    lon_bin_id = np.digitize(lon, lon_edges) - 1
    lat_bin_id = np.digitize(lat, lat_edges) - 1
    block_id   = lon_bin_id * lat_bins + lat_bin_id

    results = []
    unique_blocks = np.unique(block_id)

    for blk in unique_blocks:
        test_idx  = np.where(block_id == blk)[0]
        train_idx = np.where(block_id != blk)[0]

        if len(test_idx) < 10 or len(train_idx) < 100:
            continue

        i_lon = blk // lat_bins
        i_lat = blk  % lat_bins
        fold_name = f'Spatial-Block(lon{i_lon+1},lat{i_lat+1})'

        metrics = _run_fold(X, y, train_idx, test_idx,
                            fold_name=fold_name)
        results.append({**metrics, 'block_id': blk,
                        'lon_range': (lon_edges[i_lon], lon_edges[i_lon+1]),
                        'lat_range': (lat_edges[i_lat], lat_edges[i_lat+1]),
                        'n_test': len(test_idx)})
        print(f"  Block ({i_lon+1},{i_lat+1}): "
              f"N_test={len(test_idx):4d}  "
              f"R²={metrics['R2_global']:.4f}  "
              f"RMSE={metrics['RMSE_global']:.6f}")

    return results


def _run_fold(X, y, train_idx, test_idx, fold_name='fold',
              epochs=50):
    """单折训练+评估"""
    device = CFG['device']

    # 标准化（仅用训练集统计量）
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_tr = scaler_x.fit_transform(X[train_idx])
    X_te = scaler_x.transform(X[test_idx])
    y_tr = scaler_y.fit_transform(y[train_idx])
    y_te = y[test_idx]   # 评估时用原始尺度

    # 数据集
    ds_tr = TensorDataset(
        torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
    ds_te = TensorDataset(
        torch.FloatTensor(X_te),
        torch.FloatTensor(scaler_y.transform(y_te)))

    loader_tr = DataLoader(ds_tr, batch_size=CFG['batch_size'],
                           shuffle=True,  num_workers=0)
    loader_te = DataLoader(ds_te, batch_size=CFG['batch_size'],
                           shuffle=False, num_workers=0)

    # 模型
    model = SpatioTemporalAttention(
        n_feat  = X.shape[1],
        d_model = CFG['d_model'],
        n_heads = CFG['n_heads'],
        n_layers= CFG['n_layers'],
        dropout = CFG['dropout']
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG['lr'],
        weight_decay=CFG['weight_decay']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.HuberLoss(delta=0.5)   # 对消光系数异常值鲁棒

    best_loss = np.inf
    best_state = None

    for ep in range(epochs):
        train_loss = train_one_epoch(
            model, loader_tr, optimizer, criterion, device)
        val_loss, _, _ = evaluate(
            model, loader_te, criterion, device)
        scheduler.step()

        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

    # 加载最优权重评估
    model.load_state_dict(best_state)
    _, preds_scaled, trues_scaled = evaluate(
        model, loader_te, criterion, device)

    # 反标准化
    preds = scaler_y.inverse_transform(preds_scaled)
    trues = y_te

    metrics = compute_metrics(trues, preds)
    metrics['fold_name'] = fold_name
    return metrics


# ============================================================
# 7. 可视化
# ============================================================
def plot_results(temporal_results, spatial_results, ProfileAlti):
    print("\n[6/6] Plotting results ...")
    fig = plt.figure(figsize=(18, 14), facecolor='white')
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.38, wspace=0.35)

    # 颜色方案
    c_temporal = '#2166AC'
    c_spatial  = '#D6604D'

    # ----- (a) 时间CV：逐折 R² -----
    ax1 = fig.add_subplot(gs[0, 0])
    folds_t = [r['fold_name'].replace('Temporal-', '') for r in temporal_results]
    r2_t    = [r['R2_global'] for r in temporal_results]
    bars = ax1.bar(folds_t, r2_t, color=c_temporal,
                   alpha=0.85, edgecolor='white', linewidth=0.8)
    ax1.axhline(np.mean(r2_t), color='k', ls='--', lw=1.2,
                label=f'Mean R²={np.mean(r2_t):.3f}')
    ax1.set_ylim([0, 1.05])
    ax1.set_xlabel('Temporal Fold', fontsize=11)
    ax1.set_ylabel('R²', fontsize=11)
    ax1.set_title('(a) Temporal CV – R²', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.4)
    for bar, v in zip(bars, r2_t):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    # ----- (b) 时间CV：逐折 RMSE -----
    ax2 = fig.add_subplot(gs[0, 1])
    rmse_t = [r['RMSE_global'] for r in temporal_results]
    ax2.plot(folds_t, rmse_t, 'o-', color=c_temporal,
             lw=2, ms=8, markerfacecolor='white', markeredgewidth=2)
    ax2.fill_between(range(len(folds_t)), rmse_t,
                     alpha=0.15, color=c_temporal)
    ax2.set_xlabel('Temporal Fold', fontsize=11)
    ax2.set_ylabel('RMSE (km⁻¹)', fontsize=11)
    ax2.set_title('(b) Temporal CV – RMSE', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.4)

    # ----- (c) 空间CV：逐块 R² -----
    ax3 = fig.add_subplot(gs[0, 2])
    folds_s = [f"B{i+1}" for i in range(len(spatial_results))]
    r2_s    = [r['R2_global'] for r in spatial_results]
    bars_s  = ax3.bar(folds_s, r2_s, color=c_spatial,
                      alpha=0.85, edgecolor='white', linewidth=0.8)
    ax3.axhline(np.mean(r2_s), color='k', ls='--', lw=1.2,
                label=f'Mean R²={np.mean(r2_s):.3f}')
    ax3.set_ylim([0, 1.05])
    ax3.set_xlabel('Spatial Block', fontsize=11)
    ax3.set_ylabel('R²', fontsize=11)
    ax3.set_title('(c) Spatial Block CV – R²', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.4)

    # ----- (d) 逐高度层 R²（时间CV均值）-----
    ax4 = fig.add_subplot(gs[1, 0])
    r2_alt_mean = np.mean(
        np.vstack([r['R2_per_alt'] for r in temporal_results]), axis=0)
    ax4.plot(r2_alt_mean, ProfileAlti, color=c_temporal, lw=1.8)
    ax4.axvline(0, color='gray', ls='--', lw=0.8)
    ax4.set_xlabel('R² per Altitude Level', fontsize=11)
    ax4.set_ylabel('Altitude (km)', fontsize=11)
    ax4.set_title('(d) R² Profile (Temporal CV)', fontsize=12, fontweight='bold')
    ax4.set_xlim([-0.1, 1.05])
    ax4.grid(alpha=0.4)

    # ----- (e) 逐高度层 R²（空间CV均值）-----
    ax5 = fig.add_subplot(gs[1, 1])
    r2_alt_s = np.mean(
        np.vstack([r['R2_per_alt'] for r in spatial_results]), axis=0)
    ax5.plot(r2_alt_s, ProfileAlti, color=c_spatial, lw=1.8)
    ax5.axvline(0, color='gray', ls='--', lw=0.8)
    ax5.set_xlabel('R² per Altitude Level', fontsize=11)
    ax5.set_ylabel('Altitude (km)', fontsize=11)
    ax5.set_title('(e) R² Profile (Spatial CV)', fontsize=12, fontweight='bold')
    ax5.set_xlim([-0.1, 1.05])
    ax5.grid(alpha=0.4)

    # ----- (f) 综合指标汇总表 -----
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    summary_data = [
        ['Metric', 'Temporal CV\n(mean±std)', 'Spatial CV\n(mean±std)'],
        ['R²',
         f"{np.mean(r2_t):.4f}±{np.std(r2_t):.4f}",
         f"{np.mean(r2_s):.4f}±{np.std(r2_s):.4f}"],
        ['RMSE\n(km⁻¹)',
         f"{np.mean(rmse_t):.5f}±{np.std(rmse_t):.5f}",
         f"{np.mean([r['RMSE_global'] for r in spatial_results]):.5f}±"
         f"{np.std([r['RMSE_global'] for r in spatial_results]):.5f}"],
        ['MAE\n(km⁻¹)',
         f"{np.mean([r['MAE_global'] for r in temporal_results]):.5f}",
         f"{np.mean([r['MAE_global'] for r in spatial_results]):.5f}"],
    ]
    tbl = ax6.table(cellText=summary_data[1:],
                    colLabels=summary_data[0],
                    cellLoc='center', loc='center',
                    bbox=[0.0, 0.2, 1.0, 0.7])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#EBF5FB')
    ax6.set_title('(f) Summary Metrics', fontsize=12, fontweight='bold')

    fig.suptitle(
        'SpatioTemporal Attention DL Model – ACDL 532nm Extinction Profile Retrieval\n'
        'Cross-Validation Results',
        fontsize=14, fontweight='bold', y=1.01
    )

    plt.savefig('CV_Results.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    print("  Saved: CV_Results.png")
    plt.show()


# ============================================================
# 8. 主流程
# ============================================================
def main():
    print("=" * 60)
    print("  ACDL ERA5 SpatioTemporal Attention DL Training")
    print("=" * 60)

    # 8.1 加载数据
    X_raw, y_raw, ProfileAlti = load_data(CFG['mat_file'])

    # 8.2 特征工程
    X_eng, feat_names = feature_engineering(X_raw)

    # 8.3 标准化（全局，用于最终模型）
    print("\n[3/6] Normalizing features and targets ...")
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X_eng).astype(np.float32)
    y_scaled = scaler_y.fit_transform(y_raw).astype(np.float32)

    # 8.4 时间交叉验证
    temporal_results = temporal_cv(
        X_eng, y_raw,
        time_col_idx=2,
        n_folds=CFG['n_time_folds']
    )

    # 8.5 空间块交叉验证
    spatial_results = spatial_cv(
        X_eng, y_raw,
        lon_bins=CFG['lon_bins'],
        lat_bins=CFG['lat_bins']
    )

    # 8.6 打印汇总
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  Temporal CV  R² : {np.mean([r['R2_global'] for r in temporal_results]):.4f} "
          f"± {np.std([r['R2_global'] for r in temporal_results]):.4f}")
    print(f"  Spatial  CV  R² : {np.mean([r['R2_global'] for r in spatial_results]):.4f} "
          f"± {np.std([r['R2_global'] for r in spatial_results]):.4f}")

    # 8.7 可视化
    plot_results(temporal_results, spatial_results, ProfileAlti)

    # 8.8 训练最终全量模型并保存
    print("\n[+] Training final model on full dataset ...")
    ds_full = TensorDataset(
        torch.FloatTensor(X_scaled),
        torch.FloatTensor(y_scaled)
    )
    loader_full = DataLoader(ds_full, batch_size=CFG['batch_size'],
                             shuffle=True, num_workers=0)

    final_model = SpatioTemporalAttention(
        n_feat  = X_eng.shape[1],
        d_model = CFG['d_model'],
        n_heads = CFG['n_heads'],
        n_layers= CFG['n_layers'],
        dropout = CFG['dropout']
    ).to(CFG['device'])

    optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG['epochs'])
    criterion = nn.HuberLoss(delta=0.5)

    for ep in range(CFG['epochs']):
        loss = train_one_epoch(
            final_model, loader_full, optimizer, criterion, CFG['device'])
        scheduler.step()
        if (ep + 1) % 20 == 0:
            print(f"  Epoch [{ep+1:3d}/{CFG['epochs']}]  Loss: {loss:.6f}")

    torch.save({
        'model_state': final_model.state_dict(),
        'scaler_x'   : scaler_x,
        'scaler_y'   : scaler_y,
        'ProfileAlti': ProfileAlti,
        'feat_names' : feat_names,
        'cfg'        : CFG,
    }, 'ACDL_SpatioTempAttn_final.pth')
    print("  Saved: ACDL_SpatioTempAttn_final.pth")
    print("\nDone!")


if __name__ == '__main__':
    main()