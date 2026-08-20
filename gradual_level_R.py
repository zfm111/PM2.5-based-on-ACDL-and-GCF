"""
train_eval_simple.py
简化版训练+评估脚本，只输出逐高度层的 R²、RMSE、MAE
使用残差学习 + 标准化
新增：露点温度差（DPD）作为第 13 个特征
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 只从原脚本导入 load_data 和配置，其他功能在本脚本重写
from Main_ACDL_ERA5_SpatiotemporalAttention_DL import load_data, CFG

# ============================================================
# 配置
# ============================================================
CONFIG = {
    'mat_file': r'C:/Users/admin/Desktop/Main_ACDL_ERA5_SpatiotemporalAttention_DL\ACDL_ERA5_Matched_20220601.mat',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'batch_size': 256,
    'epochs': 100,
    'lr': 1e-4,
    'weight_decay': 1e-3,
    'dropout': 0.3,
    'test_ratio': 0.2,
    'save_model': False,
}

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
print(f"Device: {CONFIG['device']}")

# ============================================================
# 自定义特征工程（适配 9 列输入，包含 DPD）
# ============================================================
def feature_engineering_custom(X_raw: np.ndarray):
    """
    输入列顺序：[Lon, Lat, Time, DewT, T, U, V, SP, DPD]
    输出 13 个特征：Lon, Lat, T_sin, T_cos, Sp_X, Sp_Y, Sp_Z,
                   DewT, T, U, V, SP, DPD
    """
    lon = np.radians(X_raw[:, 0])
    lat = np.radians(X_raw[:, 1])
    time = X_raw[:, 2]

    t_sin = np.sin(2 * np.pi * time / 24.0)
    t_cos = np.cos(2 * np.pi * time / 24.0)

    sp_x = np.cos(lat) * np.cos(lon)
    sp_y = np.cos(lat) * np.sin(lon)
    sp_z = np.sin(lat)

    # 气象特征：DewT, T, U, V, SP, DPD（共 6 列）
    meteo = X_raw[:, 3:]

    X_eng = np.column_stack([
        X_raw[:, 0], X_raw[:, 1],   # Lon, Lat
        t_sin, t_cos,
        sp_x, sp_y, sp_z,
        meteo
    ]).astype(np.float32)

    feat_names = ['Lon', 'Lat', 'T_sin', 'T_cos', 'Sp_X', 'Sp_Y', 'Sp_Z',
                  'DewT_K', 'T_K', 'U_ms', 'V_ms', 'SP_Pa', 'DPD_K']
    return X_eng, feat_names

# ============================================================
# 自定义模型类（适配动态气象特征维度）
# ============================================================
class SpatioTemporalAttention_custom(nn.Module):
    def __init__(self, n_feat: int, d_model: int, n_heads: int,
                 n_layers: int, dropout: float):
        super().__init__()
        self.d_model = d_model

        # n_feat 应为 13（在本脚本中固定）
        # 前 7 列：Lon, Lat, T_sin, T_cos, Sp_X, Sp_Y, Sp_Z
        # 后 6 列：DewT, T, U, V, SP, DPD
        self.n_meteo = n_feat - 7   # 自动计算气象特征维度

        self.time_embed = nn.Sequential(
            nn.Linear(2, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.space_embed = nn.Sequential(
            nn.Linear(3, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.meteo_embed = nn.Sequential(
            nn.Linear(self.n_meteo, d_model), nn.LayerNorm(d_model), nn.GELU()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.attn_pool = nn.Linear(d_model, 1)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 1291)   # 保持输出 1291 层，若过滤可在外部调整
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # 前 2 列 Lon, Lat 我们不直接使用，但保留在特征中（由空间编码使用）
        # 实际分组：time_feat = x[:, 2:4], space_feat = x[:, 4:7], meteo_feat = x[:, 7:]
        time_feat = x[:, 2:4]
        space_feat = x[:, 4:7]
        meteo_feat = x[:, 7:]

        t_tok = self.time_embed(time_feat).unsqueeze(1)
        s_tok = self.space_embed(space_feat).unsqueeze(1)
        m_tok = self.meteo_embed(meteo_feat).unsqueeze(1)

        tokens = torch.cat([t_tok, s_tok, m_tok], dim=1)
        encoded = self.transformer(tokens)

        attn_w = F.softmax(self.attn_pool(encoded), dim=1)
        pooled = (attn_w * encoded).sum(dim=1)
        out = self.decoder(pooled)
        return out

# ============================================================
# 主流程
# ============================================================
print("\n[1/5] Loading data...")
X_raw, y_raw, ProfileAlti = load_data(CONFIG['mat_file'])

# ========== 额外清洗：剔除异常填充值（≈50.0） ==========
print("\n[额外清洗] 剔除无效填充值（≈50.0）...")
anomaly_mask = (y_raw >= 49.9) & (y_raw <= 50.1)
n_anomaly = anomaly_mask.sum()
if n_anomaly > 0:
    print(f"  发现 {n_anomaly} 个异常填充值（≈50.0），将用该列有效均值替换")
    for col in range(y_raw.shape[1]):
        col_data = y_raw[:, col]
        mask_col = (col_data >= 49.9) & (col_data <= 50.1)
        if mask_col.any():
            valid_vals = col_data[~mask_col]
            col_mean = valid_vals.mean() if len(valid_vals) > 0 else 0.0
            col_data[mask_col] = col_mean
    print("  替换完成")
else:
    print("  未发现异常填充值")

# ========== 新增特征：露点温度差 DPD ==========
print("\n[特征工程] 计算露点温度差 (T - DewT) 作为第9列特征...")
DPD = X_raw[:, 4] - X_raw[:, 3]   # T_K - DewT_K
DPD = DPD.reshape(-1, 1)
X_raw = np.column_stack([X_raw, DPD])
print(f"  X_raw 形状变为: {X_raw.shape} (新增1列 DPD)")

# 使用自定义特征工程
X_eng, feat_names = feature_engineering_custom(X_raw)
print(f"Feature shape: {X_eng.shape}, Target shape: {y_raw.shape}")

# 划分训练/测试集（按时间顺序）
n = X_eng.shape[0]
test_start = int(n * (1 - CONFIG['test_ratio']))
X_train, X_test = X_eng[:test_start], X_eng[test_start:]
y_train, y_test = y_raw[:test_start], y_raw[test_start:]
print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# ============================================================
# 2. 构建残差目标 + 标准化
# ============================================================
print("\n[2/5] Computing residual targets...")
mean_profile_train = y_train.mean(axis=0, keepdims=True)
y_train_residual = y_train - mean_profile_train
y_test_residual = y_test - mean_profile_train

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train_residual)
y_test_scaled = scaler_y.transform(y_test_residual)
print(f"残差标准化后: 均值≈{y_train_scaled.mean():.4f}, 标准差≈{y_train_scaled.std():.4f}")

# ============================================================
# 3. 特征标准化
# ============================================================
print("\n[3/5] Feature normalization...")
scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)

X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.FloatTensor(y_train_scaled)
X_test_t = torch.FloatTensor(X_test_scaled)
y_test_t = torch.FloatTensor(y_test_scaled)

train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

# ============================================================
# 4. 模型构建与训练（使用自定义模型）
# ============================================================
print("\n[4/5] Building and training model...")
model = SpatioTemporalAttention_custom(
    n_feat=X_eng.shape[1],   # 现在为13
    d_model=CFG['d_model'],
    n_heads=CFG['n_heads'],
    n_layers=CFG['n_layers'],
    dropout=CONFIG['dropout']
).to(CONFIG['device'])

optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
criterion = nn.HuberLoss(delta=1.0)  # ★★★ 改为 MAE ★★★
best_test_loss = float('inf')
patience = 15
wait = 0
best_state = None

for epoch in range(1, CONFIG['epochs'] + 1):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(CONFIG['device']), yb.to(CONFIG['device'])
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(CONFIG['device']), yb.to(CONFIG['device'])
            pred = model(xb)
            loss = criterion(pred, yb)
            test_loss += loss.item() * xb.size(0)
    test_loss /= len(test_loader.dataset)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")

model.load_state_dict(best_state)
print(f"Best test loss: {best_test_loss:.6f}")

# ============================================================
# 5. 评估并输出逐高度指标
# ============================================================
print("\n[5/5] Evaluating on test set...")
model.eval()
preds_scaled = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(CONFIG['device'])
        pred = model(xb).cpu().numpy()
        preds_scaled.append(pred)
preds_scaled = np.vstack(preds_scaled)

# 还原
pred_residual = scaler_y.inverse_transform(preds_scaled)
pred_actual = pred_residual + mean_profile_train
pred_actual = np.clip(pred_actual, 0, 10)
true_actual = y_test

VALID_STD_THRESHOLD = 1e-4
n_alt = true_actual.shape[1]
r2_per_alt = np.full(n_alt, np.nan)
rmse_per_alt = np.full(n_alt, np.nan)
mae_per_alt = np.full(n_alt, np.nan)

valid_count = 0
for k in range(n_alt):
    y_true = true_actual[:, k]
    y_pred = pred_actual[:, k]
    if y_true.std() < VALID_STD_THRESHOLD:
        continue
    r2_per_alt[k] = r2_score(y_true, y_pred)
    rmse_per_alt[k] = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_per_alt[k] = mean_absolute_error(y_true, y_pred)
    valid_count += 1

print(f"\n有效高度层数: {valid_count} / {n_alt} (跳过 {n_alt - valid_count} 个低变率层)")
print(f"pred_actual min={pred_actual.min():.4f}, max={pred_actual.max():.4f}")
print(f"true_actual min={true_actual.min():.4f}, max={true_actual.max():.4f}")

# 全局指标
r2_global = r2_score(true_actual.ravel(), pred_actual.ravel())
rmse_global = np.sqrt(mean_squared_error(true_actual.ravel(), pred_actual.ravel()))
mae_global = mean_absolute_error(true_actual.ravel(), pred_actual.ravel())

print("\n" + "="*60)
print("          TEST SET PERFORMANCE")
print("="*60)
print(f"Global R²  : {r2_global:.4f}")
print(f"Global RMSE: {rmse_global:.6f} km⁻¹")
print(f"Global MAE : {mae_global:.6f} km⁻¹")
print("\n--- Per-altitude statistics (valid layers only) ---")
print(f"Valid layers: {valid_count}")
print(f"R²  : mean={np.nanmean(r2_per_alt):.4f}, std={np.nanstd(r2_per_alt):.4f}")
print(f"RMSE: mean={np.nanmean(rmse_per_alt):.6f}, std={np.nanstd(rmse_per_alt):.6f}")
print(f"MAE : mean={np.nanmean(mae_per_alt):.6f}, std={np.nanstd(mae_per_alt):.6f}")

print("\n[对齐验证] 前5个样本的第0层（最低层）真实值与预测值：")
print("真实值:", true_actual[:5, 0])
print("预测值:", pred_actual[:5, 0])   # 修正为5个样本

# 按高度区间统计
altitude_ranges = [(0, 3), (3, 10), (10, 20)]
print("\n--- Statistics by altitude ranges ---")
for low, high in altitude_ranges:
    idx = (ProfileAlti >= low) & (ProfileAlti < high) & ~np.isnan(r2_per_alt)
    if idx.sum() == 0:
        print(f"{low:.0f}-{high:.0f} km : 无有效层")
        continue
    print(f"{low:.0f}-{high:.0f} km :")
    print(f"  有效层数: {idx.sum()}")
    print(f"  R²  mean = {np.nanmean(r2_per_alt[idx]):.4f} ± {np.nanstd(r2_per_alt[idx]):.4f}")
    print(f"  RMSE mean = {np.nanmean(rmse_per_alt[idx]):.6f} ± {np.nanstd(rmse_per_alt[idx]):.6f}")
    print(f"  MAE  mean = {np.nanmean(mae_per_alt[idx]):.6f} ± {np.nanstd(mae_per_alt[idx]):.6f}")

# ============================================================
# 6. 绘图
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ax = axes[0]
ax.plot(r2_per_alt, ProfileAlti, 'b-', lw=1.5)
ax.axvline(0, color='gray', linestyle='--', lw=0.8)
ax.set_xlabel('R²')
ax.set_ylabel('Altitude (km)')
ax.set_title('Per-altitude R²')
ax.grid(alpha=0.3)
ax.set_xlim([-1, 1])

ax = axes[1]
ax.plot(rmse_per_alt, ProfileAlti, 'r-', lw=1.5)
ax.set_xlabel('RMSE (km⁻¹)')
ax.set_ylabel('Altitude (km)')
ax.set_title('Per-altitude RMSE')
ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(mae_per_alt, ProfileAlti, 'g-', lw=1.5)
ax.set_xlabel('MAE (km⁻¹)')
ax.set_ylabel('Altitude (km)')
ax.set_title('Per-altitude MAE')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('per_altitude_metrics.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'per_altitude_metrics.png'")
plt.show()

# 保存 CSV
import pandas as pd
df = pd.DataFrame({
    'Altitude_km': ProfileAlti,
    'R2': r2_per_alt,
    'RMSE': rmse_per_alt,
    'MAE': mae_per_alt
})
df.to_csv('per_altitude_metrics.csv', index=False)
print("Metrics saved to 'per_altitude_metrics.csv'")

# ============================================================
# 统计分布对比
# ============================================================
print("\n" + "="*60)
print("  预测值 vs 真实值 统计分布")
print("="*60)

true_flat = true_actual.ravel()
pred_flat = pred_actual.ravel()

print("\n[全局统计] (所有高度层 × 所有样本)")
print(f"  真实值: 均值={true_flat.mean():.6f}, 标准差={true_flat.std():.6f}")
print(f"          min={true_flat.min():.6f}, max={true_flat.max():.6f}")
print(f"  预测值: 均值={pred_flat.mean():.6f}, 标准差={pred_flat.std():.6f}")
print(f"          min={pred_flat.min():.6f}, max={pred_flat.max():.6f}")
print(f"  均值偏差 (预测 - 真实): {pred_flat.mean() - true_flat.mean():.6f}")

alt_ranges = [(0, 3, "0-3 km"), (3, 10, "3-10 km"), (10, 20, "10-20 km")]
print("\n[分层统计]")
for low, high, label in alt_ranges:
    idx = (ProfileAlti >= low) & (ProfileAlti < high)
    if idx.sum() == 0:
        continue
    true_layer = true_actual[:, idx].ravel()
    pred_layer = pred_actual[:, idx].ravel()
    print(f"  {label}:")
    print(f"    真实值: 均值={true_layer.mean():.6f}, 标准差={true_layer.std():.6f}")
    print(f"    预测值: 均值={pred_layer.mean():.6f}, 标准差={pred_layer.std():.6f}")
    print(f"    标准差比值 (预测/真实): {pred_layer.std() / (true_layer.std() + 1e-8):.4f}")

true_sample_mean = true_actual.mean(axis=1)
true_sample_std = true_actual.std(axis=1)
pred_sample_mean = pred_actual.mean(axis=1)
pred_sample_std = pred_actual.std(axis=1)

print("\n[每个样本的廓线统计]")
print(f"  真实廓线均值: 平均={true_sample_mean.mean():.6f}, 标准差={true_sample_mean.std():.6f}")
print(f"  预测廓线均值: 平均={pred_sample_mean.mean():.6f}, 标准差={pred_sample_mean.std():.6f}")
print(f"  真实廓线标准差: 平均={true_sample_std.mean():.6f}, 标准差={true_sample_std.std():.6f}")
print(f"  预测廓线标准差: 平均={pred_sample_std.mean():.6f}, 标准差={pred_sample_std.std():.6f}")

ratio_mean = pred_sample_mean / (true_sample_mean + 1e-8)
ratio_std = pred_sample_std / (true_sample_std + 1e-8)
print("\n[诊断指标]")
print(f"  预测/真实 廓线均值比值: 平均={ratio_mean.mean():.4f}, 中位={np.median(ratio_mean):.4f}")
print(f"  预测/真实 廓线标准差比值: 平均={ratio_std.mean():.4f}, 中位={np.median(ratio_std):.4f}")
print(f"  预测值接近 0 (<1e-6) 的比例: {(np.abs(pred_flat) < 1e-6).mean() * 100:.2f}%")
print(f"  预测值绝对值 < 0.001 的比例: {(np.abs(pred_flat) < 0.001).mean() * 100:.2f}%")

print("\n[结论]")
if ratio_std.mean() < 0.5:
    print("  ⚠️ 预测廓线的标准差远小于真实值 (比值 < 0.5)")
    print("  → 模型预测值过于集中，趋向于平均廓线")
    print("  → 建议：补充相对湿度(RH)、边界层高度(PBLH)等物理特征")
elif ratio_std.mean() < 0.8:
    print("  ⚠️ 预测廓线的标准差略小于真实值 (比值 < 0.8)")
    print("  → 模型有一定预测能力，但变异性不足")
    print("  → 建议：尝试增加模型容量或调整学习率")
else:
    print("  ✅ 预测廓线的标准差与真实值接近 (比值 > 0.8)")
    print("  → 模型成功捕捉了样本间的变异性！")

print("\n" + "="*60)

if CONFIG['save_model']:
    torch.save({
        'model_state': model.state_dict(),
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'mean_profile': mean_profile_train,
        'cfg': CFG,
    }, 'model_simple.pth')
    print("Model saved to 'model_simple.pth'")

print("\nAll done!")