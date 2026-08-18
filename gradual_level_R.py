"""
train_eval_simple.py
简化版训练+评估脚本，只输出逐高度层的 R²、RMSE、MAE
使用残差学习 + 标准化
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from Main_ACDL_ERA5_SpatiotemporalAttention_DL import (
    load_data, feature_engineering, SpatioTemporalAttention, CFG
)

# ============================================================
# 配置
# ============================================================
CONFIG = {
    'mat_file': r'G:/ACDL/Data/MatchResult/ACDL_ERA5_Matched_20220601.mat',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'batch_size': 256,
    'epochs': 100,
    'lr': 1e-4,              # 降低学习率
    'weight_decay': 1e-3,    # 增加权重衰减
    'dropout': 0.3,          # 增加 Dropout
    'test_ratio': 0.2,
    'save_model': False,
}

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
print(f"Device: {CONFIG['device']}")

# ============================================================
# 1. 数据加载与划分
# ============================================================
print("\n[1/5] Loading data...")
X_raw, y_raw, ProfileAlti = load_data(CONFIG['mat_file'])
X_eng, feat_names = feature_engineering(X_raw)
print(f"Feature shape: {X_eng.shape}, Target shape: {y_raw.shape}")

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
# 4. 模型构建与训练
# ============================================================
print("\n[4/5] Building and training model...")
model = SpatioTemporalAttention(
    n_feat=X_eng.shape[1],
    d_model=CFG['d_model'],
    n_heads=CFG['n_heads'],
    n_layers=CFG['n_layers'],
    dropout=CONFIG['dropout']   # 使用新的 dropout
).to(CONFIG['device'])

optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
criterion = nn.MSELoss()

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
# 5. 评估并输出逐高度指标（改进版）
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
pred_actual = np.clip(pred_actual, 0, 50)
true_actual = y_test

# ========== 修改开始 ==========
# 配置：忽略真实值标准差小于该阈值的层
VALID_STD_THRESHOLD = 1e-4
# ==============================

n_alt = true_actual.shape[1]
r2_per_alt = np.full(n_alt, np.nan)
rmse_per_alt = np.full(n_alt, np.nan)
mae_per_alt = np.full(n_alt, np.nan)

valid_count = 0
valid_layers = []
for k in range(n_alt):
    y_true = true_actual[:, k]
    y_pred = pred_actual[:, k]
    
    # 检查真实值是否有足够的变化
    if y_true.std() < VALID_STD_THRESHOLD:
        continue
    
    r2_per_alt[k] = r2_score(y_true, y_pred)
    rmse_per_alt[k] = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_per_alt[k] = mean_absolute_error(y_true, y_pred)
    valid_count += 1
    valid_layers.append(k)

print(f"\n有效高度层数: {valid_count} / {n_alt} (跳过 {n_alt - valid_count} 个低变率层)")

# -------- 诊断打印 ----------
print(f"pred_actual min={pred_actual.min():.4f}, max={pred_actual.max():.4f}")
print(f"true_actual min={true_actual.min():.4f}, max={true_actual.max():.4f}")

# -------- 全局指标 ----------
r2_global = r2_score(true_actual.ravel(), pred_actual.ravel())
rmse_global = np.sqrt(mean_squared_error(true_actual.ravel(), pred_actual.ravel()))
mae_global = mean_absolute_error(true_actual.ravel(), pred_actual.ravel())

# -------- 打印结果 ----------
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
ax.set_xlim([-1, 1])  # 聚焦在合理范围

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