"""
train_eval_simple.py
简化版训练+评估脚本，只输出逐高度层的 R²、RMSE、MAE
使用残差学习 + 量级放大（1000倍）
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 从原训练脚本导入数据加载、特征工程和模型类
# 请确保原文件 Main_ACDL_ERA5_SpatiotemporalAttention_DL.py 在同一目录
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
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'test_ratio': 0.2,          # 用最后20%数据作为测试集（时间顺序）
    'SCALE': 1000.0,            # 残差放大倍数
    'save_model': False,        # 是否保存模型
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

# 按时间顺序划分训练/测试集（前80%训练，后20%测试）
n = X_eng.shape[0]
test_start = int(n * (1 - CONFIG['test_ratio']))
X_train, X_test = X_eng[:test_start], X_eng[test_start:]
y_train, y_test = y_raw[:test_start], y_raw[test_start:]
print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# ============================================================
# 2. 构建残差目标（训练集平均廓线）
# ============================================================
print("\n[2/5] Computing residual targets...")
mean_profile_train = y_train.mean(axis=0, keepdims=True)   # 训练集平均廓线
y_train_residual = y_train - mean_profile_train
y_test_residual = y_test - mean_profile_train

# 放大残差（解决梯度量级问题）
SCALE = CONFIG['SCALE']
y_train_scaled = y_train_residual * SCALE
y_test_scaled = y_test_residual * SCALE

# ============================================================
# 3. 特征标准化（仅用训练集）
# ============================================================
print("\n[3/5] Feature normalization...")
scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)

# 转换为 Tensor
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
    dropout=CFG['dropout']
).to(CONFIG['device'])

optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
criterion = nn.MSELoss()   # 放大后的残差量级~1，MSE合适

best_test_loss = float('inf')
best_state = None

for epoch in range(1, CONFIG['epochs'] + 1):
    # 训练
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

    # 验证（测试集）
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

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")

# 加载最佳模型
model.load_state_dict(best_state)
print(f"Best test loss: {best_test_loss:.6f}")

# ============================================================
# 5. 评估并输出逐高度指标
# ============================================================
print("\n[5/5] Evaluating on test set...")
model.eval()
preds_scaled = []
trues_scaled = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(CONFIG['device'])
        pred = model(xb).cpu().numpy()
        preds_scaled.append(pred)
        trues_scaled.append(yb.numpy())
preds_scaled = np.vstack(preds_scaled)
trues_scaled = np.vstack(trues_scaled)

# 还原真实消光系数
pred_residual = preds_scaled / SCALE
pred_actual = pred_residual + mean_profile_train
true_actual = y_test

# -------- 逐高度指标计算 ----------
n_alt = true_actual.shape[1]
r2_per_alt = np.zeros(n_alt)
rmse_per_alt = np.zeros(n_alt)
mae_per_alt = np.zeros(n_alt)

for k in range(n_alt):
    y_true = true_actual[:, k]
    y_pred = pred_actual[:, k]
    r2_per_alt[k] = r2_score(y_true, y_pred)
    rmse_per_alt[k] = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_per_alt[k] = mean_absolute_error(y_true, y_pred)

# 全局指标（参考）
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
print("\n--- Per-altitude statistics ---")
print(f"R²  : mean={r2_per_alt.mean():.4f}, std={r2_per_alt.std():.4f}, min={r2_per_alt.min():.4f}, max={r2_per_alt.max():.4f}")
print(f"RMSE: mean={rmse_per_alt.mean():.6f}, std={rmse_per_alt.std():.6f}")
print(f"MAE : mean={mae_per_alt.mean():.6f}, std={mae_per_alt.std():.6f}")

# 按高度区间统计（示例：0-3km, 3-10km, 10-20km）
altitude_ranges = [(0, 3), (3, 10), (10, 20)]
print("\n--- Statistics by altitude ranges ---")
for low, high in altitude_ranges:
    idx = (ProfileAlti >= low) & (ProfileAlti < high)
    if idx.sum() == 0:
        continue
    print(f"{low:.0f}-{high:.0f} km :")
    print(f"  R²  mean = {r2_per_alt[idx].mean():.4f} ± {r2_per_alt[idx].std():.4f}")
    print(f"  RMSE mean = {rmse_per_alt[idx].mean():.6f} ± {rmse_per_alt[idx].std():.6f}")
    print(f"  MAE  mean = {mae_per_alt[idx].mean():.6f} ± {mae_per_alt[idx].std():.6f}")

# ============================================================
# 6. 绘图：逐高度 R²、RMSE、MAE
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].plot(r2_per_alt, ProfileAlti, 'b-', lw=1.5)
axes[0].axvline(0, color='gray', linestyle='--', lw=0.8)
axes[0].set_xlabel('R²')
axes[0].set_ylabel('Altitude (km)')
axes[0].set_title('Per-altitude R²')
axes[0].grid(alpha=0.3)

axes[1].plot(rmse_per_alt, ProfileAlti, 'r-', lw=1.5)
axes[1].set_xlabel('RMSE (km⁻¹)')
axes[1].set_ylabel('Altitude (km)')
axes[1].set_title('Per-altitude RMSE')
axes[1].grid(alpha=0.3)

axes[2].plot(mae_per_alt, ProfileAlti, 'g-', lw=1.5)
axes[2].set_xlabel('MAE (km⁻¹)')
axes[2].set_ylabel('Altitude (km)')
axes[2].set_title('Per-altitude MAE')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('per_altitude_metrics.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'per_altitude_metrics.png'")
plt.show()

# 可选：保存逐高度指标到CSV
import pandas as pd
df = pd.DataFrame({
    'Altitude_km': ProfileAlti,
    'R2': r2_per_alt,
    'RMSE': rmse_per_alt,
    'MAE': mae_per_alt
})
df.to_csv('per_altitude_metrics.csv', index=False)
print("Metrics saved to 'per_altitude_metrics.csv'")

# 保存模型（可选）
if CONFIG['save_model']:
    torch.save({
        'model_state': model.state_dict(),
        'scaler_x': scaler_x,
        'mean_profile': mean_profile_train,
        'scale_factor': SCALE,
        'cfg': CFG,
    }, 'model_simple.pth')
    print("Model saved to 'model_simple.pth'")

print("\nAll done!")