"""
IPL Score Prediction – Deep Learning Model Training (PyTorch)
Architecture: Multi-layer Dense Neural Network (MLP) with BatchNorm + Dropout
Features: batting/bowling team, venue, over, ball, current score,
          wickets, runs/wickets in last 5 overs
Target: final_score (regression)
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

print(f"PyTorch version : {torch.__version__}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device    : {DEVICE}")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "ipl_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print("\n[1/6] Loading data …")
df = pd.read_csv(DATA_PATH)
print(f"  Rows: {len(df):,}  |  Cols: {df.shape[1]}")

# ── Encode categoricals ────────────────────────────────────────────────────────
print("[2/6] Encoding categorical features …")
CAT_COLS = ["batting_team", "bowling_team", "venue", "batsman", "bowler"]
encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ── Feature / target split ─────────────────────────────────────────────────────
FEATURE_COLS = [
    "batting_team", "bowling_team", "venue", "batsman", "bowler",
    "over", "ball", "current_score", "wickets",
    "runs_last_5_overs", "wickets_last_5_overs"
]
TARGET_COL = "final_score"

X = df[FEATURE_COLS].values.astype(np.float32)
y = df[TARGET_COL].values.astype(np.float32).reshape(-1, 1)

# ── Scale ──────────────────────────────────────────────────────────────────────
print("[3/6] Scaling features …")
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ── Train / val / test split ───────────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

def to_tensor(*arrays):
    return [torch.tensor(a, dtype=torch.float32).to(DEVICE) for a in arrays]

Xt, yt       = to_tensor(X_train, y_train)
Xv, yv       = to_tensor(X_val,   y_val)
Xte, yte     = to_tensor(X_test,  y_test)

train_loader = DataLoader(TensorDataset(Xt, yt), batch_size=256, shuffle=True)
val_loader   = DataLoader(TensorDataset(Xv, yv), batch_size=512)

# ── Model ──────────────────────────────────────────────────────────────────────
print("[4/6] Building model …")
INPUT_DIM = X_train.shape[1]

class IPLScoreNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)   # regression output
        )

    def forward(self, x):
        return self.net(x)

model = IPLScoreNet(INPUT_DIM).to(DEVICE)
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable parameters: {total_params:,}")

criterion  = nn.HuberLoss(delta=10.0)
optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                 optimizer, mode='min', factor=0.5, patience=5)

# ── Training loop ──────────────────────────────────────────────────────────────
print("[5/6] Training …")
EPOCHS       = 120
PATIENCE     = 12
best_val     = float('inf')
patience_cnt = 0
best_state   = None

history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}

for epoch in range(1, EPOCHS + 1):
    # --- train ---
    model.train()
    t_loss, t_mae = 0.0, 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred  = model(xb)
        loss  = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        t_loss += loss.item() * len(xb)
        t_mae  += (pred - yb).abs().sum().item()

    t_loss /= len(Xt);  t_mae /= len(Xt)

    # --- val ---
    model.eval()
    v_loss, v_mae = 0.0, 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred   = model(xb)
            v_loss += criterion(pred, yb).item() * len(xb)
            v_mae  += (pred - yb).abs().sum().item()
    v_loss /= len(Xv);  v_mae /= len(Xv)

    scheduler.step(v_loss)
    history["train_loss"].append(t_loss)
    history["val_loss"].append(v_loss)
    history["train_mae"].append(t_mae)
    history["val_mae"].append(v_mae)

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {t_loss:.4f}  MAE: {t_mae:.4f} | "
              f"Val Loss: {v_loss:.4f}  MAE: {v_mae:.4f}")

    # Early stopping
    if v_loss < best_val - 1e-6:
        best_val     = v_loss
        best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

# Restore best weights
if best_state:
    model.load_state_dict(best_state)
    model.to(DEVICE)

# ── Evaluate ───────────────────────────────────────────────────────────────────
print("[6/6] Evaluating on test set …")
model.eval()
with torch.no_grad():
    y_pred_t = model(Xte).cpu().numpy().flatten()
y_true_t = yte.cpu().numpy().flatten()

mae  = mean_absolute_error(y_true_t, y_pred_t)
rmse = np.sqrt(mean_squared_error(y_true_t, y_pred_t))
r2   = r2_score(y_true_t, y_pred_t)

metrics = {"MAE": round(float(mae), 4),
           "RMSE": round(float(rmse), 4),
           "R2": round(float(r2), 4)}
print(f"\n  MAE : {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²  : {r2:.4f}")

# ── Save artefacts ─────────────────────────────────────────────────────────────
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "ipl_model.pt"))
# Save model architecture info so Flask can reconstruct
arch_info = {"input_dim": INPUT_DIM}
with open(os.path.join(MODEL_DIR, "arch.json"), "w") as f:
    json.dump(arch_info, f)

joblib.dump(scaler,   os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))

with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# Save label options for UI dropdowns
label_opts = {col: list(le.classes_) for col, le in encoders.items()}
with open(os.path.join(MODEL_DIR, "label_options.json"), "w") as f:
    json.dump(label_opts, f, indent=2)

print("  Saved: ipl_model.pt | scaler.pkl | encoders.pkl | label_options.json | arch.json")

# ── Plots ──────────────────────────────────────────────────────────────────────
sns.set_style("darkgrid")
plt.rcParams["figure.facecolor"] = "#1a1b2e"
plt.rcParams["axes.facecolor"]   = "#1a1b2e"
plt.rcParams["axes.edgecolor"]   = "#444"
plt.rcParams["text.color"]       = "#e0e0f0"
plt.rcParams["axes.labelcolor"]  = "#e0e0f0"
plt.rcParams["xtick.color"]      = "#aaa"
plt.rcParams["ytick.color"]      = "#aaa"

VC  = "#7c5cfc"
PC  = "#ff6584"

# Loss curve
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(history["train_loss"], color=VC, lw=2, label="Train Loss")
ax.plot(history["val_loss"],   color=PC, lw=2, linestyle="--", label="Val Loss")
ax.set_title("Training vs Validation Loss (Huber)", fontsize=13, fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss")
ax.legend(facecolor="#222", edgecolor="#444")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "loss_curve.png"), dpi=130)
plt.close(fig)

# MAE curve
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(history["train_mae"], color=VC, lw=2, label="Train MAE")
ax.plot(history["val_mae"],   color=PC, lw=2, linestyle="--", label="Val MAE")
ax.set_title("Training vs Validation MAE", fontsize=13, fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Mean Absolute Error (runs)")
ax.legend(facecolor="#222", edgecolor="#444")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "mae_curve.png"), dpi=130)
plt.close(fig)

# Actual vs Predicted
sample = min(1200, len(y_true_t))
idx    = np.random.choice(len(y_true_t), sample, replace=False)
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_true_t[idx], y_pred_t[idx], alpha=0.4, color=VC, s=22, label="Predictions")
lims = [min(y_true_t.min(), y_pred_t.min()), max(y_true_t.max(), y_pred_t.max())]
ax.plot(lims, lims, "r--", lw=1.5, label="Perfect Prediction")
ax.set_xlabel("Actual Score"); ax.set_ylabel("Predicted Score")
ax.set_title("Actual vs Predicted Final Score", fontsize=13, fontweight="bold")
ax.legend(facecolor="#222", edgecolor="#444")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "actual_vs_pred.png"), dpi=130)
plt.close(fig)

# Residual distribution
residuals = y_pred_t - y_true_t
fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(residuals, bins=60, kde=True, color=VC, ax=ax, alpha=0.85)
ax.axvline(0, color="red", linestyle="--", lw=1.5)
ax.set_title("Residual Distribution (Predicted − Actual)", fontsize=13, fontweight="bold")
ax.set_xlabel("Residual (runs)"); ax.set_ylabel("Frequency")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "residuals.png"), dpi=130)
plt.close(fig)

print(f"\n  Plots saved to: {PLOTS_DIR}")
print("\n✅  Training complete!")
