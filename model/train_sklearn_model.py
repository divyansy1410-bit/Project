"""
Train a scikit-learn GradientBoostingRegressor for IPL Score Prediction.
Replaces the PyTorch model — produces a lightweight model compatible with Vercel.
Run: python model/train_sklearn_model.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "ipl_data.csv")
MODEL_DIR = BASE_DIR

FEATURE_ORDER = [
    "batting_team", "bowling_team", "venue", "batsman", "bowler",
    "over", "ball", "current_score", "wickets",
    "runs_last_5_overs", "wickets_last_5_overs"
]
CATEGORICAL = ["batting_team", "bowling_team", "venue", "batsman", "bowler"]
TARGET = "final_score"

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data …")
df_raw = pd.read_csv(DATA_PATH)
df = df_raw.copy()

# ── Build label encoders ───────────────────────────────────────────────────────
print("Encoding categorical features …")
encoders = {}
for col in CATEGORICAL:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ── Build label_options.json (for dropdowns) ───────────────────────────────────
label_options = {
    "batting_team": list(encoders["batting_team"].classes_),
    "bowling_team": list(encoders["bowling_team"].classes_),
    "venue":        list(encoders["venue"].classes_),
    "batsman":      list(encoders["batsman"].classes_),
    "bowler":       list(encoders["bowler"].classes_),
}

# team → players mapping (used by frontend)
team_batsmen = {
    team: list(df_raw[df_raw["batting_team"] == team]["batsman"].unique())
    for team in df_raw["batting_team"].unique()
}
team_bowlers = {
    team: list(df_raw[df_raw["bowling_team"] == team]["bowler"].unique())
    for team in df_raw["bowling_team"].unique()
}
label_options["team_batsmen"] = team_batsmen
label_options["team_bowlers"] = team_bowlers

# ── Train / test split ─────────────────────────────────────────────────────────
X = df[FEATURE_ORDER].values
y = df[TARGET].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Train model ────────────────────────────────────────────────────────────────
print("Training GradientBoostingRegressor (this may take ~1-2 min) …")
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    min_samples_leaf=10,
    random_state=42,
    verbose=1
)
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\n  MAE  : {mae:.2f}")
print(f"  RMSE : {rmse:.2f}")
print(f"  R²   : {r2:.4f}")

# ── Save artefacts ─────────────────────────────────────────────────────────────
joblib.dump(model,    os.path.join(MODEL_DIR, "ipl_model_sklearn.pkl"), compress=3)
joblib.dump(scaler,   os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))

with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump({"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}, f)

with open(os.path.join(MODEL_DIR, "label_options.json"), "w") as f:
    json.dump(label_options, f, indent=2)

print("\n✅ Saved: ipl_model_sklearn.pkl, scaler.pkl, encoders.pkl, metrics.json, label_options.json")
