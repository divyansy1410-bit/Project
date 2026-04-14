"""
Flask API – IPL Score Prediction  (scikit-learn backend)
Endpoints:
  GET  /             → serve web dashboard
  GET  /api/options  → label options for dropdowns
  GET  /api/metrics  → model performance metrics
  POST /api/predict  → predict final score
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "model")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# ── Load model artefacts (scikit-learn) ───────────────────────────────────────
print("Loading model artefacts …")

model    = joblib.load(os.path.join(MODEL_DIR, "ipl_model_sklearn.pkl"))
scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))

with open(os.path.join(MODEL_DIR, "label_options.json")) as f:
    label_options = json.load(f)

with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
    model_metrics = json.load(f)

print("Model loaded")

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
CORS(app)

FEATURE_ORDER = [
    "batting_team", "bowling_team", "venue", "batsman", "bowler",
    "over", "ball", "current_score", "wickets",
    "runs_last_5_overs", "wickets_last_5_overs"
]

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/api/options")
def options():
    # Construct exact mapping from real data
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "ipl_data.csv"))
    
    mapping = {
        "batting_team": list(df["batting_team"].unique()),
        "bowling_team": list(df["bowling_team"].unique()),
        "venue": list(df["venue"].unique()),
        "team_batsmen": {team: list(df[df["batting_team"] == team]["batsman"].unique()) for team in df["batting_team"].unique()},
        "team_bowlers": {team: list(df[df["bowling_team"] == team]["bowler"].unique()) for team in df["bowling_team"].unique()}
    }
    return jsonify(mapping)

@app.route("/api/player_stats")
def player_stats():
    player_name = request.args.get("player")
    role = request.args.get("role")  # batsman or bowler
    
    if not player_name or not role:
        return jsonify({"error": "Missing player name or role"}), 400
        
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, "data", "ipl_data.csv"))
        if role == "batsman":
            pdf = df[df["batsman"] == player_name]
            score_dist = pdf["final_score"].value_counts(bins=5).sort_index()
            labels = [f"{int(b.left)}-{int(b.right)}" for b in score_dist.index]
            data = score_dist.values.tolist()
            
            avg_score = round(pdf['final_score'].mean(), 1)
            high_score = int(pdf['final_score'].max())
            matches = len(pdf)
            return jsonify({
                "labels": labels, "data": data, "avg": avg_score, 
                "stats": {"Total Scenarios": matches, "Avg Team Score": avg_score, "Highest Projection": high_score}
            })
        else: # bowler
            pdf = df[df["bowler"] == player_name]
            score_dist = pdf["final_score"].value_counts(bins=5).sort_index()
            labels = [f"{int(b.left)}-{int(b.right)}" for b in score_dist.index]
            data = score_dist.values.tolist()
            
            avg_score = round(pdf['final_score'].mean(), 1)
            low_score = int(pdf['final_score'].min())
            matches = len(pdf)
            return jsonify({
                "labels": labels, "data": data, "avg": avg_score,
                "stats": {"Total Scenarios": matches, "Avg Conceded Score": avg_score, "Best Restriction": low_score}
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/metrics")
def metrics():
    return jsonify(model_metrics)

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    missing = [f for f in FEATURE_ORDER if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        row = []
        for col in FEATURE_ORDER:
            val = data[col]
            if col in encoders:
                le = encoders[col]
                val = str(val)
                if val not in le.classes_:
                    val = le.classes_[0]
                val = int(le.transform([val])[0])
            else:
                val = float(val)
            row.append(val)

        X = np.array([row], dtype=np.float32)
        X = scaler.transform(X)

        pred = model.predict(X)[0]

        current = float(data.get("current_score", 0))
        pred = max(current, round(pred))

        mae   = model_metrics.get("MAE", 8)
        low   = max(0, int(pred - mae))
        high  = int(pred + mae)

        return jsonify({
            "predicted_score": int(pred),
            "range_low":  low,
            "range_high": high,
            "confidence": round(model_metrics.get("R2", 0) * 100, 1)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

import pandas as pd

@app.route("/api/charts")
def chart_data():
    try:
        # Load real data directly
        df = pd.read_csv(os.path.join(BASE_DIR, "data", "ipl_data.csv"))
        
        # 1. Average Final Score by Venue
        venue_avg = df.groupby("venue")["final_score"].mean().sort_values(ascending=False).to_dict()
        
        # 2. Average Final Score by Batting Team
        team_avg = df.groupby("batting_team")["final_score"].mean().sort_values(ascending=False).to_dict()
        
        # 3. Wickets vs Final Score correlation (simple binning)
        # We group records by the number of wickets fallen and get the average score
        wickets_score = df.groupby("wickets")["final_score"].mean().to_dict()
        
        return jsonify({
            "venue_avg": {"labels": list(venue_avg.keys()), "data": list(venue_avg.values())},
            "team_avg":  {"labels": list(team_avg.keys()), "data": list(team_avg.values())},
            "wickets_score": {"labels": list(wickets_score.keys()), "data": list(wickets_score.values())}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n  Server running → http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
