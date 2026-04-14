# IPL Score Predictor

An end-to-end **Deep Learning** project that predicts the final score of an IPL T20 cricket match based on the current match situation.

---

## 🗂️ Project Structure

```
AI PROJECT/
├── data/
│   ├── generate_data.py    ← synthetic IPL dataset generator
│   └── ipl_data.csv        ← generated dataset (15,000 records)
├── model/
│   ├── train_model.py      ← deep learning training pipeline
│   ├── ipl_model.keras     ← trained model (after training)
│   ├── scaler.pkl          ← feature scaler
│   ├── encoders.pkl        ← label encoders
│   ├── label_options.json  ← dropdown values for UI
│   └── metrics.json        ← MAE / RMSE / R² scores
├── static/
│   ├── index.html          ← web dashboard
│   ├── style.css           ← premium UI styles
│   ├── app.js              ← frontend JavaScript
│   └── plots/              ← training charts (auto-generated)
├── app.py                  ← Flask REST API server
├── requirements.txt        ← Python dependencies
└── run.ps1                 ← one-click setup & run script
```

---

## 🚀 Quick Start

### Option A – One-click (PowerShell)
```powershell
cd "c:\Users\Dell\Desktop\AI PROJECT"
.\run.ps1
```

### Option B – Step by step
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
python data/generate_data.py

# 3. Train the model
python model/train_model.py

# 4. Start the web server
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## 🧠 Model Architecture

| Layer | Units | Activation | Extras |
|-------|-------|-----------|--------|
| Input | 11 | — | StandardScaler pre-processing |
| Dense | 256 | ReLU | L2 reg + BatchNorm + Dropout(0.3) |
| Dense | 128 | ReLU | L2 reg + BatchNorm + Dropout(0.25) |
| Dense | 64  | ReLU | BatchNorm + Dropout(0.2) |
| Dense | 32  | ReLU | — |
| Output| 1   | Linear | Regression |

- **Loss**: Huber (robust to outliers)  
- **Optimizer**: Adam (lr=1e-3)  
- **Callbacks**: EarlyStopping · ReduceLROnPlateau · ModelCheckpoint  

---

## 📊 Features Used

| Feature | Type |
|---------|------|
| Batting Team | Categorical |
| Bowling Team | Categorical |
| Venue | Categorical |
| Batsman | Categorical |
| Bowler | Categorical |
| Over | Numeric |
| Ball | Numeric |
| Current Score | Numeric |
| Wickets Fallen | Numeric |
| Runs (last 5 overs) | Numeric |
| Wickets (last 5 overs) | Numeric |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web dashboard |
| GET | `/api/options` | Dropdown option lists |
| GET | `/api/metrics` | Model performance metrics |
| POST | `/api/predict` | Predict final score |

**POST /api/predict** example body:
```json
{
  "batting_team": "Mumbai Indians",
  "bowling_team": "Chennai Super Kings",
  "venue": "Wankhede Stadium",
  "batsman": "Rohit Sharma",
  "bowler": "Jasprit Bumrah",
  "over": 12,
  "ball": 3,
  "current_score": 98,
  "wickets": 2,
  "runs_last_5_overs": 48,
  "wickets_last_5_overs": 1
}
```

**Response:**
```json
{
  "predicted_score": 172,
  "range_low": 164,
  "range_high": 180,
  "confidence": 87.3
}
```
