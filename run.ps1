# IPL Score Predictor – Full Pipeline Runner
# Run this script once to: install deps → generate data → train model → start server

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host " IPL Score Predictor – Setup & Run " -ForegroundColor Cyan
Write-Host "============================================`n" -ForegroundColor Cyan

Set-Location $ProjectDir

# 1. Install dependencies
Write-Host "[Step 1/4] Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet
if ($LASTEXITCODE -ne 0) { Write-Error "pip install failed"; exit 1 }
Write-Host "  ✅ Dependencies installed" -ForegroundColor Green

# 2. Generate dataset
Write-Host "`n[Step 2/4] Generating synthetic IPL dataset..." -ForegroundColor Yellow
python data/generate_data.py
if ($LASTEXITCODE -ne 0) { Write-Error "Data generation failed"; exit 1 }
Write-Host "  ✅ Dataset created: data/ipl_data.csv" -ForegroundColor Green

# 3. Train model
Write-Host "`n[Step 3/4] Training deep learning model (this may take a few minutes)..." -ForegroundColor Yellow
python model/train_model.py
if ($LASTEXITCODE -ne 0) { Write-Error "Model training failed"; exit 1 }
Write-Host "  ✅ Model trained and saved to model/" -ForegroundColor Green

# 4. Start Flask server
Write-Host "`n[Step 4/4] Starting Flask server..." -ForegroundColor Yellow
Write-Host "  🚀 Dashboard → http://localhost:5000" -ForegroundColor Magenta
Write-Host "  Press Ctrl+C to stop`n" -ForegroundColor Gray
python app.py
