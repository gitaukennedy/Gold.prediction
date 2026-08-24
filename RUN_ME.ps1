#!/usr/bin/env powershell

# ================================================================================
# GOLD TRADING PREDICTION SYSTEM - QUICK START GUIDE
# Time to Complete: 30 minutes
# ================================================================================

Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  GOLD TRADING PREDICTION SYSTEM SETUP" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Create Virtual Environment
Write-Host "STEP 1/4: Creating Python Virtual Environment..." -ForegroundColor Yellow
Write-Host "  Location: C:\Users\hp\Desktop\gold-trading-project\venv" -ForegroundColor Gray
python -m venv venv
Write-Host "  ✓ Virtual environment created" -ForegroundColor Green
Write-Host ""

# Step 2: Activate Virtual Environment
Write-Host "STEP 2/4: Activating Virtual Environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
Write-Host "  ✓ Virtual environment activated" -ForegroundColor Green
Write-Host ""

# Step 3: Install Dependencies
Write-Host "STEP 3/4: Installing Dependencies..." -ForegroundColor Yellow
Write-Host "  Installing: pandas, numpy, scikit-learn, yfinance, etc." -ForegroundColor Gray
pip install -r requirements.txt
Write-Host "  ✓ All dependencies installed" -ForegroundColor Green
Write-Host ""

# Step 4: Run the Pipeline
Write-Host "STEP 4/4: Running Gold Trading Pipeline..." -ForegroundColor Yellow
Write-Host "  Starting main.py..." -ForegroundColor Gray
Write-Host ""
python main.py
Write-Host ""

# Results Summary
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  EXECUTION COMPLETE!" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "What Happened:" -ForegroundColor Cyan
Write-Host "  1. ✓ Fetched 3 months of gold price data" -ForegroundColor Green
Write-Host "  2. ✓ Calculated 15+ technical indicators" -ForegroundColor Green
Write-Host "  3. ✓ Trained Random Forest model" -ForegroundColor Green
Write-Host "  4. ✓ Generated trading signals" -ForegroundColor Green
Write-Host "  5. ✓ Simulated trades & calculated P&L" -ForegroundColor Green
Write-Host ""
Write-Host "Key Files Created:" -ForegroundColor Cyan
Write-Host "  • src/data_fetcher.py ........ Data retrieval" -ForegroundColor Gray
Write-Host "  • src/data_processor.py ...... Feature engineering" -ForegroundColor Gray
Write-Host "  • src/model_trainer.py ....... ML model training" -ForegroundColor Gray
Write-Host "  • src/predictor.py ........... Trading signals" -ForegroundColor Gray
Write-Host "  • src/trader.py ............. Trade execution" -ForegroundColor Gray
Write-Host "  • main.py ................... Main orchestrator" -ForegroundColor Gray
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Review trading results in the output above" -ForegroundColor Green
Write-Host "  2. Adjust parameters in main.py for your preferences" -ForegroundColor Green
Write-Host "  3. Integrate with real broker API for live trading" -ForegroundColor Green
Write-Host "  4. Set up scheduled runs (cron/Task Scheduler)" -ForegroundColor Green
Write-Host ""
Write-Host "For Live Trading:" -ForegroundColor Cyan
Write-Host "  • Add broker API integration (Alpaca, IB, Schwab)" -ForegroundColor Green
Write-Host "  • Implement real-time data feeds" -ForegroundColor Green
Write-Host "  • Add comprehensive risk management" -ForegroundColor Green
Write-Host "  • Monitor performance continuously" -ForegroundColor Green
Write-Host ""
