@echo off
echo =====================================================================
echo  Starting Smart Sales Prediction & Demand Forecasting System
echo =====================================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo Launching Streamlit Dashboard on http://localhost:8501 ...
streamlit run dashboard\app.py
pause
