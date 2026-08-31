@echo off
echo ===================================================
echo   Starting IQSYS MOM AI Agent (Backend + Streamlit)
echo ===================================================

cd /d "%~dp0aiml-automation"

echo Starting FastAPI Backend on Port 8100...
start "FastAPI AI Backend" .venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8100

timeout /t 2 /nobreak >nul

echo Starting Streamlit UI on Port 8501...
start "Streamlit AI Studio" .venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.port 8501

echo.
echo ===================================================
echo   App is running!
echo   Streamlit UI: http://localhost:8501
echo   FastAPI Docs: http://127.0.0.1:8100/docs
echo ===================================================
