@echo off
echo ========================================
echo  BVC Surveillance — Dashboard Streamlit
echo ========================================
echo.

REM Utiliser Anaconda (Python 3.13 avec tous les packages)
set STREAMLIT=C:\Users\sella\Anaconda3\Scripts\streamlit.exe

REM Si Anaconda n'est pas disponible, utiliser le streamlit du PATH
if not exist "%STREAMLIT%" (
    set STREAMLIT=streamlit
)

cd /d "%~dp0\app"
echo Lancement sur http://localhost:8501
echo.
"%STREAMLIT%" run app.py --server.port 8501 --browser.gatherUsageStats false
pause
