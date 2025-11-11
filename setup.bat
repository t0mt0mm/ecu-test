@echo off
REM --- Minimal-Setup: venv + Pakete installieren ---

REM 1) venv erstellen (falls nicht vorhanden)
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Erstelle virtuelle Umgebung...
    python.exe -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Konnte venv nicht erstellen. Ist Python im PATH?
        pause
        exit /b 1
    )
)

REM 2) pip in der venv upgraden
echo [INFO] Upgrade pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip

REM 3) Requirements installieren (entweder aus Datei oder Standardliste)
if exist "requirements.txt" (
    echo [INFO] Installiere Anforderungen aus requirements.txt...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
) else (
    echo [INFO] Installiere Standard-Anforderungen...
    ".venv\Scripts\python.exe" -m pip install PyQt5 pyqtgraph numpy python-can cantools pyyaml pandas j1939 asyncqt uptime
)

echo [INFO] Setup fertig.
echo [HINWEIS] Starte das Programm mit: run.bat
pause