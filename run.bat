@echo off
echo Starting Caption Generation Server...

cd /d "%~dp0"

if not exist gen-env (
    echo Creating Python virtual environment...
    python -m venv gen-env
)

call .\gen-env\Scripts\activate.bat

if exist requirements.txt (
    echo Installing dependencies from requirements.txt...
    pip install --upgrade pip
    pip install -r requirements.txt
) else (
    echo WARNING: requirements.txt not found! Skipping dependency installation.
)

echo Running script.py...
python script.py

pause
