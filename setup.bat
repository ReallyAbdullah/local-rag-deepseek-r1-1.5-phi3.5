@echo off
setlocal

echo Starting the setup process for Local Agentic RAG Application...
echo ---------------------------------------------------------------

REM --- Configuration ---
set PYTHON_COMMAND=python
set MIN_PYTHON_MAJOR=3
set MIN_PYTHON_MINOR=9
set VENV_DIR=.venv
set REQUIREMENTS_FILE=requirements_forWindows.txt

REM --- 1. Check for Python ---
echo.
echo Step 1: Checking Python installation...

%PYTHON_COMMAND% --version >nul 2>&1
if errorlevel 1 (
    echo Error: %PYTHON_COMMAND% is not installed or not in PATH.
    echo Please install Python %MIN_PYTHON_MAJOR%.%MIN_PYTHON_MINOR% or later and try again.
    goto:eof
)

REM Basic Python version check (less robust than the bash script's version_ge)
REM This checks if major is >= 3 and minor is >= 9 if major is 3.
FOR /F "tokens=2 delims=." %%V IN ('%PYTHON_COMMAND% --version 2^>^&1') DO set PY_MAJOR=%%V
FOR /F "tokens=3 delims=." %%W IN ('%PYTHON_COMMAND% --version 2^>^&1') DO set PY_MINOR=%%W


echo Found Python version: %PY_MAJOR%.%PY_MINOR% (approx)

if "%PY_MAJOR%" LSS "%MIN_PYTHON_MAJOR%" (
    echo Error: Python version %PY_MAJOR%.%PY_MINOR% is less than the required %MIN_PYTHON_MAJOR%.%MIN_PYTHON_MINOR%.
    echo Please upgrade your Python installation and try again.
    goto:eof
)
if "%PY_MAJOR%" EQU "%MIN_PYTHON_MAJOR%" (
    if "%PY_MINOR%" LSS "%MIN_PYTHON_MINOR%" (
        echo Error: Python version %PY_MAJOR%.%PY_MINOR% is less than the required %MIN_PYTHON_MAJOR%.%MIN_PYTHON_MINOR%.
        echo Please upgrade your Python installation and try again.
        goto:eof
    )
)
echo Python version check passed (basic check).

REM --- 2. Create Virtual Environment ---
echo.
echo Step 2: Setting up Python virtual environment...
if exist "%VENV_DIR%" (
    echo Virtual environment '%VENV_DIR%' already exists.
    choice /C YN /M "Do you want to remove and recreate it?"
    if errorlevel 2 (
        echo Skipping virtual environment creation. Using existing one.
    ) else (
        echo Removing existing virtual environment...
        rd /s /q "%VENV_DIR%"
        echo Creating new virtual environment in '%VENV_DIR%'...
        %PYTHON_COMMAND% -m venv "%VENV_DIR%"
        if errorlevel 1 (
            echo Error: Failed to create virtual environment.
            goto:eof
        )
    )
) else (
    echo Creating new virtual environment in '%VENV_DIR%'...
    %PYTHON_COMMAND% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Error: Failed to create virtual environment.
        goto:eof
    )
)
echo Virtual environment setup complete.

REM --- 3. Activate Virtual Environment and Install Dependencies ---
echo.
echo Step 3: Installing dependencies...

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo Error: Failed to activate virtual environment.
    echo Please try activating it manually: %VENV_DIR%\Scripts\activate.bat
    goto:eof
)
echo Virtual environment activated.

if not exist "%REQUIREMENTS_FILE%" (
    echo Error: %REQUIREMENTS_FILE% not found.
    echo Please ensure you are in the root directory of the project and the file exists.
    goto:eof
)

echo Installing dependencies from %REQUIREMENTS_FILE%...
pip install -r "%REQUIREMENTS_FILE%"
if errorlevel 1 (
    echo Error: Failed to install dependencies.
    echo Please check the output above for errors. You might need to install some system libraries manually or ensure Build Tools for Visual Studio are installed.
    goto:eof
)
echo Dependencies installed successfully.

REM --- 4. Reminders for Manual Steps ---
echo.
echo ---------------------------------------------------------------
echo Step 4: Manual Prerequisites - IMPORTANT!
echo ---------------------------------------------------------------
echo Please ensure you have the following installed and configured:
echo.
echo 1. Ollama:
echo    - Install from: https://ollama.ai/
echo    - Ensure the Ollama application is running before starting the app.
echo.
echo 2. Ollama Models (run these commands in your terminal/PowerShell):
echo    ollama pull nomic-embed-text
echo    ollama pull phi3.5
echo    ollama pull deepseek-r1:1.5b
echo.
echo 3. Tesseract OCR:
echo    - Install from the UBMannheim project: https://github.com/UB-Mannheim/tesseract/wiki
echo    - IMPORTANT: Add the Tesseract installation directory (e.g., C:\Program Files\Tesseract-OCR) to your system's PATH environment variable.
echo ---------------------------------------------------------------

REM --- 5. How to Run ---
echo.
echo Setup complete!
echo To run the application:
echo 1. Ensure Ollama is running.
echo 2. Activate the virtual environment (if not already active in this session):
echo    %VENV_DIR%\Scripts\activate.bat
echo 3. Run the application:
echo    %PYTHON_COMMAND% app.py
echo.
echo Access the UI at http://localhost:7860 (or the configured port).
echo ---------------------------------------------------------------

endlocal
exit /b 0

:eof
echo.
echo Setup failed. Please check the error messages above.
endlocal
exit /b 1
