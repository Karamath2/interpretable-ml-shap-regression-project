@echo off
echo ==========================================
echo  Interpretable ML - Full Pipeline Runner
echo ==========================================

REM ----- CHECK IF VENV EXISTS -----
IF NOT EXIST "venv\" (
    echo [ERROR] Virtual environment not found!
    echo Please create it first using:
    echo     python -m venv venv
    echo.
    pause
    exit /b
)

REM ----- ACTIVATE VENV -----
echo Activating virtual environment...
call venv\Scripts\activate.bat
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b
)

echo.
echo ======== Running preprocessing ========
python src\preprocess.py
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Preprocessing failed.
    pause
    exit /b
)

echo.
echo ======== Training XGBoost Model ========
python src\train_xgb.py
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] XGBoost training failed.
    pause
    exit /b
)

echo.
echo ======== Training DNN Model ========
python src\train_dnn.py
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] DNN training failed.
    pause
    exit /b
)

echo.
echo ======== Evaluating Models ========
python src\evaluate.py
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Evaluation failed.
    pause
    exit /b
)

echo.
echo ======== Running SHAP Analysis ========
python src\shap_analysis.py
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] SHAP analysis failed.
    pause
    exit /b
)

echo.
echo ==========================================
echo   ALL TASKS COMPLETED SUCCESSFULLY! 🎉
echo   Outputs saved in /outputs directory.
echo ==========================================

pause
