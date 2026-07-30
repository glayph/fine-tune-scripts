@echo off
setlocal enabledelayedexpansion

REM Set working directory to script location
cd /d "%~dp0"

REM Check for Python and pip
python --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.8+ is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Upgrade pip and install dependencies
python -m pip install --upgrade pip >nul 2>&1
if errorlevel 1 (
    echo Failed to upgrade pip
    pause
    exit /b 1
)

python -m pip install -r requirements.txt >nul 2>&1
if errorlevel 1 (
    echo Failed to install dependencies from requirements.txt
    pause
    exit /b 1
)

REM Create necessary directories
if not exist ".cache" mkdir ".cache"
if not exist "output" mkdir "output"
if not exist "checkpoints" mkdir "checkpoints"

REM Check for dataset argument
if "%1"=="" (
    echo.
    echo Usage: start_finetune.bat [DATASET_NAME]
    echo Example: start_finetune.bat "dataset_name"
    echo.
    set /p dataset_name="Enter dataset name (or press Enter to use dummy dataset): "
    if not "%dataset_name%"=="" set "DATASET_ARG=%dataset_name%"
) else (
    set "DATASET_ARG=%1"
)

REM Run training script
if defined DATASET_ARG (
    echo.
    echo Starting training with dataset: !DATASET_ARG!
    echo.
    python train.py --dataset !DATASET_ARG! --output_dir "output/finetuned_model"
) else (
    echo.
    echo Starting training with dummy dataset...
    echo.
    python train.py --output_dir "output/finetuned_model"
)

REM Keep window open
if errorlevel 1 (
    echo.
    echo Training completed with errors
    pause
) else (
    echo.
    echo Training completed successfully!
    pause
)