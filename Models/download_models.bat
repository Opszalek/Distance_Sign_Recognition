@echo off
REM Check if gdown is installed
where gdown >nul 2>&1
if %errorlevel% neq 0 (
    echo gdown could not be found, installing...
    pip install gdown
)

REM Set output directory
set "OUTPUT_DIR=."

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
)

REM Download version file
gdown --fuzzy https://docs.google.com/document/d/1n2NpuAOEweBrwn-rftuyUz2dGmns0JosEMvYdXhu-kU/edit?usp=sharing -O "%OUTPUT_DIR%\version_drive.txt"

REM Compare version files and download folder if they differ
fc /b version.txt "%OUTPUT_DIR%\version_drive.txt" >nul
if %errorlevel% neq 0 (
    gdown --folder https://drive.google.com/drive/folders/1F-84oPjwY6zqReKcOvymW7Q_SfKByVAD?usp=sharing -O "%OUTPUT_DIR%"
    move "%OUTPUT_DIR%\version_drive.txt" "%OUTPUT_DIR%\version.txt"
) else (
    echo Models are up to date
)
