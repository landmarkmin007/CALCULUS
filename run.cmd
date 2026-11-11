@echo off
REM =====================================================================
REM  Calculus Python Tutor - one-click launcher (double-click to run)
REM  Works on any Windows PC with Python 3.10 or newer installed.
REM  First run installs the 4 free libraries (internet needed once);
REM  later runs start offline. If a "_libs" folder exists it is used first.
REM =====================================================================
setlocal
cd /d "%~dp0"

REM ---- 1. find a usable Python interpreter -----------------------------
set PYCMD=
py -3 -c "import sys" >nul 2>nul
if not errorlevel 1 set PYCMD=py -3
if not defined PYCMD (
    python -c "import sys" >nul 2>nul
    if not errorlevel 1 set PYCMD=python
)
if not defined PYCMD (
    if exist "%LOCALAPPDATA%\Qianwen\User Data\qwen-agent\resources\runtimes\python\3.14.6\python.exe" (
        set PYCMD="%LOCALAPPDATA%\Qianwen\User Data\qwen-agent\resources\runtimes\python\3.14.6\python.exe"
    )
)
if not defined PYCMD (
    echo [!] No usable Python found on this computer.
    echo     Install Python 3.10+ from https://www.python.org/downloads/
    echo     and tick "Add python.exe to PATH" during setup, then re-run this file.
    pause
    exit /b 1
)

REM ---- 2. use bundled libraries when this folder ships them ------------
if exist "%~dp0_libs" set "PYTHONPATH=%~dp0_libs;%PYTHONPATH%"

REM ---- 3. first run: install the dependencies once ----------------------
%PYCMD% -c "import streamlit, sympy, numpy, matplotlib" >nul 2>nul
if errorlevel 1 (
    echo First run: installing streamlit / sympy / numpy / matplotlib ...
    echo This needs an internet connection and takes a few minutes.
    %PYCMD% -m pip install --disable-pip-version-check -r requirements.txt
    if errorlevel 1 (
        echo [!] Dependency install failed - check the internet connection and retry.
        pause
        exit /b 1
    )
)

set "MPLCONFIGDIR=%~dp0_mplcache"
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%"

REM ---- 4. silence Streamlit's first-run email prompt --------------------
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
    >"%USERPROFILE%\.streamlit\credentials.toml" echo [general]
    >>"%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)

echo.
echo Calculus Python Tutor starting... your browser should open
echo http://localhost:8501 - if it does not, type that address in by hand.
echo Keep this window open while using the app; close it to stop the server.
echo Only this computer can reach the server (--server.address localhost).
echo.
%PYCMD% -m streamlit run calculus_tutor.py --server.address localhost --global.developmentMode false --browser.gatherUsageStats false
pause
