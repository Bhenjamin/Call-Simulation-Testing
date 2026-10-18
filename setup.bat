@echo off
setlocal
cd /d "%~dp0"

REM ============================================================
REM Call Simulation Prototype - setup script (Windows)
REM Run this from the repo folder. Needs Miniconda and an NVIDIA GPU.
REM ============================================================

REM Change TORCH_INDEX if your GPU or driver needs a different CUDA build
REM of PyTorch (see https://pytorch.org/get-started/locally/)
set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
set "ENV_NAME=call_simulation_prototype"
set "LLM_MODEL=qwen3:4b-instruct"

echo.
echo === 1/6 Checking conda ===
where conda >nul 2>nul
if not errorlevel 1 goto have_conda
if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" goto add_conda
echo Conda was not found. Install Miniconda, then run this script again.
goto fail
:add_conda
set "PATH=%USERPROFILE%\miniconda3\condabin;%PATH%"
:have_conda

echo.
echo === 2/6 Creating the conda environment (%ENV_NAME%) ===
call conda env list | findstr /b /c:"%ENV_NAME% " >nul
if not errorlevel 1 goto env_exists
call conda create -y -n %ENV_NAME% python=3.12
if errorlevel 1 goto fail
goto env_ready
:env_exists
echo Environment already exists, skipping creation.
:env_ready
call conda activate %ENV_NAME%
if errorlevel 1 goto fail

echo.
echo === 3/6 Installing PyTorch (CUDA build) ===
python -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url %TORCH_INDEX%
if errorlevel 1 goto fail

echo.
echo === 4/6 Installing the other packages ===
python -m pip install -r requirements.txt
if errorlevel 1 goto fail

echo.
echo === 5/6 Checking the GPU ===
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA GPU FOUND')"

echo.
echo === 6/6 Setting up Ollama and the response model ===
where ollama >nul 2>nul
if not errorlevel 1 goto have_ollama
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" goto add_ollama
echo Installing Ollama with winget...
winget install -e --id Ollama.Ollama --accept-package-agreements --accept-source-agreements
:add_ollama
set "PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%"
:have_ollama
where ollama >nul 2>nul
if errorlevel 1 goto ollama_missing
ollama list >nul 2>nul
if not errorlevel 1 goto ollama_running
echo Starting the Ollama server...
start "" /min ollama serve
timeout /t 8 /nobreak >nul
:ollama_running
ollama pull %LLM_MODEL%
if errorlevel 1 goto fail

echo.
echo === Creating folders ===
if not exist user_audio mkdir user_audio
if not exist caller_audio_output mkdir caller_audio_output

echo.
echo ============================================================
echo Setup finished.
echo Still to do by hand:
echo   - Copy the voice reference .wav into voice_references\
echo   - Check personas\persona1.txt exists
echo Then run:  conda activate %ENV_NAME%  and  python main.py
echo ============================================================
pause
exit /b 0

:ollama_missing
echo Ollama could not be found after install. Open a new terminal and run this script again.
goto fail

:fail
echo.
echo Setup stopped because of an error. Read the messages above.
pause
exit /b 1