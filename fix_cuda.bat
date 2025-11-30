@echo off
cd /d E:\git\ai\lerobot-scratch

echo Uninstalling PyTorch...
venv\Scripts\python.exe -m pip uninstall torch torchvision torchaudio -y

echo.
echo Installing PyTorch NIGHTLY with CUDA 12.8 (for RTX 5090 Blackwell)...
venv\Scripts\python.exe -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

echo.
echo Testing CUDA...
venv\Scripts\python.exe check_cuda.py

pause
