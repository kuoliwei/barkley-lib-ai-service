@echo off
REM ============================================================
REM Barkley WebSocket Client 啟動腳本
REM 自動進入 conda 環境 barkley 並執行，使用 Elgato Facecam
REM ============================================================

REM 切換到本 .bat 所在目錄
cd /d "%~dp0"

REM 啟動 conda 環境 barkley（用完整路徑，避免 conda 不在 PATH 上）
call "C:\Users\liweikuo\AppData\Local\miniconda3\Scripts\activate.bat" barkley
if errorlevel 1 (
    echo [錯誤] 無法啟動 conda 環境 barkley
    pause
    exit /b 1
)

REM 執行 WebSocket client
REM   --camera 0     = Elgato Facecam MK.2（本機正確相機）
REM   --backend openvino = 本機推論後端（無 CUDA）
python websocket_client_server.py --camera 0 --backend openvino

REM ============================================================
REM 若要改用 CUDA 推論（需在有 NVIDIA GPU + CUDA 版 PyTorch 的電腦）：
REM   1. 註解掉上面那行 openvino 指令（前面加 REM）
REM   2. 取消下面這行的註解（移除 REM）
REM python websocket_client_server.py --camera 0 --backend cuda
REM ============================================================

pause
