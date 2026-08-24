@echo off
title 风控建模助手 - Risk Modeling Assistant
cd /d "%~dp0"

echo ============================================================
echo   风控建模助手 启动脚本
echo ============================================================
echo.

REM 检查Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到Python，请先安装Python 3.10+
    pause
    exit /b 1
)

REM 检查依赖
python -c "import fastapi, pandas, sklearn" >nul 2>&1
if %errorlevel% neq 0 (
    echo [提示] 正在安装依赖包...
    pip install -r requirements.txt
)

REM 启动后端服务
echo [启动] 后端API服务 (端口 8080)...
start /min "Risk API Server" cmd /c "cd /d %~dp0\backend && python main.py"

REM 等待API就绪
timeout /t 3 /nobreak >nul

REM 启动前端
echo [启动] 前端界面...
start "" "%~dp0\frontend\index.html"

echo.
echo ============================================================
echo   服务已启动！
echo   后端API: http://localhost:8080
echo   API文档: http://localhost:8080/docs
echo   前端界面: 已在浏览器中打开
echo ============================================================
echo.
pause
