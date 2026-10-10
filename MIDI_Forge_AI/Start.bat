@echo off
setlocal enabledelayedexpansion
title MIDI Forge AI - Iniciar Aplicativo

echo =====================================================================
echo                MIDI FORGE AI - GERADOR E TREINADOR LOCAL
echo =====================================================================
echo.

:: 1. Ir para a pasta do proprio aplicativo
cd /d "%~dp0"

:: 2. Identificar interpretador Python (embutido ou .venv ou sistema)
set "PY_EXE="

if exist "%~dp0python\python.exe" (
    set "PY_EXE=%~dp0python\python.exe"
    echo [*] Utilizando Python portatil embutido: !PY_EXE!
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_EXE=%~dp0.venv\Scripts\python.exe"
    echo [*] Utilizando ambiente virtual local: !PY_EXE!
) else (
    where python >nul 2>nul
    if !errorlevel! equ 0 (
        set "PY_EXE=python"
        echo [*] Utilizando Python do sistema operacional.
    )
)

if "%PY_EXE%"=="" (
    echo.
    echo [ERRO CRITICO] Nenhum interpretador Python encontrado!
    echo.
    echo Por favor, execute primeiro o script 'Setup_First_Run.bat' para
    echo configurar o ambiente e as dependencias necessarias.
    echo.
    pause
    exit /b 1
)

:: 3. Verificar dependencias criticas (PyTorch e Mido)
echo [*] Verificando integridade das dependencias locais...
"%PY_EXE%" -c "import torch; import mido; print('[OK] PyTorch e Mido detectados com sucesso.')" >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo [AVISO] Dependencias ausentes ou corrompidas!
    echo Sera executado o instalador local Setup_First_Run.bat.
    echo.
    call "%~dp0Setup_First_Run.bat"
    if %errorlevel% neq 0 (
        echo [ERRO] Falha ao preparar dependencias.
        pause
        exit /b 1
    )
)

:: 4. Protecao contra multiplas instancias simultaneas
set "PORT=8080"
netstat -ano | findstr ":%PORT% " >nul 2>nul
if %errorlevel% equ 0 (
    echo [AVISO] A porta %PORT% ja esta em uso.
    echo O MIDI Forge AI pode ja estar em execucao.
    echo Abrindo a interface no navegador...
    start http://127.0.0.1:%PORT%
    pause
    exit /b 0
)

:: 5. Iniciar o servidor local na interface de loopback
echo.
echo [*] Iniciando servidor local na porta %PORT% (127.0.0.1)...
echo [*] Pressione Ctrl+C nesta janela quando desejar encerrar o aplicativo.
echo.

:: Aguarda 2 segundos e abre o navegador
start /b cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:%PORT%"

:: Inicia o processo Python
"%PY_EXE%" "%~dp0app\server.py" --host 127.0.0.1 --port %PORT% --root "%~dp0."

if %errorlevel% neq 0 (
    echo.
    echo [ERRO] O servidor encerrou com codigo de erro %errorlevel%.
    pause
)

exit /b 0
