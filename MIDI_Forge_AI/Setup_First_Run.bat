@echo off
setlocal enabledelayedexpansion
title MIDI Forge AI - Preparacao Inicial do Ambiente

echo =====================================================================
echo          MIDI FORGE AI - CONFIGURACAO E PREPARACAO LOCAL
echo =====================================================================
echo.
echo Este assistente prepara o ambiente Python independente e as
echo dependencias locais necessarias para treinamento e geracao musical.
echo.

cd /d "%~dp0"

:: 1. Verificar presenca do Python no sistema
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERRO CRITICO] Python 3.9+ nao foi detectado no sistema!
    echo.
    echo Para utilizar o MIDI Forge AI:
    echo 1. Baixe o instalador oficial do Python (versao 3.10 ou 3.11 recomendada):
    echo    https://www.python.org/downloads/windows/
    echo 2. Marque a opcao "[x] Add Python to PATH" durante a instalacao.
    echo 3. Execute este arquivo novamente.
    echo.
    pause
    exit /b 1
)

echo [*] Python detectado:
python --version
echo.

:: 2. Criar ambiente virtual isolado (.venv)
if not exist "%~dp0.venv" (
    echo [*] Criando ambiente virtual isolado em .venv...
    python -m venv "%~dp0.venv"
    if %errorlevel% neq 0 (
        echo [ERRO] Falha ao criar ambiente virtual .venv.
        pause
        exit /b 1
    )
    echo [*] Ambiente virtual criado com sucesso!
) else (
    echo [*] Ambiente virtual .venv ja existente.
)

:: 3. Atualizar pip e instalar dependencias
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "VENV_PIP=%~dp0.venv\Scripts\pip.exe"

echo.
echo [*] Instalando dependencias essenciais do requirements.txt...
echo [*] Isso pode levar alguns minutos na primeira execucao (download do PyTorch)...
echo.

"%VENV_PIP%" install --upgrade pip
"%VENV_PIP%" install -r "%~dp0requirements.txt"

if %errorlevel% neq 0 (
    echo.
    echo [ERRO] Falha durante a instalacao das dependencias.
    echo Verifique sua conexao ou tente instalar manualmente via pip.
    pause
    exit /b 1
)

:: 4. Executar testes de integridade basicos
echo.
echo [*] Executando verificacao de integridade e diagnósticos...
"%VENV_PY%" -c "import torch; import mido; print('[OK] PyTorch:', torch.__version__, '| Mido:', mido.__file__, '| GPU CUDA:', torch.cuda.is_available())"

echo.
echo =====================================================================
echo   [SUCESSO] Configuracao concluida!
echo   Voce ja pode iniciar o aplicativo dando um duplo clique em:
echo   Start.bat
echo =====================================================================
echo.
pause
exit /b 0
