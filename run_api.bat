@echo off
cd /d "%~dp0"
echo ====================================================================
echo  Iniciando Microservicio FastAPI (California Housing + CQR 95%)
echo  Lead Architect: Guillen Concepcion
echo ====================================================================
python scripts\serve.py
pause
