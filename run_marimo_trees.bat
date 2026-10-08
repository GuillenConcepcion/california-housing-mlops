@echo off
cd /d "%~dp0"
echo ====================================================================
echo  Iniciando Marimo Editor para Visualising Trees (James Gibbins)
echo ====================================================================
python -m marimo edit "external\visualising_trees\decision_trees.py"
pause
