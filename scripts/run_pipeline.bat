@echo off
echo ====================================================================
echo  Odysseus MLOps Execution Pipeline: Ajuste de Hiperparametros Vivienda California
echo  Lead Architect: Guillén Concepción
echo ====================================================================

echo [1/3] Entrenando modelo...
python scripts\train.py
if errorlevel 1 (
    echo Error durante el entrenamiento.
    exit /b 1
)

echo [2/3] Evaluando métricas del modelo...
python scripts\evaluate.py
if errorlevel 1 (
    echo Error durante la evaluación.
    exit /b 1
)

echo [3/3] Ejecutando pruebas unitarias...
pytest tests\
if errorlevel 1 (
    echo Fallaron las pruebas unitarias.
    exit /b 1
)

echo ====================================================================
echo  Pipeline completado exitosamente!
echo ====================================================================
pause
