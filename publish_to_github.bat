@echo off
chcp 65001 > nul
echo ==============================================================================
echo  ODYSSEUS AI PLATFORM - PUBLICACIÓN EN GITHUB (CALIFORNIA HOUSING MLOPS)
echo  Lead Architect: Guillén Concepción
echo  Repositorio Remoto: https://github.com/GuillenConcepcion/california-housing-mlops.git
echo ==============================================================================
echo.

cd /d "%~dp0"

:: 1. Inicializar Git si no existe
if not exist ".git" (
    echo [*] Inicializando repositorio Git en rama 'main'...
    git init -b main
) else (
    echo [*] Repositorio Git ya detectado.
)

:: 2. Configurar Autor localmente
echo [*] Configurando autor para este repositorio...
git config user.name "Guillén Concepción"
git config user.email "guillenconcepcion@gmail.com"

:: 3. Añadir remote origin
git remote remove origin 2>nul
echo [*] Vinculando repositorio remoto: https://github.com/GuillenConcepcion/california-housing-mlops.git
git remote add origin https://github.com/GuillenConcepcion/california-housing-mlops.git

:: 4. Añadir archivos respetando .gitignore
echo [*] Agregando archivos al commit inicial...
git add .

:: 5. Commit inicial
echo [*] Creando commit inicial...
git commit -m "feat: initial commit - California Housing MLOps, Optuna, CQR & PowerPoint presentation"

:: 6. Push a GitHub
echo.
echo [*] Publicando en GitHub (rama main)...
git push -u origin main

echo.
if %ERRORLEVEL% EQU 0 (
    echo [OK] Repositorio publicado exitosamente en:
    echo      https://github.com/GuillenConcepcion/california-housing-mlops
) else (
    echo [!] Si el push solicita autenticacion o el repo remoto ya contiene un README:
    echo     1. Asegurate de haber iniciado sesion en Git / GitHub Credential Manager.
    echo     2. Si el repo remoto no estaba vacio, ejecuta: git pull origin main --rebase
    echo        y luego: git push -u origin main
)
echo.
pause
