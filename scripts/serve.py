import sys
from pathlib import Path
import uvicorn

# Asegurar que la raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings

if __name__ == "__main__":
    print(f"-> Iniciando servidor FastAPI en {settings.API_HOST}:{settings.API_PORT}...")
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )
