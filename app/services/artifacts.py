# Helper that checks a required file exists before we load it
from pathlib import Path  # Path type

from app.services.errors import ModelNotFoundError  # our error type


def require_file(path: Path) -> Path:
    if not path.exists():  # file is missing
        raise ModelNotFoundError(f"File not found: {path.name}. Follow GUIDE.md to create it.")  # clear message
    return path  # file exists, give the path back
