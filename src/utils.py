"""Utility helpers shared by the training scripts."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"
FIGURES_DIR = REPO_ROOT / "figures"


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_file(path: Path, help_message: str | None = None) -> Path:
    """Return an existing file path or raise a clear FileNotFoundError."""
    if path.exists() and path.is_file():
        return path

    message = f"Required file not found: {path}"
    if help_message:
        message = f"{message}\n{help_message}"
    raise FileNotFoundError(message)


def read_json(path: Path) -> dict[str, Any]:
    """Load a JSON file using UTF-8 encoding."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_zip_if_needed(zip_path: Path, extract_dir: Path) -> Path:
    """Extract a zip archive into extract_dir only when the directory is absent."""
    if extract_dir.exists():
        return extract_dir

    require_file(zip_path)
    ensure_directory(extract_dir)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    return extract_dir
