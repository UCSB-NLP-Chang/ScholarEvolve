"""Shared helpers: JSON files, names, module loading, and JSON extraction from model text."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_write_json(path: str | Path, value: Any) -> Path:
    """Write JSON through a temporary file so a reader never sees a partial document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str),
                             encoding="utf-8")
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


def safe_name(value: str, *, fallback: str = "run") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip()).strip("-._")
    return cleaned or fallback


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def load_module_from_path(path: str | Path, *, prefix: str) -> ModuleType:
    """Import a Python file under a name derived from its path and content."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256(str(path).encode("utf-8") + path.read_bytes()).hexdigest()[:16]
    module_name = f"{prefix}_{safe_name(path.stem)}_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    if "```" in text:
        text = max(text.split("```"), key=len)
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    return text


def extract_json_object(text: str) -> dict[str, Any]:
    """The JSON object in a model reply, tolerating code fences and surrounding prose."""
    body = _strip_fences(text)
    start, end = body.find("{"), body.rfind("}")
    if not 0 <= start < end:
        return {}
    try:
        value = json.loads(body[start:end + 1])
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def extract_json_array(text: str) -> list[Any]:
    """The JSON array in a model reply, tolerating code fences and surrounding prose."""
    body = _strip_fences(text)
    match = re.search(r"\[.*\]", body, re.DOTALL)
    if not match:
        return []
    try:
        value = json.loads(match.group(0))
    except ValueError:
        return []
    return value if isinstance(value, list) else []
