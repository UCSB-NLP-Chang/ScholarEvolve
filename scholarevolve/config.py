"""Run configuration: environment, task backbone, research model, coding agent and search budget.

A configuration is a JSON file with five sections::

    {
      "environment":    {"name": "appworld" | "tau2", ...adapter settings...},
      "backbone":       {"model": ..., "base_url": ..., "api_key_env": ..., "model_args": {...}},
      "research_model": {"model": ..., "base_url": ..., "api_key_env": ..., "reasoning_effort": ...},
      "coding_agent":   {"backend": "codex" | "chat", "model": ..., "base_url": ..., ...},
      "search":         {...SearchSettings...}
    }

String values may reference environment variables (``$VAR`` or ``${VAR}``) and ``~``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from scholarevolve.llm import ChatModel
from scholarevolve.mutation.coding_agent import CodingSettings
from scholarevolve.search.evolve import SearchSettings
from scholarevolve.utils import load_json


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


@dataclass
class RunConfig:
    environment: dict[str, Any]
    backbone: dict[str, Any]
    research_model: ChatModel
    coding: CodingSettings
    search: SearchSettings

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        raw = _expand(load_json(path))
        missing = [key for key in ("environment", "backbone", "research_model") if key not in raw]
        if missing:
            raise ValueError(f"configuration {path} lacks section(s): {', '.join(missing)}")
        coding_fields = {item.name for item in fields(CodingSettings)}
        coding = dict(raw.get("coding_agent") or {})
        unknown = sorted(set(coding) - coding_fields)
        if unknown:
            raise ValueError(f"unknown coding_agent setting(s): {', '.join(unknown)}")
        return cls(environment=dict(raw["environment"]), backbone=dict(raw["backbone"]),
                   research_model=ChatModel.from_config(dict(raw["research_model"])),
                   coding=CodingSettings(**coding),
                   search=SearchSettings.from_config(dict(raw.get("search") or {})))
