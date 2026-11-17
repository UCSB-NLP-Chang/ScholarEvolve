"""Memories slot: task-specific experience retrieved by similarity.

Implements a lightweight version of the paper's memory lifecycle:
- ingest solved episodes via ``write`` / ``fit``
- extract durable summaries from trajectories
- consolidate them into a frozen JSON store via ``save``
- retrieve task-scoped memories by lexical similarity in ``recall``
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from evolve.slots.interfaces import Evidence

_WORD_RE = re.compile(r"[a-z0-9]+")
_CALL_RE = re.compile(r"apis\.([a-z_]+)\.([a-z_]+)\s*\(", re.IGNORECASE)
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "do", "for", "from", "i", "in", "is", "it", "me", "my",
    "of", "on", "or", "please", "that", "the", "this", "to", "with", "your",
})


def _tokenize(text: str) -> set[str]:
    return {
        token for token in _WORD_RE.findall((text or "").lower())
        if token not in _STOPWORDS
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class OracleStyleMemories:
    def __init__(
        self,
        *,
        artifact_path: str | Path | None = None,
        top_k: int = 3,
        min_similarity: float = 0.08,
        max_summary_steps: int = 8,
        max_chars: int = 1400,
    ) -> None:
        self.artifact_path = Path(artifact_path) if artifact_path else None
        self.top_k = int(top_k)
        self.min_similarity = float(min_similarity)
        self.max_summary_steps = int(max_summary_steps)
        self.max_chars = int(max_chars)
        self.active_memories: list[dict[str, Any]] = []
        self.passive_memories: list[dict[str, Any]] = []
        self._index: list[tuple[set[str], dict[str, Any]]] = []
        if self.artifact_path and self.artifact_path.is_file():
            payload = json.loads(self.artifact_path.read_text(encoding="utf-8"))
            self.passive_memories = list(payload.get("entries") or [])
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        combined = [*self.passive_memories, *self.active_memories]
        self._index = [
            (_tokenize(str(entry.get("instruction") or "")), entry)
            for entry in combined
        ]

    def _scope_key(self, instruction: str) -> str:
        tokens = sorted(_tokenize(instruction))
        if not tokens:
            return "global"
        return "|".join(tokens[:12])

    def _summarize_episode(self, episode_result: Any) -> str:
        instruction = str((episode_result.metadata or {}).get("instruction") or "").strip()
        lines: list[str] = []
        if instruction:
            lines.append(f"Task: {instruction}")
        for step in list(episode_result.trajectory or ())[:self.max_summary_steps]:
            code = str(step.get("code") or "").strip()
            output = str(step.get("output") or "").strip()
            if not code and not output:
                continue
            api_calls = [f"apis.{app}.{api}()" for app, api in _CALL_RE.findall(code)]
            action = ", ".join(api_calls) if api_calls else code.splitlines()[0][:120]
            detail = f"Step {step.get('step')}: {action}"
            if output:
                detail += f" -> {output.splitlines()[0][:120]}"
            lines.append(detail)
        if getattr(episode_result, "completed", False):
            lines.append("Outcome: completed successfully.")
        elif float(getattr(episode_result, "reward", 0.0) or 0.0) > 0.0:
            lines.append("Outcome: partially successful.")
        else:
            lines.append("Outcome: unsuccessful.")
        return "\n".join(lines)[:self.max_chars]

    def _make_entry(self, episode_result: Any) -> dict[str, Any]:
        instruction = str((episode_result.metadata or {}).get("instruction") or "").strip()
        return {
            "task_id": str(getattr(episode_result, "task_id", "")),
            "instruction": instruction,
            "scope": self._scope_key(instruction),
            "reward": float(getattr(episode_result, "reward", 0.0) or 0.0),
            "steps": int(getattr(episode_result, "steps", 0) or 0),
            "completed": bool(getattr(episode_result, "completed", False)),
            "summary": self._summarize_episode(episode_result),
        }

    def recall(self, task: Any, budget: Any | None = None) -> list[Evidence]:
        instruction = str((task or {}).get("instruction") or "")
        wanted = _tokenize(instruction)
        if not wanted or not self._index:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for tokens, entry in self._index:
            similarity = _jaccard(wanted, tokens)
            if similarity < self.min_similarity:
                continue
            reward = float(entry.get("reward", 0.0) or 0.0)
            completed_bonus = 0.05 if entry.get("completed") else 0.0
            score = similarity + min(0.15, max(0.0, reward) * 0.05) + completed_bonus
            scored.append((score, entry))
        scored.sort(key=lambda item: (-item[0], str(item[1].get("task_id") or "")))
        char_cap = self.max_chars
        if budget is not None and getattr(budget, "max_steps", 0):
            # Keep memory payload compact late in the episode.
            char_cap = max(300, min(self.max_chars, 900 - (int(budget.step_index) * 10)))
        evidence: list[Evidence] = []
        for score, entry in scored[:self.top_k]:
            evidence.append(
                Evidence(
                    task_id=str(entry.get("task_id") or ""),
                    content=str(entry.get("summary") or "")[:char_cap],
                    score=round(score, 6),
                    source="oracle_memory_store",
                    metadata={
                        "scope": entry.get("scope"),
                        "reward": entry.get("reward"),
                        "steps": entry.get("steps"),
                        "completed": entry.get("completed"),
                    },
                )
            )
        return evidence

    def write(self, episode_result: Any) -> None:
        entry = self._make_entry(episode_result)
        if not entry["instruction"] or not entry["summary"]:
            return
        self.active_memories.append(entry)
        self._rebuild_index()

    def fit(self, episodes: Any) -> None:
        self.active_memories = []
        self.passive_memories = []
        for episode in episodes:
            if float(getattr(episode, "reward", 0.0) or 0.0) <= 0.0:
                continue
            self.write(episode_result=episode)

    def save(self, path: str) -> dict[str, Any]:
        entries = list(self.active_memories)
        payload = {
            "schema_version": "oracle_style_memory.v1",
            "entries": entries,
            "memory_lifecycle": [
                "ingestion",
                "extraction",
                "consolidation",
                "retrieval",
            ],
            "n_entries": len(entries),
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.passive_memories = entries
        self.active_memories = []
        self._rebuild_index()
        return payload


def create_slot(*, context: Any, **params: Any) -> OracleStyleMemories:
    artifact_path = (context.artifacts or {}).get("memories")
    if artifact_path is None:
        artifact_path = (context.artifacts or {}).get("memory_pool")
    return OracleStyleMemories(artifact_path=artifact_path, **params)
