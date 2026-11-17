from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from evolve.slots.interfaces import Evidence


_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "the a an my i me we our you your of to in on for and or is are was were with from "
    "that this it be as at by if then than".split()
)
_API_CALL = re.compile(r"apis\.([a-z_]+)\.([a-z_]+)\s*\(", re.IGNORECASE)


def _tokens(text: str) -> set[str]:
    return {word for word in _WORD.findall((text or "").lower()) if word not in _STOP}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _step_actions(step: Any) -> list[str]:
    calls = step.get("tool_calls") if isinstance(step, dict) else None
    if calls:
        names: list[str] = []
        for call in calls:
            if isinstance(call, dict):
                name = str(call.get("name") or "")
            else:
                name = str(getattr(call, "name", "") or "")
            if name:
                names.append(name)
        if names:
            return names
    code = str((step.get("code") if isinstance(step, dict) else "") or "")
    return [f"{app}.{api}" for app, api in _API_CALL.findall(code)]


class EpisodicMemorySlot:
    def __init__(
        self,
        *,
        artifact_path: Path | None = None,
        top_k: int = 3,
        min_similarity: float = 0.08,
        max_entries: int = 128,
        max_chars: int = 700,
        seed: int = 0,
    ) -> None:
        self.artifact_path = artifact_path
        self.top_k = int(top_k)
        self.min_similarity = float(min_similarity)
        self.max_entries = max(1, int(max_entries))
        self.max_chars = max(120, int(max_chars))
        self.seed = int(seed)
        self.entries: list[dict[str, Any]] = []
        self._seen = 0
        self._rng = random.Random(self.seed)
        if artifact_path and artifact_path.is_file():
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.entries = list(payload.get("entries") or [])
            self._seen = max(int(payload.get("seen") or 0), len(self.entries))
        self._reindex()

    def _reindex(self) -> None:
        self._index = [(_tokens(str(entry.get("instruction") or "")), entry) for entry in self.entries]

    def recall(self, task: Any) -> list[Evidence]:
        instruction = ""
        if isinstance(task, dict):
            instruction = str(task.get("instruction") or "")
        wanted = _tokens(instruction)
        scored: list[tuple[float, dict[str, Any]]] = []
        for stored_tokens, entry in self._index:
            score = _jaccard(wanted, stored_tokens)
            if score >= self.min_similarity:
                scored.append((score, entry))
        scored.sort(key=lambda item: (-item[0], -float(item[1].get("reward") or 0.0), str(item[1].get("task_id") or "")))
        return [
            Evidence(
                task_id=str(entry.get("task_id") or ""),
                content=str(entry.get("summary") or "")[: self.max_chars],
                score=round(score, 6),
                source=f"episode {entry.get('task_id')}",
                metadata={
                    "reward": entry.get("reward"),
                    "steps": entry.get("steps"),
                },
            )
            for score, entry in scored[: self.top_k]
            if str(entry.get("summary") or "").strip()
        ]

    def write(self, episode_result: Any) -> None:
        entry = self._make_entry(episode_result)
        self._seen += 1
        if not entry["instruction"] or not entry["summary"]:
            return
        if len(self.entries) < self.max_entries:
            self.entries.append(entry)
            self._reindex()
            return
        keep_index = self._rng.randrange(self._seen)
        if keep_index < self.max_entries:
            self.entries[keep_index] = entry
            self._reindex()

    def _make_entry(self, episode_result: Any) -> dict[str, Any]:
        instruction = ""
        metadata = getattr(episode_result, "metadata", {}) or {}
        if isinstance(metadata, dict):
            instruction = str(metadata.get("instruction") or "")
        return {
            "task_id": str(getattr(episode_result, "task_id", "") or ""),
            "instruction": instruction,
            "reward": float(getattr(episode_result, "reward", 0.0) or 0.0),
            "steps": int(getattr(episode_result, "steps", 0) or 0),
            "summary": self._summarize_episode(episode_result),
        }

    def _summarize_episode(self, episode_result: Any) -> str:
        lines: list[str] = []
        for step in list(getattr(episode_result, "trajectory", ()) or ())[:12]:
            step_no = ""
            if isinstance(step, dict) and step.get("step") is not None:
                step_no = f"{step.get('step')}. "
            actions = _step_actions(step)
            if actions:
                lines.append(f"{step_no}{', '.join(name + '()' for name in actions)}")
                continue
            text = ""
            if isinstance(step, dict):
                text = str(step.get("raw_text") or step.get("output") or step.get("code") or "").strip()
            if text:
                lines.append(f"{step_no}{text.splitlines()[0][:120]}")
        return "\n".join(lines)[: self.max_chars]

    def fit(self, episodes: Any) -> None:
        self.entries = []
        self._seen = 0
        self._rng = random.Random(self.seed)
        self._reindex()
        for episode in episodes:
            self.write(episode_result=episode)

    def save(self, path: str) -> None:
        payload = {
            "schema_version": "episodic-memory-slot.v1",
            "entries": self.entries,
            "seen": self._seen,
            "top_k": self.top_k,
            "min_similarity": self.min_similarity,
            "max_entries": self.max_entries,
            "max_chars": self.max_chars,
            "seed": self.seed,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def create_slot(*, context: Any, **params: Any) -> EpisodicMemorySlot:
    artifact_path = None
    if getattr(context, "artifacts", None):
        artifact_path = context.artifacts.get("memories") or context.artifacts.get("memory_pool")
    return EpisodicMemorySlot(
        artifact_path=Path(artifact_path) if artifact_path else None,
        **params,
    )
