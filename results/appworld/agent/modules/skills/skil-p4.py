from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from evolve.slots.interfaces import Skill


API_CALL_RE = re.compile(r"apis\.([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*\(")
TOKEN_RE = re.compile(r"[a-z0-9_]+")
STOP_WORDS = {
    "a", "an", "and", "are", "for", "from", "how", "i", "in", "into", "is", "it",
    "me", "my", "of", "on", "or", "please", "show", "task", "that", "the", "to",
    "up", "use", "with", "you",
}


def _tokenize(text: Any) -> list[str]:
    words = TOKEN_RE.findall(str(text or "").lower())
    return [word for word in words if len(word) > 1 and word not in STOP_WORDS]


def _sequence_from_trajectory(trajectory: Any) -> list[str]:
    sequence: list[str] = []
    for step in trajectory or ():
        code = str((step or {}).get("code") or "")
        for app, api in API_CALL_RE.findall(code):
            sequence.append(f"{app}.{api}")
    return sequence


def _best_trigger_words(counter: Counter[str], support: int, limit: int) -> list[str]:
    minimum_hits = max(1, support // 2)
    return [
        word for word, hits in counter.most_common(limit)
        if hits >= minimum_hits
    ]


def _skill_name(calls: tuple[str, ...]) -> str:
    if len(calls) == 1:
        return f"sap::{calls[0]}"
    return "sap::" + " -> ".join(calls)


def _skill_body(calls: tuple[str, ...], triggers: list[str], support: int) -> str:
    typed_steps = "\n".join(
        f"{index + 1}. emit Python that calls `apis.{call}(...)`"
        for index, call in enumerate(calls)
    )
    examples = "\n".join(
        f"- `apis.{call}(...)`"
        for call in calls
    )
    trigger_text = ", ".join(triggers) if triggers else "any overlapping task wording"
    return (
        "typed contract:\n"
        "- input.task_instruction: str\n"
        "- input.current_observation: str\n"
        "- output.python_action: str\n"
        f"- applies_when: task wording overlaps with {trigger_text}\n"
        f"- confidence_signal: seen in {support} successful training episodes\n"
        "algorithm:\n"
        f"{typed_steps}\n"
        "action template:\n"
        f"{examples}"
    )


def _overlap_score(task_tokens: set[str], triggers: list[str]) -> float:
    if not triggers:
        return 0.01
    trigger_set = set(triggers)
    intersection = len(task_tokens & trigger_set)
    if not intersection:
        return 0.0
    return intersection / float(len(trigger_set))


class SkillAsPseudocode:
    def __init__(
        self,
        *,
        artifact_path: Path | None,
        min_support: int = 2,
        max_ngram: int = 3,
        top_k: int = 8,
        max_chars: int = 1400,
        min_match_score: float = 0.0,
    ):
        self.artifact_path = artifact_path
        self.min_support = max(1, int(min_support))
        self.max_ngram = max(1, int(max_ngram))
        self.top_k = max(1, int(top_k))
        self.max_chars = max(200, int(max_chars))
        self.min_match_score = float(min_match_score)
        self.induced_from = 0
        self.skills: list[dict[str, Any]] = []
        if artifact_path and artifact_path.is_file():
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.induced_from = int(payload.get("induced_from_episodes") or 0)
            loaded = payload.get("skills") or []
            self.skills = [dict(item) for item in loaded if isinstance(item, dict)]

    def fit(self, episodes: Any) -> None:
        solved = [
            episode for episode in episodes or ()
            if bool(getattr(episode, "completed", False)) or float(getattr(episode, "reward", 0.0)) > 0.0
        ]
        self.induced_from = len(solved)
        pattern_support: Counter[tuple[str, ...]] = Counter()
        pattern_triggers: dict[tuple[str, ...], Counter[str]] = {}

        for episode in solved:
            sequence = _sequence_from_trajectory(getattr(episode, "trajectory", ()))
            if not sequence:
                continue
            instruction = ""
            metadata = getattr(episode, "metadata", {}) or {}
            if isinstance(metadata, dict):
                instruction = str(metadata.get("instruction") or "")
            tokens = _tokenize(instruction)
            seen_here: set[tuple[str, ...]] = set()
            for width in range(1, min(self.max_ngram, len(sequence)) + 1):
                for start in range(0, len(sequence) - width + 1):
                    gram = tuple(sequence[start:start + width])
                    if gram in seen_here:
                        continue
                    seen_here.add(gram)
                    pattern_support[gram] += 1
                    bucket = pattern_triggers.setdefault(gram, Counter())
                    bucket.update(tokens)

        ranked = sorted(
            pattern_support.items(),
            key=lambda item: (-item[1], -len(item[0]), item[0]),
        )
        self.skills = []
        for calls, support in ranked:
            if support < self.min_support:
                continue
            triggers = _best_trigger_words(pattern_triggers.get(calls, Counter()), support, 8)
            self.skills.append(
                {
                    "name": _skill_name(calls),
                    "calls": list(calls),
                    "support": support,
                    "triggers": triggers,
                    "form": "pseudocode",
                    "body": _skill_body(calls, triggers, support),
                }
            )
            if len(self.skills) >= self.top_k:
                break

    def save(self, path: str) -> None:
        payload = {
            "schema_version": "ripe.skills.sap.v1",
            "source_split": "train",
            "method": "skill_as_pseudocode",
            "induced_from_episodes": self.induced_from,
            "min_support": self.min_support,
            "max_ngram": self.max_ngram,
            "top_k": self.top_k,
            "skills": self.skills,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def match(self, task: Any) -> Skill | None:
        if not self.skills:
            return None
        instruction = ""
        if isinstance(task, dict):
            instruction = str(task.get("instruction") or "")
        task_tokens = set(_tokenize(instruction))
        best_skill: dict[str, Any] | None = None
        best_score = -1.0
        for skill in self.skills:
            triggers = [str(item) for item in skill.get("triggers") or ()]
            overlap = _overlap_score(task_tokens, triggers)
            support = float(skill.get("support") or 0.0)
            score = overlap + min(0.25, support / 100.0)
            if score > best_score:
                best_skill = skill
                best_score = score
        if best_skill is None or best_score < self.min_match_score:
            return None
        return Skill(
            name=str(best_skill.get("name") or "sap::procedure"),
            body=str(best_skill.get("body") or "")[:self.max_chars],
            form=str(best_skill.get("form") or "pseudocode"),
            score=round(best_score, 6),
            provenance={
                "method": "skill_as_pseudocode",
                "support": int(best_skill.get("support") or 0),
                "calls": list(best_skill.get("calls") or ()),
            },
        )


def create_slot(*, context: Any, **params: Any) -> SkillAsPseudocode:
    artifacts = getattr(context, "artifacts", {}) or {}
    artifact = (
        params.pop("artifact_path", None)
        or artifacts.get("skills")
        or artifacts.get("skill_set")
        or artifacts.get("skill_library")
    )
    return SkillAsPseudocode(
        artifact_path=Path(artifact) if artifact else None,
        **params,
    )
