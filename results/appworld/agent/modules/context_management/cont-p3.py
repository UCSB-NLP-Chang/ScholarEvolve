"""Query-aware, variable-rate context compression for the context_management slot.

The paper's key result for this slot is that compression must depend on the downstream query.
This implementation follows that spirit with an Adaptive QuerySelect-style policy:

* preserve the host policy;
* score candidate context blocks by task relevance plus recency;
* keep blocks in their original order;
* allocate more characters to more relevant blocks under the advertised budget.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from evolve.slots.interfaces import Budget, Evidence, Proposal, Skill, ToolSpec

_WORD_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i", "if", "in",
    "is", "it", "me", "my", "of", "on", "or", "please", "show", "that", "the", "this", "to",
    "up", "use", "what", "with", "you", "your",
}
_ELLIPSIS = "\n[... compressed ...]\n"
_MESSAGE_OVERHEAD = 32


@dataclass
class _Segment:
    source: str
    role: str
    content: str
    score: float
    order: int
    min_chars: int
    max_chars: int


def _tokenize(text: str) -> list[str]:
    return [token for token in _WORD_RE.findall(text.lower()) if token not in _STOPWORDS]


def _content(message: dict[str, Any]) -> str:
    value = message.get("content")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(value or "")


def _clip_center(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    if limit <= len(_ELLIPSIS) + 8:
        return text[:limit]
    half = max(1, (limit - len(_ELLIPSIS)) // 2)
    return text[:half] + _ELLIPSIS + text[-half:]


def _query_focused_excerpt(text: str, query_terms: list[str], limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    lowered = text.lower()
    positions = [lowered.find(term) for term in query_terms if term and lowered.find(term) >= 0]
    if not positions:
        return _clip_center(text, limit)
    center = min(positions)
    window = max(1, (limit - len(_ELLIPSIS)) // 2)
    start = max(0, center - window // 2)
    end = min(len(text), start + limit - len(_ELLIPSIS))
    excerpt = text[start:end]
    if start <= 0 and end >= len(text):
        return excerpt
    if start <= 0:
        return excerpt + _ELLIPSIS.strip()
    if end >= len(text):
        return _ELLIPSIS.strip() + excerpt
    return _ELLIPSIS + excerpt + _ELLIPSIS


def _compress(text: str, limit: int, query_terms: list[str]) -> str:
    text = str(text or "").strip()
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    focused = _query_focused_excerpt(text, query_terms, limit)
    return focused[:limit]


def _render_tools(tools: list[ToolSpec]) -> str:
    extra = [tool for tool in tools if tool.name != "python_code"]
    if not extra:
        return ""
    lines = ["Available actions besides a plain Python block:"]
    for tool in extra:
        if tool.parameters:
            params = ", ".join(f"{key}: {value}" for key, value in tool.parameters.items())
            lines.append(f"- {tool.name}({params}): {tool.description}")
        else:
            lines.append(f"- {tool.name}: {tool.description}")
    return "\n".join(lines)


def _render_evidence_item(item: Evidence) -> str:
    label = item.source or item.task_id or "memory"
    return f"[{label}]\n{item.content}"


def _render_skill_item(item: Skill) -> str:
    return f"[{item.name}]\n{item.body}"


class AdaptiveQuerySelect:
    def __init__(
        self,
        *,
        min_history_chars: int = 160,
        min_memory_chars: int = 220,
        recent_turns: int = 4,
        history_recency_weight: float = 0.35,
        memory_score_weight: float = 0.15,
    ):
        self.min_history_chars = int(min_history_chars)
        self.min_memory_chars = int(min_memory_chars)
        self.recent_turns = max(0, int(recent_turns))
        self.history_recency_weight = float(history_recency_weight)
        self.memory_score_weight = float(memory_score_weight)

    def _score_text(self, text: str, query_terms: list[str]) -> float:
        if not text:
            return 0.0
        if not query_terms:
            return 0.0
        text_terms = set(_tokenize(text))
        if not text_terms:
            return 0.0
        overlap = sum(1 for term in query_terms if term in text_terms)
        density = overlap / max(1, len(query_terms))
        return density

    def _history_segments(
        self,
        history: list[dict[str, Any]],
        query_terms: list[str],
    ) -> list[_Segment]:
        total = len(history)
        segments: list[_Segment] = []
        forced_recent_start = max(0, total - self.recent_turns)
        for index, turn in enumerate(history):
            content = _content(turn)
            if not content:
                continue
            relevance = self._score_text(content, query_terms)
            recency = 0.0 if total <= 1 else index / (total - 1)
            if index >= forced_recent_start:
                recency = max(recency, 0.85)
            score = relevance + self.history_recency_weight * recency
            segments.append(
                _Segment(
                    source="history",
                    role=str(turn.get("role") or "user"),
                    content=content,
                    score=score,
                    order=index,
                    min_chars=min(self.min_history_chars, len(content)),
                    max_chars=len(content),
                )
            )
        return segments

    def _memory_segments(
        self,
        evidence: list[Evidence],
        skills: list[Skill],
        query_terms: list[str],
        tools: list[ToolSpec],
        start_order: int,
    ) -> list[_Segment]:
        segments: list[_Segment] = []
        tool_text = _render_tools(tools)
        if tool_text:
            score = max(0.25, self._score_text(tool_text, query_terms))
            segments.append(
                _Segment(
                    source="tools",
                    role="user",
                    content=tool_text,
                    score=score,
                    order=start_order,
                    min_chars=min(240, len(tool_text)),
                    max_chars=len(tool_text),
                )
            )
            start_order += 1
        if skills:
            body = "Procedures that may help:\n\n" + "\n\n".join(
                _render_skill_item(item) for item in skills
            )
            score = max(
                (self._score_text(body, query_terms) + max((item.score for item in skills), default=0.0)
                 * self.memory_score_weight),
                0.2,
            )
            segments.append(
                _Segment(
                    source="skills",
                    role="user",
                    content=body,
                    score=score,
                    order=start_order,
                    min_chars=min(self.min_memory_chars, len(body)),
                    max_chars=len(body),
                )
            )
            start_order += 1
        if evidence:
            body = "Relevant prior experience:\n\n" + "\n\n".join(
                _render_evidence_item(item) for item in evidence
            )
            score = max(
                (
                    self._score_text(body, query_terms)
                    + max((item.score for item in evidence), default=0.0) * self.memory_score_weight
                ),
                0.2,
            )
            segments.append(
                _Segment(
                    source="evidence",
                    role="user",
                    content=body,
                    score=score,
                    order=start_order,
                    min_chars=min(self.min_memory_chars, len(body)),
                    max_chars=len(body),
                )
            )
        return segments

    def _allocate(self, segments: list[_Segment], available_chars: int) -> dict[int, int]:
        if available_chars <= 0 or not segments:
            return {}
        ranked = sorted(segments, key=lambda item: (-item.score, item.order))
        chosen: list[_Segment] = []
        spent = 0
        for segment in ranked:
            need = segment.min_chars + _MESSAGE_OVERHEAD
            if chosen and spent + need > available_chars:
                continue
            if not chosen and need > available_chars:
                chosen.append(
                    _Segment(
                        source=segment.source,
                        role=segment.role,
                        content=segment.content,
                        score=segment.score,
                        order=segment.order,
                        min_chars=max(1, available_chars - _MESSAGE_OVERHEAD),
                        max_chars=segment.max_chars,
                    )
                )
                spent = available_chars
                break
            chosen.append(segment)
            spent += need
        if not chosen:
            return {}
        remaining = max(0, available_chars - spent)
        weights = [max(0.05, item.score) for item in chosen]
        total_weight = sum(weights) or float(len(chosen))
        allocation = {item.order: item.min_chars for item in chosen}
        expandable = {item.order: max(0, item.max_chars - item.min_chars) for item in chosen}
        for item, weight in zip(chosen, weights):
            extra = int(math.floor(remaining * weight / total_weight))
            extra = min(extra, expandable[item.order])
            allocation[item.order] += extra
            remaining -= extra
            total_weight -= weight
        if remaining > 0:
            for item in sorted(chosen, key=lambda entry: (-entry.score, entry.order)):
                if remaining <= 0:
                    break
                room = expandable[item.order] - (allocation[item.order] - item.min_chars)
                if room <= 0:
                    continue
                bump = min(room, remaining)
                allocation[item.order] += bump
                remaining -= bump
        return allocation

    def compose(
        self,
        policy: list[dict[str, Any]],
        history: list[dict[str, Any]],
        evidence: list[Evidence],
        skills: list[Skill],
        tools: list[ToolSpec],
        budget: Budget,
        task: dict[str, Any],
    ) -> list[dict[str, Any]]:
        query_terms = _tokenize(str((task or {}).get("instruction") or ""))
        messages = [
            {"role": str(message.get("role") or "user"), "content": _content(message)}
            for message in policy
        ]
        policy_chars = sum(len(message["content"]) + _MESSAGE_OVERHEAD for message in messages)
        limit = budget.max_chars if budget.max_chars is not None else None
        if limit is None or policy_chars >= limit:
            return messages + [
                {"role": str(message.get("role") or "user"), "content": _content(message)}
                for message in history
            ]

        segments = self._memory_segments(
            evidence=evidence,
            skills=skills,
            query_terms=query_terms,
            tools=tools,
            start_order=-10,
        ) + self._history_segments(history=history, query_terms=query_terms)
        allocation = self._allocate(segments, max(0, limit - policy_chars))
        for segment in sorted(segments, key=lambda item: item.order):
            allowed = allocation.get(segment.order, 0)
            if allowed <= 0:
                continue
            content = _compress(segment.content, allowed, query_terms)
            if not content:
                continue
            messages.append({"role": segment.role, "content": content})
        return messages


def create_slot(*, context: Any, **params: Any) -> AdaptiveQuerySelect:
    del context
    return AdaptiveQuerySelect(**params)
