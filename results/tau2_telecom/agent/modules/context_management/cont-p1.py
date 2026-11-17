from __future__ import annotations

from typing import Any

from evolve.slots.interfaces import Budget, Evidence, Skill, ToolSpec


def _clean(text: Any) -> str:
    return " ".join(str(text or "").split())


def _shorten(text: Any, limit: int) -> str:
    cleaned = _clean(text)
    if limit <= 0 or len(cleaned) <= limit:
        return cleaned
    if limit <= 3:
        return cleaned[:limit]
    return cleaned[: limit - 3] + "..."


def _message_size(message: dict[str, Any]) -> int:
    size = len(str(message.get("role") or "")) + len(str(message.get("content") or ""))
    for call in message.get("tool_calls") or ():
        if isinstance(call, dict):
            function = call.get("function") or {}
            size += len(str(call.get("id") or ""))
            size += len(str(function.get("name") or ""))
            size += len(str(function.get("arguments") or ""))
        else:
            function = getattr(call, "function", None)
            size += len(str(getattr(call, "id", "") or ""))
            size += len(str(getattr(function, "name", "") or ""))
            size += len(str(getattr(function, "arguments", "") or ""))
    return size


def _unit_size(unit: list[dict[str, Any]]) -> int:
    return sum(_message_size(message) for message in unit)


def _history_units(history: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    units: list[list[dict[str, Any]]] = []
    index = 0
    while index < len(history):
        message = history[index]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            unit = [message]
            index += 1
            while index < len(history) and history[index].get("role") == "tool":
                unit.append(history[index])
                index += 1
            units.append(unit)
            continue
        units.append([message])
        index += 1
    return units


def _tool_names(message: dict[str, Any]) -> str:
    names: list[str] = []
    for call in message.get("tool_calls") or ():
        if isinstance(call, dict):
            function = call.get("function") or {}
            name = str(function.get("name") or "")
        else:
            function = getattr(call, "function", None)
            name = str(getattr(function, "name", "") or "")
        if name:
            names.append(name)
    return ", ".join(names)


def _history_notes(history: list[dict[str, Any]], limit: int = 8) -> tuple[list[str], int]:
    notes: list[str] = []
    for message in reversed(history):
        role = str(message.get("role") or "")
        content = _clean(message.get("content") or "")
        if role == "tool" and content:
            notes.append("Verified tool feedback: " + _shorten(content, 180))
        elif role == "assistant" and message.get("tool_calls"):
            names = _tool_names(message)
            if names:
                notes.append("Assistant checked: " + _shorten(names, 120))
        elif role == "assistant" and content:
            notes.append("Assistant said: " + _shorten(content, 160))
        elif role == "user" and content:
            notes.append("User wants: " + _shorten(content, 160))
        if len(notes) >= limit:
            break
    notes.reverse()
    omitted = max(0, len(history) - len(notes))
    return notes, omitted


def _render_tools(tools: list[ToolSpec], limit: int = 12) -> str:
    if not tools:
        return ""
    lines = ["Available actions:"]
    for tool in tools[:limit]:
        params = ", ".join(sorted((tool.parameters or {}).keys()))
        signature = f"{tool.name}({params})" if params else f"{tool.name}()"
        detail = _shorten(tool.description, 120)
        lines.append(f"- {signature}: {detail}")
    extra = len(tools) - min(len(tools), limit)
    if extra:
        lines.append(f"- ... plus {extra} more action(s)")
    return "\n".join(lines)


def _render_evidence(evidence: list[Evidence], skills: list[Skill]) -> list[str]:
    lines: list[str] = []
    for skill in skills[:2]:
        body = _shorten(skill.body, 180)
        if body:
            lines.append(f"Reusable procedure [{skill.name}]: {body}")
    for item in evidence[:3]:
        content = _shorten(item.content, 180)
        if content:
            source = item.source or item.task_id
            lines.append(f"Retrieved prior [{source}]: {content}")
    return lines


def _fit_text(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return _shorten(text, limit)


class ExplicitWorkingMemory:
    def __init__(self, *, memory_chars: int = 2200, note_limit: int = 8):
        self.memory_chars = int(memory_chars)
        self.note_limit = int(note_limit)

    def _memory_block(
        self,
        *,
        history: list[dict[str, Any]],
        evidence: list[Evidence],
        skills: list[Skill],
        tools: list[ToolSpec],
        state: Any,
        budget: Budget,
    ) -> str:
        step_text = f"Step {budget.step_index}/{budget.max_steps}" if budget.max_steps else f"Step {budget.step_index}"
        lines = ["Explicit working memory", step_text]
        instruction = _clean(getattr(state, "instruction", ""))
        if instruction:
            lines.append("Task: " + _shorten(instruction, 220))
        notes, omitted = _history_notes(history, limit=self.note_limit)
        lines.extend(notes)
        lines.extend(_render_evidence(evidence, skills))
        tool_text = _render_tools(tools)
        if tool_text:
            lines.append(tool_text)
        if omitted:
            lines.append(f"Older dialogue compressed: {omitted} earlier message(s) omitted from the exact window.")
        return _fit_text("\n".join(line for line in lines if line), self.memory_chars)

    def compose(
        self,
        policy: list[dict[str, Any]],
        history: list[dict[str, Any]],
        evidence: list[Evidence],
        skills: list[Skill],
        tools: list[ToolSpec],
        budget: Budget,
        state: Any,
    ) -> list[dict[str, Any]]:
        messages = [dict(message) for message in policy]
        max_chars = budget.max_chars
        used = sum(_message_size(message) for message in messages)
        remaining = None if max_chars is None else max(0, max_chars - used)

        memory_limit = self.memory_chars if remaining is None else min(self.memory_chars, max(0, remaining // 3 or remaining))
        memory_text = self._memory_block(
            history=history, evidence=evidence, skills=skills, tools=tools, state=state, budget=budget
        )
        if memory_text and memory_limit:
            memory_text = _fit_text(memory_text, memory_limit)
            if memory_text:
                memory_message = {"role": "user", "content": memory_text}
                messages.append(memory_message)
                used += _message_size(memory_message)
                remaining = None if max_chars is None else max(0, max_chars - used)

        units = _history_units(history)
        kept: list[list[dict[str, Any]]] = []
        for unit in reversed(units):
            unit_size = _unit_size(unit)
            if remaining is not None and kept and unit_size > remaining:
                continue
            if remaining is not None and not kept and unit_size > remaining:
                trimmed = [dict(message) for message in unit]
                tail = trimmed[-1]
                trimmed[-1] = {**tail, "content": _fit_text(str(tail.get("content") or ""), max(0, remaining))}
                if _unit_size(trimmed) > max(0, remaining):
                    break
                unit = trimmed
                unit_size = _unit_size(unit)
            kept.append([dict(message) for message in unit])
            if remaining is not None:
                remaining = max(0, remaining - unit_size)

        for unit in reversed(kept):
            messages.extend(unit)
        return messages


def create_slot(*, context: Any, **params: Any) -> ExplicitWorkingMemory:
    del context
    return ExplicitWorkingMemory(**params)
