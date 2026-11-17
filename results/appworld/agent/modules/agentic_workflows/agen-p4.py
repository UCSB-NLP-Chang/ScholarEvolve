"""TMAS-inspired agentic workflow slot.

This module keeps the slot boundary intact: it only controls intra-step reasoning flow.
It runs several specialist rollouts, aggregates them through a verifier, and carries
forward a tiny hierarchical memory built from prior step outputs:

* experience bank: concrete observations from earlier actions and outputs
* guideline bank: high-level strategy reminders to reduce redundant retries
"""

from __future__ import annotations

import re
from typing import Any

from evolve.slots.interfaces import Proposal


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_FAILURE_RE = re.compile(
    r"traceback|exception|\berror\b|failed|invalid|not found|denied|unsupported",
    re.IGNORECASE,
)


def _extract_code(text: str) -> str:
    match = _CODE_BLOCK_RE.search(text or "")
    if match:
        return match.group(1).strip()
    stripped = (text or "").strip()
    if not stripped:
        return ""
    lines = stripped.splitlines()
    if lines and lines[0].startswith("python"):
        return "\n".join(lines[1:]).strip()
    return stripped


def _usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return dict(usage)
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    result: dict[str, Any] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, name, None)
        if value is not None:
            result[name] = value
    return result


def _merge_usage(total: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(total)
    for key, value in update.items():
        if isinstance(value, (int, float)) and isinstance(merged.get(key), (int, float)):
            merged[key] = merged[key] + value
        elif isinstance(value, (int, float)) and key not in merged:
            merged[key] = value
        elif key not in merged:
            merged[key] = value
    return merged


class TMASWorkflow:
    def __init__(
        self,
        *,
        model_name: str,
        model_args: dict[str, Any],
        temperature: float | None,
        num_agents: int = 3,
    ) -> None:
        self.model_name = model_name
        self.model_args = dict(model_args or {})
        self.temperature = temperature
        self.num_agents = max(2, int(num_agents))
        self.experience_bank: list[str] = []
        self.guideline_bank: list[str] = []

    def _complete(self, messages: list[dict[str, Any]], client: Any) -> tuple[str, dict[str, Any]]:
        kwargs = {"model": self.model_name, "messages": messages, **self.model_args}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        response = client.chat.completions.create(**kwargs)
        choices = getattr(response, "choices", None) or []
        if not choices:
            return "", _usage_to_dict(getattr(response, "usage", None))
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        return content or "", _usage_to_dict(getattr(response, "usage", None))

    def _remember(self, state: Any) -> None:
        if state is None:
            return
        last_action = str(getattr(state, "last_action", "") or "").strip()
        last_output = str(getattr(state, "last_output", "") or "").strip()
        if last_action:
            note = "Prior action:\n" + last_action[:500]
            if note not in self.experience_bank:
                self.experience_bank.append(note)
        if last_output:
            prefix = "Failure signal" if _FAILURE_RE.search(last_output) else "Observed result"
            note = prefix + ":\n" + last_output[:500]
            if note not in self.experience_bank:
                self.experience_bank.append(note)
            if _FAILURE_RE.search(last_output):
                guideline = "Avoid repeating the previous failing API pattern unchanged."
            else:
                guideline = "Build on successful observations instead of restarting blindly."
            if guideline not in self.guideline_bank:
                self.guideline_bank.append(guideline)
        if len(self.experience_bank) > 6:
            self.experience_bank = self.experience_bank[-6:]
        if len(self.guideline_bank) > 4:
            self.guideline_bank = self.guideline_bank[-4:]

    def _memory_text(self) -> str:
        parts: list[str] = []
        if self.guideline_bank:
            parts.append("Guideline bank:\n" + "\n".join(f"- {item}" for item in self.guideline_bank))
        if self.experience_bank:
            parts.append("Experience bank:\n" + "\n\n".join(self.experience_bank))
        return "\n\n".join(parts).strip()

    def decide(self, messages: list[dict[str, Any]], client: Any, state: Any = None) -> Proposal:
        self._remember(state)

        personas = (
            "Planner agent: propose a direct low-risk next action that gathers the missing fact or executes the task.",
            "Solver agent: propose the strongest task-completing next action using available APIs.",
            "Critic agent: propose a next action that avoids redundant work and fixes likely failure modes.",
        )
        memory = self._memory_text()
        candidate_texts: list[str] = []
        usage_total: dict[str, Any] = {}

        for persona in personas[:self.num_agents]:
            prompt = (
                persona
                + "\nReturn exactly one executable Python code block for the next environment action."
            )
            if memory:
                prompt += "\n\nUse this shared memory to coordinate with prior attempts:\n" + memory
            text, usage = self._complete([*messages, {"role": "user", "content": prompt}], client)
            candidate_texts.append(text)
            usage_total = _merge_usage(usage_total, usage)

        synthesis = [
            "Verifier agent: choose the best candidate action.",
            "Selection criteria: executable now, non-redundant, and most likely to advance the task.",
        ]
        if memory:
            synthesis.append("Shared memory:\n" + memory)
        for index, candidate in enumerate(candidate_texts, start=1):
            synthesis.append(f"Candidate {index}:\n{candidate}")
        synthesis.append("Return only the chosen next action as one Python code block.")

        final_text, usage = self._complete(
            [*messages, {"role": "user", "content": "\n\n".join(synthesis)}],
            client,
        )
        usage_total = _merge_usage(usage_total, usage)

        action = _extract_code(final_text)
        if not action:
            for candidate in candidate_texts:
                action = _extract_code(candidate)
                if action:
                    final_text = candidate
                    break

        return Proposal(
            text=final_text,
            action=action,
            usage=usage_total,
            metadata={
                "workflow": "tmas",
                "agents_used": min(self.num_agents, len(personas)),
                "experience_items": len(self.experience_bank),
                "guideline_items": len(self.guideline_bank),
            },
        )


def create_slot(*, context: Any, **params: Any) -> TMASWorkflow:
    return TMASWorkflow(
        model_name=context.model_name,
        model_args=context.model_args,
        temperature=context.temperature,
        **params,
    )
