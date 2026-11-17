"""MatrixCoT-style workflow for the ``agentic_workflows`` slot."""

from __future__ import annotations

import json
import re
from typing import Any

from evolve.slots.interfaces import Budget, Proposal, ToolSpec

FAILURE = re.compile(
    r"traceback|exception|\berror\b|failed|invalid|denied|missing|not found|unable",
    re.IGNORECASE,
)

PLAN_PROMPT = """
Build an internal MatrixCoT plan for the next turn.

Return plain text with exactly these sections:
1. Normalized Facts
   - each line: F<n> | type | statement | citation
2. Plan Steps
   - each line: S<n> | objective | expected tool or reply
3. Dependency Matrix
   - rows are steps, columns reference F<n>/S<n>, values are needed/supports
4. Next Step
   - name the single best next step to execute now and why

Rules:
- This is an internal planning artifact, not the final assistant reply.
- Normalize ambiguous language into concrete entities and constraints.
- Keep it compact and actionable.
- Do not call any tools.
- Do not answer the user yet.
""".strip()

REPLAN_PROMPT = """
Review the prior MatrixCoT plan under a semantic-equivalence constraint.

Tasks:
- identify omissions, contradictions, or unsupported steps;
- rewrite and compress the dependency matrix if needed;
- keep only information that is still relevant after the latest feedback;
- end with one corrected Next Step to execute now.

Return the same four sections as before.
Do not call any tools.
Do not answer the user yet.
""".strip()

ACT_PROMPT = """
Use the internal MatrixCoT plan above to take exactly one next action.

Rules:
- If a domain tool is needed, emit the appropriate tool call.
- If the user needs information, clarification, or confirmation before any change, reply to the user instead of calling a tool.
- Do not describe the whole plan.
- Keep the response concise and grounded in the plan.
""".strip()


class MatrixPlanWorkflow:
    def __init__(
        self,
        *,
        model_name: str,
        model_args: dict[str, Any],
        temperature: float | None,
        replan_on_feedback: bool = True,
        replan_every: int = 3,
    ) -> None:
        self.model_name = model_name
        self.model_args = dict(model_args)
        self.temperature = temperature
        self.replan_on_feedback = bool(replan_on_feedback)
        self.replan_every = max(1, int(replan_every))

    def _usage_dict(self, usage: Any) -> dict[str, Any]:
        if usage is None:
            return {}
        if isinstance(usage, dict):
            return dict(usage)
        if hasattr(usage, "model_dump"):
            return dict(usage.model_dump())
        if hasattr(usage, "dict"):
            return dict(usage.dict())
        return {}

    def _merge_usage(self, total: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
        merged = dict(total)
        for key, value in extra.items():
            if isinstance(value, (int, float)) and isinstance(merged.get(key), (int, float)):
                merged[key] = merged[key] + value
            elif isinstance(value, (int, float)) and key not in merged:
                merged[key] = value
            else:
                merged[key] = value
        return merged

    def _extract_tool_calls(self, message: Any) -> list[dict[str, str]]:
        calls: list[dict[str, str]] = []
        for call in list(getattr(message, "tool_calls", None) or []):
            function = getattr(call, "function", None)
            name = getattr(function, "name", "") or ""
            arguments = getattr(function, "arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments or {})
            calls.append(
                {
                    "id": getattr(call, "id", "") or "",
                    "name": name,
                    "arguments": arguments,
                }
            )
        return calls

    def _request(
        self, messages: list[dict[str, Any]], client: Any, tools: list[ToolSpec] | None
    ) -> tuple[str, list[dict[str, str]], dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": list(messages),
            **self.model_args,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if tools is not None:
            kwargs["tools"] = list(tools)
        response = client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        raw_text = getattr(message, "content", None) or ""
        return raw_text, self._extract_tool_calls(message), self._usage_dict(
            getattr(response, "usage", None)
        )

    def _needs_replan(self, state: Any, budget: Budget | None) -> bool:
        if not self.replan_on_feedback or state is None:
            return False
        latest = "\n".join(
            part
            for part in (
                str(getattr(state, "last_action", "") or ""),
                str(getattr(state, "last_output", "") or ""),
            )
            if part
        )
        if latest and FAILURE.search(latest):
            return True
        if budget is None:
            return False
        return budget.step_index > 1 and budget.step_index % self.replan_every == 0

    def decide(
        self,
        messages: list[dict[str, Any]],
        client: Any,
        state: Any = None,
        tools: list[ToolSpec] | None = None,
        budget: Budget | None = None,
    ) -> Proposal:
        usage_total: dict[str, Any] = {}

        plan_text, _, usage = self._request(
            [*messages, {"role": "user", "content": PLAN_PROMPT}],
            client,
            [],
        )
        usage_total = self._merge_usage(usage_total, usage)

        if self._needs_replan(state, budget):
            replan_messages = list(messages)
            if plan_text.strip():
                replan_messages.append({"role": "assistant", "content": plan_text})
            replan_messages.append({"role": "user", "content": REPLAN_PROMPT})
            revised, _, usage = self._request(replan_messages, client, [])
            usage_total = self._merge_usage(usage_total, usage)
            if revised.strip():
                plan_text = revised

        acting_messages = list(messages)
        if plan_text.strip():
            acting_messages.append({"role": "assistant", "content": plan_text.strip()})
        acting_messages.append({"role": "user", "content": ACT_PROMPT})
        raw_text, tool_calls, usage = self._request(acting_messages, client, tools)
        usage_total = self._merge_usage(usage_total, usage)

        if not tool_calls and not raw_text.strip():
            raw_text = "I need one detail before I can continue. Please confirm the exact request."

        return Proposal(
            text=raw_text,
            action={"tool_calls": tool_calls},
            usage=usage_total,
            metadata={"planned": bool(plan_text.strip()), "replanned": self._needs_replan(state, budget)},
        )


def create_slot(*, context: Any, **params: Any) -> MatrixPlanWorkflow:
    return MatrixPlanWorkflow(
        model_name=context.model_name,
        model_args=context.model_args,
        temperature=context.temperature,
        **params,
    )
