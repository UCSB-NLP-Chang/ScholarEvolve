"""AppWorld dialect: one executable Python block per step, extracted from the model reply."""

from __future__ import annotations

from typing import Any

from scholarevolve.envs.appworld.host import completion_kwargs, extract_python_code
from scholarevolve.harness.interfaces import Proposal, ToolSpec


class AppWorldDialect:
    name = "appworld"

    def tool_specs(self) -> list[ToolSpec]:
        return [ToolSpec(
            name="python_code",
            description=("One executable Python block per step. `apis.<app>.<api>(**params)` "
                         "calls the environment; `apis.supervisor.complete_task(...)` ends the "
                         "episode."),
            parameters={"code": "str"})]

    def task_text(self, messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if str(message.get("role")) == "user":
                return str(message.get("content") or "")
        return ""

    def native_tools(self, tools: list[ToolSpec]) -> list[dict[str, Any]] | None:
        """AppWorld has no native tool channel: the action space is a code block."""
        del tools
        return None

    def repair_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return messages

    def render_extra_tools(self, tools: list[ToolSpec]) -> str:
        """Describe actions a tool-interface module registered beside the code block."""
        stock = {tool.name for tool in self.tool_specs()}
        extra = [tool for tool in tools if tool.name not in stock]
        if not extra:
            return ""
        lines = ["Besides the usual Python block, these actions are available:"]
        for tool in extra:
            arguments = ", ".join(f"{key}: {value}" for key, value in tool.parameters.items())
            lines.append(f"- {tool.name}({arguments}): {tool.description}")
        return "\n".join(lines)

    def completion_kwargs(self, *, model: str, messages: list[dict[str, Any]],
                          model_args: dict[str, Any], temperature: float | None,
                          tools: list[ToolSpec] | None = None) -> dict[str, Any]:
        del tools
        return completion_kwargs(model=model, messages=messages, model_args=model_args,
                                 temperature=temperature)

    def proposal(self, response: Any) -> Proposal:
        message = response.choices[0].message
        raw_text = message.content or ""
        usage = response.usage.model_dump() if getattr(response, "usage", None) is not None else {}
        return Proposal(text=raw_text, action=extract_python_code(raw_text), usage=usage)

    def act(self, proposal: Proposal) -> Any:
        if proposal.action is not None:
            return proposal.action
        return extract_python_code(proposal.text)

    def host_step(self, action: Any, proposal: Proposal, metadata: dict[str, Any]) -> dict[str, Any]:
        return {"code": str(action or ""), "raw_text": proposal.text,
                "usage": dict(proposal.usage), "metadata": metadata}

    def concede_step(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {"code": "apis.supervisor.complete_task(status='fail')",
                "raw_text": "[context exhausted]", "usage": {}, "metadata": metadata}
