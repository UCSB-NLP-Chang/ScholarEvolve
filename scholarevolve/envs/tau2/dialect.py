"""τ²-bench dialect: one assistant turn of a half-duplex dialogue with a simulated user.

τ²-bench drives the conversation: the agent is called once per incoming user or tool message, and
the model call goes through τ²'s own ``generate``. Messages are converted to the chat-dictionary
form every module is written against.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from scholarevolve.harness.interfaces import Proposal, ToolSpec


def repair_tool_pairing(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-pair tool calls and tool results after context management rewrote the history.

    A tool result whose call is gone, or a call whose result is gone, is rejected by chat
    endpoints. Unanswered calls are removed from the assistant turn; the turn's text is kept.
    """
    answered = {str(message.get("tool_call_id") or "") for message in messages
                if message.get("role") == "tool" and message.get("tool_call_id")}
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for message in messages:
        calls = list(message.get("tool_calls") or ())
        if calls:
            kept = []
            for call in calls:
                identifier = str((call.get("id") if isinstance(call, dict)
                                  else getattr(call, "id", None)) or "")
                if identifier and identifier in answered:
                    kept.append(call)
                    seen.add(identifier)
            if len(kept) != len(calls):
                message = {key: value for key, value in message.items() if key != "tool_calls"}
                if kept:
                    message["tool_calls"] = kept
                elif not str(message.get("content") or "").strip():
                    continue
        if message.get("role") == "tool" and str(message.get("tool_call_id") or "") not in seen:
            continue
        out.append(message)
    return out


class Tau2Dialect:
    name = "tau2"

    def __init__(self) -> None:
        self._types: Any = None
        self._env_tools: list[Any] = []

    def bind_tools(self, tools: list[Any]) -> None:
        """Hand the harness the domain's tools, which τ² owns and passes to the model itself."""
        self._env_tools = list(tools or [])

    @property
    def types(self) -> Any:
        if self._types is None:
            from tau2.data_model.message import (
                AssistantMessage, MultiToolMessage, SystemMessage, ToolCall, ToolMessage,
                UserMessage,
            )

            self._types = SimpleNamespace(
                AssistantMessage=AssistantMessage, MultiToolMessage=MultiToolMessage,
                SystemMessage=SystemMessage, ToolCall=ToolCall, ToolMessage=ToolMessage,
                UserMessage=UserMessage)
        return self._types

    def task_text(self, messages: list[dict[str, Any]]) -> str:
        """The user's opening request, which states the task of the dialogue."""
        for message in messages:
            if str(message.get("role")) == "user":
                return str(message.get("content") or "")
        return ""

    # ------------------------------------------------------------------ message conversion
    def to_chat(self, messages: list[Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, self.types.MultiToolMessage):
                out.extend(self.to_chat(message.tool_messages))
                continue
            if isinstance(message, self.types.ToolMessage):
                out.append({"role": "tool", "tool_call_id": message.id,
                            "content": message.content or ""})
                continue
            entry: dict[str, Any] = {"role": str(getattr(message, "role", "user")),
                                     "content": getattr(message, "content", None) or ""}
            calls = getattr(message, "tool_calls", None)
            if calls:
                entry["tool_calls"] = [
                    {"id": call.id, "type": "function",
                     "function": {"name": call.name, "arguments": json.dumps(call.arguments)}}
                    for call in calls]
            out.append(entry)
        return out

    def from_chat(self, messages: list[dict[str, Any]]) -> list[Any]:
        out = []
        for message in messages:
            role, content = message.get("role"), message.get("content") or ""
            if role == "system":
                out.append(self.types.SystemMessage(role="system", content=content))
            elif role == "tool":
                out.append(self.types.ToolMessage(id=str(message.get("tool_call_id") or ""),
                                                  role="tool", content=content,
                                                  requestor="assistant"))
            elif role == "assistant":
                calls = [self.types.ToolCall(
                    id=call.get("id", ""), name=call["function"]["name"],
                    arguments=json.loads(call["function"].get("arguments") or "{}"))
                    for call in (message.get("tool_calls") or [])]
                out.append(self.types.AssistantMessage(role="assistant", content=content,
                                                       tool_calls=calls or None))
            else:
                out.append(self.types.UserMessage(role="user", content=content))
        return out

    # ------------------------------------------------------------------ module-facing surface
    def tool_specs(self, tools: list[Any] | None = None) -> list[ToolSpec]:
        specs = []
        for tool in (self._env_tools if tools is None else tools):
            function = tool.openai_schema["function"]
            properties = (function.get("parameters") or {}).get("properties") or {}
            specs.append(ToolSpec(name=function["name"],
                                  description=function.get("description", ""),
                                  parameters=dict(properties)))
        return specs

    def native_tools(self, tools: list[ToolSpec]) -> list[dict[str, Any]] | None:
        """τ² passes its own tool objects to the model, so nothing is added here."""
        del tools
        return None

    def render_extra_tools(self, tools: list[ToolSpec]) -> str:
        """Describe actions a tool-interface module registered beside the domain tools."""
        stock = {spec.name for spec in self.tool_specs()}
        extra = [tool for tool in tools if tool.name not in stock]
        if not extra:
            return ""
        lines = ["Besides the domain tools, these actions are available:"]
        for tool in extra:
            arguments = ", ".join(f"{key}: {value}" for key, value in tool.parameters.items())
            lines.append(f"- {tool.name}({arguments}): {tool.description}")
        return "\n".join(lines)

    def repair_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return repair_tool_pairing(messages)

    def completion_kwargs(self, *, model: str, messages: list[dict[str, Any]],
                          model_args: dict[str, Any], temperature: float | None,
                          tools: list[ToolSpec] | None = None) -> dict[str, Any]:
        del tools
        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        kwargs.update(model_args or {})
        return kwargs

    def proposal(self, response: Any) -> Proposal:
        message = response.choices[0].message
        calls = [{"id": getattr(call, "id", ""), "name": call.function.name,
                  "arguments": call.function.arguments or "{}"}
                 for call in (getattr(message, "tool_calls", None) or [])]
        usage = getattr(response, "usage", None)
        usage = usage if isinstance(usage, dict) else dict(getattr(usage, "__dict__", {}) or {})
        return Proposal(text=message.content or "", action={"tool_calls": calls}, usage=usage)

    def act(self, proposal: Proposal) -> Any:
        return proposal.action if isinstance(proposal.action, dict) else {"tool_calls": []}

    def host_step(self, action: Any, proposal: Proposal, metadata: dict[str, Any]) -> dict[str, Any]:
        action = action if isinstance(action, dict) else {}
        return {"tool_calls": list(action.get("tool_calls") or []), "raw_text": proposal.text,
                "usage": dict(proposal.usage), "metadata": metadata}

    def assistant_message(self, step: dict[str, Any]) -> Any:
        """The object τ²'s orchestrator expects back from one agent turn."""
        calls = [self.types.ToolCall(
            id=call.get("id") or "", name=call["name"],
            arguments=(call["arguments"] if isinstance(call.get("arguments"), dict)
                       else json.loads(call.get("arguments") or "{}")))
            for call in (step.get("tool_calls") or [])]
        return self.types.AssistantMessage(role="assistant", content=step.get("raw_text") or "",
                                           tool_calls=calls or None)

    def concede_step(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {"tool_calls": [], "raw_text": "[context exhausted]", "usage": {},
                "metadata": metadata}
