"""AppWorld host helpers: task prompts, code extraction, request arguments and agent loading.

Only public task context is used. Reference solutions, evaluator code and private task data are
never read or supplied to the model.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scholarevolve.utils import load_module_from_path

#: The agent prompt: a conversation transcript with USER/ASSISTANT role markers and template
#: variables for the task instruction and the supervisor's identity.
DEFAULT_PROMPT = Path(__file__).resolve().with_name("prompt.txt")


@dataclass
class AgentStep:
    code: str
    raw_text: str
    usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


#: Some open-weight backbones return harmony-formatted text through chat endpoints.
_HARMONY_SEGMENT = re.compile(
    r"<\|channel\|>(?P<channel>\w+)<\|message\|>(?P<body>.*?)(?=<\|(?:call|end|return|start)\|>|\Z)",
    flags=re.DOTALL)


def _harmony_payload(text: str) -> str | None:
    segments = [(m.group("channel"), m.group("body")) for m in _HARMONY_SEGMENT.finditer(text)]
    if not segments:
        return None
    for wanted in ("commentary", "final"):
        for channel, body in segments:
            if channel == wanted and body.strip():
                return body.strip()
    return segments[-1][1].strip() or None


def extract_python_code(text: str) -> str:
    """The first Python fence, else the first generic fence, else the text itself."""
    text = (text or "").strip()
    if "<|channel|>" in text:
        payload = _harmony_payload(text)
        if payload:
            text = payload
    for pattern in (r"```(?:python|py)\s*\n(.*?)```", r"```\s*\n(.*?)```"):
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return text


def public_task_context(task: Any) -> dict[str, Any]:
    supervisor = dict(task.supervisor) if task.supervisor is not None else {}
    app_descriptions = dict(task.app_descriptions) if task.app_descriptions is not None else {}
    return {"task_id": str(task.id), "instruction": str(task.instruction),
            "supervisor": supervisor, "datetime": str(task.datetime),
            "app_descriptions": app_descriptions}


_ROLE_MARKER = re.compile(r"(USER|ASSISTANT|SYSTEM):\n", re.IGNORECASE)
_TEMPLATE_VAR = re.compile(r"\{\{\s*([A-Za-z_][\w.]*)\s*\}\}")


def messages_from_transcript(template: str, context: dict[str, Any]) -> list[dict[str, str]]:
    """Render an AppWorld prompt transcript and split it on its role markers."""
    values: dict[str, Any] = {
        "instruction": context["instruction"],
        "app_descriptions": json.dumps(
            [{"name": name, "description": text}
             for name, text in (context.get("app_descriptions") or {}).items()],
            indent=1, ensure_ascii=False, default=str),
        "main_user": context.get("supervisor") or {},
    }

    def resolve(match: re.Match[str]) -> str:
        head, _, attribute = match.group(1).partition(".")
        value = values.get(head)
        if attribute:
            value = value.get(attribute) if isinstance(value, dict) else None
        return "" if value is None else str(value)

    text = _TEMPLATE_VAR.sub(resolve, template).lstrip()
    messages: list[dict[str, str]] = []
    start = 0
    for match in _ROLE_MARKER.finditer(text):
        if messages:
            messages[-1]["content"] = text[start:match.start()]
        elif match.start() != 0:
            raise ValueError("prompt transcript starts with text that has no role marker")
        messages.append({"role": match.group(1).lower(), "content": ""})
        start = match.end()
    if not messages:
        raise ValueError("prompt transcript has no USER:/ASSISTANT:/SYSTEM: role markers")
    messages[-1]["content"] = text[start:]
    return messages


def append_observation(messages: list[dict[str, Any]], *, raw_text: str, output: str) -> None:
    messages.append({"role": "assistant", "content": raw_text})
    messages.append({"role": "user",
                     "content": "Environment output:\n```text\n" + output.rstrip() + "\n```\n"
                                "Continue with the next Python action, or complete the task."})


def completion_kwargs(*, model: str, messages: list[dict[str, Any]],
                      model_args: dict[str, Any], temperature: float | None) -> dict[str, Any]:
    protected = {"model", "messages", "store"} & set(model_args)
    if protected:
        raise ValueError(f"model_args may not override: {', '.join(sorted(protected))}")
    kwargs: dict[str, Any] = {"model": model, "messages": copy.deepcopy(messages), "store": False}
    kwargs.update(model_args)
    reasoning = "reasoning_effort" in kwargs or "reasoning" in kwargs
    if temperature is not None and not reasoning and "temperature" not in kwargs:
        kwargs["temperature"] = temperature
    return kwargs


def coerce_step(value: Any) -> AgentStep:
    if isinstance(value, AgentStep):
        return value
    if isinstance(value, str):
        return AgentStep(code=extract_python_code(value), raw_text=value)
    if isinstance(value, dict):
        raw_text = str(value.get("raw_text") or value.get("content") or value.get("code") or "")
        code = str(value.get("code") or extract_python_code(raw_text))
        usage = value.get("usage") if isinstance(value.get("usage"), dict) else {}
        metadata = value.get("metadata") if isinstance(value.get("metadata"), dict) else {}
        return AgentStep(code=code, raw_text=raw_text, usage=usage, metadata=metadata)
    raise TypeError(f"next_step returned unsupported value: {type(value).__name__}")


class _FrozenCompletions:
    """Chat completions whose backbone model and declared arguments cannot be changed."""

    def __init__(self, completions: Any, model: str, frozen_args: dict[str, Any]):
        self._completions = completions
        self._model = model
        self._frozen = dict(frozen_args)

    def create(self, **kwargs: Any) -> Any:
        requested = kwargs.get("model", self._model)
        if requested != self._model:
            raise ValueError(f"the backbone is frozen: cannot switch {self._model!r} to "
                             f"{requested!r}")
        for name, value in self._frozen.items():
            if kwargs.get(name, value) != value:
                raise ValueError(f"the backbone is frozen: cannot change {name}")
        if kwargs.get("stream") or int(kwargs.get("n", 1)) != 1:
            raise ValueError("requests must use stream=false and n=1")
        kwargs.update(self._frozen)
        kwargs["model"] = self._model
        kwargs["store"] = False
        if "temperature" not in self._frozen:
            kwargs.pop("temperature", None)
        return self._completions.create(**kwargs)


class FrozenBackboneClient:
    """The client a harness receives: the backbone and its arguments are fixed."""

    def __init__(self, client: Any, *, model: str, model_args: dict[str, Any]):
        frozen = {key: value for key, value in (model_args or {}).items() if value is not None}
        completions = _FrozenCompletions(client.chat.completions, model, frozen)
        self.chat = type("Chat", (), {"completions": completions})()


def load_agent(*, module_path: str | Path, client: Any, model_name: str,
               model_args: dict[str, Any], temperature: float | None) -> Any:
    module = load_module_from_path(module_path, prefix="scholarevolve_appworld_agent")
    factory = getattr(module, "create_agent", None)
    if not callable(factory):
        raise AttributeError(f"{module_path} must define create_agent(...)")
    agent = factory(client=FrozenBackboneClient(client, model=model_name, model_args=model_args),
                    model_name=model_name, model_args=dict(model_args),
                    temperature=temperature)
    if not callable(getattr(agent, "next_step", None)):
        raise TypeError("the agent returned by create_agent must implement next_step(...)")
    return agent
