"""Compose five modules into one agent step, and load modules from candidate specifications.

The environment host calls ``Harness.next_step(messages, context)`` once per agent turn. The
harness runs the modules in a fixed order through their interfaces:

    tool_interface.tools -> memories.recall, skills.match -> context_management.compose
    -> agentic_workflows.decide -> tool_interface.act

Three guarantees are enforced here rather than trusted to a candidate:

* ``recall`` and ``match`` hand back ``Evidence`` and ``Skill`` objects, never chat messages;
* every model request a workflow sends starts with the messages ``compose`` returned;
* the memory write path only accepts episodes from the evolution set, and is frozen while a
  harness is evaluated.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scholarevolve.harness import defaults
from scholarevolve.harness.interfaces import (
    Budget, EpisodeResult, Evidence, Proposal, Skill, StepState, ToolSpec, looks_like_messages,
)
from scholarevolve.harness.modules import (
    MODULE_NAMES, ModuleContractError, accepted_args, validate_module,
)
from scholarevolve.utils import load_module_from_path

#: Split label carried by episodes from the evolution set D_evo.
EVOLUTION_SPLIT = "devo"

#: Characters per token, used only to state the context budget to context management.
CHARS_PER_TOKEN = 3.5


class LeakageError(RuntimeError):
    """An attempt to write experience from outside the evolution set, or into a frozen store."""


class ContextExhausted(RuntimeError):
    """The composed input no longer fits the model's context window."""


@dataclass
class ModuleRuntime:
    """What a module receives when it is built (the model client is passed to ``decide``)."""

    model_name: str
    model_args: dict[str, Any] = field(default_factory=dict)
    temperature: float | None = None
    work_dir: Path | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    environment: str = "appworld"
    context_limit_tokens: int = 128_000
    empty_retries: int = 10
    _dialect: Any = field(default=None, repr=False, compare=False)

    @property
    def dialect(self) -> Any:
        """One dialect instance per runtime, shared by the harness and every module."""
        if self._dialect is None:
            from scholarevolve.envs import get_dialect

            self._dialect = get_dialect(self.environment)
        return self._dialect


class MemoryWriteGuard:
    """Wrap a memory module so its write path cannot see an episode outside the evolution set.

    ``frozen`` is the evaluation posture: the store is built once from the evolution set and is not
    written while any harness is evaluated.
    """

    def __init__(self, inner: Any, *, evolution_ids: Any, frozen: bool = True):
        self.inner = inner
        self.evolution_ids = {str(task_id) for task_id in evolution_ids}
        self.frozen = bool(frozen)
        self.writes = 0

    def recall(self, task: Any, state: Any) -> list[Evidence]:
        wanted = accepted_args(self.inner, "memories", "recall")
        offered = {"task": task, "state": state}
        return self.inner.recall(**{name: value for name, value in offered.items()
                                    if name in wanted})

    def fit(self, episodes: Any) -> None:
        for episode in episodes:
            self.write(episode)

    def save(self, path: Any, **kwargs: Any) -> Any:
        return self.inner.save(path, **kwargs)

    def write(self, episode_result: EpisodeResult) -> None:
        task_id = str(getattr(episode_result, "task_id", "") or "")
        if self.frozen:
            raise LeakageError(f"the memory store is frozen; refusing to write task {task_id!r}")
        split = str(getattr(episode_result, "split", "") or "")
        if split != EVOLUTION_SPLIT or task_id not in self.evolution_ids:
            raise LeakageError(
                f"refusing to write experience from task {task_id!r} (split {split!r}); memories "
                "may only be built from the evolution set")
        self.inner.write(episode_result=episode_result)
        self.writes += 1


def _as_evidence(values: Any, *, who: str) -> list[Evidence]:
    if values is None:
        return []
    if looks_like_messages(values):
        raise ModuleContractError(f"{who}.recall returned chat messages")
    items = list(values)
    bad = sorted({type(item).__name__ for item in items if not isinstance(item, Evidence)})
    if bad:
        raise ModuleContractError(f"{who}.recall must return Evidence objects, got {bad}")
    return items


def _as_skills(value: Any, *, who: str) -> list[Skill]:
    if value is None:
        return []
    if looks_like_messages(value):
        raise ModuleContractError(f"{who}.match returned chat messages")
    items = [value] if isinstance(value, Skill) else list(value)
    bad = sorted({type(item).__name__ for item in items if not isinstance(item, Skill)})
    if bad:
        raise ModuleContractError(f"{who}.match must return a Skill or None, got {bad}")
    return items


def task_from_messages(messages: list[dict[str, Any]], *, dialect: Any) -> dict[str, Any]:
    """Recover the task statement from the environment-built input.

    Where the statement sits is environment-specific, so the dialect locates it.
    """
    text = dialect.task_text(messages)
    instruction = ""
    for line in text.splitlines():
        if line.startswith("Task:"):
            instruction = line[len("Task:"):].strip()
    return {"instruction": instruction or text.strip(), "prompt_tail": text}


class PrefixGuardedClient:
    """The client handed to ``decide``.

    A workflow may issue several model calls per step and append transient turns, but every request
    must start with the messages context management composed. Empty replies from the backbone are
    retried, and a context-window overflow is raised as ``ContextExhausted``.
    """

    def __init__(self, inner: Any, composed: list[dict[str, Any]], stats: dict[str, int], *,
                 dialect: Any, empty_retries: int, default_kwargs: dict[str, Any] | None = None):
        self._inner = inner
        self._composed = [dict(message) for message in composed]
        self._stats = stats
        self._dialect = dialect
        self._empty_retries = int(empty_retries)
        self._default_kwargs = dict(default_kwargs or {})
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _count(self, key: str) -> None:
        self._stats[key] = self._stats.get(key, 0) + 1

    def _create(self, **kwargs: Any) -> Any:
        for key, value in self._default_kwargs.items():
            kwargs.setdefault(key, value)
        tools = kwargs.get("tools")
        if tools and any(isinstance(tool, ToolSpec) for tool in tools):
            native = self._dialect.native_tools(
                [tool for tool in tools if isinstance(tool, ToolSpec)])
            if native:
                kwargs["tools"] = native
            else:
                kwargs.pop("tools", None)
        sent = list(kwargs.get("messages") or [])
        if [dict(message) for message in sent[:len(self._composed)]] != self._composed:
            raise ModuleContractError(
                "agentic_workflows.decide sent a request that does not start with the composed "
                "messages; a workflow may append turns but may not rebuild the input")
        response = None
        for _ in range(self._empty_retries + 1):
            try:
                response = self._inner.chat.completions.create(**kwargs)
            except Exception as error:
                text = str(error)
                if ("context_length_exceeded" in text or "maximum context length" in text
                        or "Input tokens exceed" in text):
                    self._count("context_exhausted")
                    raise ContextExhausted(text) from error
                raise
            choices = getattr(response, "choices", None) or []
            message = choices[0].message if choices else None
            content = getattr(message, "content", None) if message is not None else None
            if choices and getattr(choices[0], "finish_reason", None) == "length":
                self._count("truncated" if (content or "").strip() else "reasoning_exhausted")
            if (content or "").strip() or getattr(message, "tool_calls", None):
                return response
            self._count("empty_responses")
        self._count("empty_responses_exhausted")
        return response


class Harness:
    """One ``next_step`` assembled from the five modules."""

    def __init__(self, *, modules: dict[str, Any], client: Any, runtime: ModuleRuntime,
                 manifest: dict[str, Any] | None = None):
        missing = sorted(set(MODULE_NAMES) - set(modules))
        if missing:
            raise ModuleContractError(f"harness is missing module(s): {', '.join(missing)}")
        for name in MODULE_NAMES:
            validate_module(modules[name], name)
        self.modules = dict(modules)
        self.client = client
        self.runtime = runtime
        self.manifest = dict(manifest or {})
        self.dialect = runtime.dialect
        self._policy_len: int | None = None
        self._task: dict[str, Any] = {}
        self.trace: list[dict[str, Any]] = []
        self.stats: dict[str, int] = {}

    def _invoke(self, name: str, method: str, **available: Any) -> Any:
        module = self.modules[name]
        wanted = accepted_args(module, name, method)
        return getattr(module, method)(**{key: available[key] for key in wanted
                                          if key in available})

    def max_chars(self) -> int:
        reserved = int(self.runtime.model_args.get("max_completion_tokens")
                       or self.runtime.model_args.get("max_tokens") or 0)
        return int(max(0, self.runtime.context_limit_tokens - reserved) * CHARS_PER_TOKEN)

    def next_step(self, *, messages: list[dict[str, Any]],
                  context: dict[str, Any]) -> dict[str, Any]:
        if self._policy_len is None:
            # The host only appends, so the first input fixes where the policy ends.
            self._policy_len = len(messages)
            self._task = task_from_messages(messages, dialect=self.dialect)
        policy = list(messages[:self._policy_len])
        history = list(messages[self._policy_len:])
        step_index = int(context.get("step_index") or 1)
        max_steps = int(context.get("max_steps") or 0)
        budget = Budget(step_index=step_index, max_steps=max_steps, max_chars=self.max_chars())
        last_output = str(history[-1].get("content") or "") if history else ""
        last_action = str(history[-2].get("content") or "") if len(history) > 1 else ""

        def state(**extra: Any) -> StepState:
            return StepState(step_index=step_index, max_steps=max_steps, task=dict(self._task),
                             policy=tuple(policy), history=tuple(history),
                             last_action=last_action, last_output=last_output, **extra)

        shared: dict[str, Any] = {"task": self._task, "budget": budget, "client": self.client}
        tools = list(self._invoke("tool_interface", "tools", state=state(), **shared) or [])
        base = state(tools=tuple(tools))
        evidence = _as_evidence(
            self._invoke("memories", "recall", state=base, tools=tools, **shared),
            who=type(self.modules["memories"]).__name__)
        skills = _as_skills(
            self._invoke("skills", "match", state=base, tools=tools, **shared),
            who=type(self.modules["skills"]).__name__)
        enriched = state(tools=tuple(tools), evidence=tuple(evidence), skills=tuple(skills))
        composed = self._invoke("context_management", "compose", policy=policy, history=history,
                                evidence=evidence, skills=skills, tools=tools, state=enriched,
                                **shared)
        if not looks_like_messages(composed):
            raise ModuleContractError(
                f"{type(self.modules['context_management']).__name__}.compose must return a "
                "list of chat messages")
        composed = self.dialect.repair_messages(list(composed))
        native = self.dialect.native_tools(tools)
        client = PrefixGuardedClient(
            self.client, list(composed), self.stats, dialect=self.dialect,
            empty_retries=self.runtime.empty_retries,
            default_kwargs={"tools": native} if native else None)
        try:
            proposal = self._invoke("agentic_workflows", "decide", messages=list(composed),
                                    client=client, state=enriched, tools=tools,
                                    evidence=evidence, skills=skills, task=self._task,
                                    budget=budget)
        except ContextExhausted:
            # Running out of context is an outcome of the harness: the episode is conceded.
            return self.dialect.concede_step({"context_exhausted": True,
                                              "harness_stats": dict(self.stats)})
        if not isinstance(proposal, Proposal):
            raise ModuleContractError(
                f"{type(self.modules['agentic_workflows']).__name__}.decide must return a "
                "Proposal")
        action = self._invoke("tool_interface", "act", proposal=proposal, tools=tools,
                              env_actions=tools, state=enriched, **shared)
        self.trace.append({"step": step_index, "evidence": len(evidence), "skills": len(skills),
                           "tools": len(tools), "composed_messages": len(composed)})
        return self.dialect.host_step(action, proposal, {
            **dict(proposal.metadata), "modules": self.manifest.get("modules", []),
            "evidence": len(evidence), "skills": len(skills), "tools": len(tools),
            **({"harness_stats": dict(self.stats)} if self.stats else {})})


def load_module(spec: Any, name: str, *, runtime: ModuleRuntime) -> Any:
    """Build one module from a spec entry.

    An entry is ``None`` (the default module) or ``{"module": <path or dotted name>,
    "params": {...}}``; the referenced file must define ``create_module(*, context, **params)``.
    """
    if spec in (None, "", "default"):
        return defaults.build(name, runtime=runtime)
    if isinstance(spec, str):
        spec = {"module": spec}
    reference = str(spec.get("module") or "")
    if not reference:
        raise ModuleContractError(f"module {name!r} spec has no module reference")
    if reference.endswith(".py"):
        loaded = load_module_from_path(reference, prefix=f"scholarevolve_{name}")
    else:
        loaded = importlib.import_module(reference)
    factory = getattr(loaded, "create_module", None)
    if not callable(factory):
        raise ModuleContractError(f"{reference} must define create_module(*, context, **params)")
    built = factory(context=runtime, **dict(spec.get("params") or {}))
    validate_module(built, name)
    return built


def build_harness(*, spec: dict[str, Any], client: Any, model_name: str,
                  model_args: dict[str, Any], temperature: float | None,
                  work_dir: str | Path | None = None, dialect: Any = None) -> Harness:
    """Assemble the harness described by a specification dictionary.

    A specification names the module file and parameters for each module (``None`` for the
    default), the frozen artifacts the modules read, and the evolution-set ids for the memory guard.
    ``dialect`` lets the environment host pass a dialect it has already bound to its tools.
    """
    runtime = ModuleRuntime(
        model_name=model_name, model_args=dict(model_args or {}), temperature=temperature,
        work_dir=Path(work_dir) if work_dir else None,
        artifacts=dict(spec.get("artifacts") or {}),
        environment=str(spec.get("environment") or "appworld"),
        context_limit_tokens=int(spec.get("context_limit_tokens") or 128_000),
        empty_retries=int(spec.get("empty_retries", 10)), _dialect=dialect)
    module_specs = dict(spec.get("modules") or {})
    unknown = sorted(set(module_specs) - set(MODULE_NAMES))
    if unknown:
        raise ModuleContractError(f"spec names unknown module(s): {', '.join(unknown)}")
    modules = {name: load_module(module_specs.get(name), name, runtime=runtime)
               for name in MODULE_NAMES}
    guard = dict(spec.get("memory_guard") or {})
    modules["memories"] = MemoryWriteGuard(modules["memories"],
                                           evolution_ids=guard.get("evolution_ids") or [],
                                           frozen=bool(guard.get("frozen", True)))
    return Harness(modules=modules, client=client, runtime=runtime,
                   manifest=dict(spec.get("manifest") or {}))
