"""The initial harness H(0): one default implementation per module.

With all five defaults in place, the composed agent issues the same request as a plain ReAct agent
on the environment: the policy and the full history go to the model in one call per step, and the
action is taken from the reply. When a memory or skill module returns content, the default context
manager places it in one user message between the policy and the history.
"""

from __future__ import annotations

from typing import Any

from scholarevolve.harness.interfaces import Budget, Evidence, Proposal, Skill, ToolSpec


class DefaultToolInterface:
    def __init__(self, *, dialect: Any):
        self.dialect = dialect

    def tools(self) -> list[ToolSpec]:
        return self.dialect.tool_specs()

    def act(self, proposal: Proposal) -> Any:
        return self.dialect.act(proposal)


class DefaultMemories:
    def recall(self, task: Any, state: Any) -> list[Evidence]:
        del task, state
        return []

    def write(self, episode_result: Any) -> None:
        del episode_result


class DefaultSkills:
    def match(self, task: Any, state: Any) -> Skill | None:
        del task, state
        return None


def render_evidence(evidence: list[Evidence], skills: list[Skill]) -> str:
    parts: list[str] = []
    if skills:
        parts.append("Procedures that worked on tasks of this kind:\n" + "\n\n".join(
            f"[{skill.name}]\n{skill.body}" for skill in skills))
    if evidence:
        parts.append("Experience from earlier episodes:\n" + "\n\n".join(
            f"[{item.source or item.task_id}]\n{item.content}" for item in evidence))
    return "\n\n".join(parts)


class DefaultContextManagement:
    def __init__(self, *, dialect: Any):
        self.dialect = dialect

    def compose(self, policy: list[dict[str, Any]], history: list[dict[str, Any]],
                evidence: list[Evidence], skills: list[Skill], tools: list[ToolSpec],
                budget: Budget) -> list[dict[str, Any]]:
        del budget
        block = "\n\n".join(part for part in (self.dialect.render_extra_tools(list(tools)),
                                              render_evidence(evidence, skills)) if part)
        if not block:
            return [*policy, *history]
        return [*policy, {"role": "user", "content": block}, *history]


class DefaultAgenticWorkflows:
    def __init__(self, *, model_name: str, model_args: dict[str, Any],
                 temperature: float | None, dialect: Any):
        self.model_name = model_name
        self.model_args = dict(model_args)
        self.temperature = temperature
        self.dialect = dialect

    def decide(self, messages: list[dict[str, Any]], client: Any,
               tools: list[ToolSpec]) -> Proposal:
        response = client.chat.completions.create(**self.dialect.completion_kwargs(
            model=self.model_name, messages=messages, model_args=self.model_args,
            temperature=self.temperature, tools=list(tools)))
        return self.dialect.proposal(response)


def build(name: str, *, runtime: Any) -> Any:
    """The default implementation of one module."""
    dialect = runtime.dialect
    if name == "tool_interface":
        return DefaultToolInterface(dialect=dialect)
    if name == "memories":
        return DefaultMemories()
    if name == "skills":
        return DefaultSkills()
    if name == "context_management":
        return DefaultContextManagement(dialect=dialect)
    if name == "agentic_workflows":
        return DefaultAgenticWorkflows(model_name=runtime.model_name,
                                       model_args=runtime.model_args,
                                       temperature=runtime.temperature, dialect=dialect)
    raise ValueError(f"no default for module {name!r}")
