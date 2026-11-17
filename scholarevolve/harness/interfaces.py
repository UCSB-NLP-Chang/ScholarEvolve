"""Data objects exchanged between harness modules.

The types draw the boundary between modules: ``recall`` and ``match`` return data
(``Evidence`` / ``Skill``), and only context management turns data into chat messages. This keeps
the effect of a mutation attributable to the module it changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    """One action the agent may emit, described as structured data.

    A tool-interface module can register composite actions beside the environment's own actions;
    ``composite_of`` names the environment actions such an action expands into.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    composite_of: tuple[str, ...] = ()


@dataclass(frozen=True)
class Evidence:
    """Task-specific experience returned by a memory module. Never a chat message."""

    task_id: str
    content: str
    score: float = 0.0
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Skill:
    """A reusable procedure returned by a skill module. Never a chat message.

    ``form`` is unconstrained (text, markdown, pseudocode or code) so that a mutation can choose
    its own representation.
    """

    name: str
    body: str
    form: str = "text"
    score: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Proposal:
    """What a workflow module returns: the model output plus the action it implies.

    ``action`` is whatever the environment dialect executes, e.g. a Python code string or a
    ``{"tool_calls": [...]}`` dictionary.
    """

    text: str
    action: Any = None
    usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Budget:
    """Where in the episode a module is called, and what the composed input must fit inside."""

    step_index: int
    max_steps: int
    max_messages: int | None = None
    max_chars: int | None = None


@dataclass(frozen=True)
class EpisodeResult:
    """One finished episode, as offered to the offline construction phase of a module.

    ``split`` travels with the episode so the harness can refuse experience from any split other
    than the evolution set without trusting the module.
    """

    task_id: str
    split: str
    reward: float
    steps: int
    completed: bool
    trajectory: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepState:
    """Everything a module may read about the episode so far.

    All fields are derived from what the environment already shows the agent; none of them carries
    privileged information about the task's solution.
    """

    step_index: int
    max_steps: int
    task: dict[str, Any]
    policy: tuple[dict[str, Any], ...] = ()
    history: tuple[dict[str, Any], ...] = ()
    last_action: str = ""
    last_output: str = ""
    #: Filled by the harness for modules that run after the one that produced them.
    evidence: tuple[Evidence, ...] = ()
    skills: tuple[Skill, ...] = ()
    tools: tuple[ToolSpec, ...] = ()

    @property
    def steps_left(self) -> int:
        return max(0, self.max_steps - self.step_index)

    @property
    def instruction(self) -> str:
        return str(self.task.get("instruction") or "")


def looks_like_messages(value: Any) -> bool:
    """Whether a value is a chat message or a sequence of them."""
    if isinstance(value, dict):
        return "role" in value and "content" in value
    if isinstance(value, (list, tuple)):
        return bool(value) and all(looks_like_messages(item) for item in value)
    return False
