"""The five harness modules, their interfaces, and the contracts that keep mutations separable.

A harness is ``Compose(m1, ..., m5)``. Each module implements a fixed interface, so a mutation can
replace one module while the other four stay fixed, and crossover can recombine modules taken from
different mutations. Two rules are enforced:

1. Only context management produces the model input. Memory modules return ``Evidence`` and skill
   modules return ``Skill`` objects; a workflow module may append transient turns to the input it
   receives but may not rebuild it.
2. A candidate declares the module it modifies, so "this mutation changes exactly one module" can be
   checked.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any

#: Ordered from the environment inwards.
MODULE_NAMES: tuple[str, ...] = (
    "tool_interface",
    "context_management",
    "skills",
    "memories",
    "agentic_workflows",
)

#: Short prefixes used to name candidates, e.g. ``memo-p2`` for the second memory candidate.
MODULE_PREFIX: dict[str, str] = {
    "tool_interface": "tool",
    "context_management": "cont",
    "skills": "skil",
    "memories": "memo",
    "agentic_workflows": "agen",
}

MODULE_RESPONSIBILITIES: dict[str, str] = {
    "tool_interface": (
        "Controls the actions available to the agent and translates model proposals into "
        "executable calls. Mutation directions include tool selection, tool composition and call "
        "validation, such as exposing task-relevant tools or packaging related operations into a "
        "composite action."),
    "context_management": (
        "Assembles the model input from instructions, interaction history, memories and skills "
        "through context selection, compression and ordering, preserving the evidence needed for "
        "subsequent decisions within the available context budget."),
    "skills": (
        "Maintains reusable procedures through skill induction, retrieval and curation: extracting "
        "procedures from successful trajectories, retrieving them by applicability, and "
        "consolidating overlapping entries in a skill library."),
    "memories": (
        "Maintains experience through memory formation, retrieval and consolidation, for example "
        "by recording episode outcomes, recalling relevant situations and merging related "
        "records."),
    "agentic_workflows": (
        "Organizes planning and action generation through task decomposition, agent coordination "
        "and subagent delegation, including assigning subtasks and integrating specialist "
        "proposals."),
}

#: Arguments a module method must accept.
MODULE_INTERFACES: dict[str, dict[str, tuple[str, ...]]] = {
    "tool_interface": {"tools": (), "act": ("proposal",)},
    "context_management": {"compose": ("policy", "history")},
    "skills": {"match": ("task",)},
    "memories": {"recall": ("task",), "write": ("episode_result",)},
    "agentic_workflows": {"decide": ("messages", "client")},
}

#: Arguments the harness additionally offers. A module receives exactly the subset it declares.
OPTIONAL_ARGS: dict[str, dict[str, tuple[str, ...]]] = {
    "tool_interface": {
        "tools": ("state", "task"),
        "act": ("tools", "env_actions", "state", "task", "client"),
    },
    "context_management": {
        "compose": ("evidence", "skills", "tools", "budget", "state", "task", "client"),
    },
    "skills": {"match": ("state", "tools", "budget")},
    "memories": {"recall": ("state", "tools", "budget"), "write": ("state",)},
    "agentic_workflows": {
        "decide": ("state", "tools", "evidence", "skills", "budget", "task"),
    },
}

#: Methods allowed to build chat messages: context management composes the input, and a workflow
#: may append transient turns for its own model calls.
MESSAGE_PRODUCERS: tuple[str, ...] = ("compose", "decide")

#: Optional offline construction phase, available to every module. ``fit`` consumes successful
#: episodes from the evolution set and ``save`` freezes the result, so evaluation reads a fixed
#: artifact. The phase is all-or-nothing.
OFFLINE_INTERFACE: dict[str, tuple[str, ...]] = {"fit": ("episodes",), "save": ("path",)}


class ModuleContractError(TypeError):
    """A module or candidate manifest violates the module contract."""


def accepted_args(module: Any, name: str, method: str) -> tuple[str, ...]:
    """Required arguments plus whichever optional arguments the module declares."""
    required = MODULE_INTERFACES[name][method]
    optional = OPTIONAL_ARGS.get(name, {}).get(method, ())
    function = getattr(module, method, None)
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return required
    if any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
        return (*required, *optional)
    return (*required, *(arg for arg in optional if arg in parameters))


def supports_offline_fit(module: Any) -> bool:
    return all(callable(getattr(module, method, None)) for method in OFFLINE_INTERFACE)


def _bind_problem(function: Any, parameters: tuple[str, ...]) -> str | None:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return None
    try:
        signature.bind(**{parameter: None for parameter in parameters})
    except TypeError as error:
        return str(error)
    return None


def _assembles_messages(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Dict):
            keys = {key.value for key in child.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)}
            if {"role", "content"} <= keys:
                return True
        if (isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                and child.func.id == "dict"):
            if {"role", "content"} <= {keyword.arg for keyword in child.keywords}:
                return True
    return False


def _message_violations(module: Any, name: str) -> list[str]:
    violations = []
    for method in MODULE_INTERFACES[name]:
        if method in MESSAGE_PRODUCERS:
            continue
        try:
            source = textwrap.dedent(inspect.getsource(getattr(module, method, None)))
            tree = ast.parse(source)
        except (OSError, TypeError, SyntaxError):
            continue
        if _assembles_messages(tree):
            violations.append(method)
    return violations


def validate_module(module: Any, name: str) -> None:
    """Reject a module that does not implement its interface or that writes chat messages."""
    if name not in MODULE_INTERFACES:
        raise ModuleContractError(
            f"unknown module {name!r}; the modules are {', '.join(MODULE_NAMES)}")
    for method, parameters in MODULE_INTERFACES[name].items():
        function = getattr(module, method, None)
        if not callable(function):
            raise ModuleContractError(
                f"a {name!r} module must implement {method}({', '.join(parameters)}); "
                f"{type(module).__name__} does not")
        will_send = accepted_args(module, name, method)
        problem = _bind_problem(function, will_send)
        if problem:
            raise ModuleContractError(
                f"{type(module).__name__}.{method} cannot be called as "
                f"{method}({', '.join(will_send)}): {problem}")
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            signature = None
        if signature is not None:
            offered = {*parameters, *OPTIONAL_ARGS.get(name, {}).get(method, ())}
            unfillable = [
                parameter.name for parameter in signature.parameters.values()
                if parameter.default is inspect.Parameter.empty
                and parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
                and parameter.name not in offered and parameter.name != "self"]
            if unfillable:
                raise ModuleContractError(
                    f"{type(module).__name__}.{method} requires {', '.join(unfillable)}, which "
                    f"the harness does not supply; available: {', '.join(sorted(offered))}")
    offline = [method for method in OFFLINE_INTERFACE if callable(getattr(module, method, None))]
    if offline and len(offline) != len(OFFLINE_INTERFACE):
        missing = sorted(set(OFFLINE_INTERFACE) - set(offline))
        raise ModuleContractError(
            f"{type(module).__name__} implements {', '.join(offline)} but not "
            f"{', '.join(missing)}; the offline phase must implement both fit and save")
    for method in offline:
        problem = _bind_problem(getattr(module, method), OFFLINE_INTERFACE[method])
        if problem:
            raise ModuleContractError(
                f"{type(module).__name__}.{method} cannot be called as "
                f"{method}({', '.join(OFFLINE_INTERFACE[method])}): {problem}")
    violations = _message_violations(module, name)
    if violations:
        raise ModuleContractError(
            f"{type(module).__name__} assembles chat messages inside "
            f"{', '.join(sorted(violations))}; only compose() builds the model input. Return "
            "Evidence or Skill data and let context management place it.")


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Normalise a candidate manifest, refusing one that does not declare its module set."""
    if not isinstance(manifest, dict):
        raise ModuleContractError(f"manifest must be an object, got {type(manifest).__name__}")
    name = str(manifest.get("name") or "").strip()
    if not name:
        raise ModuleContractError("manifest must carry a non-empty name")
    declared = manifest.get("modules")
    if isinstance(declared, str) or not isinstance(declared, (list, tuple)) or not declared:
        raise ModuleContractError(f"candidate {name!r} must declare its modules as a list")
    modules = [str(module) for module in declared]
    if len(set(modules)) != len(modules):
        raise ModuleContractError(f"candidate {name!r} declares a module twice: {modules}")
    unknown = sorted(set(modules) - set(MODULE_NAMES))
    if unknown:
        raise ModuleContractError(f"candidate {name!r} declares unknown module(s): {unknown}")
    return {**manifest, "name": name, "modules": sorted(modules)}


def interface_contract(name: str) -> str:
    """A plain-text description of one module's interface, for prompts and coding sessions."""
    lines = [f"MODULE: {name}", f"RESPONSIBILITY: {MODULE_RESPONSIBILITIES[name]}", "",
             "REQUIRED methods (called by keyword):"]
    for method, parameters in MODULE_INTERFACES[name].items():
        optional = OPTIONAL_ARGS.get(name, {}).get(method, ())
        lines.append(f"    def {method}({', '.join(('self', *parameters))}, ...)")
        if optional:
            lines.append(f"        # may also declare: {', '.join(optional)}")
    lines += [
        "",
        "A method receives exactly the optional arguments it declares:",
        "  state    StepState: .step_index .max_steps .steps_left .task .instruction .policy",
        "           .history .last_action .last_output .evidence .skills .tools",
        "  tools    the current list[ToolSpec]",
        "  budget   Budget: .step_index .max_steps .max_chars",
        "  client   an OpenAI-compatible client (client.chat.completions.create)",
        "",
        "OPTIONAL offline phase - implement both methods or neither:",
        "    def fit(self, episodes): consume successful episodes from the evolution set",
        "    def save(self, path): write the built artifact as JSON to `path`",
        "  Each episode is an EpisodeResult with .task_id .split .reward .steps .completed",
        "  .trajectory (a tuple of step dicts) and .metadata['instruction'].",
        "  create_module must reload the artifact from context.artifacts[<module name>] and treat",
        "  a missing key or file as 'nothing has been built yet'.",
        "",
        "The file MUST define:",
        "    def create_module(*, context, **params) -> <module object>",
        "  `context` has .model_name .model_args .temperature .work_dir .artifacts (dict of paths).",
    ]
    return "\n".join(lines)
