"""τ²-bench adapter.

D_evo is an experience pool sampled from the domain's full task collection after excluding the
base collection, stratified by issue type. D_val is the official training split and D_test the
official test split. A task's reward is its success rate over τ²'s internal trials.
"""

from __future__ import annotations

import os
import random
import re
import subprocess
from pathlib import Path
from typing import Any

from scholarevolve.envs.base import EVOLUTION, PACKAGE_ROOT, Environment
from scholarevolve.utils import atomic_write_json, load_json, safe_name

DEFAULT_SPLITS = {"val": "train", "test": "test"}

#: Stratified experience pool for the Telecom domain: all free service-issue tasks plus samples of
#: the mobile-data and MMS issue types.
DEFAULT_TELECOM_POOL = {"seed": 2026,
                        "counts": {"service_issue": 18, "mobile_data_issue": 60, "mms_issue": 172}}

_BASE_BEHAVIOUR = {
    "tool_interface": "the domain's own function schemas, passed to the model exactly as the "
                      "environment supplies them. No composite or macro action, no argument "
                      "validation beyond the schema, no routing, no gating of a write behind a "
                      "check.",
    "context_management": "the domain policy document followed by the whole dialogue verbatim - "
                          "user turns, assistant turns and tool results in the order they "
                          "happened. Nothing is dropped, summarised or re-ordered, and no note "
                          "about the user's request is kept outside the transcript.",
    "skills": "nothing at all - `match` returns None.",
    "memories": "nothing at all - `recall` returns an empty list.",
    "agentic_workflows": "exactly one model call per incoming user or tool message, whose output "
                         "is the next assistant turn. No plan, no verification of a write before "
                         "it is issued, no re-reading of the policy, no critic, no retry, no "
                         "self-consistency. The dialogue ends when the user stops.",
}

_ACTION_FORMAT = (
    "`act` MUST return `{\"tool_calls\": [...]}`, where a call is `{\"id\": str, \"name\": str, "
    "\"arguments\": <JSON string>}`. The environment executes only the domain's own tools; an "
    "unknown name is an environment error and repeated errors abort the dialogue, so you may "
    "register extra ToolSpecs but `act` has to translate them back into domain tools. Returning "
    "no tool call is a legal action: it sends the assistant's text to the user, which is how "
    "information reaches the user, so a module must never drop that text.")

_AGENT_TEMPLATE = '''"""τ²-bench agent generated for one harness specification."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, {package_root!r})

from tau2.agent.llm_agent import LLMAgent  # noqa: E402
from tau2.data_model.message import MultiToolMessage  # noqa: E402
from tau2.utils.llm_utils import generate  # noqa: E402

from scholarevolve.envs.tau2.dialect import Tau2Dialect  # noqa: E402
from scholarevolve.harness.runtime import build_harness  # noqa: E402

SPEC_PATH = Path({spec_path!r})


class _Tau2Client:
    """``client.chat.completions.create`` over τ²'s ``generate`` with a frozen backbone."""

    _OVERRIDABLE = ("temperature", "max_tokens", "max_completion_tokens")

    def __init__(self, *, dialect, model, tools, model_args):
        self._dialect = dialect
        self._model = model
        self._tools = tools
        self._model_args = dict(model_args or {{}})
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        args = dict(self._model_args)
        for key in self._OVERRIDABLE:
            if key not in args and kwargs.get(key) is not None:
                args[key] = kwargs[key]
        reply = generate(model=self._model, messages=self._dialect.from_chat(kwargs["messages"]),
                         tools=self._tools, call_name="agent_response", **args)
        calls = [SimpleNamespace(id=call.id, function=SimpleNamespace(
            name=call.name, arguments=json.dumps(call.arguments)))
            for call in (reply.tool_calls or [])]
        message = SimpleNamespace(content=reply.content or "", tool_calls=calls or None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                               usage=None)


class EvolvedAgent(LLMAgent):
    def __init__(self, tools, domain_policy, llm, llm_args=None):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        self._dialect = Tau2Dialect()
        self._dialect.bind_tools(list(tools or []))
        self._max_steps = int(spec.get("max_steps") or 0)
        client = _Tau2Client(dialect=self._dialect, model=llm, tools=list(tools or []),
                             model_args=llm_args)
        self._harness = build_harness(spec=spec, client=client, model_name=llm,
                                      model_args=dict(llm_args or {{}}), temperature=None,
                                      work_dir=SPEC_PATH.parent, dialect=self._dialect)
        self._turn = 0

    def _generate_next_message(self, message, state):
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        self._turn += 1
        chat = self._dialect.to_chat(list(state.system_messages) + list(state.messages))
        step = self._harness.next_step(
            messages=chat, context={{"step_index": self._turn, "max_steps": self._max_steps}})
        return self._dialect.assistant_message(step)
'''

_PROBE_ASSERT = '''\
calls = list(step.get("tool_calls") or [])
text = str(step.get("raw_text") or "").strip()
if not calls and not text:
    print("FAILED no action: neither a tool call nor a reply, so the turn is empty")
    return 1
for call in calls:
    if not isinstance(call, dict) or not call.get("name"):
        print(f"FAILED malformed tool call: {call!r}")
        return 1
    if not isinstance(call.get("arguments"), str):
        print("FAILED arguments must be a JSON string, got "
              f"{type(call.get('arguments')).__name__}")
        return 1
spoken = make_harness(_Client(prose_only=True)).next_step(
    messages=list(messages), context={"step_index": 2, "max_steps": 200})
if not str(spoken.get("raw_text") or "").strip():
    print("FAILED cannot speak: the model replied to the user and the module returned no text")
    return 1
summary = f"{[call['name'] for call in calls]} text={text[:60]!r}"[:160]'''

_TYPE_PREFIX = re.compile(r"\[(\w+)\]")


def _truncate(text: Any, limit: int) -> str:
    value = str(text or "")
    return value if len(value) <= limit else value[:limit] + f"... [+{len(value) - limit} chars]"


class Tau2Environment(Environment):
    name = "tau2"

    def __init__(self, settings: dict[str, Any], *, backbone: dict[str, Any],
                 work_dir: str | Path):
        super().__init__(settings, backbone=backbone, work_dir=work_dir)
        self.root = Path(os.path.expanduser(str(settings["root"]))).resolve()
        self.domain = str(settings.get("domain", "telecom"))
        self.splits = {**DEFAULT_SPLITS, **dict(settings.get("splits") or {})}

    # ------------------------------------------------------------------ splits
    def _domain_dir(self) -> Path:
        return self.root / "data" / "tau2" / "domains" / self.domain

    def _split_tasks(self) -> dict[str, list[str]]:
        return {name: [str(task_id) for task_id in ids]
                for name, ids in load_json(self._domain_dir() / "split_tasks.json").items()}

    def experience_pool(self) -> list[str]:
        """Stratified D_evo sample from the full collection minus the base collection."""
        listed = self.settings.get("devo_task_ids_file")
        if listed:
            value = load_json(Path(listed).expanduser())
            return [str(task_id) for task_id in (value.get("task_ids")
                                                 if isinstance(value, dict) else value)]
        pool = dict(self.settings.get("devo_pool") or DEFAULT_TELECOM_POOL)
        splits = self._split_tasks()
        candidates = sorted(set(splits["full"]) - set(splits.get("base") or []))
        by_type: dict[str, list[str]] = {}
        for task_id in candidates:
            match = _TYPE_PREFIX.match(task_id)
            by_type.setdefault(match.group(1) if match else "other", []).append(task_id)
        rng = random.Random(int(pool.get("seed", 2026)))
        chosen: list[str] = []
        for issue, count in sorted(dict(pool["counts"]).items()):
            members = list(by_type.get(issue, []))
            rng.shuffle(members)
            chosen += members[:int(count)]
        return sorted(chosen)

    def task_ids(self, split: str) -> list[str]:
        if split == EVOLUTION:
            return self.experience_pool()
        return self._split_tasks()[self.splits.get(split, split)]

    # ------------------------------------------------------------------ descriptions
    def base_behaviour(self) -> dict[str, str]:
        return dict(_BASE_BEHAVIOUR)

    def interface_description(self) -> str:
        return (f"τ²-bench, {self.domain} domain. The agent talks with a simulated user who "
                "reports a problem, follows the domain policy, calls the domain's tools and, "
                "where the policy requires it, guides the user through actions on the user's "
                "side. The dialogue ends when the user stops or a step or error limit is "
                "reached. A dialogue scores 1 only if the final environment state satisfies the "
                "task's goal, otherwise 0.")

    def action_format(self) -> str:
        return _ACTION_FORMAT

    def render_trajectory(self, status: dict[str, Any]) -> str:
        dialogues = list(status.get("dialogues") or [])
        failed = [d for d in dialogues if float(d.get("reward") or 0.0) < 1.0] or dialogues
        if not failed:
            return ""
        dialogue = failed[0]
        lines = [f"(termination: {dialogue.get('termination_reason')})"]
        for message in dialogue.get("messages") or []:
            calls = message.get("tool_calls") or []
            rendered = "; ".join(f"{call.get('name')}({_truncate(call.get('arguments'), 200)})"
                                 for call in calls)
            body = _truncate(message.get("content"), 500)
            lines.append(f"[{message.get('role')}] {body}" + (f" -> calls: {rendered}"
                                                              if rendered else ""))
        text = "\n".join(lines)
        if len(text) > 6000 and len(lines) > 12:
            text = "\n".join(lines[:4] + [f"... [{len(lines) - 12} turns omitted]"] + lines[-8:])
        return _truncate(text, 6000)

    def probe_fixture(self) -> dict[str, str]:
        return {
            "task": ("You are a support agent. Follow the policy and confirm before any "
                     "change.\n\nMy phone has no mobile data."),
            "reply": '"Let me check your line status."',
            "tool_calls": ('[types.SimpleNamespace(id="c1", function=types.SimpleNamespace('
                           'name="get_customer_by_phone", '
                           'arguments=\'{"phone_number": "555-0100"}\'))]'),
            "content_with_calls": "False",
            "episode_steps": (
                '({"step": 1, "raw_text": "Let me check.", "tool_calls": [{"id": "c1", '
                '"name": "get_customer_by_phone", "arguments": \'{"phone_number": "555-0100"}\'}], '
                '"observations": ["{}"]}, {"step": 2, "raw_text": "Fixed. Anything else?", '
                '"tool_calls": [], "observations": []})'),
            "instruction": "my phone has no data {i}",
            "assert_action": _PROBE_ASSERT,
        }

    # ------------------------------------------------------------------ execution
    def agent_module_source(self, spec_path: Path) -> str:
        return _AGENT_TEMPLATE.format(package_root=str(PACKAGE_ROOT), spec_path=str(spec_path))

    def write_agent(self, spec: dict[str, Any], out_dir: str | Path, name: str = "agent") -> Path:
        spec = {**spec, "max_steps": int(self.settings.get("max_steps", 200))}
        return super().write_agent(spec, out_dir, name)

    def run_trial(self, agent_module: Path, *, split: str, task_ids: list[str], label: str,
                  out_dir: Path) -> dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "runner_output.json"
        trials = int(self.settings.get("devo_num_trials", 1) if split == EVOLUTION
                     else self.settings.get("num_trials", 4))
        config = {
            "experiment_name": safe_name(f"{self.settings.get('experiment_prefix', 'se')}-"
                                         f"{label}"),
            "domain": self.domain,
            "task_split_name": self.settings.get("task_split_name", "full"),
            "task_ids": list(task_ids),
            "agent_module": str(agent_module),
            "model": self.backbone["model"],
            "model_args": dict(self.backbone.get("model_args") or {}),
            "user_model": self.settings["user_model"],
            "user_model_args": dict(self.settings.get("user_model_args") or {}),
            "num_trials": trials,
            "max_steps": int(self.settings.get("max_steps", 200)),
            "max_errors": int(self.settings.get("max_errors", 10)),
            "seed": int(self.settings.get("seed", 300)),
            "max_concurrency": int(self.settings.get("max_concurrency", 8)),
            "max_connections": self.settings.get("max_connections"),
            "out_path": str(out_path),
        }
        config_path = atomic_write_json(out_dir / "runner_config.json", config)
        environment = {**os.environ, "PYTHONPATH": os.pathsep.join(
            filter(None, [str(PACKAGE_ROOT), os.environ.get("PYTHONPATH", "")]))}
        with (out_dir / "runner.log").open("ab") as log:
            completed = subprocess.run(
                [self.probe_python, "-m", "scholarevolve.envs.tau2.runner",
                 "--config", str(config_path)],
                cwd=str(self.root), env=environment, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode != 0 or not out_path.is_file():
            raise RuntimeError(f"τ²-bench runner failed (exit {completed.returncode}); "
                               f"see {out_dir / 'runner.log'}")
        return load_json(out_path)


__all__ = ["Tau2Environment"]
