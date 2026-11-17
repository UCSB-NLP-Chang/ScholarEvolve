"""AppWorld adapter.

D_evo is AppWorld's train split, D_val its dev split, and the test splits are test_normal and
test_challenge. The agent writes one executable Python action per step against documented app APIs,
and AppWorld's native state evaluator scores the episode.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from scholarevolve.envs.appworld.host import DEFAULT_PROMPT
from scholarevolve.envs.base import PACKAGE_ROOT, Environment
from scholarevolve.utils import atomic_write_json, load_json, safe_name

DEFAULT_SPLITS = {"devo": "train", "val": "dev", "test": "test_normal",
                  "test_challenge": "test_challenge"}

_BASE_BEHAVIOUR = {
    "tool_interface": "one ```python block per step, executed verbatim. No schema, no argument "
                      "validation, no composite or macro actions, no routing.",
    "context_management": "the full policy followed by the full untouched history. Nothing is "
                          "dropped, summarised, re-ordered or stored outside the transcript.",
    "skills": "nothing at all - `match` returns None.",
    "memories": "nothing at all - `recall` returns an empty list.",
    "agentic_workflows": "exactly one model call per step, whose output is the action. No "
                         "planning stage, no verification, no critic, no retry, no branching, "
                         "no replanning. Observe -> think -> act, which is the ReAct loop.",
}

_INTERFACE = (
    "AppWorld. The agent writes one executable Python action per step against documented app "
    "APIs, observes the interpreter output, and must complete a multi-app task within a step "
    "budget. Scoring checks the final database state and the task-completion protocol.")

_ACTION_FORMAT = (
    "`act` MUST return a string that the environment can execute: Python source. Returning a "
    "dict, a JSON blob or an error object produces a line that parses, executes and changes "
    "nothing in the environment.")

_AGENT_TEMPLATE = '''"""Agent module generated for one harness specification."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, {package_root!r})

from scholarevolve.harness.runtime import build_harness  # noqa: E402

SPEC_PATH = Path({spec_path!r})


def create_agent(*, client, model_name, model_args, temperature):
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    return build_harness(spec=spec, client=client, model_name=model_name,
                         model_args=model_args, temperature=temperature,
                         work_dir=SPEC_PATH.parent)
'''

_PROBE_ASSERT = '''\
code = str(step.get("code") or "").strip()
if not code:
    print("FAILED no action: the module produced an empty action")
    return 1
import ast
try:
    tree = ast.parse(code)
except SyntaxError as error:
    print(f"FAILED action is not executable Python: {error}")
    print(f"  action was: {code[:300]}")
    return 1
effectful = any(not isinstance(node, ast.Expr) or isinstance(node.value, ast.Call)
                for node in tree.body)
if not effectful:
    print("FAILED inert action: it parses but calls nothing, so the environment would report "
          "success and nothing would happen")
    print(f"  action was: {code[:300]}")
    return 1
summary = code[:160]'''


def _truncate(text: Any, limit: int) -> str:
    value = str(text or "")
    return value if len(value) <= limit else value[:limit] + f"... [+{len(value) - limit} chars]"


class AppWorldEnvironment(Environment):
    name = "appworld"

    def __init__(self, settings: dict[str, Any], *, backbone: dict[str, Any],
                 work_dir: str | Path):
        super().__init__(settings, backbone=backbone, work_dir=work_dir)
        self.root = Path(os.path.expanduser(str(settings["root"]))).resolve()
        self.splits = {**DEFAULT_SPLITS, **dict(settings.get("splits") or {})}

    # ------------------------------------------------------------------ splits
    def task_ids(self, split: str) -> list[str]:
        native = self.splits.get(split, split)
        path = self.root / "data" / "datasets" / f"{native}.txt"
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    def group_of(self, task_id: str) -> str:
        """AppWorld task ids are ``<scenario>_<variant>``; variants of a scenario stay together."""
        return str(task_id).rsplit("_", 1)[0]

    # ------------------------------------------------------------------ descriptions
    def base_behaviour(self) -> dict[str, str]:
        return dict(_BASE_BEHAVIOUR)

    def interface_description(self) -> str:
        return _INTERFACE

    def action_format(self) -> str:
        return _ACTION_FORMAT

    def render_trajectory(self, status: dict[str, Any]) -> str:
        lines: list[str] = []
        for step in status.get("trajectory") or []:
            lines.append(f"--- step {step.get('step')} ---")
            lines.append("code: " + _truncate(step.get("code"), 700))
            lines.append("output: " + _truncate(step.get("output"), 500))
        return _truncate("\n".join(lines), 6000)

    def probe_fixture(self) -> dict[str, str]:
        return {
            "task": "You are an agent.\n\nTask: count my spotify playlists",
            "reply": ('("Let me look up the apps.\\n\\n```python\\n"\n'
                      '         "print(apis.api_docs.show_app_descriptions())\\n```")'),
            "tool_calls": "None",
            "content_with_calls": "True",
            "episode_steps": (
                '({"step": 1, "code": "print(apis.api_docs.show_app_descriptions())", '
                '"output": "spotify, venmo", "raw_text": "..."}, '
                '{"step": 2, "code": "apis.supervisor.complete_task(answer=3)", '
                '"output": "done", "raw_text": "..."})'),
            "instruction": "count my playlists {i}",
            "assert_action": _PROBE_ASSERT,
        }

    # ------------------------------------------------------------------ execution
    def agent_module_source(self, spec_path: Path) -> str:
        return _AGENT_TEMPLATE.format(package_root=str(PACKAGE_ROOT), spec_path=str(spec_path))

    def _prompt_template(self) -> str:
        value = self.settings.get("prompt_template")
        return str(Path(value).expanduser().resolve() if value else DEFAULT_PROMPT)

    def run_trial(self, agent_module: Path, *, split: str, task_ids: list[str], label: str,
                  out_dir: Path) -> dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "runner_output.json"
        config = {
            "appworld_root": str(self.root),
            "experiment_name": safe_name(f"{self.settings.get('experiment_prefix', 'se')}-"
                                         f"{label}"),
            "dataset": self.splits.get(split, split),
            "task_ids": list(task_ids),
            "agent_module": str(agent_module),
            "model": self.backbone["model"],
            "base_url": self.backbone["base_url"],
            "api_key_env": self.backbone.get("api_key_env", "OPENAI_API_KEY"),
            "model_args": dict(self.backbone.get("model_args") or {}),
            "temperature": self.backbone.get("temperature"),
            "request_timeout": self.backbone.get("request_timeout", 1200),
            "max_steps": int(self.settings.get("max_steps", 100)),
            "num_workers": int(self.settings.get("workers", 8)),
            "infra_retries": int(self.settings.get("infra_retries", 2)),
            "task_deadline_seconds": self.settings.get("task_deadline_seconds"),
            "prompt_template": self._prompt_template(),
            "appworld_config": dict(self.settings.get("appworld_config") or {
                "random_seed": 100, "raise_on_unsafe_syntax": True,
                "null_patch_unsafe_execution": True, "raise_on_extra_parameters": True}),
            "out_path": str(out_path),
        }
        config_path = atomic_write_json(out_dir / "runner_config.json", config)
        environment = {**os.environ, "PYTHONPATH": os.pathsep.join(
            filter(None, [str(PACKAGE_ROOT), os.environ.get("PYTHONPATH", "")]))}
        with (out_dir / "runner.log").open("ab") as log:
            completed = subprocess.run(
                [self.probe_python, "-m", "scholarevolve.envs.appworld.runner",
                 "--config", str(config_path)],
                cwd=str(PACKAGE_ROOT), env=environment, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode != 0 or not out_path.is_file():
            raise RuntimeError(f"AppWorld runner failed (exit {completed.returncode}); "
                               f"see {out_dir / 'runner.log'}")
        return load_json(out_path)


__all__ = ["AppWorldEnvironment"]
