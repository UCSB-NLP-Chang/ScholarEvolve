"""Health probes for a candidate module.

Before a mutation is measured, the probe checks importability and interface compatibility, runs
the offline phase and reloads the frozen artifact, and composes the candidate into the harness
with controlled model responses to verify that a valid action is produced. The probe reports
liveness only; it never reports a reward or a task identity.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

from scholarevolve.envs.base import PACKAGE_ROOT

_PROBE = '''"""Health probe for a module candidate. Reports liveness, never a score."""
import sys
import types
from pathlib import Path

sys.path.insert(0, {package_root!r})

from scholarevolve.harness import defaults
from scholarevolve.harness.interfaces import EpisodeResult
from scholarevolve.harness.modules import MODULE_NAMES, supports_offline_fit
from scholarevolve.harness.runtime import Harness, ModuleRuntime, load_module

MODULE = {module!r}
ENVIRONMENT = {environment!r}
CANDIDATE = Path(__file__).with_name("candidate.py")
REPLY = {reply}


class _Client:
    def __init__(self, prose_only=False):
        self.sent = []
        self._prose_only = prose_only
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.sent.append(kwargs)
        calls = None if self._prose_only else {tool_calls}
        content = REPLY if ({content_with_calls} or not calls) else None
        message = types.SimpleNamespace(content=content, tool_calls=calls)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message, finish_reason="stop")], usage=None)


def _runtime(work, artifacts=None):
    return ModuleRuntime(model_name="probe", model_args={{}}, temperature=None, work_dir=work,
                         artifacts=dict(artifacts or {{}}), environment=ENVIRONMENT)


def main() -> int:
    work = Path(__file__).parent / "_probe"
    work.mkdir(exist_ok=True)
    runtime = _runtime(work)
    try:
        module = load_module({{"module": str(CANDIDATE)}}, MODULE, runtime=runtime)
    except Exception as error:
        print(f"FAILED contract: {{type(error).__name__}}: {{error}}")
        return 1
    if supports_offline_fit(module):
        episodes = [EpisodeResult(
            task_id=f"devo_{{i}}", split="devo", reward=1.0, steps=2, completed=True,
            trajectory={episode_steps},
            metadata={{"instruction": {instruction!r}.format(i=i)}}) for i in range(4)]
        artifact = work / "artifact.json"
        try:
            module.fit(episodes=episodes)
            module.save(str(artifact))
        except Exception as error:
            print(f"FAILED offline fit/save: {{type(error).__name__}}: {{error}}")
            return 1
        runtime = _runtime(work, {{MODULE: str(artifact)}})
        try:
            module = load_module({{"module": str(CANDIDATE)}}, MODULE, runtime=runtime)
        except Exception as error:
            print(f"FAILED reload after save: {{type(error).__name__}}: {{error}}")
            return 1

    def make_harness(client):
        modules = {{name: defaults.build(name, runtime=runtime) for name in MODULE_NAMES}}
        modules[MODULE] = module
        return Harness(modules=modules, client=client, runtime=runtime)

    client = _Client()
    messages = [{{"role": "user", "content": {task!r}}}]
    try:
        step = make_harness(client).next_step(messages=list(messages),
                                              context={{"step_index": 1, "max_steps": 30}})
    except Exception as error:
        print(f"FAILED first step: {{type(error).__name__}}: {{error}}")
        return 1
    if not client.sent:
        print("FAILED the model was never called")
        return 1

{assertions}

    print("RUNS")
    print(f"  action: {{summary}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

_CHECK = '''"""Run the health probe, within a fixed number of checks."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COUNTER = HERE / ".check_budget.json"

state = json.loads(COUNTER.read_text(encoding="utf-8"))
if state["used"] >= state["budget"]:
    print(f"FAILED the check budget of {{state['budget']}} is used up")
    sys.exit(2)
state["used"] += 1
COUNTER.write_text(json.dumps(state), encoding="utf-8")
print(f"[check {{state['used']}}/{{state['budget']}}]", flush=True)
sys.exit(subprocess.run([{python!r}, "-B", str(HERE / "probe.py")]).returncode)
'''


def write_probe(directory: Path, *, module: str, environment: Any) -> Path:
    """Write ``probe.py`` for one candidate into ``directory``."""
    fixture = environment.probe_fixture()
    source = _PROBE.format(
        package_root=str(PACKAGE_ROOT), module=module, environment=environment.name,
        reply=fixture["reply"], tool_calls=fixture["tool_calls"],
        content_with_calls=fixture["content_with_calls"],
        episode_steps=fixture["episode_steps"], instruction=fixture["instruction"],
        task=fixture["task"], assertions=textwrap.indent(fixture["assert_action"], " " * 4))
    path = Path(directory) / "probe.py"
    path.write_text(source, encoding="utf-8")
    return path


def write_check(directory: Path, *, python: str, budget: int) -> Path:
    """Write the ``./check`` command a coding session uses, limited to ``budget`` runs."""
    directory = Path(directory)
    (directory / ".check_budget.json").write_text(
        f'{{"budget": {int(budget)}, "used": 0}}', encoding="utf-8")
    (directory / "check.py").write_text(_CHECK.format(python=python), encoding="utf-8")
    script = directory / "check"
    script.write_text('#!/usr/bin/env bash\ncd "$(dirname "$0")" || exit 1\n'
                      f'exec {python!r} -B check.py\n'.replace("'", '"'), encoding="utf-8")
    script.chmod(0o755)
    return script
