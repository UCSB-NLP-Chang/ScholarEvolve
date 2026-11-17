"""Coding agent: implement a mutation blueprint as a candidate module.

The coding agent works in a workspace holding the blueprint; the paper's abstract, method sections,
appendix and verbatim listings; the official-code findings when a repository exists; the current
implementation of the module; and the module interface and environment action format. A ``./check``
command runs the health probe, and the agent repairs reported errors until the probe passes or its
check budget is used up.

Two back ends are provided: a Codex CLI session (``codex exec``) and a chat model that writes the
module and repairs it from probe feedback.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from scholarevolve.harness import interfaces as interfaces_module
from scholarevolve.harness.modules import interface_contract
from scholarevolve.mutation.blueprint import render_blueprint
from scholarevolve.mutation.code import CodeContext
from scholarevolve.mutation.paper import PaperContext
from scholarevolve.mutation.probes import write_check, write_probe

CANDIDATE = "candidate.py"
SEED = "# Implement the module described in BLUEPRINT.md. Run ./check to test its health.\n"

AGENTS_MD = """# Implement one harness module

Write `{candidate}` so that it implements the mutation described in `BLUEPRINT.md` for the
`{module}` module of the agent harness.

## Inputs
- `BLUEPRINT.md`: the plan to implement - source mechanism, interface methods, state and
  artifacts, runtime operations, auxiliary model calls, termination conditions and host
  adaptations.
- `paper/abstract.md` and `paper/method.md`: the paper's abstract, method sections and appendix.
  `paper/verbatim.md`: listings, algorithms and prompts copied from the paper.
- `paper/code.md`: {code_note}
- `reference/current_module.py`: the current implementation of this module in the harness.
- `reference/interfaces.py`: the data types that modules exchange.

Follow the blueprint, and consult the paper and its code for details the blueprint leaves open.

## Loop
1. Edit `{candidate}`.
2. Run `./check`. It reports health only, never a score.
3. Repeat until `./check` prints RUNS. Then stop.

`./check` verifies, in order: the module imports and satisfies its interface; its offline phase
(`fit`/`save`) runs and the saved artifact reloads, if it has one; and, composed into the harness
with a controlled model reply, it produces a valid action. You have {budget} checks; read the
failure text before editing again.

## Interface
{contract}

## Rules
1. Only context management builds chat messages. `recall` returns Evidence objects and `match`
   returns a Skill or None; where they appear in the model input is context management's choice.
2. `decide` may issue several model calls through `client`, but every request must start with the
   messages it was given. Appending turns is allowed; rebuilding or reordering them is not.
3. Import only the standard library plus
       from scholarevolve.harness.interfaces import ToolSpec, Evidence, Skill, Proposal, Budget
4. Never read benchmark data, ground truth, or anything about a task beyond the method arguments.
5. {action_format}
6. A ToolSpec you register reaches the model only through context management's rendering of it;
   give it a description the model can act on.
7. Implement the paper's core mechanism in one self-contained file of moderate length; do not
   implement every variant or ablation.
"""


@dataclass
class CodingSettings:
    backend: str = "codex"
    model: str = "gpt-5.4"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    reasoning_effort: str = "medium"
    check_budget: int = 12
    timeout: float = 2400.0
    executable: str = "codex"


def current_module_source(spec_entry: Any, module: str) -> str:
    """Source code of the module currently installed at ``module`` in the base harness."""
    if spec_entry and isinstance(spec_entry, dict) and str(spec_entry.get("module", "")).endswith(".py"):
        return Path(spec_entry["module"]).read_text(encoding="utf-8")
    from scholarevolve.harness import defaults

    names = {"tool_interface": "DefaultToolInterface", "context_management":
             "DefaultContextManagement", "skills": "DefaultSkills", "memories": "DefaultMemories",
             "agentic_workflows": "DefaultAgenticWorkflows"}
    source = Path(defaults.__file__).read_text(encoding="utf-8")
    return f"# The {module} module currently uses {names[module]} from this file.\n\n{source}"


def materialise(run_dir: Path, *, module: str, blueprint: dict[str, Any],
                paper_context: PaperContext, code_context: CodeContext | None,
                current_source: str, environment: Any, budget: int) -> Path:
    """Lay out the coding workspace."""
    sandbox = Path(run_dir) / "sandbox"
    for sub in ("paper", "reference"):
        (sandbox / sub).mkdir(parents=True, exist_ok=True)
    (sandbox / CANDIDATE).write_text(SEED, encoding="utf-8")
    (sandbox / "BLUEPRINT.md").write_text(render_blueprint(blueprint), encoding="utf-8")
    (sandbox / "paper" / "abstract.md").write_text(
        f"# {paper_context.title}\n\narXiv {paper_context.arxiv_id}\n\n{paper_context.abstract}\n",
        encoding="utf-8")
    (sandbox / "paper" / "method.md").write_text(
        paper_context.method_text or "(the full text could not be retrieved)\n", encoding="utf-8")
    (sandbox / "paper" / "verbatim.md").write_text(
        paper_context.verbatim_blocks or "(no listings were extracted)\n", encoding="utf-8")
    if code_context:
        code_note = f"findings from the official code at {code_context.repository}."
        (sandbox / "paper" / "code.md").write_text(
            f"# Official code: {code_context.repository}\n\nFiles read: "
            f"{', '.join(code_context.files_read) or '(none)'}\n\n{code_context.findings}\n",
            encoding="utf-8")
    else:
        code_note = "no official code was found for this paper."
        (sandbox / "paper" / "code.md").write_text("(no official code was found)\n",
                                                    encoding="utf-8")
    (sandbox / "reference" / "current_module.py").write_text(current_source, encoding="utf-8")
    shutil.copyfile(interfaces_module.__file__, sandbox / "reference" / "interfaces.py")
    (sandbox / "AGENTS.md").write_text(AGENTS_MD.format(
        candidate=CANDIDATE, module=module, code_note=code_note, budget=budget,
        contract=interface_contract(module), action_format=environment.action_format()),
        encoding="utf-8")
    write_probe(sandbox, module=module, environment=environment)
    write_check(sandbox, python=environment.probe_python, budget=budget)
    return sandbox


def run_probe(sandbox: Path, python: str, *, timeout: float = 600.0) -> tuple[bool, str]:
    """Run the health probe once, outside the session's check budget."""
    try:
        completed = subprocess.run([python, "-B", str(sandbox / "probe.py")], cwd=str(sandbox),
                                   capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "FAILED the probe timed out"
    output = (completed.stdout + completed.stderr).strip()
    return completed.returncode == 0 and "RUNS" in completed.stdout, output[-2000:]


# ---------------------------------------------------------------------------- Codex back end
_CODEX_CONFIG = """model = "{model}"
model_provider = "scholarevolve"
model_reasoning_effort = "{effort}"
approval_policy = "never"
sandbox_mode = "workspace-write"
web_search = "disabled"

[model_providers.scholarevolve]
name = "ScholarEvolve coding endpoint"
base_url = "{base_url}"
env_key = "{env_key}"
request_max_retries = 4
stream_max_retries = 10
stream_idle_timeout_ms = 600000

[sandbox_workspace_write]
network_access = false

[shell_environment_policy]
inherit = "core"

[history]
persistence = "none"
"""


def run_codex(sandbox: Path, run_dir: Path, settings: CodingSettings, *, on_log=print) -> None:
    executable = shutil.which(settings.executable)
    if not executable:
        raise RuntimeError(f"{settings.executable!r} is not on PATH; install the Codex CLI")
    home = Path(run_dir) / "codex_home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.toml").write_text(_CODEX_CONFIG.format(
        model=settings.model, effort=settings.reasoning_effort, base_url=settings.base_url,
        env_key=settings.api_key_env), encoding="utf-8")
    keep = ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TERM", "TMPDIR")
    environment = {name: os.environ[name] for name in keep if name in os.environ}
    environment["CODEX_HOME"] = str(home)
    if os.environ.get(settings.api_key_env):
        environment[settings.api_key_env] = os.environ[settings.api_key_env]
    command = [executable, "exec", "--cd", str(sandbox), "--sandbox", "workspace-write",
               "--skip-git-repo-check", "--json", "--color", "never",
               "-o", str(Path(run_dir) / "last_message.txt"),
               "Read AGENTS.md and do what it says. Finish when ./check prints RUNS."]
    on_log(f"    coding session ({settings.model}, {settings.check_budget} checks)")
    with (Path(run_dir) / "session.jsonl").open("w", encoding="utf-8") as log:
        try:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                           timeout=settings.timeout)
        except subprocess.TimeoutExpired:
            on_log(f"    coding session timed out after {settings.timeout:.0f}s")


# ---------------------------------------------------------------------------- chat back end
def _extract_module(text: str) -> str:
    import ast

    attempts = [match.group(1) for pattern in (r"```(?:python|py)?\s*\n(.*)```",
                                               r"```(?:python|py)?\s*\n(.*?)```")
                for match in [re.search(pattern, text or "", re.DOTALL)] if match]
    attempts.append(text or "")
    for attempt in attempts:
        try:
            ast.parse(attempt.strip())
            return attempt.strip()
        except SyntaxError:
            continue
    return attempts[0].strip()


def run_chat(sandbox: Path, llm: Callable[..., str], settings: CodingSettings, *,
             python: str, on_log=print) -> None:
    files = ["AGENTS.md", "BLUEPRINT.md", "paper/abstract.md", "paper/method.md",
             "paper/verbatim.md", "paper/code.md", "reference/current_module.py"]
    context = "\n\n".join(f"===== {name} =====\n{(sandbox / name).read_text(encoding='utf-8')}"
                          for name in files)
    messages = [{"role": "system", "content": "You implement one module of an LLM agent harness "
                                              "as a single Python file. Reply with one python "
                                              "code block and nothing else."},
                {"role": "user", "content": context + f"\n\nWrite {CANDIDATE} now."}]
    for attempt in range(max(1, settings.check_budget)):
        code = _extract_module(llm(messages, temperature=0.2))
        (sandbox / CANDIDATE).write_text(code, encoding="utf-8")
        ok, detail = run_probe(sandbox, python)
        on_log(f"    attempt {attempt + 1}: {'RUNS' if ok else detail.splitlines()[-1][:120]}")
        if ok:
            return
        messages += [{"role": "assistant", "content": f"```python\n{code}\n```"},
                     {"role": "user", "content": textwrap.dedent(f"""\
                         ./check reported:
                         {detail}

                         Return the corrected module. Keep the mechanism; fix only what the
                         report names. One python code block, nothing else.""")}]


# ---------------------------------------------------------------------------- orchestration
def implement(*, run_dir: Path, module: str, blueprint: dict[str, Any],
              paper_context: PaperContext, code_context: CodeContext | None,
              current_source: str, environment: Any, settings: CodingSettings,
              chat_llm: Callable[..., str] | None = None, on_log=print) -> dict[str, Any]:
    """Run the coding agent and return the candidate code with its final health verdict."""
    sandbox = materialise(run_dir, module=module, blueprint=blueprint,
                          paper_context=paper_context, code_context=code_context,
                          current_source=current_source, environment=environment,
                          budget=settings.check_budget)
    if settings.backend == "codex":
        run_codex(sandbox, run_dir, settings, on_log=on_log)
    elif settings.backend == "chat":
        if chat_llm is None:
            raise ValueError("the chat coding back end needs a chat model")
        run_chat(sandbox, chat_llm, settings, python=environment.probe_python, on_log=on_log)
    else:
        raise ValueError(f"unknown coding back end {settings.backend!r}")
    code = (sandbox / CANDIDATE).read_text(encoding="utf-8")
    if code.strip() == SEED.strip():
        return {"ok": False, "code": code, "detail": "the coding session left the seed untouched"}
    ok, detail = run_probe(sandbox, environment.probe_python)
    return {"ok": ok, "code": code, "detail": detail}
