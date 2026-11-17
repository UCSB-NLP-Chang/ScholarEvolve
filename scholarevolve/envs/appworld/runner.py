"""Run a harness on AppWorld tasks and score them with AppWorld's native state evaluator.

Executed with the interpreter of the AppWorld installation:

    python -m scholarevolve.envs.appworld.runner --config trial_config.json

Writes ``{"cells": [...], "statuses": [...]}`` to the config's ``out_path``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from scholarevolve.envs.appworld.host import (  # noqa: E402
    DEFAULT_PROMPT, append_observation, coerce_step, load_agent, messages_from_transcript,
    public_task_context,
)
from scholarevolve.utils import atomic_write_json, utc_now  # noqa: E402

#: An episode that repeats the same action with the same result this many times in a row, after
#: this many steps, has stopped making progress and is ended by the host.
STALL_REPEATS = 10
STALL_MIN_STEP = 30
#: Wall-clock bound on one episode, in seconds.
TASK_DEADLINE_SECONDS = 5400


def _configure(config: dict[str, Any]) -> None:
    os.environ["APPWORLD_ROOT"] = str(config["appworld_root"])
    key = os.environ.get(str(config.get("api_key_env") or "OPENAI_API_KEY"))
    os.environ["OPENAI_API_KEY"] = key or "EMPTY"


def _client(config: dict[str, Any]) -> Any:
    from openai import OpenAI

    return OpenAI(base_url=config["base_url"], api_key=os.environ["OPENAI_API_KEY"],
                  timeout=float(config.get("request_timeout") or 1200))


def _is_infrastructure_error(error: BaseException, stage: str) -> bool:
    if stage in {"initialization", "persistence"}:
        return True
    text = str(error).lower()
    if "content_filter" in text or "usage policy" in text:
        return False  # a deterministic refusal of the prompt is a task outcome
    module = type(error).__module__.split(".", 1)[0]
    if module in {"openai", "httpx", "httpcore"}:
        return True
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    name = type(error).__name__.lower()
    return any(token in name or token in text for token in (
        "apiconnection", "apitimeout", "ratelimit", "serviceunavailable",
        "connection refused", "connection reset"))


def run_one_task(config: dict[str, Any], task_id: str) -> dict[str, Any]:
    from appworld import AppWorld, update_root

    _configure(config)
    update_root(str(config["appworld_root"]))
    trajectory: list[dict[str, Any]] = []
    status: dict[str, Any] = {"task_id": task_id, "started_at": utc_now(), "completed": False,
                              "steps": 0, "instruction": "", "error": None,
                              "error_stage": None, "infra_error": False, "stopped_reason": None}
    stage = "initialization"
    try:
        with AppWorld(task_id=task_id, experiment_name=config["experiment_name"],
                      **dict(config.get("appworld_config") or {})) as world:
            stage = "agent_setup"
            context = public_task_context(world.task)
            status["instruction"] = context["instruction"]
            template = Path(config.get("prompt_template") or DEFAULT_PROMPT)
            messages: list[dict[str, Any]] = messages_from_transcript(
                template.read_text(encoding="utf-8"), context)
            agent = load_agent(module_path=config["agent_module"], client=_client(config),
                               model_name=config["model"],
                               model_args=dict(config.get("model_args") or {}),
                               temperature=config.get("temperature"))
            repeats, previous = 0, None
            deadline = time.monotonic() + float(config.get("task_deadline_seconds")
                                                or TASK_DEADLINE_SECONDS)
            max_steps = int(config["max_steps"])
            for step_index in range(1, max_steps + 1):
                if time.monotonic() > deadline:
                    status["stopped_reason"] = "deadline"
                    break
                status["steps"] = step_index
                stage = "model_request"
                step = coerce_step(agent.next_step(
                    messages=messages,
                    context={"step_index": step_index, "max_steps": max_steps}))
                if not step.code.strip():
                    raise RuntimeError("the agent returned no executable Python code")
                stage = "environment_execute"
                output = str(world.execute(step.code))
                append_observation(messages, raw_text=step.raw_text, output=output)
                trajectory.append({"step": step_index, "raw_text": step.raw_text,
                                   "code": step.code, "output": output, "usage": step.usage,
                                   "metadata": step.metadata})
                if world.task_completed():
                    status["completed"] = True
                    break
                signature = (step.code.strip(), output)
                repeats = repeats + 1 if signature == previous else 1
                previous = signature
                if repeats >= STALL_REPEATS and step_index >= STALL_MIN_STEP:
                    status["stopped_reason"] = "no_progress"
                    break
            stage = "persistence"
            save = getattr(world, "save", None)
            if callable(save):
                save()
            world.save_logs()
    except Exception as error:
        status["error"] = f"{type(error).__name__}: {error}"
        status["error_stage"] = stage
        status["infra_error"] = _is_infrastructure_error(error, stage)
        status["traceback"] = traceback.format_exc(limit=20)
    status["finished_at"] = utc_now()
    status["trajectory"] = trajectory
    return status


def _execute(config: dict[str, Any], task_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not task_ids:
        return {}
    workers = max(1, int(config.get("num_workers") or 1))
    if workers == 1:
        return {task_id: run_one_task(config, task_id) for task_id in task_ids}
    results: dict[str, dict[str, Any]] = {}
    # AppWorld's SQLite state is not thread-safe, so every task runs in its own process.
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = {executor.submit(run_one_task, config, task_id): task_id
                   for task_id in task_ids}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


def run(config: dict[str, Any]) -> dict[str, Any]:
    _configure(config)
    from appworld import evaluate_tasks, update_root

    update_root(str(config["appworld_root"]))
    task_ids = [str(task_id) for task_id in config["task_ids"]]
    experiment = Path(config["appworld_root"]) / "experiments" / "outputs" / \
        config["experiment_name"]
    if experiment.exists():
        shutil.rmtree(experiment)

    statuses = _execute(config, task_ids)
    for _ in range(int(config.get("infra_retries", 2))):
        retry = [task_id for task_id, status in statuses.items() if status.get("infra_error")]
        if not retry:
            break
        statuses.update(_execute(config, retry))

    is_test = str(config.get("dataset") or "").startswith("test")
    native = evaluate_tasks(task_ids=task_ids, experiment_name=config["experiment_name"],
                            suppress_errors=True, include_details=True,
                            save_reports=not is_test)
    individual = (native or {}).get("individual") or {}
    cells = []
    for task_id in task_ids:
        outcome = individual.get(task_id) if isinstance(individual.get(task_id), dict) else {}
        status = statuses.get(task_id) or {}
        scored = isinstance(outcome.get("success"), bool) and not status.get("infra_error")
        cells.append({"task_id": task_id,
                      "reward": 1.0 if scored and outcome["success"] else 0.0,
                      "scored": scored, "infra_error": bool(status.get("infra_error")),
                      "steps": status.get("steps"), "completed": status.get("completed")})
    return {"cells": cells, "statuses": [statuses[task_id] for task_id in task_ids]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a harness on AppWorld tasks")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run(config)
    atomic_write_json(config["out_path"], result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
