"""Run a harness on τ²-bench tasks with τ²'s own runner and evaluator.

Executed with the interpreter of the τ²-bench installation:

    python -m scholarevolve.envs.tau2.runner --config trial_config.json

The generated agent is registered under a new name; τ²'s registry defaults, user simulator and
evaluator are used unchanged. A task's reward is its success rate over τ²'s internal trials.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from scholarevolve.utils import atomic_write_json  # noqa: E402

AGENT_NAME = "scholarevolve_agent"
_INFRA_REASONS = ("", "error", "invalid", "infrastructure_error")


def _register(agent_module: str) -> None:
    from tau2.registry import registry

    spec = importlib.util.spec_from_file_location("scholarevolve_tau2_agent", agent_module)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {agent_module}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    agent_class = getattr(module, "EvolvedAgent")

    def factory(tools, domain_policy, **kwargs):
        return agent_class(tools=tools, domain_policy=domain_policy, llm=kwargs.get("llm"),
                           llm_args=kwargs.get("llm_args"))

    if registry.get_agent_factory(AGENT_NAME) is None:
        registry.register_agent_factory(factory, AGENT_NAME)


def _reward(simulation: dict[str, Any]) -> float:
    return float((simulation.get("reward_info") or {}).get("reward") or 0.0)


def _instruction(simulation: dict[str, Any]) -> str:
    for message in simulation.get("messages") or ():
        if message.get("role") == "user":
            return str(message.get("content") or "").strip()
    return ""


def _trajectory(simulation: dict[str, Any]) -> list[dict[str, Any]]:
    """One entry per assistant turn, with the tool results that followed it."""
    steps: list[dict[str, Any]] = []
    for message in simulation.get("messages") or ():
        role = message.get("role")
        if role == "assistant":
            calls = [{"id": call.get("id", ""), "name": call.get("name", ""),
                      "arguments": json.dumps(call.get("arguments") or {})}
                     for call in (message.get("tool_calls") or ())]
            steps.append({"step": len(steps) + 1, "raw_text": str(message.get("content") or ""),
                          "tool_calls": calls, "observations": []})
        elif role == "tool" and steps:
            steps[-1]["observations"].append(str(message.get("content") or ""))
    return steps


def _dialogue(simulation: dict[str, Any]) -> list[dict[str, Any]]:
    turns = []
    for message in simulation.get("messages") or ():
        turns.append({"role": message.get("role"), "content": message.get("content") or "",
                      "tool_calls": [{"name": call.get("name"),
                                      "arguments": call.get("arguments")}
                                     for call in (message.get("tool_calls") or ())]})
    return turns


def summarize(simulations: list[dict[str, Any]],
              task_ids: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for simulation in simulations:
        by_task[str(simulation.get("task_id"))].append(simulation)
    cells, statuses = [], []
    for task_id in task_ids:
        runs = sorted(by_task.get(task_id, []), key=lambda s: int(s.get("trial") or 0))
        successes = sum(1 for run in runs if _reward(run) >= 1.0)
        reasons = [str(run.get("termination_reason") or "") for run in runs]
        infra = not runs or any(reason in _INFRA_REASONS for reason in reasons)
        solved = [run for run in runs if _reward(run) >= 1.0]
        exemplar = (solved or runs or [None])[0]
        cells.append({"task_id": task_id, "reward": successes / len(runs) if runs else 0.0,
                      "scored": bool(runs) and not infra, "infra_error": infra,
                      "trials": len(runs), "successes": successes})
        statuses.append({
            "task_id": task_id,
            "instruction": _instruction(exemplar) if exemplar else "",
            "trajectory": _trajectory(exemplar) if exemplar else [],
            "steps": len(_trajectory(exemplar)) if exemplar else 0,
            "completed": bool(runs) and not infra,
            "termination_reasons": reasons,
            "dialogues": [{"trial": run.get("trial"), "reward": _reward(run),
                           "termination_reason": run.get("termination_reason"),
                           "messages": _dialogue(run)} for run in runs],
        })
    return cells, statuses


def run(config: dict[str, Any]) -> dict[str, Any]:
    _register(config["agent_module"])
    from tau2.data_model.simulation import TextRunConfig
    from tau2.run import run_domain

    if config.get("max_connections"):
        import httpx
        import litellm

        connections = int(config["max_connections"])
        litellm.client_session = httpx.Client(limits=httpx.Limits(
            max_connections=connections, max_keepalive_connections=connections))
    from tau2.utils.utils import DATA_DIR

    task_ids = [str(task_id) for task_id in config["task_ids"]]
    save_to = config["experiment_name"]
    # A run name is reused only when a trial is repeated after an interruption.
    shutil.rmtree(Path(DATA_DIR) / "simulations" / save_to, ignore_errors=True)
    run_config = TextRunConfig(
        domain=config["domain"], agent=AGENT_NAME,
        llm_agent=config["model"], llm_args_agent=dict(config.get("model_args") or {}),
        llm_user=config["user_model"], llm_args_user=dict(config.get("user_model_args") or {}),
        user=config.get("user", "user_simulator"),
        task_split_name=config.get("task_split_name", "full"),
        task_ids=task_ids, num_trials=int(config.get("num_trials", 1)), save_to=save_to,
        max_steps=int(config["max_steps"]), max_errors=int(config["max_errors"]),
        seed=int(config["seed"]), max_concurrency=int(config.get("max_concurrency", 4)))
    results = run_domain(run_config)
    payload = results.model_dump(mode="json") if hasattr(results, "model_dump") else {}
    simulations = payload.get("simulations") or []
    cells, statuses = summarize(simulations, task_ids)
    return {"cells": cells, "statuses": statuses}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a harness on τ²-bench tasks")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    atomic_write_json(config["out_path"], run(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
