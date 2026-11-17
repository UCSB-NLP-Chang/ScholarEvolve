"""Environment base class: splits, evaluation over repeated trials, and generated agent modules."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scholarevolve.utils import atomic_write_json, load_json

#: Logical split names used by the search. Adapters map them to the environment's own splits.
EVOLUTION, VALIDATION, TEST = "devo", "val", "test"

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


class EvaluationError(RuntimeError):
    """An evaluation could not score every task, so it cannot support a search decision."""


@dataclass
class Evaluation:
    """A harness evaluated on one split: per-task rewards averaged over trials."""

    label: str
    split: str
    cells: list[dict[str, Any]]
    statuses: list[dict[str, Any]] = field(default_factory=list)
    trials: int = 1
    trial_means: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        scored = [float(cell["reward"]) for cell in self.cells if cell.get("scored")]
        return round(sum(scored) / len(scored), 6) if scored else 0.0

    def rewards(self) -> dict[str, float]:
        return {str(cell["task_id"]): float(cell["reward"]) for cell in self.cells
                if cell.get("scored")}

    def to_json(self) -> dict[str, Any]:
        return {"label": self.label, "split": self.split, "mean": self.mean,
                "trials": self.trials, "trial_means": self.trial_means, "cells": self.cells,
                "statuses": self.statuses}

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Evaluation":
        return cls(label=value["label"], split=value["split"], cells=list(value["cells"]),
                   statuses=list(value.get("statuses") or []), trials=int(value.get("trials", 1)),
                   trial_means=list(value.get("trial_means") or []))


class Environment:
    """Base class for an evaluation environment."""

    name = ""

    def __init__(self, settings: dict[str, Any], *, backbone: dict[str, Any],
                 work_dir: str | Path):
        self.settings = dict(settings)
        self.backbone = dict(backbone)
        self.work_dir = Path(work_dir)

    # ------------------------------------------------------------------ adapter surface
    def task_ids(self, split: str) -> list[str]:
        raise NotImplementedError

    def group_of(self, task_id: str) -> str:
        """Tasks in one group (e.g. variants of a scenario) must stay on one side of a split."""
        return str(task_id)

    def base_behaviour(self) -> dict[str, str]:
        """What each module of the initial harness does in this environment."""
        raise NotImplementedError

    def interface_description(self) -> str:
        """How the agent interacts with the environment and how it is scored."""
        raise NotImplementedError

    def action_format(self) -> str:
        """What ``tool_interface.act`` must return for the environment to execute it."""
        raise NotImplementedError

    def render_trajectory(self, status: dict[str, Any]) -> str:
        """A failed episode rendered for the failure audit."""
        raise NotImplementedError

    def probe_fixture(self) -> dict[str, str]:
        """Environment-shaped fixtures for the coding agent's health probe."""
        raise NotImplementedError

    @property
    def probe_python(self) -> str:
        """Interpreter that runs the health probe."""
        return str(self.settings.get("python") or shutil.which("python3") or "python3")

    def agent_module_source(self, spec_path: Path) -> str:
        """Source of the agent module the environment host loads for a harness spec."""
        raise NotImplementedError

    def run_trial(self, agent_module: Path, *, split: str, task_ids: list[str], label: str,
                  out_dir: Path) -> dict[str, Any]:
        """Run one trial and return ``{"cells": [...], "statuses": [...]}``.

        A cell holds ``task_id``, ``reward``, ``scored`` and ``infra_error``. A status holds the
        task's ``instruction``, ``trajectory``, ``steps`` and ``completed`` flag.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ shared machinery
    def write_agent(self, spec: dict[str, Any], out_dir: str | Path,
                    name: str = "agent") -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        spec_path = atomic_write_json(out_dir / f"{name}.spec.json", spec).resolve()
        module_path = out_dir / f"{name}.py"
        module_path.write_text(self.agent_module_source(spec_path), encoding="utf-8")
        return module_path.resolve()

    def evaluate(self, spec: dict[str, Any], *, split: str, trials: int, label: str,
                 out_dir: str | Path, task_ids: list[str] | None = None) -> Evaluation:
        """Evaluate a harness on a split, averaging each task's reward over ``trials`` runs.

        Results are cached in ``out_dir``; an interrupted evaluation resumes at the first trial
        without a result.
        """
        out_dir = Path(out_dir)
        cached = out_dir / "evaluation.json"
        if cached.is_file():
            return Evaluation.from_json(load_json(cached))
        ids = [str(task_id) for task_id in (task_ids or self.task_ids(split))]
        agent = self.write_agent(spec, out_dir)
        runs: list[dict[str, Any]] = []
        for trial in range(1, max(1, int(trials)) + 1):
            trial_dir = out_dir / f"trial{trial}"
            trial_path = trial_dir / "trial.json"
            if trial_path.is_file():
                runs.append(load_json(trial_path))
                continue
            result = self.run_trial(agent, split=split, task_ids=ids,
                                    label=f"{label}-t{trial}", out_dir=trial_dir)
            broken = sorted(str(cell["task_id"]) for cell in result["cells"]
                            if cell.get("infra_error") or not cell.get("scored"))
            if broken:
                raise EvaluationError(f"{label} trial {trial}: {len(broken)} task(s) were not "
                                      f"scored because of infrastructure errors: {broken[:8]}")
            atomic_write_json(trial_path, result)
            runs.append(result)
        per_task: dict[str, list[float]] = {}
        for run in runs:
            for cell in run["cells"]:
                if cell.get("scored"):
                    per_task.setdefault(str(cell["task_id"]), []).append(float(cell["reward"]))
        cells = [{"task_id": task_id, "reward": sum(values) / len(values), "scored": True,
                  "trials": len(values), "trial_rewards": values}
                 for task_id, values in sorted(per_task.items())]
        trial_means = []
        for run in runs:
            scored = [float(cell["reward"]) for cell in run["cells"] if cell.get("scored")]
            trial_means.append(round(sum(scored) / len(scored), 6) if scored else 0.0)
        evaluation = Evaluation(label=label, split=split, cells=cells,
                                statuses=list(runs[0].get("statuses") or []) if runs else [],
                                trials=len(runs), trial_means=trial_means)
        atomic_write_json(cached, evaluation.to_json())
        return evaluation
