"""Environment adapters.

The search procedure is the same for every environment. An adapter supplies what differs: the task
splits, how a model reply becomes an action (the dialect), how the initial harness behaves in each
module, how trajectories are rendered for the failure audit, the health-probe fixtures, and the
episode runner that evaluates a harness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

ENVIRONMENTS = ("appworld", "tau2")


def get_dialect(name: str) -> Any:
    if name == "appworld":
        from scholarevolve.envs.appworld.dialect import AppWorldDialect

        return AppWorldDialect()
    if name == "tau2":
        from scholarevolve.envs.tau2.dialect import Tau2Dialect

        return Tau2Dialect()
    raise ValueError(f"unknown environment {name!r}; known: {', '.join(ENVIRONMENTS)}")


def get_environment(name: str, settings: dict[str, Any], *, backbone: dict[str, Any],
                    work_dir: str | Path) -> Any:
    if name == "appworld":
        from scholarevolve.envs.appworld import AppWorldEnvironment

        return AppWorldEnvironment(settings, backbone=backbone, work_dir=work_dir)
    if name == "tau2":
        from scholarevolve.envs.tau2 import Tau2Environment

        return Tau2Environment(settings, backbone=backbone, work_dir=work_dir)
    raise ValueError(f"unknown environment {name!r}; known: {', '.join(ENVIRONMENTS)}")
