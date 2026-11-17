"""Module crossover: rank configurations by additive predicted gain, then evaluate a shortlist.

A configuration c = (c_1, ..., c_5) chooses one implementation per module and may keep the base
module (c_j = 0, with zero gain). Its additive predicted gain on D_val is

    predicted(c) = 1/|D_val| * sum_i sum_j g_{j, c_j}(i),

computed from the individual mutation measurements without further rollouts. Shortlisted
configurations are assembled as complete harnesses and evaluated; the residual between the observed
and predicted gain captures module interactions, reward ceilings and evaluation variation.
"""

from __future__ import annotations

import itertools
from typing import Any

from scholarevolve.harness.modules import MODULE_NAMES
from scholarevolve.search.stats import mean

GainVector = dict[str, float]


def predicted_vector(vectors: list[GainVector]) -> GainVector:
    tasks = sorted(set().union(*(set(vector) for vector in vectors))) if vectors else []
    return {task: round(sum(vector.get(task, 0.0) for vector in vectors), 6) for task in tasks}


def enumerate_configurations(candidates: dict[str, list[str]], gains: dict[str, GainVector], *,
                             min_changed: int = 2) -> list[dict[str, Any]]:
    """Every configuration of at least ``min_changed`` mutated modules, best predicted first.

    ``candidates`` maps each module to the names of its evaluated mutations; ``gains`` maps each
    mutation name to its per-task gain vector against the base harness.
    """
    modules = sorted(module for module, names in candidates.items() if names)
    options = [[None, *candidates[module]] for module in modules]
    rows: list[dict[str, Any]] = []
    for choice in itertools.product(*options):
        chosen = [(module, name) for module, name in zip(modules, choice) if name is not None]
        if len(chosen) < min_changed:
            continue
        vector = predicted_vector([gains[name] for _, name in chosen])
        rows.append({"members": dict(chosen), "predicted_vector": vector,
                     "predicted_gain": mean(vector),
                     "member_gains": {module: mean(gains[name]) for module, name in chosen}})
    rows.sort(key=lambda row: (-row["predicted_gain"], len(row["members"]),
                               sorted(row["members"].items())))
    return rows


def shortlist(rows: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    return rows[:max(0, int(budget))]


def residual(observed: GainVector, predicted: GainVector) -> dict[str, float]:
    """Observed minus additively predicted gain, epsilon(c)."""
    observed_gain, predicted_gain = mean(observed), mean(predicted)
    return {"observed_gain": observed_gain, "predicted_gain": predicted_gain,
            "residual": round(observed_gain - predicted_gain, 6)}


def configuration_name(members: dict[str, str]) -> str:
    return "__".join(members[module] for module in MODULE_NAMES if module in members)
