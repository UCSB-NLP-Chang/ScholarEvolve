"""Paired per-task gains and the paired bootstrap used by the retention rule."""

from __future__ import annotations

import random
from typing import Any


def gain_vector(base: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    """g(i) = mean reward of the candidate on task i minus that of the base, on shared tasks."""
    shared = sorted(set(base) & set(candidate))
    return {task: round(candidate[task] - base[task], 6) for task in shared}


def mean(values: dict[str, float]) -> float:
    return round(sum(values.values()) / len(values), 6) if values else 0.0


def paired_bootstrap(base: dict[str, float], candidate: dict[str, float], *,
                     confidence: float = 0.90, samples: int = 4000,
                     seed: int = 1234) -> dict[str, Any]:
    """Mean paired gain and its bootstrap interval, resampling tasks with replacement."""
    if set(base) != set(candidate):
        missing = sorted(set(base) ^ set(candidate))
        raise ValueError(f"paired comparison needs the same tasks on both sides; "
                         f"{len(missing)} differ, e.g. {missing[:5]}")
    gains = [candidate[task] - base[task] for task in sorted(base)]
    if not gains:
        raise ValueError("paired comparison over an empty task set")
    rng = random.Random(seed)
    n = len(gains)
    means = sorted(sum(gains[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    alpha = (1.0 - confidence) / 2.0
    lower = means[max(0, min(samples - 1, int(alpha * samples)))]
    upper = means[max(0, min(samples - 1, int((1.0 - alpha) * samples) - 1))]
    wins = sum(gain > 0 for gain in gains)
    losses = sum(gain < 0 for gain in gains)
    return {"delta": round(sum(gains) / n, 6), "lower": round(lower, 6),
            "upper": round(upper, 6), "confidence": confidence, "n": n, "wins": wins,
            "losses": losses, "ties": n - wins - losses}
