"""Persistent artifacts: built once from the evolution set, then frozen.

Mutations such as skill libraries or episodic memories construct an artifact from successful
trajectories in D_evo through the optional ``fit``/``save`` phase. The artifact is saved with its
source-task provenance and stays fixed while harnesses are evaluated on D_val and D_test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from scholarevolve.harness.interfaces import EpisodeResult
from scholarevolve.harness.modules import supports_offline_fit
from scholarevolve.harness.runtime import EVOLUTION_SPLIT

#: Provenance lives beside the artifact so that the module owns the artifact's namespace.
_SIDECAR_SUFFIX = ".provenance.json"


def provenance_path(artifact_path: str | Path) -> Path:
    path = Path(artifact_path)
    return path.with_name(path.name + _SIDECAR_SUFFIX)


def read_provenance(artifact_path: str | Path) -> dict[str, Any]:
    sidecar = provenance_path(artifact_path)
    try:
        stamp = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return stamp if isinstance(stamp, dict) else {}


def evolution_episodes(statuses: list[dict[str, Any]], *, rewards: dict[str, float],
                       evolution_ids: list[str],
                       solved_only: bool = True) -> list[EpisodeResult]:
    """Turn evaluated D_evo tasks into the episodes a module's ``fit`` may see.

    ``statuses`` are per-task records from an environment evaluation; each carries the task's
    instruction and trajectory. Only successful episodes are offered by default.
    """
    allowed = {str(task_id) for task_id in evolution_ids}
    episodes: list[EpisodeResult] = []
    for status in statuses:
        task_id = str(status.get("task_id"))
        if task_id not in allowed:
            continue
        reward = float(rewards.get(task_id, 0.0))
        if solved_only and reward <= 0.0:
            continue
        episodes.append(EpisodeResult(
            task_id=task_id, split=EVOLUTION_SPLIT, reward=reward,
            steps=int(status.get("steps") or 0), completed=bool(status.get("completed")),
            trajectory=tuple(status.get("trajectory") or ()),
            metadata={"instruction": str(status.get("instruction") or "")}))
    return episodes


def build_artifact(module: Any, episodes: list[EpisodeResult], *, name: str,
                   out_path: str | Path,
                   provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run one module's offline phase over D_evo episodes and freeze the result."""
    if not supports_offline_fit(module):
        raise TypeError(f"{type(module).__name__} ({name}) has no offline phase")
    foreign = sorted({episode.task_id for episode in episodes
                      if episode.split != EVOLUTION_SPLIT})
    if foreign:
        raise ValueError(f"refusing to fit {name} on episodes outside the evolution set: "
                         f"{foreign[:8]}")
    module.fit(episodes=episodes)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    returned = module.save(str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {"artifact": payload}
    except (OSError, json.JSONDecodeError):
        payload = dict(returned) if isinstance(returned, dict) else {}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
    stamp = {
        "source_split": EVOLUTION_SPLIT,
        "frozen": True,
        "built_by": type(module).__name__,
        "module": name,
        "n_episodes": len(episodes),
        "task_ids": sorted({episode.task_id for episode in episodes}),
        **dict(provenance or {}),
    }
    provenance_path(path).write_text(json.dumps(stamp, indent=2, ensure_ascii=False, default=str),
                                     encoding="utf-8")
    return payload


def assert_disjoint(artifact_path: str | Path, evaluation_ids: list[str], *,
                    group_of: Callable[[str], str]) -> None:
    """Refuse an artifact built from tasks that share a group (e.g. a scenario) with evaluation."""
    covered = read_provenance(artifact_path).get("task_ids") or []
    built = {group_of(str(task_id)) for task_id in covered}
    evaluated = {group_of(str(task_id)) for task_id in evaluation_ids}
    shared = sorted(built & evaluated)
    if shared:
        raise ValueError(f"artifact {artifact_path} shares {len(shared)} task group(s) with the "
                         f"evaluation set: {shared[:8]}")
