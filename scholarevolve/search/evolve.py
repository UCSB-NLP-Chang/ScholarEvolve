"""Generational harness search: research-guided mutation, module crossover, selection, retention.

At generation g the retained champion H_g is the base harness:

1. H_g runs on D_evo; its failed episodes are audited and abstracted into a capability brief.
2. Module-specific queries, retrieval, topic modeling and round-robin allocation select papers.
3. Each selected paper becomes a mutation H_j^(p) = H_g[j <- m_j^(p)] through full-text reading,
   official-code reading, a blueprint and the coding agent; persistent artifacts are built from
   successful D_evo episodes and frozen.
4. Every mutation is evaluated against H_g on the same D_val tasks and trials, giving per-task
   gains g_{j,p}(i).
5. Crossover ranks configurations by additive predicted gain, evaluates a shortlist as complete
   harnesses, and records each configuration's residual.
6. The evaluated candidate with the highest mean D_val reward replaces H_g only if the lower end of
   its paired bootstrap gain interval is above zero.

The search stops after ``stop_after`` consecutive generations without a confirmed improvement, or
when the generation budget is used up.
"""

from __future__ import annotations

import copy
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable

from scholarevolve.envs.base import EVOLUTION, VALIDATION, Evaluation
from scholarevolve.harness.artifacts import (
    assert_disjoint, build_artifact, evolution_episodes, provenance_path,
)
from scholarevolve.harness.modules import MODULE_NAMES, MODULE_PREFIX, supports_offline_fit
from scholarevolve.harness.runtime import ModuleRuntime, load_module
from scholarevolve.mutation import build_mutation
from scholarevolve.mutation.coding_agent import CodingSettings
from scholarevolve.research import run_research
from scholarevolve.research.audit import run_audit
from scholarevolve.research.brief import build_brief
from scholarevolve.search.crossover import (
    configuration_name, enumerate_configurations, residual, shortlist,
)
from scholarevolve.search.stats import gain_vector, mean, paired_bootstrap
from scholarevolve.utils import atomic_write_json, load_json

LLM = Callable[..., str]

DEFAULT_BACKBONE_CAPABILITIES = (
    "The backbone is a reasoning language model: it already reasons step by step internally "
    "before every action, so prompting methods whose contribution is to make a non-reasoning "
    "model reason step by step are already provided.")


@dataclass
class SearchSettings:
    papers_per_module: dict[str, int] = field(
        default_factory=lambda: {module: 4 for module in MODULE_NAMES})
    devo_trials: int = 1
    val_trials: int = 3
    crossover_budget: int = 4
    confidence: float = 0.90
    bootstrap_samples: int = 4000
    bootstrap_seed: int = 1234
    stop_after: int = 1
    max_generations: int = 5
    arm_concurrency: int = 1
    coding_concurrency: int = 4
    research_concurrency: int = 5
    brief_modes: int = 6
    max_pool: int | None = None
    topic: dict[str, Any] = field(default_factory=dict)
    read_official_code: bool = True
    code_reader_turns: int = 40
    context_limit_tokens: int = 128_000
    backbone_capabilities: str = DEFAULT_BACKBONE_CAPABILITIES

    def __post_init__(self) -> None:
        if isinstance(self.papers_per_module, int):
            self.papers_per_module = {module: self.papers_per_module for module in MODULE_NAMES}

    @classmethod
    def from_config(cls, value: dict[str, Any]) -> "SearchSettings":
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"unknown search setting(s): {', '.join(unknown)}")
        return cls(**value)


def _section(text: str, name: str, limit: int = 1200) -> str:
    match = re.search(rf"{name}\s*[-:]?\s*(.*?)(?:\n[A-Z][A-Z ]{{6,}}\s*[-:]?|\Z)", text or "",
                      re.DOTALL)
    return re.sub(r"\s+", " ", match.group(1)).strip()[:limit] if match else ""


class Search:
    def __init__(self, *, environment: Any, research_llm: LLM, coding: CodingSettings,
                 settings: SearchSettings, out_dir: str | Path,
                 on_log: Callable[[str], None] = print):
        self.env = environment
        self.llm = research_llm
        self.coding = coding
        self.settings = settings
        self.out_dir = Path(out_dir)
        self.log = on_log

    # ------------------------------------------------------------------ harness specifications
    def base_spec(self) -> dict[str, Any]:
        return {
            "environment": self.env.name,
            "context_limit_tokens": int(self.settings.context_limit_tokens),
            "modules": {module: None for module in MODULE_NAMES},
            "artifacts": {},
            "memory_guard": {"evolution_ids": self.env.task_ids(EVOLUTION), "frozen": True},
            "manifest": {"name": "initial", "members": {}, "papers": {}},
        }

    @staticmethod
    def spec_with(base: dict[str, Any], members: dict[str, dict[str, Any]],
                  artifacts: dict[str, str | None]) -> dict[str, Any]:
        """``base`` with the given candidates installed, one per module."""
        spec = copy.deepcopy(base)
        manifest = spec.setdefault("manifest", {})
        manifest.setdefault("members", {})
        manifest.setdefault("papers", {})
        for module, candidate in members.items():
            spec["modules"][module] = {"module": candidate["module"], "params": {}}
            spec["artifacts"].pop(module, None)
            if artifacts.get(candidate["name"]):
                spec["artifacts"][module] = artifacts[candidate["name"]]
            manifest["members"][module] = candidate["name"]
            manifest["papers"][module] = candidate.get("paper")
        manifest["name"] = configuration_name(manifest["members"]) or "initial"
        return spec

    def current_behaviour(self, spec: dict[str, Any]) -> dict[str, str]:
        """What each module of the given harness does, for query construction and screening."""
        base = self.env.base_behaviour()
        described: dict[str, str] = {}
        for module in MODULE_NAMES:
            entry = spec["modules"].get(module)
            if not entry:
                described[module] = base[module]
                continue
            blueprint_path = Path(entry["module"]).parent / "blueprint.json"
            if not blueprint_path.is_file():
                described[module] = "a custom module implementation."
                continue
            blueprint = load_json(blueprint_path)
            paper = blueprint.get("paper") or {}
            described[module] = (f"implements '{paper.get('title')}' (topic: "
                                 f"{paper.get('topic')}): "
                                 f"{_section(blueprint.get('blueprint', ''), 'SOURCE MECHANISM')}")
        return described

    # ------------------------------------------------------------------ steps
    def _evaluate(self, spec: dict[str, Any], *, split: str, trials: int, label: str,
                  directory: Path) -> Evaluation:
        return self.env.evaluate(spec, split=split, trials=trials, label=label,
                                 out_dir=directory)

    def _artifacts(self, candidates: list[dict[str, Any]], devo: Evaluation,
                   directory: Path) -> dict[str, str | None]:
        """Build and freeze each candidate's artifact from successful D_evo episodes."""
        episodes = evolution_episodes(devo.statuses, rewards=devo.rewards(),
                                      evolution_ids=self.env.task_ids(EVOLUTION))
        validation = self.env.task_ids(VALIDATION)
        built: dict[str, str | None] = {}
        for candidate in candidates:
            module = candidate["modules"][0]
            runtime = ModuleRuntime(model_name=str(self.env.backbone.get("model")),
                                    work_dir=directory / candidate["name"],
                                    environment=self.env.name)
            try:
                loaded = load_module({"module": candidate["module"]}, module, runtime=runtime)
                if not supports_offline_fit(loaded):
                    built[candidate["name"]] = None
                    continue
                path = directory / candidate["name"] / f"{module}.json"
                if not (path.is_file() and provenance_path(path).is_file()):
                    build_artifact(loaded, episodes, name=module, out_path=path,
                                   provenance={"built_from": devo.label})
                assert_disjoint(path, validation, group_of=self.env.group_of)
                built[candidate["name"]] = str(path.resolve())
                self.log(f"  artifact {candidate['name']}: {len(episodes)} episodes")
            except Exception as error:
                self.log(f"  artifact {candidate['name']} failed: {type(error).__name__}: {error}")
        return built

    def generation(self, index: int, champion: dict[str, Any]) -> tuple[dict[str, Any],
                                                                         dict[str, Any] | None]:
        settings = self.settings
        directory = self.out_dir / f"generation_{index}"
        spec = champion["spec"]
        base_val = Evaluation.from_json(load_json(champion["val_evaluation"]))
        base_rewards = base_val.rewards()

        self.log(f"[g{index}] evaluating the champion on D_evo")
        devo = self._evaluate(spec, split=EVOLUTION, trials=settings.devo_trials,
                              label=f"g{index}-champion-devo",
                              directory=directory / "evaluations" / "champion_devo")
        self.log(f"[g{index}] auditing {sum(1 for c in devo.cells if c['reward'] < 1)} failures")
        summary, details = run_audit(devo.cells, devo.statuses, environment=self.env,
                                     llm=self.llm, out_dir=directory / "audit", on_log=self.log)
        brief_path = directory / "brief.json"
        if brief_path.is_file():
            brief = load_json(brief_path)
        else:
            brief = build_brief(summary, details, llm=self.llm, top_k=settings.brief_modes)
            atomic_write_json(brief_path, brief)
        self.log(f"[g{index}] capability gap: {brief.get('core_challenge') or '(none)'}")

        self.log(f"[g{index}] research: queries, retrieval, topics and allocation")
        selected = run_research(
            brief=brief, current_behaviour=self.current_behaviour(spec),
            backbone_capabilities=settings.backbone_capabilities, llm=self.llm,
            out_dir=directory / "research", papers_per_module=settings.papers_per_module,
            max_pool=settings.max_pool, topic_settings=settings.topic,
            concurrency=settings.research_concurrency, on_log=self.log)

        jobs = [(module, number, paper) for module in MODULE_NAMES
                for number, paper in enumerate(selected.get(module) or [], 1)]
        self.log(f"[g{index}] building {len(jobs)} mutations")

        def mutate(job: tuple[str, int, dict[str, Any]]) -> dict[str, Any] | None:
            module, number, paper = job
            name = f"{MODULE_PREFIX[module]}-p{number}"
            try:
                return build_mutation(
                    name=name, module=module, paper=paper, out_dir=directory / "mutations",
                    current_spec_entry=spec["modules"].get(module), environment=self.env,
                    brief=brief, research_llm=self.llm, coding=self.coding,
                    read_official_code=settings.read_official_code,
                    code_reader_turns=settings.code_reader_turns, on_log=self.log)
            except Exception as error:
                self.log(f"  {name} failed: {type(error).__name__}: {error}")
                return None

        with ThreadPoolExecutor(max_workers=max(1, settings.coding_concurrency)) as pool:
            manifests = [m for m in pool.map(mutate, jobs) if m and m.get("ok")]
        artifacts = self._artifacts(manifests, devo, directory / "artifacts")
        usable = [m for m in manifests if m["name"] in artifacts]
        self.log(f"[g{index}] {len(usable)} executable mutations")

        arms: dict[str, dict[str, Any]] = {}

        def evaluate_arm(name: str, members: dict[str, dict[str, Any]]) -> None:
            arm_spec = self.spec_with(spec, members, artifacts)
            evaluation = self._evaluate(arm_spec, split=VALIDATION, trials=settings.val_trials,
                                        label=f"g{index}-{name}-val",
                                        directory=directory / "evaluations" / name)
            arms[name] = {"spec": arm_spec, "evaluation": evaluation,
                          "members": {module: m["name"] for module, m in members.items()},
                          "gain": gain_vector(base_rewards, evaluation.rewards()),
                          "path": str(directory / "evaluations" / name / "evaluation.json")}
            self.log(f"  {name}: D_val mean {evaluation.mean:.4f} "
                     f"(gain {mean(arms[name]['gain']):+.4f})")

        def run_arms(items: list[tuple[str, dict[str, dict[str, Any]]]]) -> None:
            def guarded(item: tuple[str, dict[str, dict[str, Any]]]) -> None:
                try:
                    evaluate_arm(*item)
                except Exception as error:
                    self.log(f"  {item[0]} evaluation failed: {type(error).__name__}: {error}")

            with ThreadPoolExecutor(max_workers=max(1, settings.arm_concurrency)) as pool:
                list(pool.map(guarded, items))

        run_arms([(m["name"], {m["modules"][0]: m}) for m in usable])
        by_name = {m["name"]: m for m in usable}
        mutation_arms = {name: arm for name, arm in arms.items() if name in by_name}
        candidates_by_module = {module: [name for name, arm in mutation_arms.items()
                                         if module in arm["members"]] for module in MODULE_NAMES}
        ranked = enumerate_configurations(candidates_by_module,
                                          {name: arm["gain"] for name, arm in
                                           mutation_arms.items()})
        chosen = shortlist(ranked, settings.crossover_budget)
        self.log(f"[g{index}] crossover: {len(ranked)} configurations, evaluating {len(chosen)}")
        run_arms([(configuration_name(row["members"]),
                   {module: by_name[name] for module, name in row["members"].items()})
                  for row in chosen])
        crossover = []
        for row in chosen:
            name = configuration_name(row["members"])
            if name in arms:
                crossover.append({"name": name, "members": row["members"],
                                  **residual(arms[name]["gain"], row["predicted_vector"])})

        decision: dict[str, Any] = {
            "generation": index, "champion": spec["manifest"].get("name"),
            "champion_val_mean": base_val.mean,
            "mutations": {name: {"members": arm["members"], "val_mean": arm["evaluation"].mean,
                                 "gain": mean(arm["gain"])}
                          for name, arm in arms.items() if name in by_name},
            "crossover": crossover, "best": None, "retained": False,
        }
        new_champion = None
        if arms:
            best = max(arms, key=lambda name: arms[name]["evaluation"].mean)
            test = paired_bootstrap(base_rewards, arms[best]["evaluation"].rewards(),
                                    confidence=settings.confidence,
                                    samples=settings.bootstrap_samples,
                                    seed=settings.bootstrap_seed)
            decision.update(best=best, best_val_mean=arms[best]["evaluation"].mean, test=test,
                            retained=bool(test["lower"] > 0))
            if decision["retained"]:
                new_champion = {"spec": arms[best]["spec"], "val_evaluation": arms[best]["path"],
                                "name": f"g{index}-{best}", "generation": index}
        atomic_write_json(directory / "decision.json", decision)
        verdict = (f"new champion {decision['best']}" if decision["retained"]
                   else "champion retained")
        self.log(f"[g{index}] {verdict}")
        return decision, new_champion

    # ------------------------------------------------------------------ search loop
    def run(self) -> dict[str, Any]:
        state_path = self.out_dir / "search_state.json"
        if state_path.is_file():
            state = load_json(state_path)
        else:
            spec = self.base_spec()
            self.log("[g0] evaluating the initial harness on D_val")
            self._evaluate(spec, split=VALIDATION, trials=self.settings.val_trials,
                           label="g0-initial-val",
                           directory=self.out_dir / "initial" / "evaluation_val")
            state = {"next_generation": 0, "stale": 0, "history": [],
                     "champion": {"spec": spec, "name": "initial", "generation": None,
                                  "val_evaluation": str(self.out_dir / "initial" /
                                                        "evaluation_val" / "evaluation.json")}}
            atomic_write_json(state_path, state)
        while (state["next_generation"] < self.settings.max_generations
               and state["stale"] < self.settings.stop_after):
            index = state["next_generation"]
            decision, new_champion = self.generation(index, state["champion"])
            state["history"].append({key: decision.get(key) for key in
                                     ("generation", "best", "best_val_mean",
                                      "champion_val_mean", "retained")})
            if new_champion is not None:
                state["champion"] = new_champion
                state["stale"] = 0
            else:
                state["stale"] += 1
            state["next_generation"] = index + 1
            atomic_write_json(state_path, state)
        export_champion(state["champion"], self.out_dir / "champion")
        self.log(f"final champion: {state['champion']['name']} -> {self.out_dir / 'champion'}")
        return state


def export_champion(champion: dict[str, Any], destination: str | Path) -> Path:
    """Copy the champion's modules and frozen artifacts into a self-contained directory."""
    destination = Path(destination)
    spec = copy.deepcopy(champion["spec"])
    for module, entry in (spec.get("modules") or {}).items():
        if entry and str(entry.get("module", "")).endswith(".py"):
            source = Path(entry["module"])
            target = destination / "modules" / f"{module}.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            entry["module"] = str(target.resolve())
            blueprint = source.parent / "BLUEPRINT.md"
            if blueprint.is_file():
                (destination / "blueprints").mkdir(parents=True, exist_ok=True)
                shutil.copyfile(blueprint, destination / "blueprints" / f"{module}.md")
    for module, path in list((spec.get("artifacts") or {}).items()):
        if not path:
            continue
        target = destination / "artifacts" / f"{module}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if provenance_path(path).is_file():
            shutil.copyfile(provenance_path(path), provenance_path(target))
        spec["artifacts"][module] = str(target.resolve())
    atomic_write_json(destination / "spec.json", spec)
    atomic_write_json(destination / "champion.json", {
        "name": champion.get("name"), "generation": champion.get("generation"),
        "members": spec.get("manifest", {}).get("members", {}),
        "papers": spec.get("manifest", {}).get("papers", {})})
    return destination
