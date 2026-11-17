"""Research pool construction: from audited failures to the papers selected for each module.

    failure audit -> capability brief -> module-specific queries -> arXiv retrieval per module
    -> topic modeling per module -> mechanism screening -> round-robin allocation
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from scholarevolve.harness.modules import MODULE_NAMES
from scholarevolve.research.pool import build_pool
from scholarevolve.research.queries import build_queries
from scholarevolve.research.selection import allocate
from scholarevolve.research.topics import topic_model
from scholarevolve.utils import atomic_write_json, load_json

LLM = Callable[..., str]

_PAPER_FIELDS = ("arxiv_id", "title", "abstract", "url", "published", "topic", "topics")


def _slim(paper: dict[str, Any]) -> dict[str, Any]:
    return {key: paper[key] for key in _PAPER_FIELDS if key in paper}


def run_research(*, brief: dict[str, Any] | None, current_behaviour: dict[str, str],
                 backbone_capabilities: str, llm: LLM, out_dir: str | Path,
                 papers_per_module: dict[str, int], modules: tuple[str, ...] = MODULE_NAMES,
                 max_pool: int | None = None, topic_settings: dict[str, Any] | None = None,
                 concurrency: int = 5, on_log=print) -> dict[str, list[dict[str, Any]]]:
    """Select papers for every module; each stage is cached in ``out_dir``."""
    out_dir = Path(out_dir)
    queries_path = out_dir / "queries.json"
    if queries_path.is_file():
        queries = load_json(queries_path)
    else:
        queries = build_queries(brief=brief, current_behaviour=current_behaviour, llm=llm,
                                modules=modules, concurrency=concurrency)
        atomic_write_json(queries_path, queries)
    for module in modules:
        on_log(f"  [queries] {module}: {'; '.join(queries[module]['queries'])}")

    def one(module: str) -> tuple[str, list[dict[str, Any]]]:
        selection_path = out_dir / "selection" / f"{module}.json"
        if selection_path.is_file():
            return module, load_json(selection_path)["picked"]
        pool_path = out_dir / "pools" / f"{module}.json"
        if pool_path.is_file():
            papers = load_json(pool_path)
        else:
            papers = build_pool(queries[module]["queries"], max_results=max_pool,
                                on_log=lambda message: on_log(f"[{module}]{message}"))
            atomic_write_json(pool_path, papers)
        on_log(f"  [pool] {module}: {len(papers)} papers")
        model = topic_model(papers, module, llm,
                            on_log=lambda message: on_log(f"[{module}]{message}"),
                            **dict(topic_settings or {}))
        decision = allocate(module, model, budget=int(papers_per_module.get(module, 0)),
                            current_behaviour=current_behaviour[module],
                            backbone_capabilities=backbone_capabilities, llm=llm,
                            on_log=on_log)
        atomic_write_json(out_dir / "topics" / f"{module}.json", {
            "taxonomy": model["taxonomy"], "merges": model["merges"],
            "screen": decision["screen"], "subsumed": decision["subsumed"],
            "clusters": {name: [_slim(paper) for paper in members]
                         for name, members in model["clusters"].items()},
            "unassigned": sum(1 for paper in papers if not paper.get("topic")),
        })
        picked = [_slim(paper) for paper in decision["picked"]]
        atomic_write_json(selection_path, {"module": module, "picked": picked})
        return module, picked

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        selected = dict(pool.map(one, modules))
    for module in modules:
        for index, paper in enumerate(selected[module], 1):
            on_log(f"  [select] {module} {index}: [{paper.get('topic')}] {paper.get('title')}")
    return selected
