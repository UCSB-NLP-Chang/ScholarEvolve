"""Retrieval and pool construction for one module.

Papers are retrieved from arXiv with the module's query phrases. Within the module, results are
deduplicated by normalised title and entries without an abstract are excluded. Retrieval order is
preserved because candidate allocation follows it within each topic cluster. A paper can appear in
several module pools.
"""

from __future__ import annotations

from typing import Any

from scholarevolve.research import arxiv
from scholarevolve.utils import normalize_title


def build_pool(queries: list[str], *, max_results: int | None = None,
               categories: tuple[str, ...] = arxiv.DEFAULT_CATEGORIES,
               on_log=print) -> list[dict[str, Any]]:
    papers = arxiv.search(queries, categories=categories, max_results=max_results, on_log=on_log)
    seen: set[str] = set()
    pool: list[dict[str, Any]] = []
    for paper in papers:
        title = normalize_title(paper.get("title") or "")
        if not title or title in seen or not str(paper.get("abstract") or "").strip():
            continue
        seen.add(title)
        pool.append(paper)
    return pool
