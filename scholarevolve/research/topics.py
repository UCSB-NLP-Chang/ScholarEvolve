"""Mechanism-level topic modeling of a module's paper pool (TopicGPT-style).

1. Taxonomy induction. Batches of titles and abstracts are read alongside the taxonomy accumulated
   so far; the model reuses a topic name when its mechanism fits a paper and adds a topic when a
   relevant mechanism is still uncovered.
2. Refinement. Paraphrases and overlapping topics are merged, overly specific variants absorbed,
   and vague or irrelevant topics removed, standardising mechanism granularity.
3. Assignment. Each paper receives an ordered list of topics from the fixed taxonomy, primary
   methodological contribution first, each supported by an excerpt of the abstract. The first
   retained label is the paper's selection cluster.

Every stage is conditioned on the module's responsibility.
"""

from __future__ import annotations

import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from scholarevolve.harness.modules import MODULE_RESPONSIBILITIES
from scholarevolve.utils import extract_json_array, extract_json_object

LLM = Callable[..., str]

SCOPE = ("only papers about LLM / language-model agents that contribute a concrete, implementable "
         "mechanism; out of scope: surveys, standalone benchmarks and datasets, and unrelated uses "
         "of the same vocabulary (for example reinforcement-learning skill discovery, computer "
         "vision, or mathematics)")

_CHARS_PER_TOKEN = 3.5


def module_problem(module: str) -> tuple[str, str]:
    """The goal statement topic modeling is conditioned on, for one module."""
    challenge = MODULE_RESPONSIBILITIES[module]
    summary = (f"An LLM agent harness is split into five replaceable modules (tool interface, "
               f"context management, skills, memories, agentic workflows). We catalogue the "
               f"mechanisms that could be implemented in the {module} module. The catalogue "
               "decides which papers to implement, so topics must separate papers by the "
               "MECHANISM they would contribute to this module.")
    return challenge, summary


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _paper_listing(papers: list[dict[str, Any]]) -> str:
    return "\n".join(f"{index + 1}. {paper.get('title', '?')}\n   {paper.get('abstract', '')}"
                     for index, paper in enumerate(papers))


def _taxonomy_listing(topics: list[dict[str, Any]], *, with_count: bool = False) -> str:
    if not topics:
        return "(none yet)"
    lines = []
    for index, topic in enumerate(topics, 1):
        line = f"[{index}] {topic['name']} - {topic.get('description', '')}"
        if with_count and topic.get("count"):
            line += f"  (proposed in {topic['count']} batches)"
        lines.append(line)
    return "\n".join(lines)


def _batches(papers: list[dict[str, Any]], batch_size: int, max_batch_tokens: int | None):
    """Pack batches by token budget, since abstract lengths vary widely."""
    if not max_batch_tokens:
        for start in range(0, len(papers), batch_size):
            yield papers[start:start + batch_size]
        return
    current: list[dict[str, Any]] = []
    used = 0
    for paper in papers:
        cost = int((len(str(paper.get("title") or "")) + len(str(paper.get("abstract") or "")))
                   / _CHARS_PER_TOKEN) + 8
        if current and (used + cost > max_batch_tokens or len(current) >= batch_size):
            yield current
            current, used = [], 0
        current.append(paper)
        used += cost
    if current:
        yield current


_GENERATE_SYSTEM = (
    "You induce a taxonomy of METHODOLOGICAL topics for research papers, in service of a specific "
    "engineering problem. Topics name the technique or mechanism family, not the application "
    "domain or the problem itself. They must be ORTHOGONAL - mutually exclusive with minimal "
    "semantic overlap - general, and consistently granular. Return only valid JSON.")


def generate_taxonomy(papers: list[dict[str, Any]], challenge: str, summary: str, llm: LLM, *,
                      batch_size: int = 2000, max_batch_tokens: int | None = 180_000,
                      scope: str = SCOPE, on_log=print) -> tuple[list[dict[str, Any]], int]:
    topics: list[dict[str, Any]] = []
    index: dict[str, dict[str, Any]] = {}
    calls = 0
    for number, chunk in enumerate(_batches(papers, batch_size, max_batch_tokens), 1):
        prompt = f"""GOAL PROBLEM (the mechanisms we catalogue should help solve this):
- module responsibility: {challenge}
- context: {summary}

SCOPE - {scope}

You are inducing a taxonomy of MECHANISM topics one batch at a time.

Topics discovered SO FAR (REUSE these exact names when a paper fits one; only add a NEW topic when
none of these fits):
{_taxonomy_listing(topics)}

New batch of papers (title + abstract):
{_paper_listing(chunk)}

Task:
- List the mechanism topics PRESENT in this batch that are relevant to the goal problem.
- Reuse an existing topic name VERBATIM where it applies.
- Add a new topic ONLY if no existing one fits. Keep every topic ORTHOGONAL to the others, general,
  and about the MECHANISM (2-5 word name + one-sentence description).
- Ignore papers that are irrelevant or out of scope.

Return ONLY valid JSON:
{{"categories": [{{"name": "<2-5 words>", "description": "<one sentence>", "new": true|false}}]}}"""
        try:
            raw = llm([{"role": "system", "content": _GENERATE_SYSTEM},
                       {"role": "user", "content": prompt}], temperature=0.3)
            calls += 1
            for category in extract_json_object(raw).get("categories", []):
                name = str(category.get("name") or "").strip()
                key = _norm(name)
                if not key:
                    continue
                if key in index:
                    index[key]["count"] += 1
                else:
                    topic = {"name": name,
                             "description": str(category.get("description") or "").strip(),
                             "count": 1}
                    index[key] = topic
                    topics.append(topic)
        except Exception as error:
            on_log(f"  [topics] generation batch {number} ({len(chunk)} papers) skipped: {error}")
    on_log(f"  [topics] {len(topics)} raw topics from {calls} batches")
    return topics, calls


_REFINE_SYSTEM = (
    "You consolidate a taxonomy of methodological topics into a clean, ORTHOGONAL set. Merge "
    "paraphrases and semantically overlapping topics, absorb over-narrow topics into a broader "
    "parent, drop vague or off-topic ones, and standardise names and descriptions to a consistent "
    "granularity. Return only valid JSON.")


def refine_taxonomy(topics: list[dict[str, Any]], challenge: str, llm: LLM, *,
                    target_min: int = 5, target_max: int = 15,
                    on_log=print) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    if not topics:
        return [], [], 0
    prompt = f"""GOAL PROBLEM (module responsibility): {challenge}

Raw taxonomy of mechanism topics (name - description - how many batches proposed it):
{_taxonomy_listing(topics, with_count=True)}

Consolidate it:
- MERGE topics that are paraphrases or semantically overlap into one topic (choose the clearest
  general name and combine their descriptions).
- ABSORB overly fine-grained topics into a broader parent topic.
- DROP vague topics and topics that are not about a mechanism relevant to the goal problem.
- STANDARDISE: 2-5 word names, one-sentence descriptions, consistent granularity.
- Keep the set ORTHOGONAL (mutually exclusive). Aim for about {target_min}-{target_max} topics.

Return ONLY valid JSON:
{{"categories": [{{"name": "...", "description": "..."}}],
  "merges": [{{"merged": ["<old name>", "..."], "into": "<final name>"}}]}}"""
    raw_topics = [{"name": t["name"], "description": t.get("description", "")} for t in topics]
    try:
        value = extract_json_object(llm([{"role": "system", "content": _REFINE_SYSTEM},
                                         {"role": "user", "content": prompt}], temperature=0.2))
    except Exception as error:
        on_log(f"  [topics] refinement failed ({error}); keeping the raw taxonomy")
        return raw_topics, [], 1
    refined = [{"name": str(c.get("name") or "").strip(),
                "description": str(c.get("description") or "").strip()}
               for c in value.get("categories", []) if str(c.get("name") or "").strip()]
    on_log(f"  [topics] refined {len(topics)} -> {len(refined or raw_topics)} topics")
    return (refined or raw_topics), list(value.get("merges") or []), 1


_ASSIGN_SYSTEM = (
    "You assign research papers to methodological topics from a FIXED, orthogonal taxonomy. "
    "Assignment is multi-label and ranked from most to least relevant, so the paper's primary "
    "methodological contribution comes first. Assign a topic only when the abstract clearly "
    "supports it, and justify each with a verbatim excerpt from that paper's abstract. If no "
    "topic fits, return an empty label list. Return only valid JSON.")


def assign_topics(papers: list[dict[str, Any]], taxonomy: list[dict[str, Any]], challenge: str,
                  llm: LLM, *, batch_size: int = 10, scope: str = SCOPE, max_workers: int = 8,
                  on_log=print) -> int:
    """Annotate papers in place with ``topics`` (ranked) and ``topic`` (the primary label)."""
    valid = {_norm(topic["name"]): topic["name"] for topic in taxonomy}
    listing = _taxonomy_listing(taxonomy)

    def run(start: int) -> int:
        chunk = papers[start:start + batch_size]
        prompt = f"""GOAL PROBLEM (module responsibility): {challenge}

TAXONOMY - assign ONLY these topics, using their EXACT names:
{listing}

Papers:
{_paper_listing(chunk)}

For EACH paper:
- Assign EVERY topic whose mechanism the paper actually contributes or uses.
- RANK the assigned topics from MOST to LEAST relevant (primary contribution first).
- For each assigned topic give a short VERBATIM excerpt from THAT paper's abstract.
- Out of scope ({scope}): return "labels": [] for that paper.
- If no topic applies, return "labels": [] for that paper.

Return ONLY valid JSON (labels ranked most relevant first):
[{{"n": <paper number>, "labels": [{{"topic": "<exact topic name>", "quote": "<verbatim excerpt>"}}]}}]"""
        try:
            raw = llm([{"role": "system", "content": _ASSIGN_SYSTEM},
                       {"role": "user", "content": prompt}], temperature=0)
        except Exception as error:
            on_log(f"  [topics] assignment batch at {start} skipped: {error}")
            return 0
        by_number: dict[int, list[Any]] = {}
        for item in extract_json_array(raw):
            try:
                by_number[int(item.get("n", 0)) - 1] = list(item.get("labels") or [])
            except (AttributeError, TypeError, ValueError):
                continue
        for offset, paper in enumerate(chunk):
            labels = []
            for label in by_number.get(offset, []):
                name = valid.get(_norm(str(label.get("topic") or "")))
                if name:
                    labels.append({"topic": name, "quote": str(label.get("quote") or "").strip()})
            if labels:
                paper["topics"] = labels
                paper["topic"] = labels[0]["topic"]
        return 1

    starts = list(range(0, len(papers), batch_size))
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        calls = sum(pool.map(run, starts))
    assigned = sum(1 for paper in papers if paper.get("topic"))
    on_log(f"  [topics] {assigned}/{len(papers)} papers assigned ({calls} calls)")
    return calls


def build_clusters(papers: list[dict[str, Any]],
                   taxonomy: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Papers grouped under their primary topic, in retrieval order; empty topics dropped."""
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for paper in papers:
        if paper.get("topic"):
            clusters[paper["topic"]].append(paper)
    ordered = {topic["name"]: clusters[topic["name"]] for topic in taxonomy
               if clusters.get(topic["name"])}
    for name, members in clusters.items():
        ordered.setdefault(name, members)
    return ordered


def topic_model(papers: list[dict[str, Any]], module: str, llm: LLM, *,
                generation_batch: int = 2000, generation_batch_tokens: int | None = 180_000,
                assignment_batch: int = 10, on_log=print) -> dict[str, Any]:
    """Taxonomy induction, refinement and evidence-supported assignment for one module."""
    if not papers:
        return {"taxonomy": [], "clusters": {}, "merges": [], "n_calls": 0}
    challenge, summary = module_problem(module)
    raw, generation_calls = generate_taxonomy(papers, challenge, summary, llm,
                                              batch_size=generation_batch,
                                              max_batch_tokens=generation_batch_tokens,
                                              on_log=on_log)
    taxonomy, merges, refine_calls = refine_taxonomy(raw, challenge, llm, on_log=on_log)
    assign_calls = assign_topics(papers, taxonomy, challenge, llm, batch_size=assignment_batch,
                                 on_log=on_log)
    clusters = build_clusters(papers, taxonomy)
    return {"taxonomy": taxonomy, "clusters": clusters, "merges": merges,
            "n_calls": generation_calls + refine_calls + assign_calls}
