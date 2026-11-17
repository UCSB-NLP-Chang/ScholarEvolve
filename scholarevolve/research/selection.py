"""Mechanism screening and candidate allocation.

Each topic's mechanism is compared with a description of the current module and the backbone's
capabilities; topics already subsumed by the current harness are removed from allocation, with the
reason recorded. The remaining nonempty clusters are ordered by decreasing size and papers are
selected in round-robin order, visiting every cluster before returning to one, until the module's
budget K_j is spent or the eligible pool is exhausted. Retrieval order is preserved within a
cluster, so a budget of K_j papers covers min(K_j, |T_j^+|) distinct topics.
"""

from __future__ import annotations

from typing import Any, Callable

from scholarevolve.utils import extract_json_object

LLM = Callable[..., str]


def screen_topics(module: str, taxonomy: list[dict[str, Any]], *, current_behaviour: str,
                  backbone_capabilities: str, llm: LLM,
                  on_log=print) -> dict[str, dict[str, Any]]:
    """Whether each topic would add a mechanism the current harness and backbone lack."""
    if not taxonomy:
        return {}
    listing = "\n".join(f"[{index}] {topic['name']} - {topic.get('description', '')}"
                        for index, topic in enumerate(taxonomy))
    prompt = (
        f"MODULE: {module}\nWHAT THE CURRENT HARNESS ALREADY DOES IN THIS MODULE: "
        f"{current_behaviour}\nBACKBONE: {backbone_capabilities}\n\n"
        "Below are mechanism topics induced from the literature for this module. For each, decide "
        "whether implementing it would ADD a mechanism the current harness does not already "
        "have.\n"
        "  novel=false if the current harness (or the backbone) already provides it, so a "
        "mutation implementing it would behave like the current harness;\n"
        "  novel=true if it introduces machinery that is absent today.\n"
        "Judge the MECHANISM, not the publication date.\n\n"
        "Return ONLY JSON: {\"categories\": [{\"name\": \"<exact name>\", \"novel\": true|false, "
        f"\"reason\": \"<short>\"}}]}}\n\n{listing}")
    try:
        value = extract_json_object(llm(
            [{"role": "system", "content": "You judge whether a mechanism is already implemented. "
                                           "Reply with JSON only."},
             {"role": "user", "content": prompt}], temperature=0))
    except Exception as error:
        on_log(f"  [screen] {module}: screening unavailable ({type(error).__name__}); "
               "keeping every topic")
        return {}
    by_name = {topic["name"].strip().lower(): topic["name"] for topic in taxonomy}
    screen: dict[str, dict[str, Any]] = {}
    for row in value.get("categories") or []:
        name = by_name.get(str(row.get("name") or "").strip().lower())
        if name:
            screen[name] = {"novel": bool(row.get("novel", True)),
                            "reason": str(row.get("reason") or "")[:300]}
    return screen


def round_robin(clusters: dict[str, list[dict[str, Any]]], *, budget: int,
                excluded: set[str] | None = None) -> list[dict[str, Any]]:
    """Allocate ``budget`` papers across clusters, largest cluster first, round by round."""
    excluded = excluded or set()
    groups = sorted(((name, members) for name, members in clusters.items()
                     if name not in excluded and members), key=lambda item: -len(item[1]))
    picked: list[dict[str, Any]] = []
    depth = 0
    while len(picked) < budget and any(depth < len(members) for _, members in groups):
        for _, members in groups:
            if depth < len(members):
                picked.append(members[depth])
                if len(picked) == budget:
                    break
        depth += 1
    return picked


def allocate(module: str, model: dict[str, Any], *, budget: int, current_behaviour: str,
             backbone_capabilities: str, llm: LLM, on_log=print) -> dict[str, Any]:
    """Screen a module's topics, then select its papers by round robin over clusters."""
    clusters = model.get("clusters") or {}
    screen = screen_topics(module, model.get("taxonomy") or [],
                           current_behaviour=current_behaviour,
                           backbone_capabilities=backbone_capabilities, llm=llm, on_log=on_log)
    subsumed = {name for name in clusters if not screen.get(name, {}).get("novel", True)}
    for name in sorted(subsumed):
        on_log(f"  [screen] {module}: '{name}' is already provided "
               f"({screen[name]['reason'][:100]})")
    picked = round_robin(clusters, budget=budget, excluded=subsumed)
    return {"screen": screen, "subsumed": sorted(subsumed), "picked": picked}
