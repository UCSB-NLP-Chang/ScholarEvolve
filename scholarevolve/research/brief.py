"""Capability abstraction: from audited failures to an environment-agnostic research brief.

Recurring agent-attributable failures that are amenable to inference-time changes are selected
from the audit. The research model abstracts their evidence into a general capability gap and a
problem summary, removing environment names, domain entities, and tool or field names.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from scholarevolve.utils import extract_json_object

LLM = Callable[..., str]


def _truthy(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.strip().lower() in
                             ("true", "yes", "1"))


def select_targeted_modes(summary: dict[str, Any], details: list[dict[str, Any]], *,
                          top_k: int = 6, min_pct: float = 0.0) -> list[dict[str, Any]]:
    """Failure modes with agent-caused, inference-time-fixable episodes, most frequent first."""
    distribution = summary.get("failure_mode_distribution") or {}
    total = summary.get("n_failed") or len(details) or 1
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for audit in details:
        groups[audit.get("primary_category", "unclassified")].append(audit)
    modes = []
    for mode, audits in groups.items():
        fixable = [a for a in audits
                   if a.get("fault_attribution") == "agent" and _truthy(a.get("fixable_by_prompt"))]
        if not fixable:
            continue
        info = distribution.get(mode) or {}
        modes.append({"mode": mode, "count": len(audits), "agent_fixable_count": len(fixable),
                      "pct": info.get("pct_of_failures", round(100.0 * len(audits) / total, 1)),
                      "description": info.get("description", ""), "episodes": fixable})
    modes.sort(key=lambda mode: mode["agent_fixable_count"], reverse=True)
    qualifying = [mode for mode in modes if mode["pct"] >= min_pct] or modes
    return qualifying[:top_k]


def evidence_block(modes: list[dict[str, Any]], *, per_mode: int = 4) -> str:
    """Representative evidence for each targeted failure mode."""
    lines = []
    for mode in modes:
        lines.append(f"### {mode['mode']}  (count={mode['count']}, "
                     f"agent-fixable={mode['agent_fixable_count']}, {mode['pct']}% of failures)")
        if mode.get("description"):
            lines.append(f"description: {mode['description']}")
        for episode in mode["episodes"][:per_mode]:
            cause = str(episode.get("root_cause") or episode.get("summary") or "").strip()
            evidence = str(episode.get("evidence") or "").strip()
            if cause:
                lines.append(f"- root cause: {cause}")
            if evidence:
                lines.append(f"  evidence: {evidence[:200]}")
    return "\n".join(lines)


def build_brief(summary: dict[str, Any], details: list[dict[str, Any]], *, llm: LLM,
                top_k: int = 6, min_pct: float = 0.0) -> dict[str, Any]:
    """The capability gap behind the targeted failure modes."""
    modes = select_targeted_modes(summary, details, top_k=top_k, min_pct=min_pct)
    if not modes:
        return {"targeted_modes": [], "core_challenge": "", "problem_summary": "",
                "evidence": ""}
    evidence = evidence_block(modes)
    prompt = f"""An auditor analysed why an LLM agent failed tasks in an interactive environment.
Below are the dominant agent-caused failure modes that an inference-time change could address,
with per-episode root causes taken from real trajectories.

{evidence}

Abstract these environment-specific symptoms into a GENERAL capability gap that research papers
could address:
- Do NOT mention the environment or benchmark, domain entities, particular tools, APIs, apps or
  field names. Describe the missing inference-time ability in general agent terms.
- Focus on what the agent is unable to do (e.g. losing track of an earlier observation, stopping
  before every requirement is met), not on the task content.

Return ONLY valid JSON:
{{"core_challenge": "<1-2 sentences: the general capability gap behind these failures>",
  "problem_summary": "<1-2 sentences summarising the problem for a reader of research papers>"}}"""
    value = extract_json_object(llm(
        [{"role": "system", "content": "You abstract environment-specific agent failures into "
                                       "general research problems. Return only valid JSON."},
         {"role": "user", "content": prompt}], temperature=0.2))
    return {
        "targeted_modes": [{key: mode[key] for key in
                            ("mode", "count", "agent_fixable_count", "pct", "description")}
                           for mode in modes],
        "core_challenge": str(value.get("core_challenge") or "").strip(),
        "problem_summary": str(value.get("problem_summary") or "").strip(),
        "evidence": evidence,
    }
