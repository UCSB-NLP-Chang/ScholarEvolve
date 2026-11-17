"""Module-specific query construction.

For each of the five modules, the research model receives the same brief, the module's
responsibility and interface, and the current harness behaviour in that module. It describes how a
change within that module could address the capability gap, then proposes two query groups: broad
phrases naming the capability and narrower phrases naming candidate mechanisms. The groups are
interleaved so every retrieval batch covers both problem areas and candidate solutions. Each module
receives its own queries because one capability gap admits interventions at several points of the
harness. Without an audit, the same procedure builds queries from the module description alone.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from scholarevolve.harness.modules import MODULE_NAMES, MODULE_RESPONSIBILITIES, interface_contract
from scholarevolve.utils import extract_json_object

LLM = Callable[..., str]

_SYSTEM = ("You plan literature searches for one module of an LLM agent harness. "
           "Return only valid JSON.")

_HARNESS = (
    "The agent harness is split into five modules with fixed interfaces: tool interface, context "
    "management, skills, memories and agentic workflows. The backbone language model is frozen; "
    "only the harness code changes.")

_RULES = """Rules for the query phrases:
- No benchmark, environment, domain, application or entity names, and no particular tool or field
  names. Use general LLM-agent research vocabulary.
- SHORT: 1-5 words each. Every phrase is matched as an exact phrase in paper titles and abstracts,
  so a sentence-length phrase matches nothing.
- Use ESTABLISHED research terms that appear in real titles and abstracts (for example "context
  compression", "trajectory retrieval", "skill discovery", "self-verification"), not descriptive
  phrases invented here.
- Inference-time mechanisms only: prompting, control flow, state, retrieval, verification, tool
  use. No training, fine-tuning or reinforcement learning.
- Stay inside this module's responsibility; mechanisms that belong to another module are searched
  for separately."""


def _phrases(values: Any, limit: int) -> list[str]:
    out: list[str] = []
    for value in values or []:
        phrase = re.sub(r"\s+", " ", str(value or "")).strip().strip('"')
        if phrase and len(phrase.split()) <= 6 and phrase.lower() not in {p.lower() for p in out}:
            out.append(phrase)
    return out[:limit]


def interleave(capability: list[str], mechanism: list[str], *, limit: int) -> list[str]:
    """Alternate broad capability phrases with narrower mechanism phrases."""
    merged = [phrase for pair in zip(capability, mechanism) for phrase in pair]
    merged += capability[len(mechanism):] + mechanism[len(capability):]
    return list(dict.fromkeys(merged))[:limit]


def module_queries(module: str, *, brief: dict[str, Any] | None, current_behaviour: str,
                   llm: LLM, per_group: int = 6) -> dict[str, Any]:
    """The query set for one module."""
    if brief and brief.get("core_challenge"):
        gap = ("CAPABILITY GAP, abstracted from an audit of the agent's own failures:\n"
               f"- core challenge: {brief.get('core_challenge', '')}\n"
               f"- summary: {brief.get('problem_summary', '')}\n\n"
               "Describe how a mechanism placed in THIS module could address the gap, then "
               "propose the queries.")
        lever_hint = "<1-2 sentences: how a change in this module would address the gap>"
    else:
        gap = ("No audit is available. Propose queries that cover the main mechanism families "
               "this module could implement to make the agent more reliable.")
        lever_hint = "<1-2 sentences: which kinds of mechanisms this module could add>"
    prompt = f"""{_HARNESS}

{interface_contract(module)}

CURRENT HARNESS BEHAVIOUR IN THIS MODULE:
{current_behaviour}

{gap}

{_RULES}

Return ONLY valid JSON:
{{"lever": "{lever_hint}",
  "capability_queries": ["<{per_group} broad phrases naming the capability>"],
  "mechanism_queries": ["<{per_group} narrower phrases naming candidate mechanisms>"]}}"""
    value = extract_json_object(llm([{"role": "system", "content": _SYSTEM},
                                     {"role": "user", "content": prompt}], temperature=0.2))
    capability = _phrases(value.get("capability_queries"), per_group)
    mechanism = _phrases(value.get("mechanism_queries"), per_group)
    return {"module": module,
            "responsibility": MODULE_RESPONSIBILITIES[module],
            "lever": str(value.get("lever") or "").strip()[:600],
            "capability_queries": capability,
            "mechanism_queries": mechanism,
            "queries": interleave(capability, mechanism, limit=2 * per_group)}


def build_queries(*, brief: dict[str, Any] | None, current_behaviour: dict[str, str], llm: LLM,
                  modules: tuple[str, ...] = MODULE_NAMES, per_group: int = 6,
                  concurrency: int = 5) -> dict[str, dict[str, Any]]:
    """One independent query construction call per module."""
    def one(module: str) -> tuple[str, dict[str, Any]]:
        return module, module_queries(module, brief=brief,
                                      current_behaviour=current_behaviour[module], llm=llm,
                                      per_group=per_group)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return dict(pool.map(one, modules))
