"""Mutation blueprint: the research agent's plan for implementing one paper in one module.

The research agent reads the paper's abstract, method sections and appendix together with the
official code findings, if any. The blueprint specifies the source mechanism and its assumptions,
the target module and interface, the required state and artifacts, the runtime operations, the
auxiliary model calls, and the termination conditions. It separates the source mechanism from the
adaptations the host harness requires, and keeps the paper and topic provenance.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from scholarevolve.harness.modules import interface_contract
from scholarevolve.mutation.code import CodeContext
from scholarevolve.mutation.paper import PaperContext

LLM = Callable[..., str]

SECTIONS = (
    "SOURCE MECHANISM",
    "ASSUMPTIONS",
    "TARGET MODULE AND INTERFACE",
    "STATE AND ARTIFACTS",
    "RUNTIME OPERATIONS",
    "AUXILIARY MODEL CALLS",
    "TERMINATION CONDITIONS",
    "HOST ADAPTATIONS",
    "ESSENTIAL INVARIANTS",
)

_SYSTEM = ("You plan how to implement one research paper's mechanism as one module of an LLM "
           "agent harness. You write faithful, concrete, implementable blueprints in the "
           "requested plain-text format.")

_META_REPLY = re.compile(r"previous \(truncated\) response|don.?t have access to the previous|"
                         r"continue .{0,3}exactly where it stopped", re.I)


def _defect(text: str) -> str:
    if len(text.strip()) < 400:
        return "the blueprint is too short"
    if _META_REPLY.search(text):
        return "the blueprint is conversational text, not a plan"
    missing = [section for section in SECTIONS if section not in text.upper()]
    return f"missing sections: {', '.join(missing)}" if missing else ""


def write_blueprint(*, module: str, paper: dict[str, Any], paper_context: PaperContext,
                    code_context: CodeContext | None, current_source: str,
                    environment: Any, brief: dict[str, Any] | None, llm: LLM,
                    max_method_chars: int = 120_000, attempts: int = 3) -> dict[str, Any]:
    """Write and validate the blueprint for one selected paper."""
    method = paper_context.method_text[:max_method_chars] or "(full text unavailable)"
    verbatim = paper_context.verbatim_blocks or "(none extracted)"
    code = code_context.findings if code_context else "(no official code was found)"
    gap = ""
    if brief and brief.get("core_challenge"):
        gap = (f"CAPABILITY GAP the search is addressing (context, not a requirement):\n"
               f"{brief.get('core_challenge')}\n{brief.get('problem_summary', '')}\n\n")
    prompt = f"""You are planning a MUTATION of one module of an LLM agent harness. The backbone model is
frozen; only the module's code changes. The mutation must implement the mechanism of the paper
below inside this module.

{interface_contract(module)}

ENVIRONMENT
{environment.interface_description()}
Action format: {environment.action_format()}

CURRENT IMPLEMENTATION OF THIS MODULE
```python
{current_source[:20_000]}
```

{gap}PAPER: {paper_context.title}
arXiv: {paper_context.arxiv_id}
Topic: {paper.get('topic') or '(unassigned)'}

ABSTRACT
{paper_context.abstract}

METHOD SECTIONS AND APPENDIX
{method}

VERBATIM LISTINGS, ALGORITHMS AND PROMPTS FROM THE PAPER
{verbatim}

OFFICIAL CODE FINDINGS
{code}

Write the blueprint as plain text (no JSON, no code fences) with exactly these sections, each
starting on its own line with the section name in capitals:

SOURCE MECHANISM - what the paper's method does and why it works, following the paper's own
  algorithm and, where available, its code.
ASSUMPTIONS - what the mechanism assumes about the agent, the task or the available data.
TARGET MODULE AND INTERFACE - which interface methods implement the mechanism and what each
  returns.
STATE AND ARTIFACTS - the state kept within an episode, and any persistent artifact built offline
  from successful evolution-set episodes through fit/save (or "none").
RUNTIME OPERATIONS - the step-by-step procedure executed at inference time, with decision rules.
AUXILIARY MODEL CALLS - every additional model call: when it fires, what it must elicit, and its
  output schema; at most three per agent step. Write "none" for a deterministic mechanism.
TERMINATION CONDITIONS - when each loop, retry or auxiliary procedure stops, and the fallback
  that keeps every path producing a valid result.
HOST ADAPTATIONS - what differs from the paper because of this harness and environment (for
  example a trained component replaced by prompting), kept separate from the source mechanism.
ESSENTIAL INVARIANTS - the 3-6 properties without which this would no longer be the paper's
  mechanism. Everything else (prompt wording, field names, thresholds, data structures) is
  adaptable.

Specify the mechanism, not its surface form: prompts may be reworded, but state what each prompt
must elicit. If the paper's method needs capabilities the module cannot have (training, changing
the environment, intercepting other modules), implement the part that fits this module and record
the rest under HOST ADAPTATIONS."""
    text, defect = "", "no attempt made"
    for _ in range(max(1, attempts)):
        text = llm([{"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt}], temperature=0.2).strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
        defect = _defect(text)
        if not defect:
            break
    if defect:
        raise ValueError(f"blueprint for {paper_context.title!r} is unusable: {defect}")
    return {
        "module": module,
        "paper": {"title": paper_context.title, "arxiv_id": paper_context.arxiv_id,
                  "url": paper_context.url, "topic": paper.get("topic")},
        "full_text_source": paper_context.source,
        "code_repository": code_context.repository if code_context else None,
        "blueprint": text,
    }


def render_blueprint(blueprint: dict[str, Any]) -> str:
    paper = blueprint["paper"]
    header = [f"# Mutation blueprint: {blueprint['module']}",
              "",
              f"- Paper: {paper.get('title')} (arXiv {paper.get('arxiv_id')})",
              f"- Topic: {paper.get('topic')}",
              f"- Official code: {blueprint.get('code_repository') or 'not available'}",
              ""]
    return "\n".join(header) + blueprint["blueprint"].strip() + "\n"
