"""Failure audit of the current harness on D_evo.

Every failed episode is audited individually: the auditor attributes the failure to the agent or to
the environment, names the failure mode, and records the cause and the most telling step. Free-form
mode names are then consolidated into one taxonomy. Task identifiers never enter a prompt; episodes
are exposed as anonymous references.

Output: ``audit_summary.json`` (mode distribution and attribution counts) and
``audit_details.json`` (one record per audited episode).
"""

from __future__ import annotations

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from scholarevolve.utils import atomic_write_json, extract_json_object, load_json, utc_now

LLM = Callable[..., str]

SEED_CATEGORIES = (
    "wrong_tool_selection", "wrong_argument_value", "missing_required_action",
    "premature_stop", "no_stop_signal", "state_tracking_error",
    "api_discovery_failure", "unrecovered_execution_error", "step_budget_exhausted",
    "instruction_scope_violation", "output_contract_violation",
)

_SYSTEM = "You are a precise agent-failure auditor. Return only valid JSON."


def _snake(value: Any, *, default: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return text or default


def collect_failures(cells: list[dict[str, Any]], statuses: list[dict[str, Any]], *,
                     render: Callable[[dict[str, Any]], str]) -> list[dict[str, Any]]:
    """Anonymised audit inputs for every scored episode that did not fully succeed."""
    by_id = {str(status.get("task_id")): status for status in statuses}
    failures = sorted((cell for cell in cells
                       if cell.get("scored") and float(cell.get("reward", 0.0)) < 1.0),
                      key=lambda cell: str(cell["task_id"]))
    records = []
    for index, cell in enumerate(failures, 1):
        status = by_id.get(str(cell["task_id"])) or {}
        records.append({
            "ref": f"t{index:03d}",
            "task_id": str(cell["task_id"]),
            "outcome": f"failed (reward {float(cell.get('reward') or 0.0):.2f})",
            "steps": status.get("steps"),
            "completed": status.get("completed"),
            "runtime_error": str(status.get("error") or "")[:400],
            "trajectory": render(status),
        })
    return records


def audit_one(record: dict[str, Any], *, interface: str, llm: LLM) -> dict[str, Any]:
    completion = ("" if record.get("completed") is None
                  else f" | agent signalled completion: {bool(record.get('completed'))}")
    prompt = f"""An LLM agent did not fully solve one task. Diagnose why.

ENVIRONMENT INTERFACE
{interface}

EPISODE {record['ref']}
Outcome: {record.get('outcome')}
Agent runtime error: {record.get('runtime_error') or '(none)'}
Steps taken: {record.get('steps')}{completion}

AGENT TRAJECTORY
{record.get('trajectory') or '(no trajectory available)'}

Classify the DOMINANT reason this episode lost reward. Use a short snake_case failure-mode name
describing the agent behaviour, not the task content. Reuse one of these when it fits, otherwise
invent a better one: {', '.join(SEED_CATEGORIES)}.

Return ONLY valid JSON:
{{"primary_category": "<snake_case failure mode>",
  "fault_attribution": "agent" | "environment" | "evaluation" | "underspecified_task",
  "fixable_by_prompt": true | false,
  "root_cause": "<1-2 sentences on the underlying capability gap, no task-specific nouns>",
  "evidence": "<the single most telling step, <=200 chars>",
  "summary": "<=20 words>"}}

"fault_attribution" is "agent" only when a better inference-time policy could plausibly have
succeeded. "fixable_by_prompt" means an inference-time technique (control flow, state,
verification, guidance, retrieval) could address it without training."""
    value = extract_json_object(llm([{"role": "system", "content": _SYSTEM},
                                     {"role": "user", "content": prompt}], temperature=0))
    return {
        "ref": record["ref"],
        "primary_category": _snake(value.get("primary_category"), default="unclassified"),
        "fault_attribution": _snake(value.get("fault_attribution"), default="agent"),
        "fixable_by_prompt": bool(value.get("fixable_by_prompt", False)),
        "root_cause": str(value.get("root_cause") or "")[:600],
        "evidence": str(value.get("evidence") or "")[:300],
        "summary": str(value.get("summary") or "")[:240],
        "audited": bool(value),
    }


def audit_all(records: list[dict[str, Any]], *, interface: str, llm: LLM,
              concurrency: int = 6, on_log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    def run(record: dict[str, Any]) -> dict[str, Any]:
        try:
            return audit_one(record, interface=interface, llm=llm)
        except Exception as error:
            on_log(f"  audit failed for {record['ref']}: {type(error).__name__}: {error}")
            return {"ref": record["ref"], "primary_category": "unclassified",
                    "fault_attribution": "agent", "fixable_by_prompt": False, "root_cause": "",
                    "evidence": "", "summary": "", "audited": False}

    if not records:
        return []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        audits = list(pool.map(run, records))
    on_log(f"  audited {sum(a['audited'] for a in audits)}/{len(audits)} failed episodes")
    return audits


def consolidate(audits: list[dict[str, Any]], *, environment: str, llm: LLM,
                on_log: Callable[[str], None] = print) -> tuple[list[dict[str, Any]],
                                                                dict[str, str]]:
    """Merge near-duplicate failure-mode names into one canonical taxonomy."""
    observed = sorted({a["primary_category"] for a in audits
                       if a["primary_category"] != "unclassified"})
    if len(observed) < 2:
        return audits, {}
    counts = Counter(a["primary_category"] for a in audits)
    listing = "\n".join(f"- {name} (n={counts[name]})" for name in observed)
    prompt = f"""These snake_case agent failure-mode names were produced independently while auditing
failures in {environment}. Merge names that denote the SAME underlying agent behaviour into one
canonical name, and keep genuinely distinct behaviours separate. Prefer an existing name as the
canonical form.

{listing}

Return ONLY valid JSON:
{{"canonical": {{"<observed name>": "<canonical name>", ...}},
  "descriptions": {{"<canonical name>": "<=20 words describing the behaviour>"}}}}
Every observed name must appear exactly once as a key in "canonical"."""
    try:
        value = extract_json_object(llm(
            [{"role": "system", "content": "You consolidate failure taxonomies. Return only "
                                           "valid JSON."},
             {"role": "user", "content": prompt}], temperature=0))
    except Exception as error:
        on_log(f"  failure-mode consolidation skipped: {type(error).__name__}: {error}")
        value = {}
    mapping = {name: _snake((value.get("canonical") or {}).get(name, name), default=name)
               for name in observed}
    descriptions = {_snake(key, default=key): str(text)[:200]
                    for key, text in (value.get("descriptions") or {}).items()}
    for audit in audits:
        audit["raw_category"] = audit["primary_category"]
        audit["primary_category"] = mapping.get(audit["primary_category"],
                                                audit["primary_category"])
    on_log(f"  failure taxonomy: {len(set(mapping.values()))} modes")
    return audits, descriptions


def run_audit(cells: list[dict[str, Any]], statuses: list[dict[str, Any]], *,
              environment: Any, llm: LLM, out_dir: str | Path, concurrency: int = 6,
              on_log: Callable[[str], None] = print) -> tuple[dict[str, Any],
                                                              list[dict[str, Any]]]:
    """Audit the failed D_evo episodes of one evaluation; cached in ``out_dir``."""
    out_dir = Path(out_dir)
    summary_path, details_path = out_dir / "audit_summary.json", out_dir / "audit_details.json"
    if summary_path.is_file() and details_path.is_file():
        return load_json(summary_path), load_json(details_path)
    records = collect_failures(cells, statuses, render=environment.render_trajectory)
    on_log(f"  auditing {len(records)} failed episodes")
    audits = audit_all(records, interface=environment.interface_description(), llm=llm,
                       concurrency=concurrency, on_log=on_log)
    audits, descriptions = consolidate(audits, environment=environment.name, llm=llm,
                                       on_log=on_log)
    total = len(audits) or 1
    distribution = {
        mode: {"count": count, "pct_of_failures": round(100.0 * count / total, 1),
               "description": descriptions.get(mode, "")}
        for mode, count in Counter(a["primary_category"] for a in audits).most_common()}
    scored = [cell for cell in cells if cell.get("scored")]
    summary = {
        "generated_at": utc_now(),
        "environment": environment.name,
        "n_tasks": len(scored),
        "n_failed": len(records),
        "failure_mode_distribution": distribution,
        "fault_attribution": dict(Counter(a["fault_attribution"] for a in audits)),
        "fixable_by_prompt": dict(Counter(str(a["fixable_by_prompt"]).lower() for a in audits)),
    }
    atomic_write_json(out_dir / "audit_task_map.json",
                      {record["ref"]: record["task_id"] for record in records})
    atomic_write_json(summary_path, summary)
    atomic_write_json(details_path, audits)
    return summary, audits
