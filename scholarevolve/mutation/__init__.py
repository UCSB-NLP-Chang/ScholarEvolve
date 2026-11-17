"""From a selected paper to an executable module mutation.

    full text (abstract, method sections, appendix) + official code findings
    -> mutation blueprint (research agent) -> candidate module (coding agent + health probes)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from scholarevolve.mutation.blueprint import render_blueprint, write_blueprint
from scholarevolve.mutation.code import CodeContext, read_code
from scholarevolve.mutation.coding_agent import (
    CANDIDATE, CodingSettings, current_module_source, implement,
)
from scholarevolve.mutation.paper import PaperContext, read_paper
from scholarevolve.utils import atomic_write_json, load_json

LLM = Callable[..., str]


def build_mutation(*, name: str, module: str, paper: dict[str, Any], out_dir: str | Path,
                   current_spec_entry: Any, environment: Any, brief: dict[str, Any] | None,
                   research_llm: LLM, coding: CodingSettings, read_official_code: bool = True,
                   code_reader_turns: int = 40, on_log=print) -> dict[str, Any]:
    """Build one candidate; results are cached in ``out_dir/<name>``."""
    directory = Path(out_dir) / name
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        return load_json(manifest_path)
    directory.mkdir(parents=True, exist_ok=True)

    paper_path = directory / "paper_context.json"
    if paper_path.is_file():
        paper_context = PaperContext(**load_json(paper_path))
    else:
        paper_context = read_paper(paper)
        atomic_write_json(paper_path, paper_context.to_json())
    on_log(f"  {name}: full text from {paper_context.source} "
           f"({len(paper_context.method_text)} chars)")

    code_path = directory / "code_context.json"
    code_context: CodeContext | None = None
    if code_path.is_file():
        stored = load_json(code_path)
        code_context = CodeContext(**stored) if stored else None
    elif read_official_code:
        try:
            code_context = read_code(paper, research_llm, max_turns=code_reader_turns,
                                     on_log=on_log)
        except Exception as error:
            on_log(f"  {name}: official code unavailable ({type(error).__name__}: {error})")
        atomic_write_json(code_path, code_context.to_json() if code_context else {})

    current_source = current_module_source(current_spec_entry, module)
    blueprint_path = directory / "blueprint.json"
    if blueprint_path.is_file():
        blueprint = load_json(blueprint_path)
    else:
        blueprint = write_blueprint(module=module, paper=paper, paper_context=paper_context,
                                    code_context=code_context, current_source=current_source,
                                    environment=environment, brief=brief, llm=research_llm)
        atomic_write_json(blueprint_path, blueprint)
        (directory / "BLUEPRINT.md").write_text(render_blueprint(blueprint), encoding="utf-8")

    result = implement(run_dir=directory / "session", module=module, blueprint=blueprint,
                       paper_context=paper_context, code_context=code_context,
                       current_source=current_source, environment=environment,
                       settings=coding, chat_llm=research_llm, on_log=on_log)
    (directory / CANDIDATE).write_text(result["code"], encoding="utf-8")
    manifest = {
        "name": name,
        "modules": [module],
        "module": str((directory / CANDIDATE).resolve()),
        "origin": "paper",
        "paper": blueprint["paper"],
        "code_repository": blueprint.get("code_repository"),
        "ok": bool(result["ok"]),
        "detail": str(result["detail"])[-1500:],
    }
    atomic_write_json(manifest_path, manifest)
    on_log(f"  [{'ok' if manifest['ok'] else 'FAILED'}] {name} <- "
           f"{str(blueprint['paper'].get('title'))[:70]}")
    return manifest
