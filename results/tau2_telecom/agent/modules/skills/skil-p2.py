from __future__ import annotations

from typing import Any

from evolve.slots.interfaces import Skill


class RecoverableProgramOfThought:
    """Return the RePoT procedure as a task-general skill."""

    def __init__(self, *, name: str = "repot", form: str = "markdown") -> None:
        self.name = name
        self.form = form

    def match(self, task: Any) -> Skill | None:
        instruction = str((task or {}).get("instruction") or "").strip()
        body = (
            "Use Recoverable Program-of-Thought (RePoT).\n\n"
            "1. Write a short hidden Python-style program or ordered primitive-action plan for the"
            " full task before acting.\n"
            "2. Replay that plan deterministically against the currently verified world state, one"
            " action at a time.\n"
            "3. If every step remains valid, execute the next needed environment-changing action.\n"
            "4. If replay reaches an invalid or impossible step, keep the verified prefix fixed,"
            " summarize the checkpoint state after that prefix, and repair only the remaining"
            " suffix with one additional reasoning pass.\n"
            "5. Continue from the repaired suffix instead of restarting from scratch unless the"
            " verified prefix is empty.\n"
            "6. Emit only the concrete next user reply or tool call that follows from the verified"
            " plan.\n"
        )
        score = 1.0 if instruction else 0.5
        return Skill(
            name=self.name,
            body=body,
            form=self.form,
            score=score,
            provenance={"paper": "REPOT: Recoverable Program-of-Thought via Checkpoint Repair"},
        )


def create_slot(*, context: Any, **params: Any) -> RecoverableProgramOfThought:
    del context
    return RecoverableProgramOfThought(**params)
