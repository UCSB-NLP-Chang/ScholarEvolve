import ast
import json
import re
from pathlib import Path

from evolve.slots.interfaces import Budget, Proposal, ToolSpec

PYTHON_BLOCK = ToolSpec(
    name="python_code",
    description=(
        "One executable Python block per step. `apis.<app>.<api>(**params)` calls the "
        "environment; `apis.supervisor.complete_task(...)` ends the episode."
    ),
    parameters={"code": "str"},
)

DOC_ACTION = ToolSpec(
    name="DOC_ACTION",
    description=(
        "DOC_ACTION <app> or DOC_ACTION <app>.<api> expands into Python that prints the "
        "available app APIs or one API document before the next reasoning step."
    ),
    parameters={"target": "str"},
    composite_of=("python_code",),
)

PYTHON_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
DOC_LINE = re.compile(
    r"^\s*DOC_ACTION\s+([A-Za-z_][\w]*)(?:\.([A-Za-z_][\w]*))?\s*$",
    re.MULTILINE,
)
INLINE_API_CALL = re.compile(r"(apis\.[A-Za-z_][\w.]*\([^`\n]*\))")


class ToolACERToolInterface:
    def __init__(self, *, artifact_path=None, max_refinement_rounds=3):
        self.max_refinement_rounds = max(1, int(max_refinement_rounds))
        self.learned_actions = []
        self.default_doc_action = "print(apis.api_docs.show_app_descriptions())"
        if artifact_path:
            self._load(artifact_path)

    def fit(self, episodes):
        counts = {}
        fallback = self.default_doc_action
        for episode in episodes:
            weight = 3 if getattr(episode, "completed", False) else 1
            weight += max(0, int(round(float(getattr(episode, "reward", 0.0)))))
            for step in getattr(episode, "trajectory", ()) or ():
                code = self._normalize_code(str(step.get("code") or ""))
                if not self._is_effectful_python(code):
                    continue
                counts[code] = counts.get(code, 0) + weight
                if "api_docs" in code:
                    fallback = code
        ranked = sorted(counts.items(), key=lambda item: (-item[1], len(item[0])))
        self.learned_actions = [code for code, _ in ranked[:12]]
        if self.learned_actions:
            self.default_doc_action = next(
                (code for code in self.learned_actions if "api_docs" in code),
                self.learned_actions[0],
            )
        else:
            self.default_doc_action = fallback

    def save(self, path):
        payload = {
            "max_refinement_rounds": self.max_refinement_rounds,
            "learned_actions": self.learned_actions,
            "default_doc_action": self.default_doc_action,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def tools(self, state=None):
        del state
        return [PYTHON_BLOCK, DOC_ACTION]

    def act(self, proposal, tools=None, state=None, task=None, client=None):
        del tools, task, client
        candidates = self._proposal_candidates(proposal)
        budgeted_rounds = self.max_refinement_rounds
        if state is not None:
            budgeted_rounds = min(
                budgeted_rounds,
                max(1, min(3, int(getattr(state, "steps_left", budgeted_rounds) or budgeted_rounds))),
            )
        for candidate in candidates:
            refined = candidate
            for _ in range(budgeted_rounds):
                if self._is_effectful_python(refined):
                    return refined
                updated = self._refine_once(refined, proposal=proposal, state=state)
                if updated == refined:
                    break
                refined = updated
            if self._is_effectful_python(refined):
                return refined
        return self._fallback_action(proposal=proposal, state=state)

    def _load(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.max_refinement_rounds = max(1, int(data.get("max_refinement_rounds", 3)))
        self.learned_actions = [
            self._normalize_code(str(code))
            for code in data.get("learned_actions", [])
            if self._normalize_code(str(code))
        ]
        default_action = self._normalize_code(str(data.get("default_doc_action") or ""))
        if default_action:
            self.default_doc_action = default_action

    def _proposal_candidates(self, proposal):
        raw = str(getattr(proposal, "text", "") or "")
        action = self._normalize_code(str(getattr(proposal, "action", "") or ""))
        candidates = []
        if action:
            candidates.append(action)
        for block in PYTHON_FENCE.findall(raw):
            normalized = self._normalize_code(block)
            if normalized and normalized not in candidates:
                candidates.append(normalized)
        for match in INLINE_API_CALL.findall(raw):
            normalized = self._normalize_code(match)
            if normalized and normalized not in candidates:
                candidates.append(normalized)
        doc_match = DOC_LINE.search(raw)
        if doc_match:
            normalized = self._normalize_code(self._doc_action(doc_match.group(1), doc_match.group(2)))
            if normalized and normalized not in candidates:
                candidates.append(normalized)
        raw_text = self._normalize_code(raw)
        if raw_text and raw_text not in candidates:
            candidates.append(raw_text)
        candidates.extend(code for code in self.learned_actions if code not in candidates)
        return candidates or [self.default_doc_action]

    def _refine_once(self, code, *, proposal, state):
        cleaned = self._normalize_code(code)
        if not cleaned:
            return self._fallback_action(proposal=proposal, state=state)
        doc_match = DOC_LINE.search(cleaned)
        if doc_match:
            return self._doc_action(doc_match.group(1), doc_match.group(2))
        if self._is_effectful_python(cleaned):
            return cleaned
        lines = [line.rstrip() for line in cleaned.splitlines() if line.strip()]
        if len(lines) == 1:
            line = lines[0]
            if line.startswith("apis.") and "(" in line and ")" in line:
                return f"print({line})" if "supervisor.complete_task" not in line else line
            inline = INLINE_API_CALL.search(line)
            if inline:
                api_call = inline.group(1)
                return f"print({api_call})" if "supervisor.complete_task" not in api_call else api_call
        if "show_app_descriptions" in cleaned or "show_api_descriptions" in cleaned or "show_api_doc" in cleaned:
            return cleaned
        instruction = ""
        if state is not None:
            instruction = str(getattr(state, "instruction", "") or "")
        if not instruction:
            instruction = str(getattr(proposal, "text", "") or "")
        target = self._infer_doc_target(instruction)
        if target is not None:
            return self._doc_action(*target)
        return self.default_doc_action

    def _fallback_action(self, *, proposal, state):
        instruction = ""
        if state is not None:
            instruction = str(getattr(state, "instruction", "") or "")
        if not instruction:
            instruction = str(getattr(proposal, "text", "") or "")
        target = self._infer_doc_target(instruction)
        if target is not None:
            return self._doc_action(*target)
        return self.default_doc_action

    def _infer_doc_target(self, text):
        lowered = str(text or "").lower()
        for app in (
            "spotify",
            "venmo",
            "gmail",
            "calendar",
            "slack",
            "notion",
            "drive",
            "docs",
            "sheets",
        ):
            if app in lowered:
                return (app, None)
        if "api" in lowered or "tool" in lowered or "app" in lowered:
            return (None, None)
        return None

    def _doc_action(self, app, api):
        if app and api:
            return f"print(apis.api_docs.show_api_doc(app_name={app!r}, api_name={api!r}))"
        if app:
            return f"print(apis.api_docs.show_api_descriptions(app_name={app!r}))"
        return "print(apis.api_docs.show_app_descriptions())"

    def _normalize_code(self, text):
        stripped = str(text or "").strip()
        if not stripped:
            return ""
        fence = PYTHON_FENCE.fullmatch(stripped)
        if fence:
            stripped = fence.group(1).strip()
        return stripped.replace("\r\n", "\n").strip()

    def _is_effectful_python(self, code):
        normalized = self._normalize_code(code)
        if not normalized:
            return False
        try:
            tree = ast.parse(normalized)
        except SyntaxError:
            return False
        if not tree.body:
            return False
        for node in tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    return True
                continue
            return True
        return False


def create_slot(*, context, **params):
    artifact_path = params.pop("artifact_path", None)
    if artifact_path is None:
        artifact_path = getattr(context, "artifacts", {}).get("tool_interface")
    return ToolACERToolInterface(artifact_path=artifact_path, **params)
