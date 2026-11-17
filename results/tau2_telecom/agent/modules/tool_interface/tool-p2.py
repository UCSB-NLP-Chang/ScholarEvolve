import json
import math
import re

from evolve.slots.interfaces import Budget, Evidence, Proposal, Skill, ToolSpec


_TOKEN_RE = re.compile(r"[a-z0-9_#]+")
_STOPWORDS = {
    "a", "an", "and", "at", "by", "for", "from", "get", "help", "i", "in", "into", "is", "it",
    "me", "my", "of", "on", "or", "please", "the", "to", "up", "with", "you",
}


class TinyAgentToolInterface:
    def __init__(self, *, context, max_tools=12, min_tools=6, keep_fraction=0.5):
        self._dialect = context.dialect
        self._max_tools = max(1, int(max_tools))
        self._min_tools = max(1, int(min_tools))
        self._keep_fraction = max(0.0, min(float(keep_fraction), 1.0))
        self._all_tools = []

    def tools(self, state=None, task=None):
        del task
        all_tools = list(self._dialect.tool_specs())
        self._all_tools = list(all_tools)
        if len(all_tools) <= self._max_tools or state is None:
            return all_tools

        ranked = []
        query_tokens = self._query_tokens(state)
        history_tokens = self._history_tokens(state)
        for index, tool in enumerate(all_tools):
            score = self._score_tool(tool, query_tokens, history_tokens)
            ranked.append((score, index, tool))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        target = min(
            len(all_tools),
            max(self._min_tools, min(self._max_tools, math.ceil(len(all_tools) * self._keep_fraction))),
        )
        selected = [tool for score, _, tool in ranked[:target] if score > 0.0]
        if len(selected) < self._min_tools:
            selected = [tool for _, _, tool in ranked[:target]]
        return selected or all_tools

    def act(self, proposal, tools=None, env_actions=None, state=None, task=None, client=None):
        del tools, env_actions, state, task, client
        allowed = {tool.name for tool in self._all_tools}
        action = proposal.action if isinstance(proposal.action, dict) else {}
        tool_calls = []
        for call in action.get("tool_calls") or []:
            normalized = self._normalize_call(call, allowed)
            if normalized is not None:
                tool_calls.append(normalized)
        return {"tool_calls": tool_calls}

    def _query_tokens(self, state):
        parts = [state.instruction]
        if state.last_output:
            parts.append(state.last_output)
        return set(self._tokenize(" ".join(parts)))

    def _history_tokens(self, state):
        tokens = set()
        for message in state.history[-4:]:
            tokens.update(self._tokenize(str(message.get("content") or "")))
        return tokens

    def _score_tool(self, tool, query_tokens, history_tokens):
        name_tokens = set(self._tokenize(tool.name.replace("_", " ")))
        desc_tokens = set(self._tokenize(tool.description))
        param_tokens = set()
        for key, value in (tool.parameters or {}).items():
            param_tokens.update(self._tokenize(str(key)))
            if isinstance(value, dict):
                param_tokens.update(self._tokenize(str(value.get("description") or "")))

        score = 0.0
        score += 4.0 * len(query_tokens & name_tokens)
        score += 2.0 * len(query_tokens & desc_tokens)
        score += 1.0 * len(query_tokens & param_tokens)
        score += 0.5 * len(history_tokens & (name_tokens | desc_tokens | param_tokens))

        joined_name = tool.name.lower()
        for token in query_tokens:
            if token and token in joined_name:
                score += 0.5
        return score

    def _normalize_call(self, call, allowed):
        if not isinstance(call, dict):
            return None
        name = str(call.get("name") or "").strip()
        if not name or (allowed and name not in allowed):
            return None

        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                parsed = {}
        elif isinstance(arguments, dict):
            parsed = arguments
        else:
            parsed = {}

        return {
            "id": str(call.get("id") or f"call_{len(name)}"),
            "name": name,
            "arguments": json.dumps(parsed, sort_keys=True),
        }

    def _tokenize(self, text):
        return [
            token for token in _TOKEN_RE.findall(str(text).lower())
            if token not in _STOPWORDS and not token.isdigit()
        ]


def create_slot(*, context, **params):
    return TinyAgentToolInterface(context=context, **params)
