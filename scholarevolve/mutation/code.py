"""Official code of a selected paper: repository discovery and a read-only code-reading agent.

The repository is located from an explicit link, a GitHub link in the abstract or on the arXiv
abstract page, or a GitHub search whose hit is verified by the arXiv identifier appearing in the
repository's description or README. A ReAct-style agent then reads the repository file by file
over the GitHub API - it never clones, installs or runs anything - until it can describe how the
method is implemented: key files, classes and functions, control flow, decision rules, and prompts
copied from the code. Set GITHUB_TOKEN to raise the GitHub API rate limit.
"""

from __future__ import annotations

import base64
import json
import os
import re
import ssl
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from scholarevolve.mutation.paper import arxiv_id_of

LLM = Callable[..., str]
USER_AGENT = "ScholarEvolve/0.1 (code reader)"

_GITHUB = re.compile(r"https?://github\.com/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)", re.I)
_NOT_OWNERS = {"login", "join", "sponsors", "features", "about", "topics", "search",
               "marketplace", "pricing", "site", "contact", "security", "enterprise", "team",
               "collections", "explore", "orgs", "apps", "settings", "new", "notifications"}
_SKIP_SEGMENTS = {".git", "node_modules", ".venv", "venv", "env", "__pycache__",
                  "site-packages", ".github", "dist", "build", ".idea", ".vscode"}
_KEEP_SUFFIXES = (".py", ".md", ".yaml", ".yml", ".json", ".jsonl", ".txt", ".cfg", ".toml",
                  ".sh", ".ipynb", ".rst")


@dataclass
class CodeContext:
    repository: str
    findings: str
    files_read: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _headers(accept_json: bool = True) -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT}
    if accept_json:
        headers["Accept"] = "application/json"
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get_text(url: str, *, timeout: int = 30, headers: dict[str, str] | None = None) -> str:
    request = urllib.request.Request(url, headers=headers or {"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout,
                                context=ssl.create_default_context()) as response:
        return response.read().decode("utf-8", "replace")


def _get_json(url: str, *, timeout: int = 30) -> Any:
    return json.loads(_get_text(url, timeout=timeout, headers=_headers()))


def _github_link(*texts: str) -> str | None:
    for text in texts:
        for match in _GITHUB.finditer(text or ""):
            owner = match.group(1)
            name = re.sub(r"(\.git)?[).,>\"'\]]*$", "", match.group(2))
            if owner.lower() in _NOT_OWNERS or not name or name.lower() in ("blob", "tree"):
                continue
            return f"https://github.com/{owner}/{name}"
    return None


def readme(owner: str, name: str) -> str:
    try:
        value = _get_json(f"https://api.github.com/repos/{owner}/{name}/readme")
        if value.get("encoding") == "base64":
            return base64.b64decode(value.get("content", "")).decode("utf-8", "replace")
    except Exception:
        pass
    return ""


def tree(owner: str, name: str, branch: str) -> list[dict[str, Any]]:
    try:
        value = _get_json(f"https://api.github.com/repos/{owner}/{name}/git/trees/{branch}"
                          "?recursive=1")
        return list(value.get("tree") or [])
    except Exception:
        return []


def raw_file(owner: str, name: str, branch: str, path: str) -> str:
    try:
        return _get_text(f"https://raw.githubusercontent.com/{owner}/{name}/{branch}/"
                         + urllib.parse.quote(path))
    except Exception:
        return ""


def _search_verified(title: str, arxiv_id: str, *, hits: int = 5) -> str | None:
    """A GitHub repository search hit, accepted only if it names the paper's arXiv id."""
    query = " ".join(re.sub(r"[^A-Za-z0-9 ]", " ", title or "").split()[:8])
    if not query or not arxiv_id:
        return None
    try:
        result = _get_json("https://api.github.com/search/repositories?q="
                           f"{urllib.parse.quote(query)}&sort=stars&order=desc&per_page={hits}")
    except Exception:
        return None
    for item in result.get("items") or []:
        owner = (item.get("owner") or {}).get("login", "")
        name = item.get("name", "")
        text = " ".join(str(item.get(key) or "")
                        for key in ("full_name", "description", "homepage")).lower()
        if arxiv_id in text or (owner and name and arxiv_id in readme(owner, name).lower()):
            return item.get("html_url")
    return None


def find_repository(paper: dict[str, Any]) -> dict[str, Any] | None:
    arxiv_id = arxiv_id_of(paper)
    url = paper.get("code_url") or _github_link(str(paper.get("abstract") or ""))
    if not url and arxiv_id:
        try:
            page = _get_text(f"https://arxiv.org/abs/{arxiv_id}")
            url = _github_link(page)
        except Exception:
            url = None
    if not url and arxiv_id:
        url = _search_verified(str(paper.get("title") or ""), arxiv_id)
    match = _GITHUB.search(url or "")
    if not match:
        return None
    owner, name = match.group(1), re.sub(r"\.git$", "", match.group(2))
    repository = {"url": f"https://github.com/{owner}/{name}", "owner": owner, "name": name,
                  "branch": "main"}
    try:
        meta = _get_json(f"https://api.github.com/repos/{owner}/{name}")
        repository["branch"] = meta.get("default_branch") or "main"
        repository["url"] = meta.get("html_url") or repository["url"]
    except Exception:
        pass
    return repository


_READER_SYSTEM = (
    "You are a meticulous code-reading agent. You are given the official repository of a research "
    "paper. Your goal is to understand how the paper's METHOD is implemented - its algorithm, "
    "control flow, the key classes and functions, and any exact prompts or templates in the code "
    "- so that a colleague can reimplement the method inside a different LLM agent. You only READ "
    "files; you never run, install or clone anything. Each turn you either read one file or "
    "finish. Go straight for the files that implement the method; skim configuration, tests or "
    "data only if they clarify it. Do not re-read files. Finish as soon as you can explain the "
    "implementation faithfully without guessing.")

_ACTION_SCHEMA = (
    'Respond with ONLY a JSON object:\n'
    '  {"thought": "<what you learned and what you still need>", "action": "read", '
    '"path": "<exact path from the tree>"}\n'
    'or, when you have gathered enough:\n'
    '  {"thought": "...", "action": "finish", "findings": "<the method as implemented: key files, '
    'classes and functions, control flow, decision rules, and prompts copied from the code>"}')


def _parse_action(raw: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        return {"action": "finish", "findings": (raw or "").strip()}
    try:
        return json.loads(match.group(0))
    except ValueError:
        action = re.search(r'"action"\s*:\s*"(\w+)"', match.group(0))
        path = re.search(r'"path"\s*:\s*"([^"]+)"', match.group(0))
        return {"action": action.group(1) if action else "finish",
                "path": path.group(1) if path else "",
                "findings": "" if action else (raw or "").strip()}


def read_repository(repository: dict[str, Any], paper: dict[str, Any], llm: LLM, *,
                    max_turns: int = 40, per_file_chars: int = 12_000,
                    on_log=print) -> CodeContext | None:
    owner, name, branch = repository["owner"], repository["name"], repository["branch"]
    files = sorted(((str(entry.get("path")), int(entry.get("size") or 0))
                    for entry in tree(owner, name, branch)
                    if entry.get("type") == "blob"
                    and not any(part in _SKIP_SEGMENTS for part in str(entry.get("path")).split("/"))
                    and str(entry.get("path")).lower().endswith(_KEEP_SUFFIXES)),
                   key=lambda item: (item[0].count("/"), item[0]))[:400]
    if not files:
        return None
    valid = {path for path, _ in files}
    listing = "\n".join(f"  {path}  ({size}B)" for path, size in files)
    messages = [
        {"role": "system", "content": _READER_SYSTEM},
        {"role": "user", "content": (
            f"PAPER: {paper.get('title')}\n\nREPOSITORY: {repository['url']} (read-only)\n\n"
            f"FILE TREE (path (size); use paths exactly as shown):\n{listing}\n\n"
            f"README (excerpt):\n{readme(owner, name)[:6000]}\n\n"
            f"Pick the first file to read.\n{_ACTION_SCHEMA}")},
    ]
    files_read: list[str] = []
    for turn in range(1, max_turns + 1):
        try:
            raw = llm(messages, temperature=0)
        except Exception as error:
            on_log(f"    code reader stopped: {type(error).__name__}: {error}")
            break
        messages.append({"role": "assistant", "content": raw})
        action = _parse_action(raw)
        if str(action.get("action")).lower() == "finish" and action.get("findings"):
            return CodeContext(repository=repository["url"], findings=str(action["findings"]),
                               files_read=files_read)
        if str(action.get("action")).lower() != "read" or turn == max_turns:
            break
        path = str(action.get("path") or "").strip().lstrip("/")
        if path not in valid:
            close = [candidate for candidate in valid if path and candidate.endswith(path)]
            path = close[0] if close else path
        content = raw_file(owner, name, branch, path) if path in valid else ""
        if not content.strip():
            observation = (f"(could not read '{path}'; pick another exact path from the tree, "
                           "or finish)")
        else:
            files_read.append(path)
            clipped = content[:per_file_chars]
            note = "" if len(content) <= per_file_chars else \
                f"\n...[truncated to {per_file_chars} chars]"
            observation = (f"FILE: {path}\n```\n{clipped}{note}\n```\n\nNext action?\n"
                           f"{_ACTION_SCHEMA}")
        messages.append({"role": "user", "content": observation})
    messages.append({"role": "user", "content": (
        "The reading budget is used up. Output ONLY JSON {\"findings\": \"...\"} describing the "
        "method as implemented: key files, classes and functions, control flow, decision rules, "
        "and prompts copied from the code.")})
    try:
        findings = _parse_action(llm(messages, temperature=0)).get("findings") or ""
    except Exception:
        findings = ""
    if not str(findings).strip():
        return None
    return CodeContext(repository=repository["url"], findings=str(findings),
                       files_read=files_read)


def read_code(paper: dict[str, Any], llm: LLM, *, max_turns: int = 40,
              on_log=print) -> CodeContext | None:
    """The paper's official implementation, if a repository can be found."""
    repository = find_repository(paper)
    if repository is None:
        return None
    on_log(f"    official code: {repository['url']}")
    return read_repository(repository, paper, llm, max_turns=max_turns, on_log=on_log)
