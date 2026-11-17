"""arXiv search client.

The arXiv API asks clients to make one request at a time and at most one request every three
seconds. This client serialises requests across local processes with a file lock, caches identical
queries for the day, and pages through relevance-ordered results until arXiv reports the end.
"""

from __future__ import annotations

import hashlib
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ARXIV_API = "https://export.arxiv.org/api/query"
DEFAULT_CATEGORIES = ("cs.IR", "cs.CL", "cs.AI", "cs.LG")
MIN_INTERVAL_SECONDS = 3.2
USER_AGENT = "ScholarEvolve/0.1 (literature retrieval)"
CACHE_ROOT = Path(os.environ.get("SCHOLAREVOLVE_CACHE",
                                 Path.home() / ".cache" / "scholarevolve")) / "arxiv"
_ATOM = "http://www.w3.org/2005/Atom"
_OPENSEARCH = "http://a9.com/-/spec/opensearch/1.1/"


class SearchIncomplete(RuntimeError):
    """The search did not finish, so its result is not a statement about the literature."""


def paper_key(paper: dict[str, Any]) -> str:
    """A version-insensitive arXiv identity, falling back to the normalised title."""
    arxiv_id = str(paper.get("arxiv_id") or "").strip().lower()
    if not arxiv_id:
        match = re.search(r"arxiv\.org/(?:abs|pdf)/([^?#/]+)", str(paper.get("url") or ""), re.I)
        arxiv_id = match.group(1).lower() if match else ""
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    title = re.sub(r"[^a-z0-9]+", " ", str(paper.get("title") or "").lower()).strip()
    return f"title:{title}" if title else ""


def _cache_path(url: str) -> Path:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return CACHE_ROOT / day / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.xml"


def _request(url: str, *, timeout: int) -> str:
    cached = _cache_path(url)
    if cached.is_file():
        return cached.read_text(encoding="utf-8")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    with (CACHE_ROOT / ".lock").open("a+", encoding="utf-8") as lock:
        try:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            fcntl = None  # type: ignore[assignment]
        try:
            if cached.is_file():
                return cached.read_text(encoding="utf-8")
            stamp = CACHE_ROOT / ".last_request"
            try:
                elapsed = time.time() - float(stamp.read_text(encoding="utf-8").strip())
                if elapsed < MIN_INTERVAL_SECONDS:
                    time.sleep(MIN_INTERVAL_SECONDS - elapsed)
            except (OSError, ValueError):
                pass
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(request, timeout=timeout,
                                            context=ssl.create_default_context()) as response:
                    payload = response.read().decode("utf-8")
            finally:
                stamp.write_text(str(time.time()), encoding="utf-8")
            cached.parent.mkdir(parents=True, exist_ok=True)
            temporary = cached.with_name(f".{cached.name}.{os.getpid()}.tmp")
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(cached)
            return payload
        finally:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _query_url(keywords: list[str], *, categories: tuple[str, ...], start: int,
               page_size: int) -> str:
    query = "(" + " OR ".join(f'all:"{keyword}"' for keyword in keywords) + ")"
    if categories:
        query += " AND (" + " OR ".join(f"cat:{category}" for category in categories) + ")"
    params = urllib.parse.urlencode({"search_query": query, "start": start,
                                     "max_results": page_size, "sortBy": "relevance",
                                     "sortOrder": "descending"})
    return f"{ARXIV_API}?{params}"


def _entry(node: ET.Element, ns: dict[str, str]) -> dict[str, Any]:
    arxiv_id = node.findtext("atom:id", default="", namespaces=ns).strip().split("/abs/")[-1]
    authors = [author.findtext("atom:name", default="", namespaces=ns)
               for author in node.findall("atom:author", ns)]
    return {
        "arxiv_id": arxiv_id,
        "title": re.sub(r"\s+", " ", node.findtext("atom:title", default="",
                                                    namespaces=ns).strip()),
        "abstract": re.sub(r"\s+", " ", node.findtext("atom:summary", default="",
                                                       namespaces=ns).strip()),
        "authors": authors,
        "published": node.findtext("atom:published", default="", namespaces=ns)[:10],
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "categories": [category.get("term") for category in node.findall("atom:category", ns)],
    }


def search(keywords: list[str], *, categories: tuple[str, ...] = DEFAULT_CATEGORIES,
           keyword_batch_size: int = 5, page_size: int = 100, max_pages: int = 10_000,
           max_results: int | None = None, timeout: int = 90, retries: int = 3,
           on_log=print) -> list[dict[str, Any]]:
    """Relevance-ordered arXiv results for the given phrases, deduplicated, in retrieval order.

    Phrases are ORed in batches of ``keyword_batch_size`` and each batch is paged until arXiv
    reports exhaustion. A page that cannot be fetched raises ``SearchIncomplete``.
    """
    keywords = list(dict.fromkeys(keyword.strip() for keyword in keywords if keyword.strip()))
    batches = [keywords[index:index + keyword_batch_size]
               for index in range(0, len(keywords), keyword_batch_size)]
    ns = {"atom": _ATOM, "opensearch": _OPENSEARCH}
    papers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for batch_index, batch in enumerate(batches, 1):
        before = len(papers)
        for page in range(max_pages):
            start = page * page_size
            url = _query_url(batch, categories=categories, start=start, page_size=page_size)
            root = None
            for attempt in range(retries):
                try:
                    root = ET.fromstring(_request(url, timeout=timeout))
                    break
                except (urllib.error.URLError, TimeoutError, OSError, ET.ParseError) as error:
                    code = getattr(error, "code", None)
                    wait = (60, 120, 300)[min(attempt, 2)] if code in (429, 503) \
                        else 10 * (attempt + 1)
                    on_log(f"  [arXiv] batch {batch_index} page {page + 1}: {error}; "
                           f"retrying in {wait}s")
                    time.sleep(wait)
            if root is None:
                raise SearchIncomplete(f"arXiv page {page + 1} of batch {batch_index} could not "
                                       f"be fetched ({len(papers)} papers so far)")
            entries = root.findall("atom:entry", ns)
            if not entries:
                break
            try:
                total = min(int(root.findtext("opensearch:totalResults", default="0",
                                              namespaces=ns)), 30_000)
            except ValueError:
                total = 30_000
            for node in entries:
                paper = _entry(node, ns)
                identity = paper_key(paper)
                if identity and identity not in seen:
                    seen.add(identity)
                    papers.append(paper)
                    if max_results and len(papers) >= max_results:
                        return papers
            if start + len(entries) >= total:
                break
        on_log(f"  [arXiv] batch {batch_index}/{len(batches)} ({', '.join(batch[:3])}"
               f"{'...' if len(batch) > 3 else ''}): {len(papers) - before} new papers")
    return papers
