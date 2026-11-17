"""Full-text reading of a selected paper: abstract, method sections and appendix.

The text is taken from arXiv's HTML rendering (falling back to the ar5iv mirror, then to the PDF).
Method, approach and algorithm sections and the appendix are kept; related work, experiments,
results and references are dropped. Code listings, algorithm boxes and prompt templates are also
extracted verbatim, because their exact form matters for implementation.
"""

from __future__ import annotations

import io
import re
import ssl
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

USER_AGENT = "ScholarEvolve/0.1 (paper reader)"

#: Section titles whose content is not needed to implement the method.
_DROP_TITLES = (
    "related work", "reference", "bibliography", "acknowledg", "broader impact", "ethic",
    "limitation", "experiment", "result", "evaluation", "ablation", "dataset", "benchmark",
    "baseline", "conclusion", "discussion",
)

_BLOCK_SIGNALS = ("prompt", "template", "instruction", "system message", "algorithm",
                  "pseudocode", "procedure")


@dataclass
class PaperContext:
    title: str
    arxiv_id: str
    url: str
    abstract: str
    method_text: str = ""
    verbatim_blocks: str = ""
    source: str = "abstract"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def arxiv_id_of(paper: dict[str, Any]) -> str:
    """The arXiv identifier without version suffix, from an explicit field or the paper URL."""
    value = str(paper.get("arxiv_id") or "").strip()
    if not value:
        match = re.search(r"arxiv\.org/(?:abs|pdf|html)/([0-9]{4}\.[0-9]{4,5})",
                          str(paper.get("url") or ""), re.I)
        value = match.group(1) if match else ""
    return re.sub(r"v\d+$", "", value.split("/abs/")[-1])


def _get(url: str, *, timeout: int = 60, binary: bool = False) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout,
                                context=ssl.create_default_context()) as response:
        data = response.read()
    return data if binary else data.decode("utf-8", "replace")


def _clean(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def fetch_html(arxiv_id: str) -> tuple[str | None, str | None]:
    for url in (f"https://arxiv.org/html/{arxiv_id}", f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"):
        try:
            html = _get(url)
        except Exception:
            continue
        if html and len(html) > 2000 and "ltx_" in html:
            return html, url
    return None, None


def method_text_from_html(html: str, *, max_chars: int) -> str:
    """Abstract, method sections and appendix from an arXiv HTML rendering."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        stripped = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
        return _clean(re.sub(r"<[^>]+>", " ", stripped))[:max_chars]
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(("script", "style", "nav", "header", "footer")):
        tag.decompose()
    for tag in soup.select(".ltx_bibliography, .ltx_biblist, .ltx_page_footer, .ltx_authors, "
                           ".ltx_dates, .ltx_role_affiliation"):
        tag.decompose()
    parts: list[str] = []
    abstract = soup.select_one(".ltx_abstract")
    if abstract:
        parts.append("## Abstract\n" + _clean(abstract.get_text("\n", strip=True)))
    for section in soup.select("section.ltx_section, section.ltx_appendix"):
        heading = section.find(["h1", "h2", "h3", "h4"])
        title = _clean(heading.get_text(" ", strip=True)) if heading else ""
        appendix = "ltx_appendix" in (section.get("class") or [])
        if not appendix and any(word in title.lower() for word in _DROP_TITLES):
            continue
        body = _clean(section.get_text("\n", strip=True))
        if len(body) > 60:
            parts.append(f"## {title}\n{body}" if title else body)
    text = "\n\n".join(parts)
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]"


def verbatim_blocks_from_html(html: str, *, max_chars: int, per_block: int = 8000) -> str:
    """Code listings, algorithm boxes and prompt figures, copied verbatim."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(("script", "style")):
        tag.decompose()
    picked: list[tuple[str, str]] = []
    seen: set[str] = set()

    def text_of(element: Any) -> str:
        lines = element.select(".ltx_listingline")
        if lines:
            value = "\n".join(line.get_text() for line in lines)
        elif element.name == "pre":
            value = element.get_text()
        else:
            value = element.get_text("\n", strip=True)
            if value.count("\n") > 8 and len(value) / (value.count("\n") + 1) < 4:
                value = element.get_text(" ", strip=True)
        return re.sub(r"[ \t]+\n", "\n", re.sub(r"\n{3,}", "\n\n", value)).strip()

    def label_of(element: Any) -> str:
        caption = (element.find("figcaption") or element.find(class_="ltx_caption")
                   or element.find(class_="ltx_title"))
        return _clean(caption.get_text(" ", strip=True))[:120] if caption else ""

    def add(element: Any, hint: str = "") -> None:
        value = text_of(element)
        if len(value) < 40 or value[:120] in seen:
            return
        seen.add(value[:120])
        picked.append((label_of(element) or hint, value[:per_block]))

    for element in soup.select("pre.ltx_verbatim, div.ltx_listing, div.ltx_lstlisting, "
                               ".ltx_float_algorithm, figure.ltx_algorithm, .ltx_algorithm"):
        add(element)
    for element in soup.select("figure.ltx_figure, figure.ltx_float, .ltx_framedbox"):
        caption = element.find("figcaption") or element.find(class_="ltx_caption")
        caption_text = caption.get_text(" ", strip=True).lower() if caption else ""
        head = element.get_text(" ", strip=True)[:400].lower()
        if any(signal in caption_text or signal in head for signal in _BLOCK_SIGNALS):
            add(element, caption_text[:120])
    out: list[str] = []
    used = 0
    for label, value in picked:
        if used >= max_chars:
            break
        block = f"\n### {label}\n{value}" if label else f"\n{value}"
        out.append(block)
        used += len(block)
    return "\n".join(out)[:max_chars]


def method_text_from_pdf(arxiv_id: str, *, max_chars: int) -> str:
    """Body text before the references plus any appendix, from the PDF (requires pypdf)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        reader = PdfReader(io.BytesIO(_get(f"https://arxiv.org/pdf/{arxiv_id}", binary=True)))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""
    text = _clean(text)
    match = re.search(r"\n\s*(References|REFERENCES|Bibliography)\s*\n", text)
    if match:
        body, tail = text[:match.start()], text[match.end():]
        appendix = re.search(r"\n\s*(Appendix|APPENDIX|A\s+[A-Z][A-Za-z ]{3,60})\s*\n", tail)
        text = body + ("\n\n## Appendix\n" + tail[appendix.start():] if appendix else "")
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]"


def read_paper(paper: dict[str, Any], *, max_method_chars: int = 150_000,
               max_verbatim_chars: int = 40_000) -> PaperContext:
    """Abstract, method sections, appendix and verbatim blocks of one selected paper."""
    arxiv_id = arxiv_id_of(paper)
    context = PaperContext(title=str(paper.get("title") or ""), arxiv_id=arxiv_id,
                           url=str(paper.get("url") or ""),
                           abstract=str(paper.get("abstract") or "").strip())
    if not arxiv_id:
        return context
    html, url = fetch_html(arxiv_id)
    if html:
        context.method_text = method_text_from_html(html, max_chars=max_method_chars)
        context.verbatim_blocks = verbatim_blocks_from_html(html, max_chars=max_verbatim_chars)
        context.source = url or "html"
    if not context.method_text.strip():
        context.method_text = method_text_from_pdf(arxiv_id, max_chars=max_method_chars)
        if context.method_text:
            context.source = "pdf"
    return context
