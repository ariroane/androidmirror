"""Download FOMC policy statements (and optionally minutes) from federalreserve.gov.

The Federal Reserve's FOMC pages list every meeting: the current calendar
(fomccalendars.htm) covers the last few years and fomchistoricalYYYY.htm the
earlier ones. Each statement is saved as one .txt file named by its date, with
a "# Statement, YYYY-MM-DD" header line and its year as the group, which the
documents splitter reads. Federal Reserve publications are public domain.

Run on the user's machine (`python -m androidmirror build --corpus fomc`); the
cloud sessions that write this code cannot reach federalreserve.gov.
"""

from __future__ import annotations

import re
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

BASE = "https://www.federalreserve.gov"
UA = {"User-Agent": "android-mirror/0.2 (research prototype)"}

STATEMENT = [re.compile(r"/newsevents/pressreleases/monetary(\d{8})a\.htm"),
             re.compile(r"/boarddocs/press/(?:monetary|general)/\d{4}/(\d{8})/default\.htm")]
MINUTES = [re.compile(r"/monetarypolicy/fomcminutes(\d{8})\.htm"), re.compile(r"/fomc/minutes/(\d{8})\.htm")]
SKIP = re.compile(r"^(voting for|voting against|implementation note|for media inquiries|for release at|"
                  r"last update|share|board of governors|release date|for immediate release)", re.I)


class _Text(HTMLParser):
    """Paragraph texts of a page, ignoring scripts, styles, navigation and footers."""

    def __init__(self) -> None:
        super().__init__()
        self.paras: list[str] = []
        self._buf: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "footer", "header"):
            self._skip += 1
        if tag in ("p", "br", "div", "li", "h1", "h2", "h3", "h4", "td"):
            self._flush()

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "footer", "header"):
            self._skip = max(0, self._skip - 1)
        if tag in ("p", "div", "li", "h1", "h2", "h3", "h4", "td"):
            self._flush()

    def handle_data(self, data):
        if not self._skip:
            self._buf.append(data)

    def _flush(self):
        t = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        self._buf = []
        if t:
            self.paras.append(t)


def page_text(html: str, min_chars: int = 60) -> list[str]:
    p = _Text()
    p.feed(html)
    p.close()
    return [t for t in p.paras if len(t) >= min_chars and not SKIP.match(t)]


def links(html: str, kinds: tuple[str, ...]) -> dict[str, dict[str, str]]:
    """{date: {kind: url}} for statement and minutes links on an index page."""
    out: dict[str, dict[str, str]] = {}
    pats = {"statement": STATEMENT, "minutes": MINUTES}
    for href in re.findall(r'href="([^"]+)"', html):
        for kind in kinds:
            for pat in pats[kind]:
                m = pat.search(href)
                if m:
                    url = href if href.startswith("http") else BASE + href
                    out.setdefault(m.group(1), {}).setdefault(kind, url)
    return out


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def index_pages(start: int, end: int) -> list[str]:
    pages = [f"{BASE}/monetarypolicy/fomccalendars.htm"]
    pages += [f"{BASE}/monetarypolicy/fomchistorical{y}.htm" for y in range(start, end + 1)]
    return pages


def download(folder: Path, start: int = 1994, end: int = 2100, kinds: list[str] | None = None,
             delay: float = 0.5, get=_get, log=print) -> int:
    """Save each meeting's statement (and minutes if asked) as a text file. Returns files written."""
    kinds_t = tuple(kinds or ["statement"])
    folder.mkdir(parents=True, exist_ok=True)
    found: dict[str, dict[str, str]] = {}
    end = min(end, time.localtime().tm_year)
    for page in index_pages(start, end):
        try:
            html = get(page)
        except Exception:  # recent years have no historical page yet
            continue
        for date, ks in links(html, kinds_t).items():
            if start <= int(date[:4]) <= end:
                found.setdefault(date, {}).update(ks)
    log(f"FOMC: {len(found)} meetings with documents between {start} and {end}")
    written = 0
    for date in sorted(found):
        for kind, url in found[date].items():
            path = folder / f"{date}-{kind}.txt"
            if path.exists():
                continue
            try:
                paras = page_text(get(url))
            except Exception as e:
                log(f"  {date} {kind}: {e}")
                continue
            if not paras:
                continue
            iso = f"{date[:4]}-{date[4:6]}-{date[6:]}"
            path.write_text(f"# {kind.capitalize()}, {iso}\ngroup: {date[:4]}\n" + "\n\n".join(paras) + "\n",
                            encoding="utf-8")
            written += 1
            time.sleep(delay)
    log(f"FOMC: saved {written} documents to {folder}")
    return written
