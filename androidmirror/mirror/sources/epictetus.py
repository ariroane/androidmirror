"""Download Epictetus's Discourses, Enchiridion and Fragments (tr. George Long, 1877).

The text is the Standard Ebooks edition of Long's translation, read from its
source repositories on GitHub (one XHTML file per book of the Discourses, one
each for the Enchiridion and the Fragments). Long's translation and the
Standard Ebooks edition are public domain.

Each chapter of the Discourses and each chapter of the Enchiridion is saved as
one .txt file with a "# <locator>" header and its book as the group, which the
documents splitter reads. The Fragments are short, so they are saved ten to a
file, each paragraph starting with its number. Arrian's preface, Long's
introductions and notes, and the fragments Long marks as only partly Epictetus's
(CXXXVII onward) are left out.

Run on the user's machine (`python -m androidmirror build --corpus epictetus`).
"""

from __future__ import annotations

import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

RAW = "https://raw.githubusercontent.com/standardebooks"
DISCOURSES = RAW + "/epictetus_discourses_george-long/master/src/epub/text/book-{n}.xhtml"
ENCHIRIDION = RAW + "/epictetus_short-works_george-long/master/src/epub/text/the-enchiridion.xhtml"
FRAGMENTS = RAW + "/epictetus_short-works_george-long/master/src/epub/text/fragments.xhtml"
UA = {"User-Agent": "android-mirror/0.2 (research prototype)"}
X = "{http://www.w3.org/1999/xhtml}"
EPUB_TYPE = "{http://www.idpf.org/2007/ops}type"
ROMAN = ["I", "II", "III", "IV"]
LAST_FRAGMENT = 136  # Long: "the fragments which follow are in part assigned to Epictetus, in part to others"


def _text(el: ET.Element) -> str:
    """An element's text without footnote markers; a line break in verse becomes " / "."""
    parts = [el.text or ""]
    for child in el:
        tag = child.tag.replace(X, "")
        if tag == "a" and "noteref" in child.get(EPUB_TYPE, ""):
            pass
        elif tag == "br":
            parts.append(" / ")
        else:
            parts.append(_text(child))
        parts.append(child.tail or "")
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _paragraphs(section: ET.Element) -> list[str]:
    """The paragraphs of a chapter, in order: prose, verse, and drama rows as "Speaker: line"."""
    out: list[str] = []
    for el in section:
        tag = el.tag.replace(X, "")
        if tag in ("hgroup", "header") or tag.startswith("h"):
            continue
        if tag == "p":
            t = _text(el)
        elif tag == "blockquote":
            t = " ".join(_text(p) for p in el.iter(X + "p")) or _text(el)
        elif tag == "table":
            t = " ".join(": ".join(_text(td) for td in tr.iter(X + "td")) for tr in el.iter(X + "tr"))
        else:
            t = "\n\n".join(_paragraphs(el)) if tag == "section" else _text(el)
        if t:
            out.append(t)
    return out


def _chapters(xhtml: str, prefix: str) -> list[tuple[int, str, list[str]]]:
    """(number, title, paragraphs) for each section whose id is <prefix><n>."""
    root = ET.fromstring(xhtml)
    pat = re.compile(re.escape(prefix) + r"(\d+)$")
    out = []
    for sec in root.iter(X + "section"):
        m = pat.match(sec.get("id", ""))
        if not m:
            continue
        title = next((_text(p) for p in sec.iter(X + "p") if p.get(EPUB_TYPE) == "title"), "")
        out.append((int(m.group(1)), title, _paragraphs(sec)))
    return out


def _roman(n: int) -> str:
    out = ""
    for v, s in ((100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= v:
            out += s
            n -= v
    return out


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8")


def _write(path: Path, locator: str, group: str, paras: list[str]) -> None:
    path.write_text(f"# {locator}\ngroup: {group}\n" + "\n\n".join(paras) + "\n", encoding="utf-8")


def download(folder: Path, discourses: bool = True, enchiridion: bool = True, fragments: bool = True,
             per_file: int = 10, delay: float = 0.5, get=_get, log=print) -> int:
    """Save the chapters as text files. Returns files written."""
    folder.mkdir(parents=True, exist_ok=True)
    written = 0
    if discourses:
        for b in range(1, 5):
            for n, title, paras in _chapters(get(DISCOURSES.format(n=b)), f"chapter-{b}-"):
                loc = f"Discourses {ROMAN[b - 1]}, {n}" + (f": {title}" if title else "")
                _write(folder / f"d{b}-{n:02d}.txt", loc, f"Discourses {ROMAN[b - 1]}", paras)
                written += 1
            time.sleep(delay)
    if enchiridion:
        for n, _, paras in _chapters(get(ENCHIRIDION), "the-enchiridion-"):
            _write(folder / f"e-{n:02d}.txt", f"Enchiridion {n}", "Enchiridion", paras)
            written += 1
        time.sleep(delay)
    if fragments:
        frags = [(n, paras) for n, _, paras in _chapters(get(FRAGMENTS), "fragments-") if n <= LAST_FRAGMENT]
        for i in range(0, len(frags), per_file):
            chunk = frags[i:i + per_file]
            paras = [f"{_roman(n)}. " + " ".join(p) for n, p in chunk]
            first, last = chunk[0][0], chunk[-1][0]
            _write(folder / f"f-{first:03d}.txt", f"Fragments {_roman(first)}-{_roman(last)}", "Fragments", paras)
            written += 1
    log(f"Epictetus: saved {written} files to {folder}")
    return written
