"""Download the German Society of Neurology's (DGN) English-language guidelines from Europe PMC.

The DGN publishes English versions of its guidelines as open-access papers, most in
its own journal, Neurological Research and Practice, and some in the Journal of
Neurology. Europe PMC indexes them: its search finds every paper titled
'... of the German Society of Neurology' that calls itself a guideline, and its
full-text service returns each one as JATS XML with its licence. Only papers whose
licence is in `licences` (CC BY by default) are kept, so the corpus can be reused.

Each top-level section of a paper (or each subsection, when it has them) is saved
as one .txt file with a "# <guideline> - <section>" header and the guideline as its
group, which the documents splitter reads. A manifest.json in the folder records
every paper's title, DOI and licence.

`conditions` narrows the search to guidelines whose title names one of them
(["parkinson"] by default); an empty list takes every DGN guideline.

Run on the user's machine (`python -m androidmirror build --corpus dgn`); the
cloud sessions that write this code cannot reach Europe PMC.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

REST = "https://www.ebi.ac.uk/europepmc/webservices/rest"
UA = {"User-Agent": "android-mirror/0.2 (research prototype)"}
XLINK = "{http://www.w3.org/1999/xlink}href"
SKIP_SECTIONS = re.compile(r"^(abbreviations|declarations|acknowledg|funding|author|conflicts? of interest|"
                           r"competing interests|ethics|consent|availability of|references|supplementary|"
                           r"electronic supplementary|publisher)", re.I)
# The paper's licence, as Creative Commons names it in the licence URL
LICENCE = re.compile(r"creativecommons\.org/licenses/([a-z-]+)/", re.I)


def query(conditions: list[str] | None) -> str:
    q = ('(TITLE:"German Society of Neurology" OR TITLE:"Deutsche Gesellschaft für Neurologie") '
         'AND TITLE:guideline AND OPEN_ACCESS:y AND HAS_FT:y')
    if conditions:
        q += " AND (" + " OR ".join(f"TITLE:{c}*" for c in conditions) + ")"
    return q


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def search(conditions: list[str] | None, get=_get) -> list[dict]:
    """Europe PMC hits with a PMC full text, newest first, without corrections or duplicates."""
    hits: dict[str, dict] = {}
    cursor = "*"
    while True:
        params = {"query": query(conditions), "format": "json", "resultType": "lite", "pageSize": 100,
                  "cursorMark": cursor, "sort": "P_PDATE_D desc"}
        page = json.loads(get(f"{REST}/search?{urllib.parse.urlencode(params)}"))
        results = page.get("resultList", {}).get("result", [])
        for r in results:
            title = r.get("title", "")
            if r.get("pmcid") and not re.match(r"^(correction|erratum|retraction)", title, re.I):
                hits.setdefault(r["pmcid"], r)
        nxt = page.get("nextCursorMark")
        if not results or not nxt or nxt == cursor:
            return list(hits.values())
        cursor = nxt


def licence(root: ET.Element) -> str:
    """'by', 'by-nc', ... from the paper's licence element, or '' if it has none."""
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag in ("license", "license_ref"):
            for s in [el.get(XLINK, ""), el.text or "", " ".join(el.itertext())]:
                m = LICENCE.search(s)
                if m:
                    return m.group(1).lower()
    return ""


def short_name(title: str) -> str:
    """'Pharmacotherapy of motor symptoms ...: guideline "Parkinson's disease" of the German Society
    of Neurology' -> "Parkinson's disease: pharmacotherapy of motor symptoms ..."."""
    title = re.sub(r"\s+", " ", title).strip().rstrip(".")
    m = re.search(r"guidelines? [\"“”„']([^\"“”„]+)[\"“”']", title, re.I)
    name = m.group(1).strip() if m else ""
    rest = re.sub(r"(?:the )?(?:S\dk?[- ])?guidelines? [\"“”„'][^\"“”„]+[\"“”'].*?"
                  r"(?:German Society of Neurology|Deutsche Gesellschaft für Neurologie)(?:\s*\([^)]*\))*",
                  "", title, flags=re.I)
    rest = re.sub(r"\b(?:S\dk?[- ]guideline|update \d{4})\b", "", rest, flags=re.I)
    rest = re.sub(r"^[\s:,;-]+|[\s:,;-]+$", "", rest)
    rest = re.sub(r"\s*:\s*:\s*", ": ", rest)
    if name and rest:
        return f"{name}: {rest[0].lower() + rest[1:]}"
    return name or rest or title


def _text(el: ET.Element) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _paras(sec: ET.Element) -> list[str]:
    """Paragraphs, boxed recommendations and table rows in a section, in order, skipping figures."""
    out: list[str] = []
    for el in sec:
        tag = el.tag.split("}")[-1]
        if tag in ("title", "label", "caption", "fig", "ref-list", "fn-group"):
            continue
        if tag == "p":
            # a paragraph can hold a table or box; take those apart rather than run them together
            if any(c.tag.split("}")[-1] in BLOCKS for c in el):
                buf = [el.text or ""]
                for c in el:
                    if c.tag.split("}")[-1] in BLOCKS:
                        out += _flush(buf) + _paras(_wrap(c))
                        buf = [c.tail or ""]
                    else:
                        buf += ["".join(c.itertext()), c.tail or ""]
                out += _flush(buf)
                continue
            t = _text(el)
            if t:
                out.append(t)
        elif tag == "list":
            out += [_text(li) for li in el if _text(li)]
        elif tag == "table-wrap":
            cap = el.find(".//caption")
            if cap is not None and _text(cap):
                out.append(_text(cap))
            for tr in el.iter():
                if tr.tag.split("}")[-1] == "tr":
                    cells = [_text(td) for td in tr if _text(td)]
                    if cells:
                        out.append(" | ".join(cells))
        elif tag in ("boxed-text", "disp-quote", "statement"):
            title = el.find("./caption")
            if title is not None and _text(title):
                out.append(_text(title))
            out += _paras(el)
        elif tag != "sec":
            t = _text(el)
            if t:
                out.append(t)
    return out


BLOCKS = ("table-wrap", "boxed-text", "list", "disp-quote")


def _flush(buf: list[str]) -> list[str]:
    t = re.sub(r"\s+", " ", "".join(buf)).strip()
    return [t] if t else []


def _wrap(el: ET.Element) -> ET.Element:
    box = ET.Element("box")
    box.append(el)
    return box


def _title(sec: ET.Element) -> str:
    t = sec.find("./title")
    return _text(t) if t is not None else ""


def _all_text(sec: ET.Element) -> list[str]:
    out = _paras(sec)
    for sub in sec.findall("./sec"):
        if _title(sub):
            out.append(f"{_title(sub)}.")
        out += _all_text(sub)
    return out


def sections(root: ET.Element) -> list[tuple[str, list[str]]]:
    """(section title, paragraphs) for each top-level section of the body, or each of its
    subsections when it has them; administrative sections are left out."""
    body = root.find(".//body")
    if body is None:
        return []
    out: list[tuple[str, list[str]]] = []
    loose = _paras(body)
    if loose:
        out.append(("Introduction", loose))
    for sec in body.findall("./sec"):
        title = _title(sec) or "Untitled"
        if SKIP_SECTIONS.match(title):
            continue
        subs = [s for s in sec.findall("./sec") if not SKIP_SECTIONS.match(_title(s))]
        if not subs:
            out.append((title, _all_text(sec)))
            continue
        own = _paras(sec)
        if own:
            out.append((title, own))
        for s in subs:
            out.append((f"{title} / {_title(s) or 'Untitled'}", _all_text(s)))
    return [(t, ps) for t, ps in out if ps]


def download(folder: Path, conditions: list[str] | None = None, licences: list[str] | None = None,
             delay: float = 0.5, get=_get, log=print) -> int:
    """Save each open-licensed DGN guideline, section by section. Returns files written."""
    conditions = ["parkinson"] if conditions is None else conditions
    allowed = {x.lower() for x in (licences or ["by"])}
    folder.mkdir(parents=True, exist_ok=True)
    hits = search(conditions, get=get)
    log(f"DGN: {len(hits)} open-access guideline papers on Europe PMC"
        + (f" for {', '.join(conditions)}" if conditions else ""))
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    written = 0
    for hit in hits:
        pmcid = hit["pmcid"]
        if any(folder.glob(f"{pmcid}_*.txt")):
            continue
        try:
            root = ET.fromstring(get(f"{REST}/{pmcid}/fullTextXML"))
        except Exception as e:
            log(f"  {pmcid}: {e}")
            continue
        lic = licence(root)
        name = short_name(hit.get("title", pmcid))
        entry = {"title": hit.get("title", ""), "doi": hit.get("doi", ""), "journal": hit.get("journalTitle", ""),
                 "year": hit.get("pubYear", ""), "licence": f"CC {lic.upper()}" if lic else "unknown",
                 "name": name}
        if lic not in allowed:
            log(f"  {pmcid} skipped, licence {entry['licence']}: {name}")
            manifest[pmcid] = {**entry, "kept": False}
            continue
        secs = sections(root)
        for i, (title, paras) in enumerate(secs, 1):
            (folder / f"{pmcid}_{i:02d}.txt").write_text(
                f"# {name} - {title}\ngroup: {name}\n" + "\n\n".join(paras) + "\n", encoding="utf-8")
        written += len(secs)
        manifest[pmcid] = {**entry, "kept": True, "sections": len(secs)}
        log(f"  {pmcid}: {len(secs)} sections, {name}")
        time.sleep(delay)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"DGN: saved {written} sections to {folder}")
    return written
