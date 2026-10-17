"""A corpus as a list of passages: the units of observation.

A corpus is described by a small JSON spec in data/corpora/ (title, subject,
where the text comes from, how to split it). The Meditations splitter keeps
"Book II, 1" locators; any other plain-text file is split into paragraphs.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from .. import corpus as p1corpus

SPECS = Path(__file__).resolve().parent.parent.parent / "data" / "corpora"


@dataclass
class Passage:
    id: str  # stable, e.g. "b02s001"
    locator: str  # human-readable, e.g. "Book II, 1"
    group: str  # a block for resampling and for splitting into regimes, e.g. "Book II"
    order: int
    text: str


@dataclass
class CorpusSpec:
    id: str
    title: str
    subject: str  # who or what the Mirror reflects
    splitter: str = "paragraphs"  # "meditations" | "paragraphs" | "documents"
    url: str | None = None
    file: str | None = None  # path relative to the data dir, or absolute
    min_chars: int = 300  # paragraphs splitter: merge short paragraphs up to this size
    dir: str | None = None  # documents splitter: a folder of .txt files (relative to the data dir)
    fetcher: str | None = None  # documents splitter: a module in mirror/sources that downloads them
    unit: str = "document"  # documents splitter: one passage per "document" or per "paragraph"
    options: dict | None = None  # passed to the fetcher

    @classmethod
    def load(cls, name_or_path: str) -> "CorpusSpec":
        path = Path(name_or_path)
        if not path.exists():
            path = SPECS / f"{name_or_path}.json"
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        return asdict(self)


def text_path(spec: CorpusSpec, data_dir: Path) -> Path:
    if spec.file:
        p = Path(spec.file)
        return p if p.is_absolute() else data_dir / p
    return data_dir / "corpus" / f"{spec.id}.txt"


def fetch(spec: CorpusSpec, data_dir: Path) -> str:
    """The corpus text, downloading it once if the spec has a URL.

    For a documents corpus this returns the folder path; `split` reads the files.
    """
    if spec.splitter == "documents":
        folder = docs_dir(spec, data_dir)
        if spec.fetcher and not any(folder.glob("*.txt")):
            import importlib

            importlib.import_module(f".sources.{spec.fetcher}", __package__).download(folder, **(spec.options or {}))
        if not any(folder.glob("*.txt")):
            raise FileNotFoundError(f"no .txt documents in {folder}")
        return str(folder)
    path = text_path(spec, data_dir)
    if not path.exists():
        legacy = data_dir / "corpus" / "meditations.txt"  # where Prototype 1 cached it
        if spec.splitter == "meditations" and legacy.exists():
            return legacy.read_text(encoding="utf-8", errors="replace")
        if not spec.url:
            raise FileNotFoundError(f"no text at {path} and no url in the corpus spec")
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(spec.url, timeout=60) as resp:
            path.write_bytes(resp.read())
    return path.read_text(encoding="utf-8", errors="replace")


def docs_dir(spec: CorpusSpec, data_dir: Path) -> Path:
    p = Path(spec.dir or f"corpus/{spec.id}")
    return p if p.is_absolute() else data_dir / p


def split(spec: CorpusSpec, text: str) -> list[Passage]:
    if spec.splitter == "documents":
        return split_documents(Path(text), spec.unit, spec.min_chars)
    if spec.splitter == "meditations":
        out = []
        for i, c in enumerate(p1corpus.split_sections(text)):
            sid = f"b{c.book:02d}s{_roman(c.section):03d}"
            n = sum(1 for p in out if p.id.startswith(sid))  # long sections are cut in pieces
            out.append(Passage(sid + (f"_{n}" if n else ""), c.locator + (f" ({n + 1})" if n else ""),
                               f"Book {p1corpus.ROMAN[c.book - 1]}", i, c.text))
        return out
    return split_paragraphs(text, spec.min_chars)


def split_paragraphs(text: str, min_chars: int = 300) -> list[Passage]:
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text.replace("\r\n", "\n"))]
    out: list[Passage] = []
    cur = ""
    for p in paras:
        if not p:
            continue
        cur = f"{cur}\n\n{p}" if cur else p
        if len(cur) >= min_chars:
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return [Passage(f"p{i + 1:04d}", f"¶ {i + 1}", f"part {i // 25 + 1}", i, t) for i, t in enumerate(out)]


def split_documents(folder: Path, unit: str = "document", min_chars: int = 300) -> list[Passage]:
    """One passage per .txt file (or per paragraph of each), in file-name order.

    A file may start with a header line "# <locator>" (e.g. "# Statement, 2008-12-16") and
    a line "group: <name>" (e.g. "group: 2008"); otherwise the file name stands in.
    """
    out: list[Passage] = []
    for f in sorted(folder.glob("*.txt")):
        raw = f.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
        locator, group = f.stem, f.stem[:4]
        lines = raw.split("\n")
        while lines and (lines[0].startswith("# ") or lines[0].startswith("group:")):
            head = lines.pop(0)
            if head.startswith("# "):
                locator = head[2:].strip()
            else:
                group = head.split(":", 1)[1].strip()
        body = "\n".join(lines).strip()
        if not body:
            continue
        if unit == "paragraph":
            for i, p in enumerate(split_paragraphs(body, min_chars)):
                out.append(Passage(f"{f.stem}_p{i + 1:02d}", f"{locator}, ¶ {i + 1}", group, len(out), p.text))
        else:
            out.append(Passage(f.stem, locator, group, len(out), re.sub(r"\n{3,}", "\n\n", body)))
    return out


def _roman(s: str) -> int:
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total = 0
    for a, b in zip(s, s[1:] + " "):
        total += -vals[a] if vals.get(b, 0) > vals[a] else vals[a]
    return total
