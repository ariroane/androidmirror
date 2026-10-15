"""Seed corpus: Marcus Aurelius, Meditations (George Long translation, public domain).

The text is downloaded from Project Gutenberg on first use and cached in
data/corpus/. It is split into chunks of one numbered section each, so every
patch can point at "Book II, 1" and quote it.
"""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/2680/pg2680.txt"
WORK = "Marcus Aurelius, Meditations (tr. George Long)"

BOOK_WORDS = ["FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH", "SIXTH",
              "SEVENTH", "EIGHTH", "NINTH", "TENTH", "ELEVENTH", "TWELFTH"]
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]


@dataclass
class Chunk:
    book: int
    section: str
    text: str

    @property
    def locator(self) -> str:
        return f"Book {ROMAN[self.book - 1]}, {self.section}"


def download(data_dir: Path) -> Path:
    path = data_dir / "corpus" / "meditations.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(GUTENBERG_URL, timeout=60) as resp:
            path.write_bytes(resp.read())
    return path


def load_text(data_dir: Path) -> str | None:
    path = data_dir / "corpus" / "meditations.txt"
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else None


def strip_gutenberg(text: str) -> str:
    start = re.search(r"\*\*\* ?START OF.*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF", text)
    return text[start.end() if start else 0 : end.start() if end else len(text)]


def split_sections(text: str, max_chars: int = 6000) -> list[Chunk]:
    """Split into one chunk per numbered section, grouped by book.

    Books start at headings like "THE FIRST BOOK"; sections start at lines
    like "I." or "XII. ". Text before the first book (introduction, notes) is
    skipped. Very long sections are cut at paragraph breaks.
    """
    body = strip_gutenberg(text).replace("\r\n", "\n")
    book_re = re.compile(r"^\s*THE\s+(" + "|".join(BOOK_WORDS) + r")\s+BOOK\s*$", re.M)
    heads = list(book_re.finditer(body))
    chunks: list[Chunk] = []
    for i, m in enumerate(heads):
        book = BOOK_WORDS.index(m.group(1)) + 1
        end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
        book_text = body[m.end() : end]
        parts = re.split(r"^\s*([IVXLC]+)\.\s+", book_text, flags=re.M)
        # parts = [preamble, num, text, num, text, ...]
        for num, sec in zip(parts[1::2], parts[2::2]):
            sec = re.sub(r"\n{3,}", "\n\n", sec).strip()
            if not sec:
                continue
            for piece in _cut(sec, max_chars):
                chunks.append(Chunk(book=book, section=num, text=piece))
    return chunks


def _cut(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    out, cur = [], ""
    for para in text.split("\n\n"):
        if cur and len(cur) + len(para) > max_chars:
            out.append(cur.strip())
            cur = ""
        cur += para + "\n\n"
    if cur.strip():
        out.append(cur.strip())
    return out
