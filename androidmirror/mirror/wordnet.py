"""Open English WordNet, read for anchoring: word senses, glosses, hypernyms, antonyms.

The release (CC BY 4.0, https://en-word.net) is fetched once from its GitHub
release into data/wordnet/ and indexed to a compact JSON file there; nothing of
it is kept in the repository. Only what anchoring needs is indexed:

- lemmas: a written form (lower case) -> its synsets, in WordNet's sense order;
- synsets: part of speech, lexicographer file (noun.feeling, noun.cognition ...),
  definition, member words, hypernyms (instance hypernyms too), antonyms
  (lifted from sense-level antonymy to the synsets) and, for adjectives, the
  head synsets a satellite is similar to.
"""

from __future__ import annotations

import gzip
import json
import re
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

EDITION = "2025"
URL = (f"https://github.com/globalwordnet/english-wordnet/releases/download/{EDITION}-edition/"
       f"english-wordnet-{EDITION}.xml.gz")
CITATION = f"Open English WordNet {EDITION} (CC BY 4.0), https://en-word.net"
PREFIX = "oewn:"  # how a synset is written in an anchor file (oewn:07531593-n)


def folder(data_dir: Path) -> Path:
    return data_dir / "wordnet"


def curie(synset_id: str) -> str:
    """oewn-07531593-n -> oewn:07531593-n"""
    return PREFIX + synset_id.removeprefix("oewn-")


def uncurie(c: str) -> str:
    return "oewn-" + c.removeprefix(PREFIX)


def index_xml(stream: Any) -> dict[str, Any]:
    """The anchoring index of a WN-LMF file (an open binary stream)."""
    lemmas: dict[str, list[list[str]]] = {}
    sense_synset: dict[str, str] = {}
    sense_ants: list[tuple[str, str]] = []
    synsets: dict[str, dict[str, Any]] = {}
    words: dict[str, list[str]] = {}
    for _, el in ET.iterparse(stream, events=("end",)):
        tag = el.tag
        if tag == "LexicalEntry":
            lemma = el.find("Lemma")
            form, pos = lemma.get("writtenForm", ""), lemma.get("partOfSpeech", "")
            key = form.lower()
            for s in el.findall("Sense"):
                syn = s.get("synset")
                sense_synset[s.get("id")] = syn
                row = [syn, pos]
                bucket = lemmas.setdefault(key, [])
                if row not in bucket:
                    bucket.append(row)
                words.setdefault(syn, []).append(form)
                for r in s.findall("SenseRelation"):
                    if r.get("relType") == "antonym":
                        sense_ants.append((syn, r.get("target")))
            el.clear()
        elif tag == "Synset":
            rels: dict[str, list[str]] = {}
            for r in el.findall("SynsetRelation"):
                t = r.get("relType")
                if t in ("hypernym", "instance_hypernym", "similar"):
                    rels.setdefault("similar" if t == "similar" else "hyper", []).append(r.get("target"))
            d = el.find("Definition")
            synsets[el.get("id")] = {"pos": el.get("partOfSpeech", ""), "lex": el.get("lexfile", ""),
                                     "def": (d.text or "").strip() if d is not None else "",
                                     "hyper": rels.get("hyper", []), "similar": rels.get("similar", []), "ant": []}
            el.clear()
    for syn, target in sense_ants:
        other = sense_synset.get(target)
        if other and syn in synsets and other not in synsets[syn]["ant"]:
            synsets[syn]["ant"].append(other)
    for syn, ws in words.items():
        if syn in synsets:
            synsets[syn]["words"] = list(dict.fromkeys(ws))
    for s in synsets.values():
        s.setdefault("words", [])
        if not s["similar"]:
            s.pop("similar")
    return {"edition": EDITION, "lemmas": lemmas, "synsets": synsets}


def ensure(data_dir: Path, log: Any = print) -> Path:
    """The path of the index, downloading and indexing the release the first time."""
    root = folder(data_dir)
    idx = root / f"oewn-{EDITION}.json.gz"
    if idx.exists():
        return idx
    root.mkdir(parents=True, exist_ok=True)
    raw = root / f"english-wordnet-{EDITION}.xml.gz"
    if not raw.exists():
        log(f"downloading {CITATION} (about 11 MB, once)")
        tmp = raw.with_suffix(".part")
        with urllib.request.urlopen(URL, timeout=120) as r, open(tmp, "wb") as out:  # noqa: S310 (fixed https URL)
            while chunk := r.read(1 << 20):
                out.write(chunk)
        tmp.replace(raw)
    log("indexing WordNet (once, about a minute)")
    with gzip.open(raw, "rb") as f:
        index = index_xml(f)
    tmp = idx.with_suffix(".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as out:
        json.dump(index, out, separators=(",", ":"), ensure_ascii=False)
    tmp.replace(idx)
    return idx


_LOADED: dict[str, "WordNet"] = {}


def load(data_dir: Path, log: Any = print) -> "WordNet":
    path = ensure(data_dir, log)
    key = str(path)
    if key not in _LOADED:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            _LOADED[key] = WordNet(json.load(f))
    return _LOADED[key]


# ---------- reading it ----------

DETERMINERS = {"the", "a", "an", "one's", "ones", "one", "his", "her", "their", "our", "my", "your"}
_ES = ("ses", "xes", "zes", "ches", "shes")


def variants(word: str) -> list[str]:
    """Forms to look a label up under: as written, without determiners, singular (a small morphy)."""
    w = re.sub(r"[_\s]+", " ", word.strip().lower()).strip(" .,;:'\"")
    out = [w]
    toks = w.split()
    while toks and toks[0] in DETERMINERS:
        toks = toks[1:]
    if toks and " ".join(toks) != w:
        out.append(" ".join(toks))
    for x in list(out):
        if x.endswith("ies") and len(x) > 4:
            out.append(x[:-3] + "y")
        elif x.endswith(_ES):
            out.append(x[:-2])
        elif x.endswith("s") and not x.endswith("ss") and len(x) > 3:
            out.append(x[:-1])
    return list(dict.fromkeys(o for o in out if o))


class WordNet:
    def __init__(self, index: dict[str, Any]) -> None:
        self.lemmas: dict[str, list[list[str]]] = index["lemmas"]
        self.synsets: dict[str, dict[str, Any]] = index["synsets"]
        self.edition = index.get("edition", EDITION)
        self._up: dict[str, dict[str, int]] = {}

    def senses(self, word: str, pos: Iterable[str] | None = None) -> list[str]:
        """Synsets the word (or a variant of it) is a lemma of, in sense order; pos filters ('n', 'v', 'a', 's', 'r')."""
        want = set(pos) if pos else None
        out: list[str] = []
        for v in variants(word):
            for syn, p in self.lemmas.get(v, []):
                if (want is None or p in want or (p == "s" and "a" in want)) and syn not in out:
                    out.append(syn)
            if out:
                break  # the closest form that is in WordNet wins
        return out

    def get(self, syn: str) -> dict[str, Any]:
        return self.synsets[syn]

    def name(self, syn: str) -> str:
        s = self.synsets.get(syn)
        return s["words"][0] if s and s["words"] else syn

    def describe(self, syn: str) -> str:
        """'anger, choler, ire (noun.feeling): a strong emotion; ... [kind of: emotion]'"""
        s = self.synsets[syn]
        hyp = ", ".join(self.name(h) for h in s["hyper"][:2])
        return (f"{', '.join(s['words'][:5])} ({s['lex']}): {s['def']}" + (f" [kind of: {hyp}]" if hyp else ""))

    def ancestors(self, syn: str) -> dict[str, int]:
        """Every hypernym above a synset (itself at 0) with its shortest distance."""
        if syn not in self._up:
            dist, q = {syn: 0}, deque([syn])
            while q:
                x = q.popleft()
                for h in self.synsets.get(x, {}).get("hyper", []):
                    if h not in dist:
                        dist[h] = dist[x] + 1
                        q.append(h)
            self._up[syn] = dist
        return self._up[syn]

    def subsumes(self, upper: str, lower: str) -> int | None:
        """How many hypernym steps from `lower` up to `upper` (0 if the same), or None if `upper` is not above it."""
        return self.ancestors(lower).get(upper)

    def antonyms(self, syn: str) -> set[str]:
        """Antonym synsets; an adjective satellite takes its head's antonyms and the antonyms' satellites."""
        s = self.synsets.get(syn, {})
        out = set(s.get("ant", []))
        heads = [syn] + (s.get("similar", []) if s.get("pos") in ("a", "s") else [])
        for h in heads:
            for a in self.synsets.get(h, {}).get("ant", []):
                out.add(a)
                out |= set(self.synsets.get(a, {}).get("similar", []))
        return out - {syn}

    def opposite(self, x: str, y: str) -> bool:
        return y in self.antonyms(x) or x in self.antonyms(y)

    def lowest_common(self, x: str, y: str) -> tuple[str, int] | None:
        """The nearest shared hypernym and the summed distance to it."""
        ax, ay = self.ancestors(x), self.ancestors(y)
        common = [(ax[s] + ay[s], s) for s in ax.keys() & ay.keys()]
        if not common:
            return None
        d, s = min(common)
        return s, d
