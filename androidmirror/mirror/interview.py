"""The interview: a corpus written by the person who answers it.

Two questionnaires, each grouped into stages: the OODA interview
(data/interview_ooda.json) asks how one person observes, orients, decides and
acts, in John Boyd's sense of the OODA loop; Get to Know You!
(data/interview_know.json) asks what they hold sacred, the images they carry of
themselves, their feelings, what they think doesn't matter, and what they count
as good and bad. One interview of a person holds answers to both, and both feed
the same corpus, so they land in the same knowledge graph and causal model. The
answers are saved after each one under data/interviews/<id>.json, so an interview
can be left and resumed, and any answer changed later.

`write_corpus` turns the answers into a documents corpus the usual build reads:
one passage per answer, in data/corpus/<corpus id>/, with its stage as the group,
and a spec in data/corpora/<corpus id>.json. A passage's file name carries a hash
of its text, so an answer changed after a build is read again and the others stay
cached. The question is kept in the passage as the interviewer's words, so the
coding attributes only the answer to the subject.

No Claude calls: the person's own words are the corpus.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

DATA = Path(__file__).resolve().parent.parent.parent / "data"
BANKS = {"ooda": DATA / "interview_ooda.json", "know": DATA / "interview_know.json"}  # in corpus order
PREFIX = "interview-"


def bank(bid: str = "ooda") -> dict[str, Any]:
    if bid not in BANKS:
        raise KeyError(bid)
    return {"id": bid, **json.loads(BANKS[bid].read_text(encoding="utf-8"))}


def stages() -> list[dict[str, Any]]:
    """Every stage of every questionnaire, in corpus order, each with its questionnaire's id."""
    return [{**s, "bank": bid} for bid in BANKS for s in bank(bid)["stages"]]


def questions(bid: str | None = None) -> list[dict[str, Any]]:
    """The questions of one questionnaire (or all), in order, with stage and number within the stage."""
    out = []
    for s in stages():
        if bid and s["bank"] != bid:
            continue
        for i, q in enumerate(s["questions"]):
            out.append({**q, "bank": s["bank"], "stage": s["id"], "stage_label": s["label"],
                        "n": i + 1, "of": len(s["questions"])})
    return out


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not s:
        raise ValueError("give the name of the person being interviewed")
    return s[:40]


def _path(data_dir: Path, iid: str) -> Path:
    if not re.fullmatch(r"[a-z0-9-]+", iid):
        raise KeyError(iid)
    return data_dir / "interviews" / f"{iid}.json"


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load(data_dir: Path, iid: str) -> dict[str, Any]:
    path = _path(data_dir, iid)
    if not path.exists():
        raise KeyError(iid)
    return json.loads(path.read_text(encoding="utf-8"))


def start(data_dir: Path, subject: str) -> dict[str, Any]:
    """A new interview of `subject`, or the one already started under that name."""
    subject = subject.strip()
    iid = slug(subject)
    try:
        return load(data_dir, iid)
    except KeyError:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        iv = {"id": iid, "subject": subject, "created": now, "updated": now, "answers": {}, "skipped": []}
        _write(_path(data_dir, iid), iv)
        return iv


def listing(data_dir: Path) -> list[dict[str, Any]]:
    folder = data_dir / "interviews"
    out = []
    for f in sorted(folder.glob("*.json")) if folder.exists() else []:
        iv = json.loads(f.read_text(encoding="utf-8"))
        out.append({"id": iv["id"], "subject": iv["subject"], "updated": iv["updated"], **progress(iv)})
    return sorted(out, key=lambda r: r["updated"], reverse=True)


def answer(data_dir: Path, iid: str, qid: str, text: str = "", skip: bool = False) -> dict[str, Any]:
    """Save one answer (or a skip) at once, so nothing is lost if the interview stops here."""
    iv = load(data_dir, iid)
    if qid not in {q["id"] for q in questions()}:
        raise KeyError(qid)
    text = text.strip()
    if skip or not text:
        iv["answers"].pop(qid, None)
        if qid not in iv["skipped"]:
            iv["skipped"].append(qid)
    else:
        iv["answers"][qid] = {"text": text, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        iv["skipped"] = [x for x in iv["skipped"] if x != qid]
    iv["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write(_path(data_dir, iid), iv)
    return iv


def next_question(iv: dict[str, Any], bid: str | None = None) -> str | None:
    """The first question neither answered nor skipped; then the first skipped one; None when all are answered."""
    qs = [q["id"] for q in questions(bid)]
    fresh = [q for q in qs if q not in iv["answers"] and q not in iv["skipped"]]
    later = [q for q in qs if q not in iv["answers"]]
    return (fresh or later or [None])[0]


def progress(iv: dict[str, Any]) -> dict[str, Any]:
    qs = questions()
    by_stage: dict[str, dict[str, Any]] = {}
    by_bank: dict[str, dict[str, Any]] = {}
    for q in qs:
        s = by_stage.setdefault(q["stage"], {"id": q["stage"], "label": q["stage_label"], "bank": q["bank"],
                                              "answered": 0, "skipped": 0, "total": 0})
        b = by_bank.setdefault(q["bank"], {"id": q["bank"], "title": bank(q["bank"])["title"],
                                           "answered": 0, "skipped": 0, "total": 0, "next": next_question(iv, q["bank"])})
        for r in (s, b):
            r["total"] += 1
            r["answered"] += q["id"] in iv["answers"]
            r["skipped"] += q["id"] in iv["skipped"]
    return {"answered": len(iv["answers"]), "skipped": len(iv["skipped"]), "total": len(qs),
            "stages": list(by_stage.values()), "banks": list(by_bank.values()), "next": next_question(iv)}


def view(data_dir: Path, iid: str) -> dict[str, Any]:
    iv = load(data_dir, iid)
    cid = PREFIX + iv["id"]
    return {**iv, "progress": progress(iv), "corpus": cid,
            "corpus_written": (specs_dir(data_dir) / f"{cid}.json").exists()}


def specs_dir(data_dir: Path) -> Path:
    """Where `build --corpus` looks for specs: data/corpora/ of the data directory in use."""
    return data_dir / "corpora"


def passage_text(subject: str, question: str, answer: str) -> str:
    return f"The interviewer asked: {question}\n\n{subject} answered: {answer}"


def write_corpus(data_dir: Path, iid: str) -> dict[str, Any]:
    """Write the answers so far as a corpus and its spec; returns the spec and the build command.

    Re-running it after more answers adds them; the folder is rewritten, so a changed or
    removed answer changes or leaves the corpus too.
    """
    iv = load(data_dir, iid)
    if not iv["answers"]:
        raise ValueError("answer at least one question first")
    cid = PREFIX + iv["id"]
    folder = data_dir / "corpus" / cid
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.txt"):
        old.unlink()
    order = {s["id"]: i + 1 for i, s in enumerate(stages())}  # OODA keeps 1-4, so its cached passages stay valid
    n = 0
    for q in questions():
        a = iv["answers"].get(q["id"])
        if not a:
            continue
        body = passage_text(iv["subject"], q["text"], a["text"])
        h = hashlib.sha256(body.encode()).hexdigest()[:6]
        name = f"{order[q['stage']]}-{q['id']}-{h}.txt"
        (folder / name).write_text(f"# {q['stage_label']} {q['n']}\ngroup: {q['stage_label']}\n{body}\n", encoding="utf-8")
        n += 1
    spec = {"id": cid, "title": f"Interview with {iv['subject']}", "subject": iv["subject"],
            "splitter": "documents", "dir": f"corpus/{cid}", "unit": "document"}
    _write(specs_dir(data_dir) / f"{cid}.json", spec)
    return {"spec": spec, "passages": n, "command": f"python -m androidmirror build --corpus {cid}"}


class StartIn(BaseModel):
    subject: str


class AnswerIn(BaseModel):
    qid: str
    text: str = ""
    skip: bool = False


def routes(data_dir: Path) -> Any:
    from fastapi import APIRouter, HTTPException

    r = APIRouter()

    def guard(fn: Any) -> Any:
        try:
            return fn()
        except KeyError as e:
            raise HTTPException(404, f"not found: {e.args[0]}") from None
        except ValueError as e:
            raise HTTPException(400, str(e)) from None

    @r.get("/api/interview")
    def index() -> dict:
        return {"banks": [{k: b[k] for k in ("id", "title", "about")} for b in map(bank, BANKS)],
                "stages": [{k: s[k] for k in ("id", "label", "about", "bank")} for s in stages()],
                "questions": questions(), "interviews": listing(data_dir)}

    @r.post("/api/interview")
    def post_start(s: StartIn) -> dict:
        return guard(lambda: view(data_dir, start(data_dir, s.subject)["id"]))

    @r.get("/api/interview/{iid}")
    def get_one(iid: str) -> dict:
        return guard(lambda: view(data_dir, iid))

    @r.post("/api/interview/{iid}/answer")
    def post_answer(iid: str, a: AnswerIn) -> dict:
        return guard(lambda: (answer(data_dir, iid, a.qid, a.text, a.skip), view(data_dir, iid))[1])

    @r.post("/api/interview/{iid}/corpus")
    def post_corpus(iid: str) -> dict:
        return guard(lambda: write_corpus(data_dir, iid))

    return r


def cli(data_dir: Path, subject: str, stage: str | None = None, inp: Any = input, out: Any = print,
        bid: str = "ooda") -> None:
    """Ask one questionnaire in the terminal, saving after each answer. Enter on an empty line skips; q stops."""
    iv = start(data_dir, subject)
    qs = [q for q in questions(bid) if not stage or q["stage"] == stage]
    if not qs:
        raise ValueError(f"no stage {stage!r} in {bank(bid)['title']}")
    todo = [q for q in qs if q["id"] not in iv["answers"]]
    b = next(x for x in progress(iv)["banks"] if x["id"] == bid)
    out(f"{bank(bid)['title']} with {iv['subject']}: {b['answered']} of {b['total']} answered. "
        "Enter on an empty line skips a question; q stops (everything so far is saved).\n")
    for q in todo:
        out(f"[{q['stage_label']} {q['n']}/{q['of']}] {q['text']}")
        try:
            text = inp("> ").strip()
        except EOFError:
            break
        if text.lower() in ("q", "quit"):
            break
        iv = answer(data_dir, iv["id"], q["id"], text, skip=not text)
        out("")
    p = progress(load(data_dir, iv["id"]))
    b = next(x for x in p["banks"] if x["id"] == bid)
    out(f"{b['answered']} of {b['total']} answered in {bank(bid)['title']}; {p['answered']} in all.")
    if p["answered"]:
        r = write_corpus(data_dir, iv["id"])
        out(f"Wrote {r['passages']} passage{'' if r['passages'] == 1 else 's'} as the corpus {r['spec']['id']}. Build it with:\n  {r['command']}")
