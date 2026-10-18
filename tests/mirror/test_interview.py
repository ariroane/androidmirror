from fastapi.testclient import TestClient

from androidmirror.mirror import interview as I
from androidmirror.mirror.build import build
from androidmirror.mirror.corpus import CorpusSpec
from androidmirror.server import create_app

from .fake import FakeCoder


def test_the_bank_covers_the_loop():
    qs = I.questions()
    assert len(qs) >= 100
    assert [s["id"] for s in I.bank()["stages"]] == ["observe", "orient", "decide", "act"]
    assert all(sum(q["stage"] == s for q in qs) >= 25 for s in ("observe", "orient", "decide", "act"))
    assert len({q["id"] for q in qs}) == len(qs) and len({q["text"] for q in qs}) == len(qs)


def test_answers_are_saved_as_they_come_and_resume(tmp_path):
    iv = I.start(tmp_path, "Android K")
    assert iv["id"] == "android-k" and I.next_question(iv) == "observe-01"
    I.answer(tmp_path, "android-k", "observe-01", "I notice the exits first, because I was once trapped.")
    I.answer(tmp_path, "android-k", "observe-02", skip=True)
    # a new start under the same name resumes, with the skipped question left for the end
    iv = I.start(tmp_path, "Android K")
    assert iv["answers"]["observe-01"]["text"].startswith("I notice")
    assert I.next_question(iv) == "observe-03"
    for q in I.questions()[2:]:
        I.answer(tmp_path, "android-k", q["id"], "fine")
    assert I.next_question(I.load(tmp_path, "android-k")) == "observe-02"
    # answering a skipped question takes it off the skipped list
    iv = I.answer(tmp_path, "android-k", "observe-02", "people I have watched keep their word")
    assert iv["skipped"] == [] and I.next_question(iv) is None
    assert I.progress(iv)["answered"] == len(I.questions())


def test_the_answers_become_a_corpus_the_build_reads(tmp_path):
    I.start(tmp_path, "Tester")
    tagged = "[v alpha 1] [v gamma 1] [j causes alpha gamma + 1 direct]"
    I.answer(tmp_path, "tester", "orient-23", f"When I rest, I see clearly. {tagged}")
    I.answer(tmp_path, "tester", "act-13", "I look at whether people came back. [v gamma 1]")
    I.answer(tmp_path, "tester", "decide-01", skip=True)
    r = I.write_corpus(tmp_path, "tester")
    assert r["passages"] == 2 and r["spec"]["id"] == "interview-tester" and "--corpus interview-tester" in r["command"]
    spec = CorpusSpec.load(str(tmp_path / "corpora" / "interview-tester.json"))
    coder = FakeCoder()
    m = build(spec, tmp_path, coder=coder, log=lambda *_: None)
    passages = {p["group"]: p for p in __import__("json").loads((tmp_path / "mirror" / spec.id / "passages.json").read_text())}
    # in stage order; the question is the interviewer's, the answer the subject's
    assert list(passages) == ["Orient", "Act"]
    assert passages["Orient"]["locator"] == "Orient 23"
    assert passages["Orient"]["text"].startswith("The interviewer asked: What do you think causes success")
    assert "Tester answered: When I rest" in passages["Orient"]["text"]
    assert {(e["src"], e["dst"]) for e in m["causal"]["edges"]} == {("alpha", "gamma")}
    assert coder.calls["code"] == 2
    # a changed answer is read again; the others stay cached
    I.answer(tmp_path, "tester", "act-13", "I look at whether people came back, and whether I slept. [v gamma 1]")
    I.write_corpus(tmp_path, "tester")
    build(spec, tmp_path, coder=coder, log=lambda *_: None)
    assert coder.calls["code"] == 3


def test_routes(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = TestClient(create_app(tmp_path))
    b = c.get("/api/interview").json()
    assert len(b["questions"]) >= 100 and b["interviews"] == []
    iv = c.post("/api/interview", json={"subject": "Ari"}).json()
    assert iv["progress"]["next"] == "observe-01" and not iv["corpus_written"]
    assert c.post("/api/interview/ari/corpus").status_code == 400  # nothing answered yet
    iv = c.post("/api/interview/ari/answer", json={"qid": "observe-01", "text": "Faces."}).json()
    assert iv["progress"]["answered"] == 1 and iv["progress"]["next"] == "observe-02"
    assert c.post("/api/interview/ari/answer", json={"qid": "nope", "text": "x"}).status_code == 404
    assert c.get("/api/interview/../etc").status_code == 404
    assert c.post("/api/interview/ari/corpus").json()["passages"] == 1
    assert c.get("/api/interview/ari").json()["corpus_written"]
    assert c.get("/api/interview").json()["interviews"][0]["answered"] == 1
