"""Adversarial variants of the planted-model benchmark (offline, no Claude).

Run: python adversarial.py <path to androidmirror repo>   (needs numpy, networkx)
"""
import random, sys, statistics
from collections import Counter
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")  # the androidmirror repo
from androidmirror.mirror import benchmark as b
from androidmirror.mirror.causal import discover
from androidmirror.mirror.causal.pdag import PDAG

V = [v for v in b.ORDER if v != "wrath"]
CONCEPTS = {v: {"label": v, "variable": True, "kind": "state"} for v in b.ORDER}
CFG = {"alpha": 0.05, "max_depth": 2, "bootstrap": 0, "compare_views": False, "view": "implied", "topic": True,
       "selection": {"min_support": 5, "max_vars": 25}, "variables": V, "vignettes": False}
# the subject's approved pole for each variable (what a Stoic praises): +1 good to have high, -1 good to have low
APPROVED = {"misfortune": 0, "wrongdoing": -1, "reason": 1, "mortality": 1, "desire_for_fame": -1,
            "judgment_of_harm": -1, "anger": -1, "wrath": -1, "distress": -1, "tranquillity": 1, "cooperation": 1,
            "gratitude": 1}


def run(written, truth, sits_fn=None):
    labels = b.grouped_topics(written)
    sits = (sits_fn or b._situations)(written, labels)
    r = discover(sits, V, CONCEPTS, [], [], CFG)
    if "cpdag" not in r:
        return {"edges": 0}
    out = b._score_model(PDAG.from_dict(r["cpdag"]), truth, V)
    # how many surviving edges were never tested conditionally on enough rows
    from androidmirror.mirror import situations as sit
    from androidmirror.mirror.causal import _pc
    data, _ = sit.table(sits, V, "implied", topic=True)
    from androidmirror.mirror.causal import DEFAULTS; found = _pc(data, V, {**DEFAULTS, **CFG})
    es = found["edge_support"]
    mine = [k for k in es if "topic" not in k]
    out["edges_never_tested_conditionally"] = sum(1 for k in mine if es[k]["tests"] - es[k].get("untestable", 0) <= 1)
    return out


def avg(rows):
    keys = {k for r in rows for k in r}
    return {k: round(statistics.mean(r[k] for r in rows if r.get(k) is not None), 2) for k in sorted(keys)
            if any(r.get(k) is not None for r in rows)}


def mention(rng, t, k):
    weights = {v: b.TOPICS[t].get(v, 0.3) for v in b.ORDER}
    out = []
    while len(out) < k:
        left = [v for v in b.ORDER if v not in out]
        out.append(rng.choices(left, [weights[v] for v in left])[0])
    return out


def valence_corpus(n, seed, p_exemplar=0.6, fidelity=0.8):
    """No causation. Each passage is a portrait (exemplary or cautionary) or neutral; a portrait puts each
    mentioned variable at its approved pole (or the opposite) with probability `fidelity`. Levels are +-1 only,
    as the real coder writes them (3 'typical' cells out of 230)."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        t = rng.choice(list(b.TOPICS))
        frame = rng.choices([1, -1, 0], [0.75 * p_exemplar, 0.25 * p_exemplar, 1 - p_exemplar])[0]
        st = {}
        for v in b.ORDER:
            pole = APPROVED[v]
            if frame and pole and rng.random() < fidelity:
                st[v] = pole * frame
            else:
                st[v] = rng.choice([-1, 1])
        st["wrath"] = st["anger"]
        out.append({"topic": t, "state": st, "mentioned": mention(rng, t, rng.randint(2, 5))})
    return out


def sparse_corpus(n, seed, kmin=2, kmax=3):
    """The planted model, but as sparse as the real build: about 2.8 stated variables per situation."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        t = rng.choice(list(b.TOPICS))
        st = b.sample_states(1, rng)[0]
        out.append({"topic": t, "state": st, "mentioned": mention(rng, t, rng.randint(kmin, kmax))})
    return out


# Claude's folk model: differs from the subject's on purpose (an ordinary modern reader)
FOLK = [("misfortune", "distress", 1), ("wrongdoing", "anger", 1), ("anger", "distress", 1),
        ("desire_for_fame", "cooperation", 1), ("mortality", "distress", 1), ("gratitude", "tranquillity", 1),
        ("reason", "tranquillity", 1), ("distress", "cooperation", -1)]


def leaky_situations(q):
    """The reader states one extra, unmentioned variable per situation with probability q, setting it from
    Claude's folk model given what is stated (what 'plainly shows' lets in)."""
    def fn(written, topics):
        sits = b._situations(written, topics)
        rng = random.Random(99)
        for s, w in zip(sits, written):
            if rng.random() > q:
                continue
            known = {v: c["level"] for v, c in s["cells"].items()}
            options = [(a, bb, sg) for a, bb, sg in FOLK if a in known and bb not in known and bb in V]
            if not options:
                continue
            a, bb, sg = rng.choice(options)
            s["cells"][bb] = {"level": sg * known[a], "tier": "stated", "quote": ""}
        return sits
    return fn


def folk_score(written, fn):
    labels = b.grouped_topics(written)
    r = discover(fn(written, labels), V, CONCEPTS, [], [], CFG)
    g = PDAG.from_dict(r["cpdag"])
    adj = {frozenset(e) for e in g.undirected} | {frozenset(d) for d in g.directed}
    subj = {frozenset((a, c)) for a, c, _ in b.EDGES if a in V and c in V}
    folk = {frozenset((a, c)) for a, c, _ in FOLK}
    return {"edges": len(adj), "subject_edges_found": len(adj & subj), "folk_only_edges_found": len(adj & (folk - subj)),
            "folk_only_total": len(folk - subj)}


if __name__ == "__main__":
    seeds = range(3)
    print("A. valence null (no causation; exemplary/cautionary portraits; +-1 levels), n=600")
    for p in (0.0, 0.3, 0.6):
        print(f"  p_exemplar={p}", avg([run(valence_corpus(600, 200 + s, p_exemplar=p), None) for s in seeds]))
    print("B. planted model at the real build's sparsity (2-3 stated variables per situation)")
    for n in (600, 1150, 3000):
        print(f"  n={n}", avg([run(sparse_corpus(n, 300 + s), b.EDGES) for s in seeds]))
    print("   reference: benchmark sparsity (2-5), n=600", avg([run(b.corpus(600, 100 + s), b.EDGES) for s in seeds]))
    print("C. reader leak: Claude's folk model fills one unmentioned variable with probability q (planted corpus, n=600)")
    for q in (0.0, 0.15, 0.3):
        print(f"  q={q}", avg([folk_score(b.corpus(600, 100 + s), leaky_situations(q)) for s in seeds]))
