"""Topics: what each passage is about, so the causal tests can hold it fixed.

A topic (death, the court, other people's faults) makes several concepts come up
together and can colour their levels too: a mood shared by a whole topic is a
hidden common cause, and it shows up as links between concepts that do not
affect each other in the subject's model. Conditioning on the topic asks
whether two concepts still go together within a topic.

Topics are found without Claude: each passage is a bag of the concepts coded in
it (weighted up) and its content words, weighted by TF-IDF, and the passages are
grouped by spherical k-means. Every situation takes its passage's topic. The
topic is only as good as this grouping: in the planted-model benchmark a topic
label that is wrong for 30% of passages removes little of the confounding,
while an accurate one removes most of it.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np

STOP = set("""a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each even ever every few for from further had has have having
he her here hers herself him himself his how i if in into is it its itself just let like may me might more most must my
myself no nor not now o of off on once one only or other ought our ours ourselves out over own same say says shall she
should so some such than that the thee their theirs them themselves then there these they thine this those thou thus thy
thyself to too under until unto up upon us very was we were what when where which while who whom why will with would ye
yet you your yours yourself yourselves hath doth art shalt wilt wouldst canst dost thing things man men time""".split())
WORD = re.compile(r"[a-z]{3,}")
CONCEPT_WEIGHT = 3.0


def features(text: str, concepts: list[str]) -> Counter:
    bag = Counter(w for w in WORD.findall(text.lower()) if w not in STOP)
    for c in concepts:
        bag["@" + c] += CONCEPT_WEIGHT
    return bag


def cluster(bags: list[Counter], k: int, seed: int = 7, restarts: int = 8, iters: int = 50,
            min_df: int = 2) -> dict[str, Any]:
    """Spherical k-means on TF-IDF vectors. Returns a label per bag and the top terms of each topic."""
    n = len(bags)
    df = Counter(t for b in bags for t in b)
    vocab = sorted(t for t, c in df.items() if c >= min(min_df, max(1, n // 50)) and c < 0.6 * n)
    if n == 0 or not vocab:
        return {"labels": [0] * n, "topics": [{"id": 0, "terms": [], "size": n}], "k": 1}
    k = max(1, min(k, n))
    index = {t: i for i, t in enumerate(vocab)}
    X = np.zeros((n, len(vocab)))
    for r, b in enumerate(bags):
        for t, c in b.items():
            if t in index:
                X[r, index[t]] = (1 + math.log(c)) * math.log(n / df[t])
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(norms > 0, norms, 1)
    rng = np.random.default_rng(seed)
    best, best_score = None, -1.0
    for _ in range(restarts):
        # k-means++ seeding on cosine distance
        centres = [X[rng.integers(n)]]
        for _ in range(k - 1):
            d = 1 - np.max(np.stack([X @ c for c in centres]), axis=0)
            d = np.clip(d, 0, None) ** 2
            centres.append(X[rng.choice(n, p=d / d.sum())] if d.sum() > 0 else X[rng.integers(n)])
        C = np.stack(centres)
        labels = np.zeros(n, dtype=int)
        for _ in range(iters):
            new = np.argmax(X @ C.T, axis=1)
            if _ and (new == labels).all():
                break
            labels = new
            for j in range(k):
                members = X[labels == j]
                if len(members):
                    c = members.sum(axis=0)
                    C[j] = c / (np.linalg.norm(c) or 1)
        score = float((X * C[labels]).sum())
        if score > best_score:
            best, best_score = (labels.copy(), C.copy()), score
    labels, C = best
    # number topics by size, largest first, and name each by its heaviest terms
    order = [j for j, _ in Counter(labels.tolist()).most_common()]
    remap = {j: i for i, j in enumerate(order)}
    topics = []
    for j in order:
        top = np.argsort(-C[j])[:8]
        topics.append({"id": remap[j], "size": int((labels == j).sum()),
                       "terms": [vocab[i] for i in top if C[j, i] > 0]})
    return {"labels": [remap[j] for j in labels.tolist()], "topics": topics, "k": len(topics)}


def label(topic: dict[str, Any], concept_labels: dict[str, str]) -> str:
    """A short readable name from a topic's top terms, concepts first."""
    terms = [concept_labels.get(t[1:], t[1:]) if t.startswith("@") else t for t in topic["terms"]]
    concepts = [concept_labels.get(t[1:], t[1:]) for t in topic["terms"] if t.startswith("@")]
    words = concepts[:2] + [t for t in terms if t not in concepts][: max(0, 3 - len(concepts[:2]))]
    return ", ".join(words) or f"topic {topic['id'] + 1}"


def default_k(n_passages: int) -> int:
    """About one topic per 60 passages, between 3 and 8: few enough that each topic keeps the tests' power."""
    return max(3, min(8, round(n_passages / 60)))
