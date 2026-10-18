# Open problems

As of 2026-10-18.

Four problems in Android Mirror that nobody has solved yet. Each entry says what the problem is, why it is still open, and where the existing work lives. Paths under `android-mirror/` are in the project's shared files, not in this repository.

## 1. A consistent graphical causal framework

**Problem.** The causal model mixes graph types without a single reading. Directed arrows come from the subject's judgments, two-way links mark a "shared cause the subject never names", and a PAG view of hidden causes was started on top. Each is reasonable alone, but they don't yet add up to one formal object with one set of rules.

**Why it's open.**

- A two-way link sometimes means a hidden common cause and sometimes just a clash or a feedback loop. Only the first reading licenses the PAG step.
- The model reads silence as separation (no claim means no connection). That is a closed-world assumption; silence is better treated as missing.
- The Causal tab probably needs to step through ADMG, then MAG, then PAG explicitly before a PAG view makes sense.

**Pointers.**

- PAG view, parked: PR #30.
- Shared-cause links: PR #26 and #27.
- Silence as missing data: `android-mirror/designs/silence-as-missing-data.md`.
- Skeptical review: `android-mirror/reviews/skeptical-evaluation.md`.

## 2. Empirically testing the value of the engines

**Problem.** Mutate, Transmutate, Derange and Question each produce proposals or rankings, but there is no evidence yet that a Mirror is better after using them than before.

**Why it's open.**

- "Better" is undefined. Candidates are a closer match to the subject on held-out passages, beating plain Claude on the same questions, or following an edit to the corpus (the corpus-transplant test).
- The real builds so far are small, so differences may be noise.
- Every change waits for a person's approval, so a test also has to account for which proposals the user accepts.

**Pointers.**

- Engine code: `androidmirror/mirror/mutate.py`, `transmutate.py`, `derange.py`, `question.py`.
- Skeptical review (validate must beat plain Claude; the corpus-transplant test is the decisive one): `android-mirror/reviews/skeptical-evaluation.md`.
- Offline benchmark of the causal fit: [README, Benchmark](../README.md#benchmark).

## 3. A Breaker: the opposite of Transmutate

**Problem.** Transmutate combines two Mirrors into one. Nothing does the reverse: take one Mirror and break it into the separate worldviews it may be hiding (for example, two voices in one corpus, or a subject who changed their mind over time).

**Why it's open.**

- It's not settled what the break is along: speakers, periods, topics, or clusters of judgments that clash with each other.
- Each part must stay internally consistent in both the knowledge graph and the causal model, and should be tested against the corpus it came from.
- Like every engine, it must only propose; nothing is applied without approval.

**Pointers.**

- Prototype 1 had a Breaker that split single nodes and severed links in both graphs at once: `androidmirror/breaker.py`. It works at node level, not whole-Mirror level, but its consistency rules can be reused.
- Transmutate, which this would mirror: `androidmirror/mirror/transmutate.py`.

## 4. Additional knowledge and causal dimensions

**Problem.** Today the knowledge graph holds typed relations between concepts, and the causal model holds the subject's causal judgments (causes, intervention, means to an end, "not x but y", independence). Much of what a subject holds doesn't fit either: how sure they are, whether they say it themselves or report someone else, when it applies, how strong an effect is, or how things change over time.

**Why it's open.**

- Which dimensions to add, and whether each belongs in the knowledge graph, the causal model, or a separate layer (values are already kept apart).
- Each new dimension has to be read faithfully from the corpus, not invented, and has to work with the engines and the Ask tab.

**Pointers.**

- Descriptive review (qualifiers for modality, attribution and polarity): [review doc](https://claude.ai/code/artifact/c75478d6-4c3b-4d12-a3d3-97cb16a36d12).
- Causal validity review (separating confidence from effect size): `android-mirror/reviews/causal-validity-review.md`.
- Current schema: [README, the descriptive ontology](../README.md#the-descriptive-ontology-built-for-inference) and [the causal model](../README.md#the-causal-model-the-subjects-own-judgments).
