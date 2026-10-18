# Open problems

As of 2026-10-18.

Six problems in Android Mirror that nobody has solved yet. Each entry says what the problem is, why it is still open, and where the existing work lives.

## 1. A consistent graphical causal framework

**Problem.** The causal model mixes graph types without a single reading. Directed arrows come from the subject's judgments, two-way links mark a "shared cause the subject never names", and a PAG view of hidden causes was started on top. Each is reasonable alone, but they don't yet add up to one formal object with one set of rules.

**Why it's open.**

- A two-way link sometimes means a hidden common cause and sometimes just a clash or a feedback loop. Only the first reading licenses the PAG step.
- The model reads silence as separation (no claim means no connection). That is a closed-world assumption; silence is better treated as missing.
- The Causal tab probably needs to step through ADMG, then MAG, then PAG explicitly before a PAG view makes sense.

**Pointers.**

- PAG view, parked: PR #30.
- Shared-cause links: PR #26 and #27.
- Silence as missing data: [designs/silence-as-missing-data.md](silence-as-missing-data.md).
- Skeptical review: [reviews/skeptical-evaluation.md](../reviews/skeptical-evaluation.md).

## 2. Empirically testing the value of the engines

**Problem.** Mutate, Transmutate, Derange and Question each produce proposals or rankings, but there is no evidence yet that a Mirror is better after using them than before.

**Why it's open.**

- "Better" is undefined. Candidates are a closer match to the subject on held-out passages, beating plain Claude on the same questions, or following an edit to the corpus (the corpus-transplant test).
- The real builds so far are small, so differences may be noise.
- Every change waits for a person's approval, so a test also has to account for which proposals the user accepts.

**Pointers.**

- Engine code: `androidmirror/mirror/mutate.py`, `transmutate.py`, `derange.py`, `question.py`.
- Skeptical review (validate must beat plain Claude; the corpus-transplant test is the decisive one): [reviews/skeptical-evaluation.md](../reviews/skeptical-evaluation.md).
- Offline benchmark of the causal fit: [the benchmark](causal-model.md#benchmark).

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
- Causal validity review (separating confidence from effect size): [reviews/causal-validity-review.md](../reviews/causal-validity-review.md).
- Current schema: [the descriptive ontology](descriptive-ontology.md) and [the causal model](causal-model.md).

## 5. An automated sequence of engines: descent with modification

**Problem.** Today each engine runs on its own and a person approves every change. The goal is a sequence that runs the engines automatically, generation after generation (vary with Mutate, Transmutate and Question, select with Derange and a fitness test), so the Mirror descends with modifications on its own.

**Why it's open.**

- It needs a fitness measure, which waits on problem 2: without evidence that a change makes a Mirror better, selection has nothing to select on.
- The rule so far is that no engine applies a change without approval. Running automatically means deciding what can run unattended (for example in a sandbox copy), and what still needs a person before it reaches the live Mirror.
- Generations must not drift away from the corpus: each one has to stay grounded in the subject's own words, and the lineage has to be traceable and reversible.

**Pointers.**

- Prototype 1 had an Evolver that searched sandbox copies of both graphs with the other engines' operations, scored designs on fitness, and queued the winners for approval: `androidmirror/evolver/`.
- The current engines: `androidmirror/mirror/mutate.py`, `transmutate.py`, `derange.py`, `question.py`.

## 6. Joint causes: "A and B are both needed for C"

**Problem.** In a causal graph, A → C ← B looks the same whether A and B are both needed (AND), either is enough (OR), or their effects just add up. The graph only says C depends on both; "both are needed" is an interaction, which lives in the function that sets C, not in the arrows. A subject who says "virtue needs both reason and practice" makes exactly this kind of claim, and the Mirror can't record it.

**Why it's open.**

- Each causal judgment is read as one pair, x to y (`JUDGMENT_SCHEMA` in `androidmirror/mirror/coding.py`), so "only if both" is either missed or split into two separate arrows and the AND is lost.
- There are three standard ways to show it, and the choice isn't made:
    - **Sufficient-component causes** (Rothman's "causal pies"): each pie is a set of causes that is enough by itself, so A and B in one pie means both are needed.
    - **A mechanism node:** A → M ← B, M → C, where M fires only when both are present. The graph stays a DAG, with the AND attached to M.
    - **A label on C's incoming arrows:** "all of" versus "any of", as qualitative and Boolean causal models do. This fits the Mirror best, since its arrows are already signed and weighted by passages, and the graph would be built the same way.
- The fit, Trace, the held-out check and the engines would all have to respect the label, for example by not treating one of two jointly needed causes as enough on its own.

**Pointers.**

- The judgment kinds and how they become constraints: [the causal model](causal-model.md).
- Problem 1 (a consistent graphical framework) and problem 4 (more causal dimensions) overlap this one.
