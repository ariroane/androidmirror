# Android Mirror

A reflective model of a subjective system (a person, a group, an institution) built from its own words (other data generating mechanisms in progress). Combines three knowledge structures - knowledge graphs for relational knowledge, causal graphs for observational knowledge, and language structures through Claude. Prototype 3 reads a whole corpus once and builds two linked structures over the same concepts:

- **Descriptive ontology:** a knowledge graph with a schema and rules, so it can be reasoned over, not only browsed. Every answer it gives has a proof that bottoms out in quotes.
- **Causal model:** the subject's *own* model of how things work, built only from the causal judgments the subject makes ("anger comes from opinion", "take away the opinion and the hurt is gone", "not the thing but your judgment of it disturbs you", "this makes no difference to that"). Each judgment becomes a constraint on a graph, and the graph is fitted to them. Nothing is inferred from how often ideas appear together, and Claude never answers as the subject.

What the subject wants is kept apart from what they think causes what: a third layer of **values** records what each passage urges more or less of.

Three corpora come with it:

| Corpus | Subject | Units |
|---|---|---|
| `meditations` | Marcus Aurelius, *Meditations* (Meric Casaubon translation, Project Gutenberg #2680) | about 410 numbered sections |
| `fomc` | The Federal Reserve's FOMC, every policy statement since 1994 (federalreserve.gov) | one statement per meeting |
| `dgn` | The German Society of Neurology's English-language guidelines on Parkinson's disease, the CC BY ones on Europe PMC (set `conditions` in `data/corpora/dgn.json` to add others, or `[]` for all) | one guideline section |

Any other subject works too: write a spec in `data/corpora/` pointing at a text file, or at a folder of `.txt` documents.

## Run it on your computer

You need Python 3.10 or newer and an Anthropic API key in a local `.env` (never committed).

```powershell
git clone https://github.com/ariroane/androidmirror
cd androidmirror
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -e .
copy .env.example .env          # then put your key in .env: ANTHROPIC_API_KEY=sk-ant-...

python -m androidmirror build --corpus meditations --limit 30   # a cheap trial on 30 passages
python -m androidmirror build --corpus meditations              # the whole book
python -m androidmirror build --corpus fomc                     # downloads the statements first
python -m androidmirror build --corpus dgn                      # downloads the guidelines first
python -m androidmirror serve
```

Open http://localhost:8000. Building makes two Claude calls per passage plus a few to consolidate the vocabulary, cached under `data/mirror/<corpus>/`; a build that stops resumes where it left off. Extending `--limit` reads only the new passages, unless the vocabulary of concepts changes, in which case the judgments pass reads every passage again against the new vocabulary. `--model` picks a cheaper model for the bulk reading. `python -m androidmirror benchmark` is free and runs offline.

## How a Mirror is built

1. **Passages.** The corpus is split into units: numbered sections for the Meditations, one statement per meeting for the FOMC.
2. **Open coding (one Claude call per passage).** Concept mentions, and descriptive statements in a fixed relation vocabulary, each with its modality (asserts, prescribes, asks, entertains), attribution (the subject, or someone it quotes) and polarity.
3. **Codebook.** All labels are consolidated into one fixed set of concepts: synonyms merged, opposites folded in with polarity −1 ("freedom from anger" is anger at −1), word senses kept apart.
4. **Causal judgments (one Claude call per passage).** With the codebook fixed, Claude lists every causal judgment the passage makes, in the subject's terms, over concepts from the codebook (see [the causal model](designs/causal-model.md)), and what the passage urges more or less of. The vocabulary is the same in every call and is prompt-cached, which lowers their cost.
5. **Knowledge graph and inference** (no Claude calls).
6. **Fitting the causal model** (no Claude calls). Re-run it any time: `python -m androidmirror causal --min-weight 2`.

There is no review queue: the whole corpus seeds the Mirror at once. To review, click any node or arrow; every fact and judgment shows the passages it came from.

## Design notes

How each part works, in `designs/`:

- [The descriptive ontology](designs/descriptive-ontology.md): kinds, relations, rules and checks that make the knowledge graph something to reason over.
- [The causal model](designs/causal-model.md): the five kinds of causal judgment, how the graph is fitted to them, the held-out check, and the benchmark.
- [The engines](designs/engines.md): Question, Mutate and Derange.
- [The face](designs/face.md): how the Face tab picks a Major Arcana card for the whole Mirror.
- [Silence as missing data](designs/silence-as-missing-data.md): a brainstorm for the hidden-cause view.

Reviews of earlier prototypes are in `reviews/`.

## The web app

- **Canvas:** the knowledge graph (node colour is the subject's evaluation) or the causal model (green arrows raise, red bars lower, thickness is the number of passages, dashed may run only through another path, a double ring is something the subject urges more or less of). Click a node for its card; click an arrow for its quotes.
- **Node:** what the concept is, its aliases and classes, stated facts with quotes, inferred facts with their rules, its arrows, every causal judgment about it with its status, and what the subject urges about it. Click a locator to read the passage and the judgments read from it.
- **Knowledge:** "does it follow?" queries with proof trees, and the list of inconsistencies.
- **Causal:** counts, the held-out check, Trace, the arrows, tensions, denials that hold, claims explained by a chain, values, reported views, and judgments about concepts outside the codebook.
- **Ask:** the Oracle answers from the knowledge graph and the causal model only (stated facts and their relations, inferred facts with the rule and facts behind them, the subject's causal judgments, links, chains of links and values), never from the passages, and cites each item it uses as [K3] or [C2]. A second step then proposes two or three ways to use the answer: an idea to change, the pathway through the subject's model it would travel, the outcome it should move, and what has to be true for it to work.
- **Derange:** where the worldview rests on least, under what matters most (see [the engines](designs/engines.md#derange-the-weakest-basis)).
- **Face:** the Major Arcana card the whole Mirror most resembles, drawn as a face. See [the face](designs/face.md).
- **Mutate:** merge concepts that are one, or split one used in two senses. See [the engines](designs/engines.md#mutate-merging-and-splitting-concepts).
- **Transmutate:** combine two Mirrors into one and trim it to what is strong; nothing is written until you approve.

## Command reference

`python -m androidmirror <command> --help` lists a command's options.

| Command | What it does |
|---|---|
| `build --corpus NAME [--limit N] [--workers 6] [--model ID]` | Code, consolidate, read judgments, reason and fit. Needs the key. |
| `causal --corpus NAME [--min-weight 1] [--folds 5]` | Re-fit only the causal model. No Claude calls. |
| `benchmark [--n 300] [--seeds 5] [--claude]` | The planted-subject benchmark. Offline and free unless `--claude`. |
| `serve [--port 8000]` | The web app. |
| `derange --corpus NAME [--top 10] [--knockouts 30]` | The weakest basis under what matters most. No Claude calls. |
| `ask "question" [--corpus NAME]` | The Oracle from the command line. |
| `mutate propose\|merge\|split\|show\|approve\|reject\|undo` | Merge or split concepts; previews, applied only on approval. Offline except `split --claude`. |
| `question [--corpus NAME] [--limit 10] [--kind K] [--list]` | Answer the Question engine's questions; approve each proposal by typing your name. `--pending`, `--approve ID --by NAME`, `--reject ID`, `--revoke ID`. No Claude calls unless you answer in words. |
| `transmutate propose A B [--claude] [--pair A=B] [--opposite A=B] [--drop C]` then `transmutate approve A B [--keep N]` | Combine two Mirrors and trim to what is strong. Writes nothing until approve. |
| `p1 ...` | The paused Prototype 1 app (patch review, Refactorer, Breaker, Chainer, Evolver). `python -m androidmirror p1 --help` lists its commands. |

## Adding a corpus

Write `data/corpora/<id>.json`:

```json
{"id": "letters", "title": "Seneca, Letters", "subject": "Seneca", "file": "corpus/letters.txt", "splitter": "paragraphs"}
```

Splitters: `paragraphs` (one text file, short paragraphs merged), `documents` (a folder of `.txt` files, one passage per file or per paragraph with `"unit": "paragraph"`; a first line `# Locator` and a line `group: name` are optional), and `meditations`.

## Code map

| Path | What |
|---|---|
| `androidmirror/mirror/corpus.py`, `sources/fomc.py`, `sources/dgn.py` | Passages; the FOMC and DGN guideline downloaders |
| `androidmirror/mirror/coding.py` | The Claude calls and their schemas |
| `androidmirror/mirror/codebook.py` | Consolidating labels into concepts |
| `androidmirror/mirror/schema.py` | Kinds, relations and their logical properties |
| `androidmirror/mirror/kg.py` | Knowledge base, rules, proofs, checks |
| `androidmirror/mirror/judgments.py` | Causal judgments into constraints; values |
| `androidmirror/mirror/causal/` | Fitting the graph to the constraints, Trace, the held-out check |
| `androidmirror/mirror/derange.py` | Derange: the weakest basis under the most important subgraphs |
| `androidmirror/mirror/benchmark.py` | The planted-subject benchmark |
| `androidmirror/mirror/build.py`, `store.py`, `ask.py` | The pipeline, the read views, the Oracle |
| `androidmirror/mirror/mutate.py`, `mutate_api.py` | The mutate engine and its routes |
| `androidmirror/mirror/question.py`, `answers.py` | The Question engine; approved answers and how they amend a Mirror |
| `androidmirror/mirror/transmutate.py` | Transmutate: align, combine and fitness-trim two Mirrors |
| `androidmirror/server.py`, `web/index.html` | The app |
| `androidmirror/p1_*`, `refactor.py`, `breaker.py`, `chainer.py`, `evolver/` | Prototype 1, paused |

## Open problems

As of 2026-10-18.

Five problems in Android Mirror that nobody has solved yet. Each entry says what the problem is, why it is still open, and where the existing work lives.

### 1. A consistent graphical causal framework

**Problem.** The causal model mixes graph types without a single reading. Directed arrows come from the subject's judgments, two-way links mark a "shared cause the subject never names", and a PAG view of hidden causes was started on top. Each is reasonable alone, but they don't yet add up to one formal object with one set of rules.

**Why it's open.**

- A two-way link sometimes means a hidden common cause and sometimes just a clash or a feedback loop. Only the first reading licenses the PAG step.
- The model reads silence as separation (no claim means no connection). That is a closed-world assumption; silence is better treated as missing.
- The Causal tab probably needs to step through ADMG, then MAG, then PAG explicitly before a PAG view makes sense.

**Pointers.**

- PAG view, parked: PR #30.
- Shared-cause links: PR #26 and #27.
- Silence as missing data: [designs/silence-as-missing-data.md](designs/silence-as-missing-data.md).
- Skeptical review: [reviews/skeptical-evaluation.md](reviews/skeptical-evaluation.md).

### 2. Empirically testing the value of the engines

**Problem.** Mutate, Transmutate, Derange and Question each produce proposals or rankings, but there is no evidence yet that a Mirror is better after using them than before.

**Why it's open.**

- "Better" is undefined. Candidates are a closer match to the subject on held-out passages, beating plain Claude on the same questions, or following an edit to the corpus (the corpus-transplant test).
- The real builds so far are small, so differences may be noise.
- Every change waits for a person's approval, so a test also has to account for which proposals the user accepts.

**Pointers.**

- Engine code: `androidmirror/mirror/mutate.py`, `transmutate.py`, `derange.py`, `question.py`.
- Skeptical review (validate must beat plain Claude; the corpus-transplant test is the decisive one): [reviews/skeptical-evaluation.md](reviews/skeptical-evaluation.md).
- Offline benchmark of the causal fit: [the benchmark](designs/causal-model.md#benchmark).

### 3. A Breaker: the opposite of Transmutate

**Problem.** Transmutate combines two Mirrors into one. Nothing does the reverse: take one Mirror and break it into the separate worldviews it may be hiding (for example, two voices in one corpus, or a subject who changed their mind over time).

**Why it's open.**

- It's not settled what the break is along: speakers, periods, topics, or clusters of judgments that clash with each other.
- Each part must stay internally consistent in both the knowledge graph and the causal model, and should be tested against the corpus it came from.
- Like every engine, it must only propose; nothing is applied without approval.

**Pointers.**

- Prototype 1 had a Breaker that split single nodes and severed links in both graphs at once: `androidmirror/breaker.py`. It works at node level, not whole-Mirror level, but its consistency rules can be reused.
- Transmutate, which this would mirror: `androidmirror/mirror/transmutate.py`.

### 4. Additional knowledge and causal dimensions

**Problem.** Today the knowledge graph holds typed relations between concepts, and the causal model holds the subject's causal judgments (causes, intervention, means to an end, "not x but y", independence). Much of what a subject holds doesn't fit either: how sure they are, whether they say it themselves or report someone else, when it applies, how strong an effect is, or how things change over time.

**Why it's open.**

- Which dimensions to add, and whether each belongs in the knowledge graph, the causal model, or a separate layer (values are already kept apart).
- Each new dimension has to be read faithfully from the corpus, not invented, and has to work with the engines and the Ask tab.

**Pointers.**

- Descriptive review (qualifiers for modality, attribution and polarity): [review doc](https://claude.ai/code/artifact/c75478d6-4c3b-4d12-a3d3-97cb16a36d12).
- Causal validity review (separating confidence from effect size): [reviews/causal-validity-review.md](reviews/causal-validity-review.md).
- Current schema: [the descriptive ontology](designs/descriptive-ontology.md) and [the causal model](designs/causal-model.md).

### 5. An automated sequence of engines: descent with modification

**Problem.** Today each engine runs on its own and a person approves every change. The goal is a sequence that runs the engines automatically, generation after generation (vary with Mutate, Transmutate and Question, select with Derange and a fitness test), so the Mirror descends with modifications on its own.

**Why it's open.**

- It needs a fitness measure, which waits on problem 2: without evidence that a change makes a Mirror better, selection has nothing to select on.
- The rule so far is that no engine applies a change without approval. Running automatically means deciding what can run unattended (for example in a sandbox copy), and what still needs a person before it reaches the live Mirror.
- Generations must not drift away from the corpus: each one has to stay grounded in the subject's own words, and the lineage has to be traceable and reversible.

**Pointers.**

- Prototype 1 had an Evolver that searched sandbox copies of both graphs with the other engines' operations, scored designs on fitness, and queued the winners for approval: `androidmirror/evolver/`.
- The current engines: `androidmirror/mirror/mutate.py`, `transmutate.py`, `derange.py`, `question.py`.

## Tests

```
pip install -e ".[dev]"
pytest
```

The tests build a Mirror end to end from a synthetic corpus whose subject holds a known causal model, using an offline stand-in for Claude, and check that the fit recovers it: repetitions don't add weight, an overall claim the chain explains adds no shortcut, a slip that closes a loop is put in tension, and other people's views and hypotheses are never fitted.
