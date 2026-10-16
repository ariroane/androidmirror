# Android Mirror

A reflective model of a subjective system (a person, a group, an institution) built only from its own words. Prototype 2 reads a whole corpus once and builds two linked structures over the same concepts:

- **Descriptive ontology:** a knowledge graph with a schema and rules, so it can be reasoned over, not only browsed. Every answer it gives has a proof that bottoms out in quotes.
- **Causal model:** the corpus treated as an observational dataset (passages are the units, concepts are the variables), from which the PC algorithm finds the **Markov equivalence class** of causal DAGs the text supports. The subject's own stated causal claims are then applied as background knowledge, so you can see what the data establishes on its own and what the subject's claims add.

Two corpora come with it:

| Corpus | Subject | Units |
|---|---|---|
| `meditations` | Marcus Aurelius, *Meditations* (Meric Casaubon translation, Project Gutenberg #2680) | about 410 numbered sections |
| `fomc` | The Federal Reserve's FOMC, every policy statement since 1994 (federalreserve.gov) | one statement per meeting |

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
python -m androidmirror serve
```

Open http://localhost:8000. Building makes two Claude calls per passage plus a few to consolidate the vocabulary, cached under `data/mirror/<corpus>/`; a build that stops resumes where it left off. Extending `--limit` codes only the new passages, but re-measures every passage if the list of candidate variables changes. `--model` picks a cheaper model for the bulk coding.

## How a Mirror is built

1. **Passages.** The corpus is split into units of observation: numbered sections for the Meditations, one statement per meeting for the FOMC.
2. **Open coding (one Claude call per passage).** Concept mentions, each with its level in the passage (+1 present or affirmed, −1 absent, negated or to be got rid of); descriptive statements in a fixed relation vocabulary, each with its modality (asserts, prescribes, asks, entertains), attribution (the subject, or someone it quotes) and polarity; and causal claims with how their direction is grounded (an explicit connective, temporal order, or inferred). Passages are coded independently, so nothing depends on reading order.
3. **Codebook.** All labels are consolidated into one fixed set of concepts: synonyms merged, opposites folded in with polarity −1 ("freedom from anger" is anger at −1), word senses kept apart, and each concept marked as a *quantity* (can be more or less present) or not.
4. **Measurement (one Claude call per passage).** With the codebook fixed, each passage is measured against the same candidate variables. This is what turns the corpus into a dataset: every passage is a row, every variable a column, each cell −1, 0 (not engaged) or +1.
5. **Knowledge graph and inference** (no Claude calls; see below).
6. **Causal discovery** (no Claude calls; see below). Re-run it with other settings any time: `python -m androidmirror causal --alpha 0.01 --max-vars 20`.

There is no review queue: the whole corpus seeds the Mirror at once. To review, click any node or edge; every fact, claim and variable shows the passages it came from.

## The descriptive ontology: built for inference

Following the two families of inference in Stanford CS520's notes ([What are some inference algorithms?](https://web.stanford.edu/class/cs520/2020/notes/What_Are_Some_Inference_Algorithms.html)):

- **Taxonomic.** Every node has a declared *kind* (agent, disposition, emotion, practice, belief, state, faculty, concept, class, ...). `is_a` is transitive, so membership and evaluation are inherited: *justice is a virtue* and *virtue is good* gives *justice is good*. The evaluations `good`, `bad` and `indifferent` are built-in classes that form a partition, so calling something both good and bad is a detectable inconsistency.
- **Rule-based.** Thirteen relations with declared domains, ranges and properties: `is_a`, `part_of`, `requires` and `same_as` are transitive; `opposes` and `same_as` are symmetric; `opposes`, `requires` and `accords_with` pass from a class to its members; `values` and `rejects` pass from a class to its members on the object side (*valuing the good* means *valuing justice*). The rules run bottom-up to a fixpoint, and every derived fact keeps the rule and premises that produced it.
- **Only what the subject holds is reasoned over.** Its own assertions and self-prescriptions count; quoted views of others, questions and hypotheses are kept as reported speech and never enter inference.
- **Evidence accumulates.** The tenth passage that says the same thing adds to that fact's support instead of being dropped or overwriting it.
- **Checks.** Disjoint evaluations, values-and-rejects, a stated denial of something that follows, taxonomy cycles, and domain/range violations are listed in the Knowledge tab with their proofs.
- **Open world.** A fact that is neither stated nor derived is "not found", never false.

## The causal model: the corpus as observational data

**Choosing the variables matters most.** A concept becomes a DAG node only if it is a quantity (people, places and events cannot be intervened on), is engaged in enough passages for the tests to have power, varies (is not in nearly every passage), and is not a near-duplicate of a stronger variable (near-deterministic pairs are usually one idea or a definition, which breaks faithfulness and belongs in the knowledge graph). The top 25 by support and by how often the subject makes causal claims about them are kept. Every candidate's decision and reason is shown in the Causal tab.

**Discovery.** PC-stable with a G² test for discrete data, conditioning sets up to size 3, and the majority rule for colliders (each unshielded triple is re-tested against every separating set, so one unlucky test cannot orient edges). The result is a CPDAG: compelled edges are the same in every DAG of the class; reversible edges are not.

**Without listing the class.** The class can be super-exponential, so it is never enumerated. Instead the app shows summaries that are cheap to compute:

- each edge's **status**: compelled, pinned by the subject's claims, implied by those pins, or reversible;
- **why** each arrow points the way it does: the v-structure and the independence test that created it, or the chain of Meek rules from one;
- the **class size**, counted exactly per chain component (He, Jia and Yu 2015), before and after the subject's claims;
- **possible effects** of one variable on another across the class (local IDA, Maathuis, Kalisch and Bühlmann 2009), from the variable's neighbourhood alone;
- each edge's **stability** across 20 block-bootstrap resamples of the passages.

**The subject's claims as background knowledge.** Claims grounded in explicit or temporal wording become pins (required orientations), strongest first. A pin is kept only if a DAG in the class still fits: it can orient a reversible edge, and Meek's rules carry it further, but it cannot reverse a compelled edge or create a collider the data did not find. Each claim is reported as agreeing with the data, applied, contradicted, or with **no link in the data** (the two variables are independent once others are accounted for).

**Assumptions, stated in the app:** passages are the units; no hidden common causes; faithfulness. Neighbouring passages share themes, so resampling is done in blocks. The model describes the writing, not the writer's life.

The structure exposes what the "Shifting the Markov equivalence class" brainstorm needs later (per-edge provenance, class size, IDA ranges, pins and tiers, resampling) without building those levers yet. Tiers are already accepted: `python -m androidmirror causal --tiers tiers.json`.

## The web app

- **Canvas:** the knowledge graph (node colour is the subject's evaluation) or the causal model (black arrows compelled by the data, blue oriented by the subject's claims, dashed reversible, thickness for stability). Click a node for its card; click a causal edge for its provenance.
- **Node:** what the concept is, its aliases and classes, stated facts with quotes, inferred facts with their rules, causal claims, its place in the causal model, and every passage it was coded from. Click a locator to read the passage and what was coded from it.
- **Knowledge:** "does it follow?" queries with proof trees, and the list of inconsistencies.
- **Causal:** class size, edges with status, effect and stability, the effect explorer, the subject's claims tested against the data, and the variable selection report.
- **Ask:** the Oracle answers from stated and inferred facts, claims, the causal model and the most relevant passages, and cites passages by locator.
- **Face:** the Major Arcana card the Mirror most resembles (from Prototype 1), now read from the new graphs.

## Command reference

| Command | What it does |
|---|---|
| `build --corpus NAME [--limit N] [--workers 6] [--model ID] [--no-measure]` | Code, consolidate, measure, reason and discover. Needs the key. |
| `causal --corpus NAME [--alpha] [--depth] [--max-vars] [--min-support] [--bootstrap] [--no-claims] [--tiers FILE]` | Re-run only the causal layer. No Claude calls. |
| `serve [--port 8000]` | The web app. |
| `ask "question" [--corpus NAME]` | The Oracle from the command line. |
| `p1 ...` | The paused Prototype 1 app (patch review, Refactorer, Breaker, Chainer, Evolver): [docs/prototype1.md](docs/prototype1.md). |

## Adding a corpus

Write `data/corpora/<id>.json`:

```json
{"id": "letters", "title": "Seneca, Letters", "subject": "Seneca", "file": "corpus/letters.txt", "splitter": "paragraphs"}
```

Splitters: `paragraphs` (one text file, short paragraphs merged), `documents` (a folder of `.txt` files, one passage per file or per paragraph with `"unit": "paragraph"`; a first line `# Locator` and a line `group: name` are optional), and `meditations`.

## Code map

| Path | What |
|---|---|
| `androidmirror/mirror/corpus.py`, `sources/fomc.py` | Passages; the FOMC downloader |
| `androidmirror/mirror/coding.py` | The three Claude calls and their schemas |
| `androidmirror/mirror/codebook.py` | Consolidating labels into concepts |
| `androidmirror/mirror/schema.py` | Kinds, relations and their logical properties |
| `androidmirror/mirror/kg.py` | Knowledge base, rules, proofs, checks |
| `androidmirror/mirror/causal/` | Variable selection, G² test, PC, Meek rules, pins, class size, IDA, bootstrap |
| `androidmirror/mirror/build.py`, `store.py`, `ask.py` | The pipeline, the read views, the Oracle |
| `androidmirror/server.py`, `web/index.html` | The app |
| `androidmirror/p1_*`, `refactor.py`, `breaker.py`, `chainer.py`, `evolver/` | Prototype 1, paused |

## Tests

```
pip install -e ".[dev]"
pytest
```

The Prototype 2 tests build a Mirror end to end from a synthetic corpus with a known causal structure, using an offline stand-in for Claude, and check that PC recovers its equivalence class.
