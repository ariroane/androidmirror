# Android Mirror

A reflective model of a subjective system (a person, a group, an institution) built only from its own words. Prototype 2 reads a whole corpus once and builds two linked structures over the same concepts:

- **Descriptive ontology:** a knowledge graph with a schema and rules, so it can be reasoned over, not only browsed. Every answer it gives has a proof that bottoms out in quotes.
- **Causal model:** the subject's *own* model of how things work, over concepts drawn from the knowledge graph. The units are the situations the subject reasons about (an insult, an illness, the world an "if" imagines), each concept low, typical, high or unknown in each. The PC algorithm finds the **Markov equivalence class** of DAGs those situations support; the subject's stated causal claims are applied as background knowledge; and **vignettes** answered "as the subject" can add more, drawn in their own colour so the corpus graph is always visible on its own.

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

Open http://localhost:8000. Building makes two Claude calls per passage plus a few to consolidate the vocabulary, cached under `data/mirror/<corpus>/`; a build that stops resumes where it left off. Extending `--limit` codes only the new passages, but re-reads situations in every passage if the list of candidate variables changes. `--model` picks a cheaper model for the bulk coding.

Then, optionally, let Claude read between the lines as the subject (section "The subject transformation" below):

```powershell
python -m androidmirror validate --corpus meditations    # about 3 calls per held-out situation (40 by default)
python -m androidmirror fill --corpus meditations        # up to 1 call per situation, plus leak checks
python -m androidmirror vignettes --corpus meditations   # 2 calls per vignette (6 per level, 6 variables: 216 calls)
python -m androidmirror benchmark                        # free: the planted-model benchmark, offline
```

Each step is cached and rewrites the Mirror when it finishes; `fill` and `vignettes` refuse to run until `validate` has.

## How a Mirror is built

1. **Passages.** The corpus is split into units of observation: numbered sections for the Meditations, one statement per meeting for the FOMC.
2. **Open coding (one Claude call per passage).** Concept mentions, each with its level in the passage (+1 present or affirmed, −1 absent, negated or to be got rid of); descriptive statements in a fixed relation vocabulary, each with its modality (asserts, prescribes, asks, entertains), attribution (the subject, or someone it quotes) and polarity; and causal claims with how their direction is grounded (an explicit connective, temporal order, or inferred). Passages are coded independently, so nothing depends on reading order.
3. **Codebook.** All labels are consolidated into one fixed set of concepts: synonyms merged, opposites folded in with polarity −1 ("freedom from anger" is anger at −1), word senses kept apart, and each concept marked as a *quantity* (can be more or less present) or not.
4. **Situations (one Claude call per passage).** With the codebook fixed, Claude lists the situations each passage reasons about (actual, hypothetical, ideal or general, and which ones it contrasts), and for each situation how every candidate variable stands there: low (−1), typical (0) or high (+1), each with a quote. A variable the situation doesn't touch is **unknown**, never 0. The same call picks out the subject's statements about their own model: "x causes y", "x directly", "not x but y", "whether or not x", "x so that y". Situations, not passages, are the rows of the dataset.
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

## The causal model: the subject's own model

The causal layer is not a model of the world, and not Claude's model of the subject. It is the model the subject's own writing shows: when the subject describes a situation where x is high, how do they describe y? Nodes are concepts from the knowledge graph.

**How each value is known.** Every cell has an evidence tier, shown as a chip in the app:

| Tier | Meaning |
|---|---|
| stated | the passage says so, with a quote |
| implied | follows from what is stated plus the subject's own beliefs in the knowledge graph (requires, opposes, same as; with the proof), or the subject transformation filled it citing a passage or belief |
| defaulted | the subject's usual level for that variable (only in the "all" view) |
| reading | Claude filled it with no citation, or it failed a prior-leak check (only in the "all" view) |
| set | a vignette fixed it |
| common sense | a vignette answer a generic adult gives too, with no citation; never used |

Discovery runs on stated + implied by default, and also on stated only and on everything, so you can see whether the answer depends on what counts as evidence (`causal --view stated|implied|all`). Missing values are handled by test-wise deletion: each test uses only the situations where all its variables are known. A link is only removed by a test that had enough data; a collider is only oriented when its own signature test confirms it (conservative PC), and undecided triples are shown and never fed to Meek's rules.

**Choosing the variables matters most.** A concept becomes a DAG node only if it is a quantity (people, places and events cannot be intervened on), is known in enough situations for the tests to have power, varies (its rarer levels are at least 10% of the known values), and is not a near-duplicate of a stronger variable (near-deterministic pairs are usually one idea or a definition, which breaks faithfulness and belongs in the knowledge graph). The top 25 by support and by how often the subject makes causal claims about them are kept. Every candidate's decision and reason is shown in the Causal tab.

**Discovery.** PC-stable with a G² test for discrete data, conditioning sets up to size 3, and the majority rule for colliders (each unshielded triple is re-tested against every separating set, so one unlucky test cannot orient edges). The result is a CPDAG: compelled edges are the same in every DAG of the class; reversible edges are not.

**Without listing the class.** The class can be super-exponential, so it is never enumerated. Instead the app shows summaries that are cheap to compute:

- each edge's **status**: compelled, pinned by the subject's claims, implied by those pins, or reversible;
- **why** each arrow points the way it does: the v-structure and the independence test that created it, or the chain of Meek rules from one;
- the **class size**, counted exactly per chain component (He, Jia and Yu 2015), before and after the subject's claims;
- **possible effects** of one variable on another across the class (local IDA, Maathuis, Kalisch and Bühlmann 2009), from the variable's neighbourhood alone;
- each edge's **stability** across 20 block-bootstrap resamples of the passages, and the number of situations where both ends are known.

**The subject's claims as background knowledge.** "x causes y" and "x so that y" become pins; "not x but y" and "whether or not x" are independence claims, tested against the situations and reported as agreeing or not. Claims grounded in explicit or temporal wording become pins (required orientations), strongest first. A pin is kept only if a DAG in the class still fits: it can orient a reversible edge, and Meek's rules carry it further, but it cannot reverse a compelled edge or create a collider the data did not find. Each claim is reported as agreeing with the data, applied, contradicted, or with **no link in the data** (the two variables are independent once others are accounted for).

**Assumptions, stated in the app:** situations are the units; no hidden common causes beyond the topic, which the tests hold fixed; faithfulness; a value is missing for reasons unrelated to its level (the weakest one: the subject mentions what is salient). Neighbouring passages share themes, so resampling is done in blocks. The model describes the subject's writing, not their life. Temporal tiers are accepted: `python -m androidmirror causal --tiers tiers.json`.

## The subject transformation

Situations leave most values unknown, and the data alone orients few edges. Claude can fill some gaps, but its own priors must not leak in as the subject's. So every Claude answer goes through a transformation into the subject, with checks that it is the subject talking:

1. **Profile.** Built from the corpus only: the subject's vocabulary and glosses, the beliefs they hold and deny, what follows from them, their causal claims, their usual levels, passages in their voice, and the views of others they quote but do not hold. Nothing from Claude's knowledge of the person is added. The same code serves any subject; for the FOMC it is the Committee, not Marcus.
2. **Answering.** For a situation, Claude is given the profile and the most relevant passages (BM25), and asked for each unknown variable: low, typical, high or "the subject's writing doesn't say". Every answer must cite a passage or a belief.
3. **Validation (`validate`).** On held-out situations whose stated values are hidden, the same questions are put three ways: as the subject, as plain Claude, and as a generic adult. A variable is trusted only where the subject answers are right at least 60% of the time they answer (at least 3 answers). The scores are in the Causal tab.
4. **Fill (`fill`).** Only trusted variables are filled. A sample of answers gets prior-leak checks: asked again with the cited passage removed, and compared with the generic adult. An answer that survives without its citation, or that anyone would give, is demoted to "reading" and left out of the default view. The leak rate is shown.
5. **Vignettes (`vignettes`).** Invented situations, each built on a real one with one variable set high or low, answered as the subject and as a generic adult. These are interventions on the subject's model, so they can orient edges the corpus cannot. Discovery runs twice, on the corpus alone and with vignettes (as context variables in the first tier, Joint Causal Inference style), and **only the difference** is drawn in orange: a new orange edge, an orange arrowhead on a direction only vignettes give, a dashed orange line where vignettes remove an edge. A toggle on the canvas hides orange and returns the pure corpus graph. Vignettes are listed separately from passages everywhere.

## The planted-model benchmark

`python -m androidmirror benchmark` runs the method on corpora where the truth is known: a 12-variable Stoic-flavoured model with selection bias in what gets mentioned and topic salience, plus two null corpora where nothing causes anything (only topics). Offline, with no Claude calls, over 600 situations and 4 seeds:

- **Situations, stated values only:** about 0.68 skeleton precision and 0.78 recall, and no compelled arrows. Mentions alone don't orient anything; that needs claims or vignettes.
- **Plus stated claims:** about 4 right and 2 wrong arrows. **Plus vignettes:** about 4 right and 1 wrong.
- **Nulls:** situations give no compelled arrows on either null; the old passage-as-row table gave 10 to 22 false arrows. But where a topic's mood shifts several concepts at once, about 7 false links remain.
- **Holding the topic fixed** (below): with the exact topic, those 7 false links drop to 2; with topics grouped from what each passage mentions (right for about 82% of passages), to 5. The planted model loses nothing either way.

`benchmark --claude` has Claude write the passages from the planted situations and runs the real pipeline on them, so it also measures how accurately Claude reads levels. It costs about the same as building a corpus of that size.

Most remaining false links in the planted model are mediated pairs (wrongdoing and cooperation, through judgment of harm and anger) where too few situations state all three for the test that would separate them. Filling values through the subject transformation targets exactly this; the benchmark does not measure it yet.

Not built yet: FCI for hidden common causes, and missingness-aware PC (MVPC).

## Holding the topic fixed

What a passage is about (death, the court, other people's faults) can make several concepts come up together and colour their levels alike: a hidden common cause that shows up as links between concepts that don't affect each other in the subject's model. So each passage is grouped into a topic, and the topic joins discovery as a background variable:

- **Grouping, without Claude.** Each passage is a bag of its words and the concepts coded in it (TF-IDF), grouped by k-means into about one topic per 60 passages (3 to 8). Every situation takes its passage's topic; a vignette takes the topic of the situation it was built on. `build --topics N` sets the number.
- **In the tests.** The topic is a variable PC can condition on, in the first tier (nothing in a situation causes what its passage is about). Two concepts stay linked only if they still go together within a topic. Effects are estimated within topics too.
- **Not in the model.** The topic is a rough stand-in, so it may test the subject's links but never orients them, and it is not drawn on the canvas. The Causal tab lists the links holding it fixed explained away, the concepts whose level depends on the topic, and the topics with their top words. `causal --topic off` turns it off for comparison.

How much this helps depends on how good the grouping is: in the benchmark, exact topics remove most topic-made links, a grouping that is right for 82% of passages removes fewer. A sharper grouping (Claude naming each passage's topic in the situations pass) is the next step if the word-based one is too rough on the real corpus.

## The web app

- **Canvas:** the knowledge graph (node colour is the subject's evaluation) or the causal model (black arrows compelled by the situations, blue oriented by the subject's claims, dashed reversible, thickness for stability, orange for what vignettes change, with a toggle to hide it). Click a node for its card; click a causal edge for its provenance.
- **Node:** what the concept is, its aliases and classes, stated facts with quotes, inferred facts with their rules, causal claims, its place in the causal model, its usual level, the situations it appears in with every value's tier, and, separately, the vignettes that involve it. Click a locator to read the passage and the situations read from it.
- **Knowledge:** "does it follow?" queries with proof trees, and the list of inconsistencies.
- **Causal:** how each value is known, class size (situations, with claims, with vignettes), edges with status, effect and stability, the effect explorer, the subject's claims and independence statements tested, the three evidence views compared, the transformation's validation scores and leak rate, the subject profile, and the variable selection report.
- **Ask:** the Oracle answers from stated and inferred facts, claims, the causal model and the most relevant passages, and cites passages by locator.
- **Face:** the Major Arcana card the Mirror most resembles (from Prototype 1), now read from the new graphs.

## Command reference

| Command | What it does |
|---|---|
| `build --corpus NAME [--limit N] [--workers 6] [--model ID] [--no-situations] [--topics N]` | Code, consolidate, read situations, reason and discover. Needs the key. `--no-situations` treats each passage as one situation, from the open coding (cheaper, cruder). |
| `causal --corpus NAME [--alpha] [--depth] [--max-vars] [--min-support] [--bootstrap] [--no-claims] [--tiers FILE] [--view stated\|implied\|all] [--topic on\|off]` | Re-run only the causal layer. No Claude calls. |
| `validate --corpus NAME [--n 40]` | Score the subject transformation on held-out situations. |
| `fill --corpus NAME [--leak-sample 30]` | Fill unknown values of trusted variables as the subject, with prior-leak checks. |
| `vignettes --corpus NAME [--per-level 6] [--max-targets 6]` | Invented situations as interventions; their contribution is drawn in orange. |
| `benchmark [--n 600] [--seeds 3] [--claude]` | The planted-model benchmark. Offline and free unless `--claude`. |
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
| `androidmirror/mirror/topics.py` | Grouping passages into topics (TF-IDF, k-means) |
| `androidmirror/mirror/situations.py` | Situations, evidence tiers, belief implications, views |
| `androidmirror/mirror/subject.py` | The subject transformation: profile, retrieval, validation, fill, leak checks, vignettes |
| `androidmirror/mirror/benchmark.py` | The planted-model benchmark |
| `androidmirror/mirror/causal/` | Variable selection, G² test with missing values, PC, Meek rules, pins, class size, IDA, bootstrap, the vignette diff |
| `androidmirror/mirror/build.py`, `store.py`, `ask.py` | The pipeline, the read views, the Oracle |
| `androidmirror/server.py`, `web/index.html` | The app |
| `androidmirror/p1_*`, `refactor.py`, `breaker.py`, `chainer.py`, `evolver/` | Prototype 1, paused |

## Tests

```
pip install -e ".[dev]"
pytest
```

The Prototype 2 tests build a Mirror end to end from a synthetic corpus with a known causal structure, using an offline stand-in for Claude, and check that PC recovers its equivalence class, that unknown values stay unknown, and that validation, fill and vignettes work and are kept apart.
