# Android Mirror

A reflective model of a subjective system (a person, a group, an institution) built from its own words (other data generating mechanisms in progress). Combines three knowledge structures - knowledge graphs for relational knowledge, causal graphs for observational knowledge, and language structures through Claude. Prototype 3 reads a whole corpus once and builds two linked structures over the same concepts:

- **Descriptive ontology:** a knowledge graph with a schema and rules, so it can be reasoned over, not only browsed. Every answer it gives has a proof that bottoms out in quotes.
- **Causal model:** the subject's *own* model of how things work, built only from the causal judgments the subject makes ("anger comes from opinion", "take away the opinion and the hurt is gone", "not the thing but your judgment of it disturbs you", "this makes no difference to that"). Each judgment becomes a constraint on a graph, and the graph is fitted to them. Nothing is inferred from how often ideas appear together, and Claude never answers as the subject.

What the subject wants is kept apart from what they think causes what: a third layer of **values** records what each passage urges more or less of.

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

Open http://localhost:8000. Building makes two Claude calls per passage plus a few to consolidate the vocabulary, cached under `data/mirror/<corpus>/`; a build that stops resumes where it left off. Extending `--limit` reads only the new passages, unless the vocabulary of concepts changes, in which case the judgments pass reads every passage again against the new vocabulary. `--model` picks a cheaper model for the bulk reading. `python -m androidmirror benchmark` is free and runs offline.

## How a Mirror is built

1. **Passages.** The corpus is split into units: numbered sections for the Meditations, one statement per meeting for the FOMC.
2. **Open coding (one Claude call per passage).** Concept mentions, and descriptive statements in a fixed relation vocabulary, each with its modality (asserts, prescribes, asks, entertains), attribution (the subject, or someone it quotes) and polarity.
3. **Codebook.** All labels are consolidated into one fixed set of concepts: synonyms merged, opposites folded in with polarity −1 ("freedom from anger" is anger at −1), word senses kept apart.
4. **Causal judgments (one Claude call per passage).** With the codebook fixed, Claude lists every causal judgment the passage makes, in the subject's terms, over concepts from the codebook (see below), and what the passage urges more or less of. The vocabulary is the same in every call and is prompt-cached, which lowers their cost.
5. **Knowledge graph and inference** (no Claude calls).
6. **Fitting the causal model** (no Claude calls). Re-run it any time: `python -m androidmirror causal --min-weight 2`.

There is no review queue: the whole corpus seeds the Mirror at once. To review, click any node or arrow; every fact and judgment shows the passages it came from.

## The descriptive ontology: built for inference

Following the two families of inference in Stanford CS520's notes ([What are some inference algorithms?](https://web.stanford.edu/class/cs520/2020/notes/What_Are_Some_Inference_Algorithms.html)):

- **Taxonomic.** Every node has a declared *kind* (agent, disposition, emotion, practice, belief, state, faculty, concept, class, ...). `is_a` is transitive, so membership and evaluation are inherited: *justice is a virtue* and *virtue is good* gives *justice is good*. The evaluations `good`, `bad` and `indifferent` are built-in classes that form a partition, so calling something both good and bad is a detectable inconsistency.
- **Rule-based.** Thirteen relations with declared domains, ranges and properties: `is_a`, `part_of`, `requires` and `same_as` are transitive; `opposes` and `same_as` are symmetric; `opposes`, `requires` and `accords_with` pass from a class to its members; `values` and `rejects` pass from a class to its members on the object side (*valuing the good* means *valuing justice*). The rules run bottom-up to a fixpoint, and every derived fact keeps the rule and premises that produced it.
- **Only what the subject holds is reasoned over.** Its own assertions and self-prescriptions count; quoted views of others, questions and hypotheses are kept as reported speech and never enter inference.
- **Evidence accumulates.** The tenth passage that says the same thing adds to that fact's support instead of being dropped or overwriting it.
- **Checks.** Disjoint evaluations, values-and-rejects, a stated denial of something that follows, taxonomy cycles, and domain/range violations are listed in the Knowledge tab with their proofs.
- **Open world.** A fact that is neither stated nor derived is "not found", never false.

## The causal model: the subject's own judgments

The causal layer is not a model of the world, and not Claude's model of the subject. It is the model the subject's own causal judgments state. An earlier prototype tried to find it in how concepts co-occur across passages; a skeptical evaluation showed that a reflective text is not a sample of situations, so that approach mostly measured the author's habits of mind. Prototype 3 reads the judgments themselves.

**Five kinds of judgment**, each read with a sign (+1 raises, −1 lowers, 0 makes no difference), a scope (direct, overall, or unsaid), a register (describes, urges, entertains, asks) and an attribution (the subject, or someone they quote):

| Kind | Example | Constraint on the graph |
|---|---|---|
| causes | "anger comes from opinion" | x reaches y with that sign; an arrow if it says "directly" |
| intervention | "take away the opinion, and the hurt is gone" | x reaches y (sign 0: no path) |
| means to an end | "do this so that you are free" | x reaches y |
| not this but that | "not things but our judgments disturb us" | x acts on z only through y; y reaches z |
| makes no difference | "whether or not they praise you, you are the same" | no path from x to y |

**Only what the subject holds is fitted.** A judgment counts when it is the subject's own and describes or urges. Views it reports from others, hypotheses and questions are kept and shown, never fitted.

**Weight is passages, not repetitions.** A constraint's weight is the number of distinct passages that make it; saying it three times in one passage counts once. Repetitions are only a tie-breaker.

**Fitting.** Constraints are taken strongest first, a denial before a claim on a tie:
- a claim the graph already implies by a path of the same sign is marked *follows* and adds no arrow;
- a claim the subject does not call direct waits until all the others are in, so an overall claim never becomes a shortcut arrow when the subject also gives the chain;
- a claim that would close a loop, or break a stronger denial, is put *in tension*, with the constraints it ran into;
- an arrow from an overall claim that the rest of the graph also explains is drawn dashed: the subject may mean only that.

The Causal tab lists every constraint with its status, the tensions, the pairs said in both directions, and the subject's values.

**What it can answer.** *Trace* in the Causal tab answers "what does more x do to y, in the subject's model?" with the paths and the passages behind each step, or "makes no difference (the subject says so)", or "not said".

**Does the rest of the corpus predict a passage?** The model is fitted with one contiguous block of passages held out at a time (5 blocks), and each held-out claim is checked: said elsewhere too, not said but follows from the rest, nothing bears on it, opposite sign, or contradicted. Claims that *follow* without being said are the evidence that the judgments form one model rather than a list.

## Benchmark

`python -m androidmirror benchmark` plants a 12-concept causal model in a synthetic subject who states pieces of it (direct claims, overall claims, thought experiments, "not this but that", denials) with Zipf-like repetition, some views of other people, and a share of misread judgments. Offline, 300 passages, 5 seeds, share of drawn arrows that are arrows of the planted model:

| Reading errors | Fitted model | Every claim an arrow |
|---|---|---|
| 0% | 1.00 | 0.43 |
| 10% | 0.85 | 0.28 |
| 20% | 0.68 | 0.22 |

`benchmark --claude` has Claude write the passages in the subject's voice and runs the real reading pipeline on them, so it also measures how accurately Claude reads judgments. It costs about as much as building a corpus of that size.

## The web app

- **Canvas:** the knowledge graph (node colour is the subject's evaluation) or the causal model (green arrows raise, red bars lower, thickness is the number of passages, dashed may run only through another path, a double ring is something the subject urges more or less of). Click a node for its card; click an arrow for its quotes.
- **Node:** what the concept is, its aliases and classes, stated facts with quotes, inferred facts with their rules, its arrows, every causal judgment about it with its status, and what the subject urges about it. Click a locator to read the passage and the judgments read from it.
- **Knowledge:** "does it follow?" queries with proof trees, and the list of inconsistencies.
- **Causal:** counts, the held-out check, Trace, the arrows, tensions, denials that hold, claims explained by a chain, values, reported views, and judgments about concepts outside the codebook.
- **Ask:** the Oracle answers from stated and inferred facts, the subject's causal judgments and model, their values and the most relevant passages, and cites passages by locator.
- **Derange:** where the worldview rests on least, under what matters most (see below).
- **Face:** the Major Arcana card the whole Mirror most resembles, drawn as a face. See [The face](#the-face).
- **Mutate:** merge concepts that are one, or split one used in two senses. See [Mutate](#mutate-merging-and-splitting-concepts).

## The face

The Face tab reads the whole Mirror, both the knowledge graph and the subject's causal model, and names the one Major Arcana card whose patterns it shows most strongly. It always reads the whole Mirror, never a single concept's neighbourhood. The code is `androidmirror/archetypes.py`; the cards are data in `data/archetypes_major_arcana.json`.

1. **The graphs, in the cards' vocabulary.** Each concept gets a type from its kind and the subject's evaluation (an agent is a person, a disposition the subject calls good is a virtue, a bad one a vice, and so on), and each type falls in a class: AIM (virtues, values), ACT (practices), PASSION (vices, emotions), MIND (beliefs, concepts), AGENT (people) or WORLD (events, places). Held facts become relations grouped into families: LINEAGE (learned_from), EMBODY (exemplifies, practices, expresses, accords_with), COMMIT (values), OPPOSE (opposes, rejects) and PART (is_a, part_of, requires, same_as). `related_to`, the catch-all for any other link, is in its own ASSOCIATE family, which no card's patterns use, so it shapes the graph but never counts toward a card. The subject's causal arrows keep their sign, strength and whether they are open.
2. **Each card is a few patterns.** A card is two or three small motifs of at most three nodes over those classes, families and signs, each with a weight and a plain-words description. The Hierophant, for example, is "a person who learned from someone who embodied an aim" plus "an aim someone embodied that raises another aim".
3. **Counted against chance.** The matcher counts every motif in the Mirror, then counts it again in 200 shuffles that keep both graphs' shape but scatter node types across nodes and causal attributes across arrows. A motif scores only by how far it beats the shuffle (tanh of half its z-score, discounted when it occurs only once or twice), so a pattern doesn't win just because the Mirror is full of virtues or arrows. A card's score is the weighted mean of its motifs, between 0 and 1.
4. **Reversed.** A card whose motifs make sense with the signs flipped (raises becomes lowers) is also scored that way; if the flipped reading is clearly stronger, the card is drawn reversed, upside down as in tarot.
5. **The top card wins.** The highest-scoring card is always the one shown, however close the runner-up. Only if even the best card scores under 0.45, meaning no card's patterns beat chance by much, does the tab show a blank card back (Undecided) and name the two closest cards. A Mirror under three concepts is not read at all.
6. **The drawing.** The winning card's face is a fixed expression stored with the card (eye openness, brow angle, mouth curve and opening, gaze, head tilt) under a small glyph of its emblem, so a face always means the same card. It is drawn fainter the narrower the margin.
7. **Why this card.** Under the face are the motifs that matched, how often each was found, up to three examples with their quotes (click one to highlight those concepts on the canvas), the runners-up, every card's score, and any relation names the cards don't read yet.

The reading is deterministic (a fixed shuffle seed), read-only, and makes no Claude calls. It is an interpretive vocabulary, not a validated psychological instrument: the narrow claim is that the Mirror contains this card's patterns more often than chance would predict, and the tab shows where. The full design is in the project's archetype face design note.

## The Question engine

The corpus leaves parts of the subject's model open. The Question engine (`androidmirror/mirror/question.py`) finds them in the built structures, with no Claude call, and ranks them by how central the concepts are (passages that mention them plus passages behind their causal judgments) and how much evidence is on the line. Answer by picking an option, or in your own words if Claude is on (one call reads the answer into an option).

An answer is a proposal, saved in `data/mirror/<corpus>/answers.json` with a preview: the arrows that would be added, removed or flipped, the judgments that would be settled differently, the evaluations and values that would change. Only approving it (with a name) adds it to the Mirror. Approved answers are a source of their own, shown as the passage "Answer": a new causal judgment, a ruling on one the corpus gives (affirmed constraints are fitted before any other; retracted ones keep their quotes but carry no weight; "directly" or "only through" settles an open arrow), a fact the subject holds, a fact that no longer enters inference, or a value. They are laid over the corpus-built Mirror on every load and fit, so a rebuild keeps them and revoking one undoes it.

## Mutate: merging and splitting concepts

The codebook decides what counts as one concept, and it can be wrong both ways: two words for one thing ("wrath", "anger"), a word and its opposite ("calm" is anger at −1), or one word for two things (the pleasure the subject calls good and the pleasure they call bad). The mutate engine finds such cases from the Mirror's own structure (the subject says two concepts are the same, each is a kind of the other, they play the same or a mirror-image causal role, one is called both good and bad or said both to raise and to lower one thing), or takes them from you, and re-evaluates the edges: everything read from each passage moves with it (signs turned around for an opposite), the knowledge base is re-run and the causal model re-fitted, so merged evidence is weighed together and a judgment between two merged concepts ("anger breeds wrath") is shown as saying nothing any more. Each proposal previews what changes in both graphs, including the held-out check, and **is applied only when you approve it**; it can be undone, and approved mutations are made again on every rebuild. Offline except `split --claude`, where Claude reads which sense each passage uses.

## Derange: the weakest basis

Some of what the subject holds is said in forty passages and some in one; some of it carries nothing else and some carries a whole account. Derange looks for little support under a lot of weight, and only reads: it never changes the Mirror.

- **What matters most.** In the causal model, the subject's ends: what the values layer says they urge more or less of, and everything that acts on it. In the knowledge graph, the concepts the corpus is most about, and everything the subject holds about them, stated or inferred.
- **The weakest basis.** Causal: for each thing that acts on an end, the minimum cut between them, each arrow's capacity being the passages that make it. Conceptual: the stated facts the proofs of those held facts rest on.
- **Knock it out and look.** Each candidate is removed (the judgments or the fact behind it, whole) and the model re-fitted, or the knowledge re-derived, as a build would. What changes is what it was holding up: things that no longer act on an end or act on it with another sign, claims that stop following, inferred facts that stop following, arrows the subject's other words would draw instead. If the rest of the corpus carries it anyway, it is not reported.
- **Score:** each important subgraph's share that falls, times its weight, per passage of support. Each finding ends in a question for the person, of the form: "Is it so that x lowers y? The subject says so in 1 passage, yet without it 7 of the 9 things that act on y would no longer act on it the same way."

`python -m androidmirror derange` prints the list; the Derange tab shows it with the quotes.

## Command reference

The full guide, with every option, examples, what each step costs and what to do when something goes wrong, is in [docs/commands.md](docs/commands.md). `python -m androidmirror <command> --help` lists a command's options.

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

## Tests

```
pip install -e ".[dev]"
pytest
```

The tests build a Mirror end to end from a synthetic corpus whose subject holds a known causal model, using an offline stand-in for Claude, and check that the fit recovers it: repetitions don't add weight, an overall claim the chain explains adds no shortcut, a slip that closes a loop is put in tension, and other people's views and hypotheses are never fitted.
