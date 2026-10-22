# Android Mirror

A reflective model of a subjective system (a person, a group, an institution) built from its own words (other data generating mechanisms in progress). Combines three knowledge structures - knowledge graphs for relational knowledge, causal graphs for observational knowledge, and language structures through Claude. 

Four corpora come with it:

| Corpus | Subject | Units |
|---|---|---|
| `meditations` | Marcus Aurelius, *Meditations* (Meric Casaubon translation, Project Gutenberg #2680) | about 410 numbered sections |
| `fomc` | The Federal Reserve's FOMC, every policy statement since 1994 (federalreserve.gov) | one statement per meeting |
| `dgn` | The German Society of Neurology's English-language guidelines on Parkinson's disease, the CC BY ones on Europe PMC (set `conditions` in `data/corpora/dgn.json` to add others, or `[]` for all) | one guideline section |
| `epictetus` | Epictetus, the *Discourses*, *Enchiridion* and *Fragments* (George Long translation, Standard Ebooks), a Stoic to set beside Marcus in Transmutate | about 360 passages of a chapter or part of one |

Any other subject works too: write a spec in `data/corpora/` pointing at a text file, or at a folder of `.txt` documents. Or make a corpus of your own words with the [interview](#the-interview) (the OODA loop, Get to Know You!, or both).

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
python -m androidmirror build --corpus epictetus                # downloads the text first
python -m androidmirror serve
```

Open http://localhost:8000. Building makes two Claude calls per passage plus a few to consolidate the vocabulary, cached under `data/mirror/<corpus>/`; a build that stops resumes where it left off. Extending `--limit` reads only the new passages, unless the vocabulary of concepts changes, in which case the judgments pass reads every passage again against the new vocabulary. `--model` picks a cheaper model for the bulk reading. `python -m androidmirror benchmark` is free and runs offline.

## How a Mirror is built

1. **Passages.** The corpus is split into units, dependent on the corpus.
2. **Open coding (one Claude call per passage).** Concept mentions, and descriptive statements in a fixed relation vocabulary, each with its modality (asserts, prescribes, asks, entertains), attribution (the subject, or someone it quotes) and polarity.
3. **Codebook.** All labels are consolidated into one fixed set of concepts: synonyms merged, opposites folded in with polarity −1 ("freedom from anger" is anger at −1), word senses kept apart.
4. **Causal judgments (one Claude call per passage).** With the codebook fixed, Claude lists every causal judgment the passage makes, in the systems's terms, over concepts from the codebook (see [the causal model](designs/causal-model.md)), and what the passage urges more or less of. The vocabulary is the same in every call and is prompt-cached, which lowers their cost.
5. **Knowledge graph and inference** (no Claude calls).
6. **Fitting the causal model** (no Claude calls). Re-run it any time: `python -m androidmirror causal --min-weight 2`.

## Design notes

How each part works, in `designs/`:

- [The descriptive ontology](designs/descriptive-ontology.md): kinds, relations, rules and checks that make the knowledge graph something to reason over.
- [The causal model](designs/causal-model.md): the five kinds of causal judgment, how the graph is fitted to them, the held-out check, and the benchmark.
- [Checking the knowledge graph](designs/knowledge-verification-validation.md): verification and validation schemes, and the standard to check against.
- [The engines](designs/engines.md): Question, Mutate and Derange.
- [The OODA model](designs/ooda-model.md): placing the Mirror's ideas in Observe, Orient, Decide and Act, and linking the stages.
- [The face](designs/face.md): how the Face tab picks a Major Arcana card for the whole Mirror.
- [Silence as missing data](designs/silence-as-missing-data.md): a brainstorm for the hidden-cause view.
- [Open problems](designs/open-problems.md): what is still unsolved.

## The web app

- **Canvas:** the knowledge graph (node color is the subject's evaluation), the OODA loop (see below) or the causal model (green arrows raise, red bars lower, thickness is the number of passages, dashed may run only through another path, a double ring is something the subject urges more or less of). Click a node for its card; click an arrow for its quotes.
- **Node:** what the concept is, its aliases and classes, stated facts with quotes, inferred facts with their rules, its arrows, every causal judgment about it with its status, and what the subject urges about it. Click a locator to read the passage and the judgments read from it.
- **Knowledge:** "does it follow?" queries with proof trees, and the list of inconsistencies.
- **Causal:** counts, the held-out check, Trace, the arrows, tensions, denials that hold, claims explained by a chain, values, reported views, and judgments about concepts outside the codebook.
- **Ask:** two modes. **As the subject** (the default) gives one Claude call the whole knowledge graph, causal model and OODA model, still with no passages, tells it to become the system they describe (a person, a committee, a body of guidance) and answers the question in that system's voice, without citations. The structures sit in the cached system prompt, so a second question soon after costs less. If they are longer than `ANDROIDMIRROR_SUBJECT_CHARS` characters (default 600,000, about 150k tokens), the least important items are left out first (facts that only follow by logic, then views mentioned without endorsing, then the least supported stated facts) and the answer says how many. **About the subject, with sources:** the Oracle answers from the knowledge graph, the causal model and the OODA model only (stated facts and their relations, inferred facts with the rule and facts behind them, the subject's causal judgments, links, chains of links and values), never from the passages, and cites each item it uses as [K3], [C2] or [O1]. Ask about a stage ("what does Marcus observe?") or the loop ("does the loop close?") and it reads that stage's ideas and the links between stages. A second step then proposes two or three ways to use the answer: an idea to change, the pathway through the subject's model it would travel, the outcome it should move, and what has to be true for it to work.
- **Derange:** where the worldview rests on least, under what matters most (see [the engines](designs/engines.md#derange-the-weakest-basis)).
- **OODA:** the subject's ideas in the four stages of the OODA loop, built from the knowledge graph and the causal model, with why each idea is where it is, the cause-and-effect between stages, and what the shape shows (does the loop close, does Orient skip straight to Act). Move an idea to another stage if the placement is wrong; that changes neither graph. See [the OODA model](designs/ooda-model.md).
- **Face:** the Major Arcana card the whole Mirror most resembles, drawn as a face. See [the face](designs/face.md).
- **Mutate:** merge concepts that are one, or split one used in two senses. See [the engines](designs/engines.md#mutate-merging-and-splitting-concepts).
- **Transmutate:** compare two Mirrors (how they differ in knowledge, causes and OODA loop), then combine them into one and trim it to what is strong; nothing is written until you approve.

## The interview

Two questionnaires, which one person can answer in any order and in any mix:

- **OODA loop interview**: 106 questions on how the person answering observes, orients, decides and acts (John Boyd's OODA loop), grouped by stage: Observe (26), Orient (28), Decide (26) and Act (26, ending on how results feed back into what is noticed next). In `data/interview_ooda.json`.
- **Get to Know You!**: 80 questions on what the person holds dear and how they picture it, 16 per section: The sacred (including what is and isn't up to them), Metaphor and inner image (each image asks what moves what inside it, so it feeds the causal model too), Feelings and what stirs them, What doesn't matter (independence and "not this but that" claims, which corpora rarely state), and Good and bad (fills the good / bad / indifferent classes, so a contradiction shows up as a tension). In `data/interview_know.json`. The reasoning behind the sections is in [designs/interview-dimensions.md](designs/interview-dimensions.md).

Answer them on the interview page, http://localhost:8000/interview, linked from the top of the app (switch questionnaires with the buttons above the question), or `python -m androidmirror interview "Your Name" [--bank know]` in a terminal, one line per answer. Each answer is saved as soon as it is given, under `data/interviews/`, so you can stop at any point and pick up where you left off; skipped questions come back at the end, and any answer can be changed later. No Claude calls.

**Write the corpus** turns the answers so far, from both questionnaires, into the one corpus `interview-<name>`, so they all land in the same knowledge graph and causal model: one passage per answer, with the question kept as the interviewer's words so that only the answer is read as the subject's view, and the stage as its group. Then build it like any other corpus:

```powershell
python -m androidmirror build --corpus interview-your-name
```

Write the corpus again after more answers; a changed answer is read again and the rest stay cached. The answers, the corpus and its spec stay on your computer (they are gitignored).

## Command reference

`python -m androidmirror <command> --help` lists a command's options.

| Command | What it does |
|---|---|
| `build --corpus NAME [--limit N] [--workers 6] [--model ID]` | Code, consolidate, read judgments, reason and fit. Needs the key. |
| `causal --corpus NAME [--min-weight 1] [--folds 5]` | Re-fit only the causal model. No Claude calls. |
| `benchmark [--n 300] [--seeds 5] [--claude]` | The planted-subject benchmark. Offline and free unless `--claude`. |
| `serve [--port 8000]` | The web app. |
| `derange --corpus NAME [--top 10] [--knockouts 30]` | The weakest basis under what matters most. No Claude calls. |
| `ask "question" [--corpus NAME] [--cited]` | Ask from the command line: as the subject, or with `--cited` the Oracle's cited answer and ways to use it. |
| `mutate propose\|merge\|split\|show\|approve\|reject\|undo` | Merge or split concepts; previews, applied only on approval. Offline except `split --claude`. |
| `question [--corpus NAME] [--limit 10] [--kind K] [--list]` | Answer the Question engine's questions; approve each proposal by typing your name. `--pending`, `--approve ID --by NAME`, `--reject ID`, `--revoke ID`. No Claude calls unless you answer in words. |
| `transmutate compare A B [--claude] [--pair A=B] [--opposite A=B] [--drop C]` | How two Mirrors differ in knowledge, causes and OODA loop. Writes nothing. |
| `transmutate propose A B [same options]` then `transmutate approve A B [--keep N]` | Combine two Mirrors and trim to what is strong; the proposal includes the comparison. Writes nothing until approve. |
| `interview NAME [--bank ooda\|know] [--stage STAGE]` | Answer the OODA interview (default) or Get to Know You! (`--bank know`) in the terminal, resuming where NAME left off, then write the corpus `interview-NAME` from all answers. No Claude calls. |
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
| `androidmirror/mirror/corpus.py`, `sources/fomc.py`, `sources/dgn.py`, `sources/epictetus.py` | Passages; the FOMC, DGN guideline and Epictetus downloaders |
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
| `androidmirror/mirror/transmutate.py`, `compare.py` | Transmutate: align, compare, combine and fitness-trim two Mirrors |
| `androidmirror/mirror/ooda.py` | The OODA model: tiers built from the two graphs |
| `androidmirror/mirror/interview.py`, `data/interview_ooda.json`, `data/interview_know.json` | The interview: both questionnaires, saved answers, the corpus they become |
| `androidmirror/server.py`, `web/index.html` | The app |
| `androidmirror/p1_*`, `refactor.py`, `breaker.py`, `chainer.py`, `evolver/` | Prototype 1, paused |

## Tests

```
pip install -e ".[dev]"
pytest
```

The tests build a Mirror end to end from a synthetic corpus whose subject holds a known causal model, using an offline stand-in for Claude, and check that the fit recovers it: repetitions don't add weight, an overall claim the chain explains adds no shortcut, a slip that closes a loop is put in tension, and other people's views and hypotheses are never fitted.
