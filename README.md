# Android Mirror

A contextual generation engine: a reflective model of one person's subjective world, built from their own writing.
It combines Claude (through the Anthropic API) with two linked ontologies:

- **Descriptive ontology:** a knowledge graph of people, virtues, vices, emotions, beliefs, practices and values, and how they relate.
- **Causal ontology:** a DAG over the same nodes saying what drives what, with a sign, a strength and a confidence on each edge.

This first prototype (Build A) is seeded with **Marcus Aurelius, *Meditations*** (George Long translation, public domain, downloaded from Project Gutenberg).

## Run it on your computer

You need Python 3.10 or newer.

```powershell
git clone https://github.com/ariroane/androidmirror
cd androidmirror
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -e .
copy .env.example .env          # then put your key in .env: ANTHROPIC_API_KEY=sk-ant-...
python -m androidmirror serve
```

Open http://localhost:8000. The key stays in `.env` on your machine; `.env` is ignored by git.

To start with something on the canvas, queue the hand-drafted patches from Books I and II and approve them in the Review tab:

```powershell
python -m androidmirror seed
```

## What you can do

| Tab | What it does | Needs Claude |
| --- | --- | --- |
| Review | Have Claude read sections of a book and propose changes; approve or reject each one with its source quote | Reading: yes. Approving: no |
| Ask | Ask a question; the Mirror finds the relevant nodes and answers from them | For a written answer (otherwise shows the matching facts) |
| Change | Raise or lower a node in the causal model and see what moves; optionally have Claude retell it in Marcus's voice | Only for the retelling |
| Investigate | Most important nodes, and the Breaker: cut points, load-bearing causes, tensions | No |
| Face | A Chernoff face drawn from the Mirror's state (coherence, tension, valence, fragility, certainty, size) | No |

Nothing reaches the ontology until someone approves it. Every approval records who approved it, and every applied patch stores its inverse so it can be rolled back.

## Evolving the design

The Evolver looks for a better *shape* of the ontology, not new facts. It takes the operations the refactor, break and chain engines propose, tries sequences of them on private copies of both graphs, and scores each resulting design on how well it mirrors the subject:

- **probes:** claims about the subject the model should reproduce, each tied to a quote (`data/probes_meditations.json`; put your own in `data/probes.json`). A design that loses a probe the current ontology passes is thrown out.
- **coverage, grounding:** keeps the quoted evidence, and stays backed by quotes
- **coherence, robustness, parsimony:** connected, few single points of failure, no bigger than it needs to be

A beam search keeps the best few designs each round. The winners go to the Review tab as one patch each (kind `evolve`, with the steps, the fitness before and after, and which probes it now passes). Nothing changes until you approve one, and a design refuses to apply if the ontology changed after it was found.

```powershell
python -m androidmirror evolve --dry-run      # show the best designs
python -m androidmirror evolve --keep 3       # queue them for approval
```

An engine plugs in by exposing `propose(snapshot: GraphSnapshot) -> list[Patch]` in `androidmirror.refactor`, `androidmirror.breaker` or `androidmirror.chainer`. Until those land, `evolve` reports that no engines are installed.

## Command line

```powershell
python -m androidmirror ingest --book 2 --start 0 --count 3   # Claude reads Book II, sections 1 to 3
python -m androidmirror pending                               # list proposed patches
python -m androidmirror approve all --as "Your Name"          # or list patch ids
python -m androidmirror investigate                           # importance, Breaker, face metrics as JSON
python -m androidmirror evolve --dry-run                       # search for better designs (see above)
```

State lives in `data/` (`ontology.json`, `patches.json`, `corpus/`) and is not committed.

## How it is built

```
androidmirror/
  ontology.py   two linked graphs; atomic apply_ops returning inverse ops; versioned snapshots
  patches.py    the Patch / Op format (open `kind`, `extra` for engine-specific fields)
  steward.py    governance: dedupe, quote check against the corpus, approval queue, rollback
  llm.py        Claude agents: Cartographer + Causal Modeler, Oracle (answers), Storyteller
  ingest.py     reads corpus chunks and queues self-contained patches
  corpus.py     downloads and splits Meditations into "Book II, 1"-style sections
  analysis.py   importance, interventions (do-operator on the DAG), Breaker, face metrics
  evolver/      searches refactor/break/chain sequences on sandboxes; queues the best designs
  query.py      finds relevant nodes and builds grounded facts for answers
  server.py     FastAPI app; web/index.html is the UI (Cytoscape.js, vendored)
```

The Mirror's agents use `claude-opus-5-5` by default with structured outputs and server-side refusal fallback; set `ANDROIDMIRROR_MODEL` in `.env` to change it.

Other engines (for example a graph refactor engine) read `Ontology.graph_snapshot()` (copies of both graphs plus the `version`; every node and edge carries `provenance`) and propose `Patch` objects with their own `kind`, `base_version`, `rationale` and anything else in `extra`. Setting `extra["require_fresh_base"] = True` makes the Steward refuse the patch if a node it touches changed after `base_version`. Patches are applied only through the Steward, only with an approval record, and each stores its `inverse`.

## Tests

```powershell
pip install -e ".[dev]"
pytest
```
