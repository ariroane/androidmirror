# Android Mirror

A contextual generation engine: a reflective model the subjective operating frame of a person, persons, or system, built from their language.
It combines works through a combination of three knowledge structures: Claude (through the Anthropic API) with two linked ontologies over the same nodes:

- **Descriptive ontology:** a knowledge graph of people, virtues, vices, emotions, beliefs, practices and values, and how they relate.
- **Causal ontology:** a DAG saying what drives what, with a sign, a strength and a confidence on each edge.

This initial prototype (Build A) is seeded with **Marcus Aurelius, *Meditations*** (George Long translation, public domain, downloaded from Project Gutenberg).

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

## The one rule: nothing changes without your approval

Every part of the Mirror, whether it is Claude reading the book or one of the engines below, only **proposes** patches. A patch lists its operations, why it was made, and the quote it rests on. Proposals wait in the Review tab (or `pending` on the command line) until a person approves them by name. The Steward then applies the patch to both graphs at once and stores its inverse, so any approved patch can be rolled back. Duplicates are dropped, and quotes are checked against the book.

Patches from the engines carry a `kind` (`refactor`, `break`, `chain` or `evolve`) so you can tell where each came from. Most also refuse to apply if a node they touch changed after they were proposed; propose again to get a fresh one.

## The web app

| Tab | What it does | Needs Claude |
| --- | --- | --- |
| Review | Have Claude read sections of a book and propose changes; approve or reject each proposal (from Claude or any engine) with its source quote | Reading: yes. Approving: no |
| Ask | Ask a question; the Mirror finds the relevant nodes and answers from them | For a written answer (otherwise shows the matching facts) |
| Change | Raise or lower a node in the causal model and see what moves; optionally have Claude retell it in Marcus's voice | Only for the retelling |
| Investigate | Most important nodes; the Breaker's cut points, load-bearing causes and tensions; and split/cut suggestions with a button to propose each | No |
| Face | A Chernoff face drawn from the Mirror's state (coherence, tension, valence, fragility, certainty, size) | No |

The buttons above the canvas show both graphs, only the descriptive one, or only the causal one.

## The engines

Four engines reshape the ontology. None of them calls Claude or writes to the ontology; each turns the current graphs into proposals for you to review.

### Refactorer: tidy up

Finds redundancy and inconsistency and proposes `refactor` patches:

- **Merge duplicates:** nodes of the same type with nearly identical labels.
- **Normalize relations:** relation names that differ only in form (`value` / `values`).
- **Reduce transitive edges:** a causal edge already explained by a longer path and with no quote of its own.
- **Archive orphans:** nodes with no edges and no quote are marked archived, never deleted.

Contradictions (a direct causal edge whose sign disagrees with a longer path) are listed as findings for you to judge; it never resolves them itself. Each patch shows a risk level and before/after metrics.

```powershell
python -m androidmirror.refactor --dry     # show suggestions only
python -m androidmirror.refactor           # queue them for approval
```

### Breaker: take apart

Finds where the world is fragile and proposes `break` patches:

- **Split** one node into two or more parts. Each relation and causal edge moves to one part (or is copied to every part), so both graphs stay in step. With no parts named, it makes one part per cluster the node alone holds together. The original is archived with `split_into`, never deleted, and a split can never create a cycle.
- **Sever** two nodes: cut the causal edge and the descriptive relations between them together. The proposal says how much causal influence is lost and warns if the cut disconnects the graph.

Suggestions are ranked by how many separate clusters a node bridges and how weak a link is (people are left out of split suggestions). You can propose them from the Investigate tab or here:

```powershell
python -m androidmirror break-candidates                      # what the Breaker would split or sever
python -m androidmirror break-split simplicity --parts "Plain living" "Plain speech"
python -m androidmirror break-sever duty calm                 # cut every link between two nodes
python -m androidmirror break-sever duty calm --rel RELATION  # only the named relations
python -m androidmirror break-sever duty calm --keep-causal   # leave the causal edge
```

### Chainer: join together

Proposes `chain` patches that union things across both graphs:

- **Merge** one or more nodes into a survivor. Their relations and causal edges move to the survivor, duplicate edges collapse (keeping every quote), and the old labels are kept as aliases.
- **Chain** another mirror's `ontology.json` into this one. Nodes that mean the same thing are aligned onto existing ones; the rest come in under a namespace (`epictetus:fear`), so several mirrors can form one shared context.

It refuses rather than guesses when a union would create a causal cycle, or when two causal edges between the same pair have opposite signs. Use `--on-conflict keep_existing` or `keep_incoming` to say which side wins.

```powershell
python -m androidmirror chain merge anger wrath --label Anger         # fold Wrath into Anger
python -m androidmirror chain align other/ontology.json               # preview which nodes would match
python -m androidmirror chain import other/ontology.json --as Epictetus
python -m androidmirror chain import other/ontology.json --align fear=anxiety --separate pride
```

### Evolver: search for a better design

The Evolver looks for a better *shape* of the ontology, not new facts. It takes the operations the Refactorer, Breaker and Chainer propose, tries sequences of them on private copies of both graphs, and scores each resulting design on how well it mirrors the subject:

- **Probes:** claims about the subject the model should reproduce, each tied to a quote (`data/probes_meditations.json`; put your own in `data/probes.json`). A design that loses a probe the current ontology passes is thrown out.
- **Fidelity, distinctness:** still carries what the current ontology says (following merges and splits), and doesn't fold together concepts the subject names differently.
- **Coverage, grounding:** keeps the quoted evidence, and stays backed by quotes.
- **Coherence, robustness, parsimony:** connected, few single points of failure, no bigger than it needs to be.

Structure alone moves the score only a little, so a design wins by keeping what the subject said or reproducing more of it. Every step in a design has to improve on the shorter design it extends. A beam search keeps the best few designs each round; the winners go to Review as one `evolve` patch each, with the steps, the fitness before and after, and which probes it now passes. A design refuses to apply if the ontology changed after it was found.

```powershell
python -m androidmirror evolve --dry-run                 # show the best designs
python -m androidmirror evolve --keep 3                  # queue them for approval
python -m androidmirror evolve --generations 5 --beam 6 --probes my_probes.json
```

## Command reference

All commands are `python -m androidmirror <command>`; add `--data DIR` before the command to use a different data folder.

| Command | What it does |
| --- | --- |
| `serve [--port 8000]` | Run the web app |
| `download` | Download the *Meditations* text |
| `seed [file]` | Queue hand-drafted patches (default `data/seed_book1.json`) |
| `ingest --book 2 --start 0 --count 3` | Claude reads Book II, sections 1 to 3, and queues proposals |
| `pending [--status proposed]` | List patches (also `approved`, `rejected`, `rolled_back`) |
| `approve ID... --as "Your Name"` | Approve patches by id, or `all` |
| `investigate` | Importance, Breaker findings and face metrics as JSON |
| `break-candidates`, `break-split`, `break-sever` | Breaker (see above) |
| `chain merge`, `chain align`, `chain import` | Chainer (see above) |
| `evolve` | Evolver (see above) |

The Refactorer runs as its own module: `python -m androidmirror.refactor [--dry] [--data DIR]`.

State lives in `data/` (`ontology.json`, `patches.json`, `corpus/`) and is not committed.

## HTTP API

The web app uses these routes; every one that changes anything only queues a proposal, except approve, reject and rollback.

| Route | What it does |
| --- | --- |
| `GET /api/status`, `GET /api/graph`, `GET /api/metrics` | Version and counts, both graphs, face metrics |
| `GET /api/patches?status=proposed` | List patches |
| `POST /api/patches/{id}/approve` · `/reject` · `/rollback` | Decide on a patch (body: `{"approver": "...", "note": "..."}`) |
| `POST /api/ingest` | Claude reads sections (`book`, `start`, `count`) |
| `POST /api/query` | Ask a question |
| `POST /api/intervene` | Raise or lower a node (`node`, `delta`, `retell`) |
| `GET /api/investigate` | Importance and Breaker findings |
| `GET /api/break/candidates`, `GET /api/break/links/{node}` | Split and sever suggestions; a node's links |
| `POST /api/break/split`, `POST /api/break/sever` | Propose a split or a cut |
| `POST /api/chain/merge`, `/align`, `/import` | Propose a merge; preview or propose chaining another ontology |
| `GET /api/probes`, `POST /api/evolve` | The probes in use; run the Evolver (`generations`, `beam`, `keep`, `queue`) |

## How it is built

```
androidmirror/
  ontology.py     two linked graphs; atomic apply_ops returning inverse ops; versioned snapshots
  patches.py      the Patch / Op format (open `kind`, `extra` for engine-specific fields)
  steward.py      governance: dedupe, quote check against the corpus, approval queue, rollback
  llm.py          Claude agents: Cartographer + Causal Modeler, Oracle (answers), Storyteller
  ingest.py       reads corpus chunks and queues self-contained patches
  corpus.py       downloads and splits Meditations into "Book II, 1"-style sections
  query.py        finds relevant nodes and builds grounded facts for answers
  analysis.py     importance, interventions (do-operator on the DAG), Breaker findings, face metrics
  refactor.py     Refactorer: duplicate merges, relation names, transitive edges, orphans
  breaker.py      Breaker: split and sever across both graphs (breaker_api.py adds routes and commands)
  chainer.py      Chainer: merge nodes; chain another mirror's ontology in
  evolver/        Evolver: search over engine sequences on sandboxes (operators, fitness, search)
  server.py       FastAPI app; web/index.html is the UI (Cytoscape.js, vendored)
  cli.py          the commands above
```

The Mirror's agents use `claude-opus-5-5` by default with structured outputs and server-side refusal fallback; set `ANDROIDMIRROR_MODEL` in `.env` to change it.

### Writing a new engine

An engine reads `Ontology.graph_snapshot()` (frozen copies of both graphs plus the `version`; every node and edge carries `provenance`) and exposes `propose(snapshot: GraphSnapshot) -> list[Patch]`. Give each patch its own `kind`, a `base_version`, a `rationale`, and anything else in `extra`. Setting `extra["require_fresh_base"] = True` makes the Steward refuse the patch if a node it touches changed after `base_version`. Patches are applied only through the Steward, only with an approval record, and each stores its `inverse`. The Evolver picks up `androidmirror.refactor`, `androidmirror.breaker` and `androidmirror.chainer` through this interface and skips any that is missing.

## Tests

```powershell
pip install -e ".[dev]"
pytest
```
