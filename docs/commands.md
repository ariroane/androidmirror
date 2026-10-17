# Command guide

Every command runs from the repository folder, inside the virtual environment:

```powershell
cd androidmirror
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
python -m androidmirror <command> [options]
python -m androidmirror <command> --help    # the options of one command
```

Commands that call Claude need `ANTHROPIC_API_KEY` in the local `.env` file. Everything Claude returns is cached under `data/mirror/<corpus>/`, so running a command again only pays for work it has not done yet, and a run that stops (a closed window, a network error) resumes where it left off.

## Quick reference

| Command | What it does | Calls Claude? |
|---|---|---|
| `build` | Read the corpus and build the Mirror | Yes, about 2 calls per passage the first time; nothing for passages already cached |
| `causal` | Re-run causal discovery with other settings | No |
| `validate` | Score the subject transformation on held-out situations | Yes, 3 per held-out situation (about 120) |
| `fill` | Fill in values the text leaves unstated, as the subject | Yes, up to 1 per situation, plus about 2 per leak check |
| `vignettes` | Ask the subject about invented situations | Yes, 2 per vignette (144 by default) |
| `benchmark` | Test the causal method on a model where the truth is known | No (`--claude`: yes) |
| `serve` | Open the web app | No (the Ask tab makes 1 call per question) |
| `ask` | Ask the Mirror a question from the command line | 1 call per question |
| `p1` | The paused Prototype 1 app | Some subcommands |

## The usual order

```powershell
python -m androidmirror build --corpus meditations --limit 30   # a cheap trial on 30 passages
python -m androidmirror serve                                    # look at it: http://localhost:8000
python -m androidmirror build --corpus meditations              # the whole book (pays only for the new passages)
python -m androidmirror validate --corpus meditations           # optional: score the subject transformation
python -m androidmirror fill --corpus meditations               # optional: fill values the text leaves unstated
python -m androidmirror vignettes --corpus meditations          # optional: invented situations, shown in orange
python -m androidmirror causal --corpus meditations --alpha 0.01   # any time: try other discovery settings, free
```

`fill` and `vignettes` refuse to run until `validate` has, because they only fill in concepts the validation trusted. After `git pull`, run `build` again: it reuses everything cached and only redoes what the update changed.

## Options for every command

| Option | Meaning |
|---|---|
| `--data FOLDER` | Where corpora and built Mirrors live (default `./data`). Goes **before** the command: `python -m androidmirror --data D:\mirrors build` |
| `--corpus NAME` | Which corpus: `meditations` (the default), `fomc`, any spec in `data/corpora/`, or a path to a spec file. Used by `build`, `causal`, `validate`, `fill`, `vignettes` and `ask` |
| `--workers N` | How many Claude calls run at once (default 6). Lower it if you hit rate limits. Used by `build`, `validate`, `fill` and `vignettes` |

---

## build

Reads the corpus with Claude and builds both graphs.

```powershell
python -m androidmirror build --corpus meditations --limit 30
python -m androidmirror build --corpus meditations
python -m androidmirror build --corpus fomc
```

What happens, in order (the log prints each step):

1. **Passages.** The text is downloaded if needed and split into passages (412 for the Meditations, one per meeting for the FOMC).
2. **Open coding**, 1 Claude call per passage: concepts, statements and causal claims.
3. **Codebook**, a few calls in total: all the labels are merged into one set of concepts.
4. **Topics**, no calls: passages are grouped by their words and concepts.
5. **Situations**, 1 Claude call per passage: the situations each passage reasons about, and how each candidate concept stands in them.
6. **Knowledge graph, inference and causal discovery**, no calls.

| Option | Meaning |
|---|---|
| `--limit N` | Only the first N passages. A cheap trial; raising it later pays only for the new passages |
| `--model ID` | Which Claude model reads the passages (default: the one in `androidmirror/llm.py`, or the `ANDROIDMIRROR_MODEL` variable in `.env`). A cheaper model lowers the cost of a big build |
| `--no-situations` | Skip step 5 and use one rough situation per passage from the open coding. Halves the cost, but every unmentioned concept is simply unknown and nothing is read between the lines |
| `--topics N` | How many topics to group passages into (default about one per 60 passages, between 3 and 8; `0` for none) |
| `--workers N` | Claude calls at once (default 6) |

Notes:

- If the list of candidate concepts changes (for example after a bigger `--limit`), the situations pass reads every passage again, because each reading is against that list.
- Building needs the key even when everything is cached, as a safety check.
- A build does not undo `fill` or `vignettes`: their results are kept and re-applied.

## causal

Re-runs only causal discovery on a built Mirror, with other settings. No Claude calls, so try as many settings as you like; reload the web app to see the result. Settings you give are remembered for later runs and builds.

```powershell
python -m androidmirror causal --corpus meditations --alpha 0.01
python -m androidmirror causal --corpus meditations --view stated --topic off
python -m androidmirror causal --corpus meditations --max-vars 15 --min-support 20
```

| Option | Meaning | Default |
|---|---|---|
| `--alpha A` | Significance level of each independence test. Lower means fewer, surer links | 0.05 |
| `--depth N` | The most concepts a test may hold fixed at once. Higher finds more indirect paths but needs more data | 3 |
| `--max-vars N` | The most concepts in the causal graph | 25 |
| `--min-support N` | Fewest situations a concept must be known in to be a candidate (or 1.5% of situations, if more) | 8 |
| `--bootstrap N` | Resamples used to score each edge's stability (`0` skips it and runs faster) | 20 |
| `--no-claims` | Ignore the subject's own causal claims, to see what the situations alone say | claims used |
| `--topic on\|off` | Hold each passage's topic fixed in the tests | on |
| `--view stated\|implied\|all` | Which values count: only what the text states; also what the subject's beliefs imply (default); or also usual levels and Claude's uncited readings | implied |
| `--tiers FILE` | A JSON list of lists of concept ids, earliest first: concepts in a later tier can never cause ones in an earlier tier. Example: `[["misfortune"], ["anger", "distress"]]` | none |

It prints the number of concepts, edges and DAGs in the class, for example `14 variables, 19 edges; class size 48 from the data, 6 with pins`.

## validate

Tests whether the subject transformation answers like the subject before it is allowed to fill anything in. Some real situations are held out: half of what they state is shown, and Claude is asked for the rest three ways: as the subject, as plain Claude, and as a generic adult. A concept is trusted only where the subject answers are right at least 60% of the time they answer.

```powershell
python -m androidmirror validate --corpus meditations
python -m androidmirror validate --corpus meditations --n 80
```

| Option | Meaning | Default |
|---|---|---|
| `--n N` | Held-out situations (at most 15% of those eligible) | 40 |

Cost: 3 calls per held-out situation, so about 120 by default. The scores appear in the Causal tab under "The subject transformation".

## fill

Fills in values of trusted concepts in situations where the text is silent, answered as the subject, each citing a passage or belief. Then a sample of answers gets prior-leak checks: asked again without the passage it cites, and compared with what a generic adult says. An answer that survives without its citation, or that anyone would give, is demoted to "reading" and left out of the default view.

```powershell
python -m androidmirror fill --corpus meditations
python -m androidmirror fill --corpus meditations --leak-sample 60
```

| Option | Meaning | Default |
|---|---|---|
| `--leak-sample N` | How many filled values to leak-check | 30 |

Cost: up to 1 call per situation with something to fill (the log says how many before it starts), plus about 2 per leak check. Needs `validate` first.

## vignettes

Invents situations, each built on a real one with one concept set high or low, and asks the subject transformation (and a generic adult) how everything else stands. Because the concept is set, these are interventions on the subject's model, so they can orient links the corpus cannot. What they change is drawn in orange; the toggle on the canvas hides it.

```powershell
python -m androidmirror vignettes --corpus meditations
python -m androidmirror vignettes --corpus meditations --per-level 4 --max-targets 4   # cheaper
```

| Option | Meaning | Default |
|---|---|---|
| `--per-level N` | Vignettes per concept and level (low, high) | 6 |
| `--max-targets N` | How many concepts to set, those on unoriented links first | 6 |

Cost: `per-level × 2 levels × max-targets` vignettes, 2 calls each: 72 vignettes and 144 calls at the defaults. Needs `validate` first.

## benchmark

Runs the causal method on corpora where the answer is known: a 12-concept Stoic-style model, plus two "null" corpora where nothing causes anything and only topics shape the text. A good method finds the planted links and almost nothing in the nulls.

```powershell
python -m androidmirror benchmark                  # offline, free, a few seconds
python -m androidmirror benchmark --seeds 10       # steadier averages
python -m androidmirror benchmark --claude --n 150 # Claude writes the passages; costs money
```

| Option | Meaning | Default |
|---|---|---|
| `--n N` | Situations to sample (offline) or passages to write (`--claude`) | 600 |
| `--seeds N` | Repeats to average over (offline only) | 3 |
| `--claude` | Have Claude write each passage in the subject's voice, then run the real pipeline on them. Also reports how accurately the situations pass reads levels | off |

How to read the output: `skeleton_precision` is the share of links found that are real, `skeleton_recall` the share of real links found, `arrows_right` and `arrows_wrong` count directions, and `topic_purity` how well the word-based topics match the true ones. In the null corpora, every edge is a false one.

Cost with `--claude`: 1 call per passage to write it, then about 2 per passage to build it (about 450 calls at `--n 150`).

## serve

Starts the web app on your computer.

```powershell
python -m androidmirror serve
python -m androidmirror serve --port 8080
```

Open http://localhost:8000 (or your port). Stop it with Ctrl+C. The app reads the built Mirrors; it doesn't build anything, so it is free apart from the Ask tab.

## ask

Asks the Mirror a question from the command line, answered from its facts, inferences, causal model and the most relevant passages, with passages cited.

```powershell
python -m androidmirror ask "What does Marcus think causes anger?"
python -m androidmirror ask "How does the Committee respond to inflation?" --corpus fomc
```

Without a key it prints the facts it found instead of an answer.

## p1

The paused Prototype 1 app (patch review, Refactorer, Breaker, Chainer, Evolver). See [prototype1.md](prototype1.md), or `python -m androidmirror p1 --help`.

## When something goes wrong

| Message | What to do |
|---|---|
| `building needs Claude: put ANTHROPIC_API_KEY in your local .env` | Add the key to `.env` in the repository folder (see `.env.example`) |
| `no Mirror built at ...` | Run `build` for that corpus first |
| `fill needs ... validate` | Run `validate` first |
| `validation trusted no variable` | The transformation didn't beat the bar on any concept; `fill` and `vignettes` have nothing they may use. Try `validate --n 80` for a larger sample |
| Rate-limit or overloaded errors | Lower `--workers`, then run the same command again; finished work is cached |
