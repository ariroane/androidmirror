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
| `causal` | Re-fit the causal model with other settings | No |
| `benchmark` | Test the causal fit on a subject whose model is known | No (`--claude`: yes) |
| `serve` | Open the web app | No (the Ask tab makes 1 call per question) |
| `ask` | Ask the Mirror a question from the command line | 1 call per question |
| `transmutate` | Combine two built Mirrors and trim the result to what is strong; writes only when you approve | No (`--claude`: 1 call) |
| `p1` | The paused Prototype 1 app | Some subcommands |

## The usual order

```powershell
python -m androidmirror build --corpus meditations --limit 30   # a cheap trial on 30 passages
python -m androidmirror serve                                    # look at it: http://localhost:8000
python -m androidmirror build --corpus meditations              # the whole book (pays only for the new passages)
python -m androidmirror causal --corpus meditations --min-weight 2   # any time: re-fit with other settings, free
```

After `git pull`, run `build` again: it reuses everything cached and only redoes what the update changed. A Mirror built by Prototype 2 must be built again (the coding prompt changed, so every passage is read once more).

## Options for every command

| Option | Meaning |
|---|---|
| `--data FOLDER` | Where corpora and built Mirrors live (default `./data`). Goes **before** the command: `python -m androidmirror --data D:\mirrors build` |
| `--corpus NAME` | Which corpus: `meditations` (the default), `fomc`, any spec in `data/corpora/`, or a path to a spec file. Used by `build`, `causal` and `ask` |

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
2. **Open coding**, 1 Claude call per passage: concepts and descriptive statements.
3. **Codebook**, a few calls in total: all the labels are merged into one set of concepts.
4. **Causal judgments**, 1 Claude call per passage: every causal judgment the passage makes over the codebook's concepts (causes, thought experiments, means to an end, "not this but that", "makes no difference"), and what it urges more or less of.
5. **Knowledge graph, inference and the causal fit**, no calls.

| Option | Meaning |
|---|---|
| `--limit N` | Only the first N passages. A cheap trial; raising it later pays only for the new passages |
| `--model ID` | Which Claude model reads the passages (default: the one in `androidmirror/llm.py`, or the `ANDROIDMIRROR_MODEL` variable in `.env`). A cheaper model lowers the cost of a big build |
| `--workers N` | Claude calls at once (default 6). Lower it if you hit rate limits |

Notes:

- The judgments pass reads each passage against the vocabulary (the 150 best-supported concepts). If the vocabulary changes (for example after a bigger `--limit`), every passage is read again.
- Building needs the key even when everything is cached, as a safety check.

## causal

Re-fits only the causal model on a built Mirror. No Claude calls, so try as many settings as you like; reload the web app to see the result.

```powershell
python -m androidmirror causal --corpus meditations --min-weight 2
```

| Option | Meaning | Default |
|---|---|---|
| `--min-weight N` | Fewest distinct passages a judgment must be made in to be fitted. Raising it keeps only what the subject says repeatedly | 1 |
| `--folds N` | Blocks of passages for the held-out check | 5 |

It prints the counts: judgments, arrows, claims that follow from the rest, denials that hold, and claims in tension.

## benchmark

Plants a 12-concept causal model in a synthetic subject who states pieces of it in every way the reader looks for, repeats some much more than others, quotes other people, and is sometimes misread. A good fit draws only the planted arrows and still entails every cause-effect pair.

```powershell
python -m androidmirror benchmark                  # offline, free, a few seconds
python -m androidmirror benchmark --seeds 10       # steadier averages
python -m androidmirror benchmark --claude --n 120 # Claude writes the passages; costs money
```

| Option | Meaning | Default |
|---|---|---|
| `--n N` | Passages to generate (offline) or to have Claude write (`--claude`) | 300 |
| `--seeds N` | Repeats to average over (offline only) | 5 |
| `--claude` | Have Claude write each passage in the subject's voice, then run the real reading pipeline. Also reports how accurately the judgments are read | off |

How to read the output: `true_arrows` is the share of drawn arrows that are arrows of the planted model; `cause_to_effect` the share that at least run from a true cause to its effect with the right sign; `recall_of_effects` the share of the model's cause-effect pairs the drawn graph entails; `in_cycles` how many arrows sit on a loop. "Every claim an arrow" is the naive reading, for comparison.

Cost with `--claude`: 1 call per passage to write it, then about 2 per passage to read it.

## serve

Starts the web app on your computer.

```powershell
python -m androidmirror serve
python -m androidmirror serve --port 8080
```

Open http://localhost:8000 (or your port). Stop it with Ctrl+C. The app reads the built Mirrors; it doesn't build anything, so it is free apart from the Ask tab.

## ask

Asks the Mirror a question from the command line, answered from its facts, inferences, the subject's causal judgments and model, their values and the most relevant passages, with passages cited.

```powershell
python -m androidmirror ask "What does Marcus think causes anger?"
python -m androidmirror ask "How does the Committee respond to inflation?" --corpus fomc
```

Without a key it prints the facts it found instead of an answer.

## transmutate

Combines two built Mirrors into one, then trims it to the relationships and causes that are strong. It works in two steps, and nothing is written into `data/mirror/` until the second.

```powershell
python -m androidmirror transmutate propose meditations fomc
python -m androidmirror transmutate propose meditations fomc --claude --opposite tranquillity=uncertainty --drop patience
python -m androidmirror transmutate approve meditations fomc --keep 3 --keep 17
```

**propose** pairs the concepts the two Mirrors share, pools their facts and causal judgments into one Mirror whose subject is both, fits one causal model to the pooled judgments, and scores everything for fitness. It saves the proposal to `data/transmutations/meditations+fomc/` (`proposal.json`, and `proposal.md` to read) and prints it:

- **Aligned concepts:** pairs with the same name (label or alias), plus Claude's pairs by meaning with `--claude`, plus yours. `--pair A=B` says B's concept is A's; `--opposite A=B` says it is A's contrary, and every sign through it flips. `--drop CONCEPT` (or `--drop A=B`) leaves a pair out. Each concept is paired at most once. Unpaired concepts keep their Mirror's name as a prefix, such as `fomc:inflation`.
- **Both Mirrors say:** causal claims made in both corpora.
- **Tensions that come from combining:** places where the Mirrors disagree that neither has on its own: opposite signs, a cycle made of one Mirror's arrow and the other's chain, a claim one makes that the other denies, a concept one calls good and the other bad, a value one urges and the other avoids. The fit takes the better-supported side and keeps the other as "in tension".
- **Proposed trims, numbered:** fitness asks whether a reader of half the combined corpus would still come away with each item. Each sample keeps every passage with probability `--share` (default 0.5) and re-fits the model; a claim survives if the re-fitted model still has a path of the same sign, a denial if it still holds, a fact if any passage holding it was kept. What survives less often than `--keep-at` (default 0.6) is proposed for trimming: in practice, what is said in only one passage and not implied by anything else. A strong claim that the model rejects is kept, so the tension stays visible. Views the subject only reports or entertains are trimmed too.

Re-run `propose` as often as you like; it is free unless `--claude`, and replaces the previous proposal.

**approve** writes the combined Mirror exactly as proposed to `data/mirror/meditations+fomc/`, where `serve` shows it in the Mirror menu like any other. `--keep N` (repeatable, by number or by id) spares a trim. Trimmed items are archived in `trimmed.json` in that folder, not deleted, and an earlier approval is moved to `history/`. If either source Mirror has been rebuilt since the proposal, approve refuses: propose again.

Two Mirrors combined this way can be combined with a third. The web app has the same steps as routes (`POST /api/transmutate/propose`, `GET /api/transmutate?a=&b=`, `POST /api/transmutate/approve`) but no buttons yet.

## p1

The paused Prototype 1 app (patch review, Refactorer, Breaker, Chainer, Evolver). See [prototype1.md](prototype1.md), or `python -m androidmirror p1 --help`.

## When something goes wrong

| Message | What to do |
|---|---|
| `building needs Claude: put ANTHROPIC_API_KEY in your local .env` | Add the key to `.env` in the repository folder (see `.env.example`) |
| `no Mirror built at ...` | Run `build` for that corpus first |
| `this Mirror was built by Prototype 2` | Run `build` again for that corpus |
| Rate-limit or overloaded errors | Lower `--workers`, then run the same command again; finished work is cached |
