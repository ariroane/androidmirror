# Checking the causal model: verification, validation, and the standard to check against

As of 2026-10-23. A proposal; nothing here is built yet. It assumes the Causal tab grows the depth it is missing now, so that each Mirror's causal layer becomes a formal object: the fitted model read as an ADMG (arrows plus two-way links), converted to a MAG, and summarised as a PAG. The parked hidden-cause view (`causal/pag.py`, PR #30) already does the last two steps; this note asks how we would know the result can be trusted.

The sibling note on [the knowledge graph](knowledge-verification-validation.md) keeps the same two questions apart, and this note follows its structure:

- **Verification: was it built right?** Does every judgment say what its passage says, does the fit satisfy what it claims to satisfy, is the MAG really a MAG, is every mark in the PAG really shared by every graph in its class, and would the same corpus give the same object again?
- **Validation: was the right thing built?** Does the formal object mirror the subject's own causal model: would the subject accept what it says, including everything it implies but never states, and does it capture what makes this subject's model different from a generic one?

The formal object changes both questions. A MAG says much more than a list of arrows: every missing link is a claim that some set of ideas separates the two ends, every arrowhead is a claim that one idea does not lead to another, and every separation implies a pattern of "once you know this, that tells you nothing more". Most of those claims were never said by anyone. That is the source of the model's power (it predicts) and of its main risk (it states things the subject doesn't hold). Most of this note is about that gap.

## 0. Where the pipeline can go wrong

From the current code (`judgments.py`, `causal/__init__.py`, and `causal/pag.py` on the parked branch):

| Stage | What can go wrong | Caught now? |
|---|---|---|
| Reading judgments (`judgments.py`, one call per passage) | A causal judgment the passage doesn't make; wrong kind, sign, scope, register or attribution; a quote that isn't in the passage | No. The benchmark measures it on synthetic text only. |
| Building constraints | A "makes no difference" that means *the effects cancel out* read as *no path at all*; a label that doesn't resolve to a codebook idea dropped | No. |
| Fitting (`fit`, greedy, strongest first) | A worse model than the best one the constraints allow, because the order of ties decided it; a claim rejected that a different order would have kept | No. The greedy fit has no optimality check. |
| Two-way links | A clash read as a hidden shared cause when it is really a feedback loop, a change of mind, or one word used in two senses | No. The reading is fixed by design. |
| ADMG to MAG (`to_mag`) | An ancestral or maximality error; inseparable pairs the subject never connects (left out by default, so the result is not a valid MAG on those pairs, and says so) | Partly: `valid` flags it. |
| MAG to PAG (`pag`, FCI rules with background knowledge) | A mark that is not shared by every MAG in the class (unsound); a circle where every MAG agrees (incomplete; expected with background knowledge) | Tests on textbook cases only. |
| Silence | The closed-world rule (no connection said, so none) turned into confident separation claims by the formalism | No. See [silence as missing data](silence-as-missing-data.md). |

## 1. Verification schemes

Free and mechanical first, costly and human last. The first four could run on every build.

### CV1. Judgment grounding (free, every build)

The causal twin of the knowledge graph's quote check: every judgment's quote must be a verbatim span of its passage, after normalising whitespace and quote marks, with a fuzzy match to tell a copying slip from an invented quote. A constraint whose only support is an ungrounded quote is held out of the fit. Report the share per corpus and mark failing items in the Causal tab.

### CV2. Constraint accounting (free, every build)

Turn silent losses and fragile results into numbers stored in the build's `meta`:

- judgments read, judgments held (the subject's own, describing or urging), judgments fitted, and the share dropped because a label didn't resolve;
- constraints by status: holds, follows, in tension, rejected, open (dashed);
- the share of arrows, two-way links and arrowheads that rest on a single passage;
- for each two-way link, what it ran into (a cycle or a denial) and the weight on each side.

### CV3. Formal invariants (free, every build, and as property tests)

These are the definitions, checked mechanically, so a bug in the conversion can't hide behind a plausible-looking picture.

**On the fitted model:**
- it is acyclic in its arrows;
- every constraint marked *holds* is satisfied: a `reaches` has a directed path of the stated sign, a `no_path` has none, an `only_through` has every path from x to z pass through m;
- every *follows* is really implied, and every *rejected* is really unsatisfiable given what was taken before it.

**On the MAG:**
- ancestral: no directed cycle, and no two-way link between an idea and one of its ancestors;
- maximal: every pair the MAG leaves unlinked is m-separated by some set of ideas, checked by brute force on small graphs and by the inducing-path test on large ones;
- it has the same m-separations among the subject's ideas as the ADMG it came from (that is what the conversion promises).

**On the PAG:**
- soundness: the MAG itself is in the PAG's class, and every mark the PAG shows (arrowhead or tail) is shared by every MAG Markov equivalent to it. On small graphs (up to about 7 ideas) enumerate the class exhaustively and check; on larger ones sample MAGs from the class.
- completeness without background knowledge: every circle varies across the class. With background knowledge, completeness isn't guaranteed (the note in `pag.py` says so), so report the circles every sampled MAG agrees on instead of treating them as failures.

As property tests: generate random ADMGs, run `to_mag` and `pag`, and check all of the above against a slow, obviously correct reference implementation (brute-force m-separation over all subsets). Tools like `pcalg` (R) or `causal-learn` and `Tetrad` (Java) can serve as an independent second implementation for the PAG on the same MAGs, offline.

### CV4. Is the greedy fit any good? (free, offline)

The fit takes constraints strongest first and never revisits a choice. That is fast and readable, but it can miss the model that satisfies the most weight. On subgraphs of real Mirrors (the ideas within two steps of each two-way link, say), solve the same weighted constraints exactly with a MaxSAT or answer-set solver, in the style of Hyttinen, Eberhardt & Järvisalo (2014), and report:

- the weight the greedy fit satisfies over the weight the best model satisfies;
- the constraints the two disagree on, with their passages;
- **tie sensitivity:** refit with ties broken in shuffled orders, and list the arrows, links and PAG marks that change. A mark that depends on which of two equally weighted passages was read first is not a finding.

If the gap is small and ties rarely matter, the greedy fit stays and the exact solver stays offline. If not, the exact solver becomes the fit for subgraphs, which also gives the three-valued pairs (forced, free, impossible) proposed in the silence note for free.

### CV5. A gold set of causal judgments (the reference for reading)

Forty passages of Meditations across all twelve books, read by hand for causal judgments with the same schema the prompt uses: kind, sign, scope, register, attribution, quote. Weight the sample toward hard cases: "not this but that", rhetorical questions, quoted opponents, thought experiments, and "makes no difference" sentences, since those turn into the strongest constraints. Two readers where possible, with agreement per field.

Measure the reading pass against it per field, and report invented judgments separately from wrong fields. Then, as in the sibling note, calibrate a blind second reader on it (a separate call seeing only the passage and one judgment) and use it for the whole corpus if it agrees well with the people.

The same 40 passages can be shared with the knowledge graph's gold set, so one afternoon of hand reading anchors both layers.

### CV6. Stability, measured at the level of the claim

Build the same corpus three times, then once on a different model, and compare at four levels:

1. judgments (after resolving to codebook ideas);
2. the fitted model's arrows and two-way links (structural Hamming distance);
3. the MAG's m-separations: the share of separation claims among the subject's ideas that agree;
4. the PAG's marks.

The levels matter because the formalism forgives some differences and amplifies others. Two runs that draw a different arrow inside the same equivalence class agree on everything the PAG claims, and should count as stable. Two runs that differ by one missing link can disagree on many separations, and should count as unstable even though the arrow count barely moved. Any validation difference smaller than run-to-run variation at the same level means nothing.

### CV7. A planted benchmark with hidden causes (known truth, offline)

The existing benchmark plants a 12-concept model and scores arrows. Extend it so the truth is a MAG, not a DAG:

- plant hidden causes the synthetic subject believes in but never names (they say "fame and fortune rise and fall together" and "fame never brings fortune"), and score whether the right two-way links appear and nothing else;
- plant the other three sources of clashes too: a real feedback loop the subject states both halves of, a change of mind across books, and one word used in two senses. Score how often each is misread as a hidden cause;
- plant cancelling paths: x raises y through a, lowers it through b, and the subject says "x makes no difference to y". The right reading is zero total effect, not no path;
- score at the mark level: precision and recall of arrowheads and of tails in the PAG, the standard measures for PAG recovery, alongside the current arrow precision.

The truth is fully known here, so this is where to decide what the two-way link should mean, how denials should be read, and how many passages a mark needs, before spending on real corpora. Its weakness is that Claude writes and reads the passages; the gold set (CV5) keeps it honest.

### CV8. Fragility of each mark (free, every build)

For every arrow, link and PAG mark, refit with each of its supporting passages left out in turn, and record the smallest number of passages whose removal changes it. The Causal tab can show this beside the mark ("rests on 1 passage; drop Book 4.3 and this arrowhead becomes a circle"), and the Ask tab can say how sure it is. This is the causal twin of the knowledge graph's support count, and the cheapest guard against one misread sentence reorienting half the graph.

### CV9. Regression fixture

Freeze in `tests/` a handful of passages with their judgments, the expected fitted model, MAG and PAG, plus the textbook graphs from Zhang (2008) and Richardson & Spirtes (2002) with their known PAGs. A change to the fit, the conversion or the rules then shows up as a failing test.

## 2. Validation schemes

The knowledge graph's four baselines carry over unchanged: **plain Claude** (the subject's name only), **a generic subject** ("a Stoic", "a central bank"), **raw passages** (retrieved text instead of the model), and **a sibling Mirror** (Epictetus for Marcus). The causal model earns its keep only where it beats all four.

### CA0. Does the subject's model have this shape at all?

Before testing the content, test the formalism. A MAG is a claim about separation: "once you know z, learning x tells you nothing more about y". It is not obvious that a person's causal beliefs have that structure. Their model may be qualitative, partial, inconsistent across contexts, or full of cancelling effects (which break the faithfulness assumption that ties separations to missing links). Two checks:

- **The reading check.** Turn a sample of the MAG's separation claims into plain sentences ("In your view, once you know whether someone judges a thing bad, whether it actually harmed them tells you nothing more about whether they're disturbed") and ask the subject (CA6) whether that is a fair statement of what they hold, independent of whether it's true. If subjects regularly answer "that isn't the kind of thing I'd say", the separations are the formalism's, not the subject's, and the layer should stop at the ADMG.
- **The cancellation check.** Count "makes no difference" judgments that sit beside a stated path between the same two ideas. Each one is either a contradiction or a cancellation the formalism can't represent. If they're common, the denial needs a second reading (no total effect) and the MAG needs to be told.

### CA1. The prediction test, at the level of marks

The causal layer already holds out blocks of passages and classifies each held-out claim as restated, follows, new, sign differs, or contradicted. With a MAG, extend it in two directions:

- **Held-out denials against implied separations.** A held-out "x makes no difference to y" is a separation the subject states. Score how often the MAG fitted without it already implies the separation, and how often it implies the opposite (x leads to y). That tests the closed-world rule directly: separations implied by silence that the subject later states are silence used correctly.
- **Held-out claims against the PAG.** For each held-out claim, ask whether the PAG built from the rest already had a tail at x and an arrowhead at y (predicted), had circles there (open), or had the opposite marks (contradicted).

Compare against a shuffled model (same arrows with endpoints permuted) for chance levels, as in the sibling note.

### CA2. Implied claims the subject never states

The prediction test only scores implied claims that happen to come up later. Most never do. Sample them from three sources, labelled by where they come from, because they fail for different reasons:

1. **Implied by chains of stated claims:** x leads to z because x leads to y and y leads to z.
2. **Implied by stated denials:** separations that follow from "makes no difference" claims.
3. **Implied by silence:** separations that exist only because the subject never connected two ideas.

Translate each into a plain sentence and ask the judge (section 3) whether the subject would accept it: yes, no, not quite, or no view. Source 3 is expected to fail most; how much it fails is the number that decides between the options in [silence as missing data](silence-as-missing-data.md). If silence-implied separations are rejected or met with "no view" far more often than chain-implied ones, the Mirror should stop treating silence as separation and switch to the three-valued pairs.

### CA3. What the subject would say about an intervention

A causal model is for answering "what if I did x?". With a PAG, some of those effects are identified from the graph alone (Jaber, Zhang & Bareinboim 2019 give the conditions) and some aren't. For each identified one, the model makes a qualitative prediction: "doing x raises y", "doing x does nothing to y", or "can't tell from what you've said". Compare against what the subject says when asked the same question directly.

This deliberately does not bring back vignettes or Claude answering as the subject. Those were dropped as a source of *data* because a simulated subject leaks Claude's own model into the fit. As a *test*, the answers must come from a real subject: an interview Mirror's owner (CA6), or, for a written corpus, passages held out of the build that describe an intervention.

### CA4. The transplant test for the causal model

The decisive test from the skeptical evaluation, applied to the formal object. Edit the corpus so the subject holds a causal view they don't, rebuild, and check the formal object follows the edit:

- **Insert a judgment:** in 10 passages, have Marcus say that pleasure drives virtue. The arrow (or the MAG ancestor relation) must appear with the edited passages as its support, and the marks it forces downstream must change accordingly.
- **Delete a judgment:** remove every passage behind one arrow. It must disappear, along with every mark that only it forced, rather than be supplied from Claude's memory of Marcus. The sharpest test of outside knowledge leaking into reading.
- **Flip attribution:** turn "I say" into "Epicurus says". The constraint must move from fitted to shown-only.
- **Insert a clash:** add a "makes no difference" that contradicts a stated chain. A two-way link must appear exactly there, and nowhere else.
- **Plant and hide a cause:** add passages where the subject says z drives both x and y, then delete every mention of z. The fit without z should leave x and y linked by a two-way link or, if they were never connected, leave an inseparable pair the MAG reports. This is the only test where the MAG's hidden-cause reading is checked against a known hidden cause in real text.

Score the share of edits the object follows at each level (arrows, separations, marks) and in Ask answers. Plain Claude, which knows the real Marcus, will not follow them.

### CA5. Telling subjects apart

Build PAGs for Marcus and Epictetus over a shared set of ideas (the codebook's overlap, or both Mirrors resolved through Transmutate). Report the marks on which they differ, and check each difference against both corpora. Then, for held-out passages from either author, score which model explains each passage's causal judgments better (predicted minus contradicted, as in CA1) and see whether the right author wins. Do the same for the FOMC across chairs or decades, where the shifts in the committee's model are documented. A causal layer that can't tell Marcus from Epictetus is a model of Stoicism.

### CA6. Asking the subject

For interview Mirrors and any living subject, show them a sample of the model and ask, in plain sentences:

- **arrows:** "You seem to hold that x leads to y. Is that right?"
- **two-way links:** "You treat x and y as moving together, but deny either drives the other. Is something you haven't named behind both? A loop? A change of mind? Two meanings of one word?" The answer validates the link's reading, not just its presence, which tells us how often the hidden-cause reading is the right one.
- **arrowheads and tails in the PAG:** "Your words say y doesn't lead back to x. Is that right?"
- **circles:** "Your words don't settle whether x leads to y or something drives both. Which is it, or don't you know?" A circle isn't an error; the answer is a new constraint.
- **separations from silence:** "Once you know z, does x tell you anything more about y?"
- **interventions** (CA3).

Answers are yes, no, not quite (with a correction), or no view. Corrections go through the Question engine's approval path like every other change, and the same questions are asked again weeks later (test-retest) so the Mirror's accuracy isn't confused with the person changing their mind. The Question engine can choose which items to ask by value of information: the circles and silence-implied pairs whose answers would settle the most marks or traced effects.

### CA7. The subject's own actions

Where the subject acts, its actions test its causal model. The FOMC's minutes say what the committee thinks drives inflation and employment, and its decisions show which levers it pulls when. If the model has the committee holding that x leads to y, the committee should move x when it wants y to move. The neurology guidelines are similar: a recommendation is an intervention the guideline's authors believe works, and its strength rating is their confidence. Score decisions against the model's identified interventional predictions (CA3). Strong for institutions, unavailable for most authors.

### CA8. Causal questions in the Ask tab

A fixed set of "what drives what" and "what if" questions per corpus, answered from the model, by plain Claude, by the generic subject and from raw passages, judged blind (section 3). Half answerable from general knowledge of the subject, half needing this corpus: a chain the subject builds across books, a minor idea's place in their model, a place where they contradict themselves. The model is validated where it wins the second half without losing the first.

## 3. What the true standard should be

The sibling note's argument carries over whole: the standard can't be the truth about the world (a faithful Mirror holds the subject's false causal beliefs), Claude's reading (it can't see its own errors), expert commentary (commentators disagree and know more than the corpus), or internal consistency (real subjects contradict themselves, and the two-way links exist to say where).

### The proposal

**The standard is the subject's own endorsement on reflection, applied to what each mark means.** A mark in the formal object is correct if the subject, shown what it says in plain words and the passages behind it, would say "yes, that is how I think it works". The same proxy ladder applies, always labelled with which rung was used:

1. **The subject, asked directly** (CA6).
2. **The subject's own words, unseen** (CA1): held-out passages and later writings.
3. **The subject's own actions** (CA7).
4. **A careful human reading of the text** (CV5): the standard for verification, which asks what the text says.
5. **Expert commentary**, as a last resort, with disagreement recorded as uncertainty.

What's new for the causal layer is that the formal object makes claims of different strengths, and each needs its own reading of "endorse":

| Mark | What it claims | Endorsed when the subject says |
|---|---|---|
| tail at x, arrowhead at y | x leads to y, perhaps through other ideas | "yes, x brings about y" |
| arrowhead at both ends | neither leads to the other; something drives both | "they go together, but neither causes the other" |
| arrowhead at y only | y doesn't lead back to x | "no, y doesn't bring about x" |
| circle | the subject's words don't settle it | at verification: the text indeed doesn't settle it. At validation: not right or wrong, but a gap to ask about |
| no link | some set of ideas separates the two | "once you know that, this tells you nothing more", or "they have nothing to do with each other" |

Three rules go with this:

- **Silence is a third answer, and it is the weakest one.** "No link" is the strongest claim a MAG makes and the one least often backed by anything the subject said. Every test must keep "no view" apart from yes and no, must label separations by their source (stated denial, chain, silence), and must never score an unasked separation as endorsed.
- **Endorse the class, not the picture.** Two drawings in the same equivalence class say the same thing, and a subject can't be asked to choose between them because nothing they could say would distinguish them. Validation scores the PAG's marks and the MAG's separations, never the particular arrows of one drawing.
- **The standard is per corpus and per question**, as in the sibling note: for an interview corpus the person; for the FOMC held-out statements and later decisions; for the guidelines the recommendations and their strength; for Marcus and Epictetus held-out passages, the transplant test and the sibling comparison.

The transplant test (CA4) sits outside the ladder, because the edit defines the truth. If the formal object doesn't follow an edit to its own corpus, it isn't a mirror of that corpus, whatever else it scores.

## 4. Where to start

In order of cost:

1. **CV3** (formal invariants as property tests): free, and the precondition for trusting anything the PAG shows. Worth doing as part of finishing PR #30.
2. **CV1, CV2, CV8** on every build: free, and they say which marks rest on one passage.
3. **CV7** (planted hidden causes, clashes and cancellations): free once written. It decides what a two-way link should mean and how to read "makes no difference", the two open design questions that everything downstream depends on.
4. **CV4** (greedy against exact on subgraphs): free, and doubles as the first step of the three-valued pairs in the silence note.
5. **CA4** (transplant: insert, delete, clash, hidden cause) on 30 to 60 passages: a few dollars of building; the clearest single answer to whether the object comes from the corpus.
6. **CA0 and CA6** on an interview Mirror: the subject spends half an hour on 40 plain sentences. This is the only test against the standard itself, and the one that says whether a MAG is the right formalism for a person's causal beliefs at all.
7. **CA2** (implied claims by source): its silence-versus-chain comparison settles the closed-world question with evidence rather than argument.
8. **CV5, CV6, CA1, CA5, CA7, CA8** once full builds of Marcus and Epictetus (and the FOMC) exist.

The first four say whether the formal object is built correctly; five and six say whether it is worth building.

## References

- Richardson & Spirtes (2002), *Ancestral graph Markov models*.
- Zhang (2008), *On the completeness of orientation rules for causal discovery in the presence of latent confounders and selection bias*.
- Hyttinen, Eberhardt & Järvisalo (2014), *Constraint-based causal discovery: conflict resolution with answer set programming*.
- Jaber, Zhang & Bareinboim (2019), *Causal identification under Markov equivalence: completeness results*.
- Ali, Richardson & Spirtes (2009), *Markov equivalence for ancestral graphs*.
- Mohan, Pearl & Tian (2013), *Graphical models for inference with missing data*.
