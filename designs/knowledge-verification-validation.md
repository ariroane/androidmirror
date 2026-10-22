# Checking the knowledge graph: verification, validation, and the standard to check against

As of 2026-10-22. A proposal; nothing here is built yet.

The knowledge graph is the Mirror's descriptive layer: ideas from the codebook, facts in thirteen relations read from every passage, and what follows from them by the rules in [the descriptive ontology](descriptive-ontology.md). The Ask tab answers from it, the engines rewrite it, and the OODA model is built on it, so whether it can be trusted matters for everything downstream.

Two questions, kept apart:

- **Verification: was it built right?** Does every fact say what its passage says, does every derived fact follow by a sound rule, can every item be traced back to its words, and would the same corpus give the same graph again?
- **Validation: was the right thing built?** Does the graph mirror the subject: would the subject recognise it, does it predict what the subject says next, and does it capture what makes this subject different from a generic one?

A graph can pass verification and fail validation (every fact quoted correctly, but the selection misses what the subject cares about), or the other way round (it captures the subject, but by leaning on what Claude already knows about Stoics). Both need their own tests, and each test needs a standard to be measured against. Section 3 argues what that standard should be; sections 1 and 2 are the schemes.

This builds on the [descriptive knowledge review](https://claude.ai/code/artifact/c75478d6-4c3b-4d12-a3d3-97cb16a36d12) (claim-level evidence, qualifiers, meaning-based identity resolution, a gold set, three-run stability) and the [skeptical evaluation](../reviews/skeptical-evaluation.md) of the causal layer (validation must beat plain Claude; the corpus-transplant test is the decisive one). The aim is to turn those recommendations into a test plan for the knowledge graph, not to repeat them.

## 0. Where the pipeline can go wrong today

Each stage is a place where an error can enter without being caught. From the current code:

| Stage | What can go wrong | Caught now? |
|---|---|---|
| Open coding (`coding.py`, one call per passage) | A fact the passage doesn't state; a wrong relation, polarity, modality or attribution; a quote that isn't in the passage | No. The prompt asks for verbatim quotes, but nothing checks them. |
| Codebook (`codebook.py`) | Two senses merged into one idea; one idea split across batches of 60 labels and never merged; an opposite folded in with the wrong polarity | No. Merges are only visible as aliases. |
| Fact assembly (`build.py: _facts`) | A statement whose label doesn't resolve to an idea is silently dropped; "self" attribution decided by string match | No count of dropped statements is kept. |
| Inference (`kg.py`) | A rule that is valid for the relation's gloss but not for this subject's use of the word (for example, `opposes` covers both "is the contrary of" and "works against", and only the first is inherited soundly) | Partly. Five checks flag contradictions, but not unsound derivations that contradict nothing. |
| Held vs reported | A quoted opponent's view marked as the subject's own, or the reverse | No. |

None of these are known to be frequent. The point of verification is to find out how frequent they are.

## 1. Verification schemes

Ordered from free and mechanical to costly and human. The first three cost nothing and could run on every build.

### V1. Quote grounding (free, every build)

Every quote on a mention or a fact must be a verbatim span of its passage, after normalising whitespace and quote marks. Report the share that are, per corpus, and mark failing items in the Knowledge tab. A fuzzy match (say, 90% of characters in order) separates a small copying slip from an invented quote. Facts with no grounded quote should not enter inference until a human or a second pass finds one.

This is the cheapest test of hallucination there is, and it is currently missing.

### V2. Pipeline accounting (free, every build)

Count what is lost or changed at each stage, so a silent loss becomes a number:

- statements dropped because a label didn't resolve to an idea (`_facts` skips them today);
- labels left to the fallback grouping because their consolidation batch failed;
- near-duplicate ideas across consolidation batches (same normal form after stemming, or a high label similarity with the same kind), which are candidate missed merges;
- aliases whose glosses disagree, which are candidate wrong merges;
- the share of facts that rest on one passage only.

Store these in the build's `meta` beside the prompt version and model, which are already recorded.

### V3. Logical audit (free, every build)

The five existing checks report contradictions. Add three measures that say whether inference is behaving:

- **Derivation depth and fan-out.** How many derived facts each stated fact produces, and the longest proof. A single `same_as` or `is_a` fact that produces hundreds of derived facts is the most likely place for a wrong merge or a misread to spread.
- **Support of derived facts.** For each derived fact, the number of distinct passages its proof rests on (`stated_support` already computes this). A derived fact resting on one passage is weaker than one resting on ten, and the Ask tab should be able to say so.
- **Rule usage.** How many derived facts each rule produces. This is the denominator for V5's per-rule precision.

### V4. A gold set read by people (the reference for everything else)

A stratified sample of passages from each corpus, coded by hand with the same schema the coding prompt uses: ideas with level, statements with relation, negation, modality, attribution and quote. For Meditations, 40 passages across all twelve books, including ones heavy in quoted opponents, rhetorical questions and dilemmas, since those test the held/reported split. Two annotators where possible, with agreement reported per field, so the gold set's own uncertainty is known.

Measure the coding pass against it per field: precision and recall of facts (a match is same idea pair and relation, after codebook resolution), and accuracy of negation, modality and attribution on matched facts. Report "the relation was right but the attribution was wrong" separately from "the fact was invented", because they have different fixes.

Hand coding 40 passages is a few hours of work, which is the price of having any ground truth at all.

### V5. A blind second reader, calibrated on the gold set

Hand coding doesn't scale to a whole corpus, so use it to calibrate an automatic checker. For each fact, a separate Claude call sees only the passage and the fact (not the coder's rationale or the other facts) and answers: does this passage state this, with this polarity, modality and attribution? Yes, partly (which field is off), or no. On the gold set, measure how often the checker agrees with the people. If it agrees well, run it on the whole corpus as the fact-precision estimate; if not, the gold set alone is the estimate.

The same checker, given a derived fact with its proof, can rate each derivation step. Combined with V3's rule counts this gives **per-rule precision**: how often each rule, applied to this subject's facts, produces something the passage-level reader would accept. A rule whose precision is low for a corpus can be switched off for that corpus, or restricted (for example, inherit `opposes` only when the stated `opposes` fact reads as a contrary, not as "works against").

The checker is a different call from the coder but the same model, so it shares the coder's blind spots. Two mitigations: run it on a different model where one is available, and keep the gold set as the final word.

### V6. Stability

Build the same corpus three times (same model, same prompts) and compare the graphs after resolving to codebook ideas: overlap of facts, of held facts, and of the codebook's merges. Then once on a different model. A fact that appears in one run out of three is a reading, not a finding. Stability doesn't prove correctness, but instability proves that some of what the graph shows is noise, and it puts a floor under every other measure: a validation difference smaller than run-to-run variation means nothing.

### V7. A planted benchmark (known truth, offline)

The causal layer has an offline benchmark with a planted model; the knowledge graph should too. Write a small known graph (30 ideas, 60 facts, a few denials, a taxonomy three levels deep), then have Claude write passages that each express a few of its facts, in varied wording, mixed with distractors: a quoted opponent who holds the opposite, a rhetorical question, a hypothesis that is entertained and dropped, a synonym for an idea used in some passages only. Run the whole pipeline and score against the planted graph: facts recovered, facts invented, held/reported mistakes, missed and wrong merges, and derived facts that the planted graph does not entail.

This is the only scheme where the truth is fully known, so it is the right place to tune prompts, batch sizes and rules before spending on real corpora. Its weakness is that Claude both writes and reads the passages, so it is easier than a real text; the gold set (V4) keeps it honest.

### V8. Regression fixture

Freeze a handful of passages with their codings and expected graph in `tests/`, so a change to `kg.py`, `codebook.py` or the build that changes the graph shows up as a failing test, not as a quietly different Mirror.

## 2. Validation schemes

Verification asks whether the graph says what the corpus says. Validation asks whether what it says is the subject. Every validation test here is a comparison against something: a test with no baseline can't tell the Mirror apart from Claude's general knowledge of the subject.

### Baselines every validation test is run against

- **Plain Claude:** the same question, with only the subject's name ("Marcus Aurelius") and no Mirror.
- **Generic subject:** the same question about a generic member of the subject's type ("a Stoic", "a central bank", "a clinical guideline").
- **Raw passages:** the same question with retrieved passages instead of the graph, which tests whether the graph's structure adds anything over the text.
- **A sibling Mirror:** the closest other Mirror (Epictetus for Marcus; another period of the FOMC for one period).

The Mirror earns its keep only where it beats all four. Beating plain Claude on Marcus is hard by design: Claude has read the Meditations. That is why the tests below are built around items Claude can't answer from memory.

### A1. The prediction test for knowledge

The causal model already has one; the knowledge graph needs the same. Build the graph with one block of consecutive passages held out (blocks, because neighbouring passages repeat each other), then classify each held fact in the held-out block:

- **restated:** the rest of the corpus states it too;
- **follows:** the rest doesn't state it, but it follows by the rules;
- **new:** the rest is silent;
- **contradicted:** the rest states or implies the denial, or puts the idea in a disjoint evaluation.

Two numbers matter. A low contradicted rate says the graph is one consistent worldview, not a pile. The follows rate says inference is predicting things the subject really goes on to say, which is the direct test of "faithful inference": a derived fact that later appears as a stated fact is a derivation the subject endorsed. Compare follows against a shuffled graph (same facts with relations permuted) to see how much of it chance gives.

### A2. Derived facts the subject never states

The prediction test only scores derived facts that happen to appear later. Most never do. Sample derived facts that are stated nowhere in the corpus and ask a judge (section 3 says who) whether the subject would accept each one when shown its proof: yes, no, or "the steps are fine but the conclusion isn't what they mean". The third answer locates a rule that is formally valid and substantively wrong for this subject, which feeds back into V5's per-rule switches.

### A3. The transplant test for knowledge

The skeptical review's decisive test, applied here. Edit the corpus so the subject holds something they do not: in 10 passages of Meditations, change "pleasure is not a good" and its paraphrases to say pleasure is a good, or insert a new doctrine (say, that grief is a virtue) in the subject's own voice. Rebuild. A Mirror that reads its corpus must now hold the edited fact, derive its consequences, flag the clashes with untouched passages, and answer Ask questions accordingly. Plain Claude, which knows the real Marcus, will not.

Three variants, each testing something different:

- **Insert:** a fact the subject never held appears in the graph with the edited passages as its support.
- **Delete:** remove every passage that supports a fact; the fact must disappear from the graph rather than be supplied from Claude's memory of the author. This is the sharpest test for leakage of outside knowledge into coding.
- **Flip attribution:** change "I say" to "Epicurus says" in passages that state a fact; it must move from held to reported.

Score: the share of edits the graph follows, and the share of Ask answers that follow. Anything below near-total on insert and delete means the graph is partly built from outside knowledge.

### A4. Telling subjects apart

A mirror of Marcus has to be a mirror of Marcus and not of Stoicism. Build Marcus and Epictetus, then for each held-out passage (not used in either build), score how well each graph fits it (share of its coded facts each graph restates or derives, minus those it contradicts) and see whether the right author wins. Do the same within the FOMC across chairs or decades, where the differences are real and dated, and within the neurology guidelines across revisions. Also report the facts that distinguish the two graphs (held by one, denied or absent in the other) and check them against the passages: those are the facts that make the subject this subject, and the ones the Ask tab should most often rest on.

### A5. Asking the subject

For a living subject, the subject is the judge. The interview corpora are built from a person's own answers, so the person can be shown a sample of their Mirror's facts and asked about each:

- **stated facts:** "Is this what you meant?"
- **derived facts, with proofs:** "This follows from things you said. Do you hold it?"
- **distinguishing facts:** "Is this what sets you apart?"
- **absences:** "The Mirror has nothing on X. Is that because you have no view?"

Each answer is yes, no, or not quite (with a correction). Corrections go through the Question engine's existing path, so they update the graph with the subject's approval, like every other change. Weeks later, ask again (test-retest), so the Mirror's measured accuracy isn't confused with the person changing their mind.

This is the best validation available anywhere in the project, and the cheapest to start: the people building the Mirror can be its first subjects.

### A6. Questions answered from the graph

The Ask tab is where the graph is used, so test it there. A fixed set of questions per corpus, answered four ways (Mirror, plain Claude, generic subject, raw passages), and judged blind against the standard in section 3. Half the questions should be ones a well-read person could answer from general knowledge, and half ones that need this corpus specifically: what the subject says about a minor figure, how two of their ideas relate in their own usage, where they contradict themselves. The Mirror is validated where it wins on the second half without losing the first.

## 3. What the true standard should be

Every test above compares the graph with something. Choosing that something is the real decision, because a test is only as good as the standard behind it. Four candidates each fail as the single standard:

- **The truth about the world.** The Mirror models a subjective system. If Marcus believes something false, a faithful Mirror holds it. Scoring against the world would reward replacing the subject's views with correct ones, which is the opposite of mirroring.
- **Claude's reading of the corpus.** Claude is the coder. Using it as the standard measures agreement with itself and can't detect its own systematic errors, such as leaning on what it already knows about the author. Claude is fine as a calibrated instrument (V5), never as the standard.
- **Experts on the subject.** Commentators disagree with each other, read the subject through their own frameworks, and know things the corpus doesn't contain. Scholars of Marcus Aurelius are a useful second opinion where the subject can't be asked, not a gold standard.
- **Internal consistency.** Real subjects contradict themselves. A Mirror tuned to be consistent would smooth away exactly the tensions that make a subject who they are. Consistency checks are diagnostics, not targets: the Mirror should be as consistent as the subject, no more and no less.

### The proposal

**The standard is the subject's own endorsement on reflection:** a fact in the Mirror is correct if the subject, shown the fact and the passages it rests on (and for a derived fact, its proof), would say "yes, that is what I hold". That is what it means for the Mirror to mirror. It covers stated facts, derived facts, absences, and the held/reported split in one definition, and it is indifferent to whether the subject is right about the world.

The subject can rarely be asked, so the standard is reached through proxies, used in this order and always labelled with which one was used:

1. **The subject, asked directly.** Interview corpora and any living subject (A5). The standard itself.
2. **The subject's own words, unseen.** Held-out passages (A1), and later writings the corpus doesn't include. These are the subject endorsing a fact without being asked. They never say "no" to a derived fact, only "yes" by restating it, so they measure recall of the subject's views better than precision.
3. **The subject's own actions.** Where the subject acts on its views, the actions are evidence: the FOMC's later decisions and minutes, a guideline's recommendation boxes and their stated strength. Strongest for institutions, unavailable for most authors.
4. **A careful human reading of the text.** The gold set (V4): people reading each passage for what it says, with agreement measured. This is the standard for verification, which asks what the text says, not what the subject holds.
5. **Expert commentary.** A last resort for validation where nothing above applies, with disagreement among experts recorded as uncertainty, not resolved.

Two rules go with this:

- **Silence is a third answer.** The graph is open-world: "not found" is not "false". Every standard must allow "the subject has no view here" alongside yes and no, and a test must never score a missing fact as a wrong denial or an absent denial as an endorsement.
- **The standard is per corpus and per question.** Verification is always measured against the text (4). Validation is measured against the highest proxy available: for an interview corpus, the person; for the FOMC, held-out statements and later decisions; for the neurology guidelines, the recommendation boxes; for Marcus and Epictetus, held-out passages, the transplant test and the sibling comparison, with commentary only as a check.

The transplant test (A3) sits outside this ladder, because it needs no external standard at all: the edit defines the truth. That is why it is the decisive test. If the graph doesn't follow an edit to its own corpus, it isn't a mirror of that corpus, whatever else it scores.

## 4. Where to start

In order of cost, with what each one unlocks:

1. **V1, V2, V3** on every build: free, and they turn invisible losses into numbers.
2. **V7** (planted benchmark): free once written; the place to compare prompts and rule settings before spending on real corpora.
3. **A3** (transplant, insert and delete) on 30 to 60 passages of Meditations: a few dollars of building; the clearest single answer to whether the graph comes from the corpus or from Claude's memory.
4. **V4 + V5** (40-passage gold set and a calibrated checker): a few hours of hand coding; gives every later number an anchor.
5. **A1** (prediction test): free once a full build exists, as in the causal layer.
6. **A5** on an interview Mirror: the subject spends half an hour judging 50 facts; the only test against the true standard itself.
7. **A4 and A6** once two full builds exist (Marcus and Epictetus).

The first three are enough to say whether the knowledge graph is trustworthy enough to keep building on; the rest say how good it is.
