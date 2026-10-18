# Causal validity review of the Android Mirror

Reviewer role: causal inference and causal discovery. Scope: the causal ontology (DAG) on `main` at 7899a00. Findings and recommendations only; nothing has been changed or pushed.

## The framing that drives everything below

The DAG is not a causal model of the world. It is a model of **what one author believes causes what**, measured from text. So "scientific validity" here splits into two questions:

1. **Measurement validity**: does each edge faithfully record a causal belief the author actually holds, with the right direction, sign and size?
2. **Inferential validity**: when the app reasons over those edges (paths, interventions, Evolver fitness), do the operations mean what the UI says they mean?

Classical causal discovery (PC, GES, FCI) needs observational data with variation across units. A single author's text has none, so those algorithms do not apply directly. What does apply is the *logic* of discovery: identifiability, Markov equivalence, testable implications, and reliability of measurement. The recommendations use those.

## Findings, most important first

### 1. Confidence is multiplied into effect size (analysis.py `propagate`)

`effect(child) = Σ effect(parent) × sign × strength × confidence`. This mixes two different things: how big the author thinks an effect is (strength) and how sure we are that the author holds that belief (confidence). A strong effect we are unsure about shows up as a weak effect we are sure about, and the two cannot be told apart in the Change tab. Along longer paths the product collapses fast: Reason to Dignity in the seed is 0.8·0.7 · 0.6·0.5 · 0.7·0.6 ≈ 0.07; from the strengths alone it would be 0.34, so most of the shrinkage is uncertainty, not magnitude.

**Recommend:** propagate `sign × strength` only, and carry uncertainty separately. The simplest sound version is Monte Carlo: in each draw keep each edge with probability `confidence` and draw its strength from a Beta centred on `strength`, then report the median effect, an 80% interval, and P(effect has this sign). The Change tab shows "Raises Dignity, moderately; the model supports this in 64% of draws." Probes in the Evolver should then pass on P(sign) ≥ a threshold rather than on any nonzero effect.

### 2. A second quote for the same edge silently overwrites the first (ontology.py `add_causal_edge`)

When an edge already exists, the new op replaces sign, strength and confidence and only appends provenance. So the last passage read wins, and a later quote with the opposite sign flips the edge without a warning (the Steward's `check_ops` passes because the op is valid).

**Recommend:** store one evidence record per quote on the edge (`evidence: [{quote, locator, sign, strength, confidence}]`) and derive the edge's values from them: strength as a confidence-weighted mean, confidence by noisy-OR across independent passages, and a `support` count. An incoming quote with the opposite sign should produce a Steward warning ("Book V, 3 contradicts the sign from Book II, 1") instead of a flip. This is the single biggest gain in evidential rigor, because repeated themes in the Meditations then strengthen an edge instead of resetting it.

### 3. "Edge" has no defined meaning: direct versus total effect

In a DAG an edge means a *direct* effect relative to the other nodes in the graph. A quote usually states a *total* effect ("looking to reason makes one dignified") that may already run through mediators the graph also contains. `propagate` then counts it twice: once on the direct edge, once along the mediated path. The Refactorer's `reduce_transitive` only removes the direct edge when it has no quote, so quoted double-counts stay.

**Recommend:** have the extractor tag each causal claim `scope: direct | overall`. For `overall` claims, the app treats the claim as a *constraint* (the total effect from A to B should have this sign) that the Refactorer checks against paths, rather than as an extra edge that adds to them. This also turns the existing `contradictions` detector from a heuristic into a precise check.

### 4. Causal versus constitutive claims are not separated

Seed example (Book II, 1): "To act against one another then is contrary to nature; and it is acting against one another to be vexed." Marcus is saying vexation *is* a form of acting against one another, which *is* contrary to nature. That is a definitional (constitutive) claim, but it is encoded as two causal edges, Vexation → Cooperation (−) → Living according to nature (+). Stoic writing is full of these identities ("the good is virtue"), and treating them as causes inflates the DAG and makes interventions on them meaningless.

**Recommend:** add `relation_kind: causal | constitutive | conditional | normative` to the extraction schema and the prompt, with one worked example of each. Constitutive and normative claims go to the knowledge graph (`is_a_form_of`, `ought_to`), never to the DAG. "Should" statements in particular are advice, not causal beliefs, and are a large share of this book.

### 5. Nodes are not always well-defined variables

A causal node has to be something that can be more or less (a quantity a do() can set). People ("Grandfather Verus"), places and events are not, yet the schema allows them as causal endpoints. Opposites also coexist as separate nodes ("anger" and "freedom from anger"), so the same belief can be stored with either sign and the Chainer's sign-conflict check misses it.

**Recommend:** restrict causal endpoints to the types that vary in degree (virtue, vice, emotion, belief, practice, value, concept) and send person/place/event links to the KG. Give the Refactorer a detector for antonym pairs that proposes merging them with a sign flip on every edge of the absorbed node, as an ordinary approval-gated patch.

### 6. Feedback loops are refused rather than represented

`add_causal_edge` raises an error when an edge would close a cycle. But reinforcing loops are central to Stoic practice (practice builds virtue, virtue sustains practice; anger feeds vexation feeds anger). Refusing them biases the model toward whichever direction happened to be read first.

**Recommend:** keep the DAG for interventions, but when a patch would close a loop, have the Steward show it as a "feedback claim" and offer two approval options: record it as a lagged edge (`X → Y` at the next step, which keeps the graph acyclic within a step and is the standard way to handle cycles in structural models) or keep it in the KG as `reinforces`. Order-dependence also drops out because neither direction is lost.

### 7. Strength and confidence are uncalibrated numbers

The extractor invents both numbers with no rubric, and nothing measures whether they are stable or meaningful.

**Recommend, in three steps:**
- **Anchor them to text.** Strength from the author's quantifier (always/necessarily ≈ 0.9, generally ≈ 0.6, can/may ≈ 0.3); confidence from how the causal link is expressed (explicit connective like "because", "so that", "from this comes" is high; juxtaposition or Claude's reading between the lines is low). Store the marker words so the user can see why.
- **Measure reliability.** Re-extract a fixed sample of passages several times and with a paraphrased prompt; report agreement on presence, direction and sign (Krippendorff's alpha) and on strength (ICC). Edges that don't reproduce get low confidence automatically.
- **Calibrate against the user.** Every Steward approve or reject is a label. Plot approval rate against stated confidence in bins; if 0.8-confidence edges are approved 50% of the time, rescale. This needs no new role: it is a read-only report over the existing patch log.

### 8. Direction is often underdetermined, and Markov equivalence is unused (backlog item)

This confirms the parked backlog idea and adds what it should rest on.
- Have the extractor record `direction_basis: explicit | temporal | inferred`.
- Compute the DAG's CPDAG (the Markov equivalence class): edges in v-structures are fixed, the rest are reversible without changing which independences the graph implies.
- An edge that is both reversible in the CPDAG **and** has `inferred` direction gets a "direction weakly grounded" flag in the UI.
- For the Evolver: flips within the equivalence class leave the observational content unchanged, so they shouldn't be rewarded as new structure. But they **do** change do() results, so they shouldn't be treated as free either. The right move is to skip them as search steps and instead show the user "this intervention's answer depends on the direction of these flagged edges."

### 9. The model's testable implications are never checked

A DAG commits to conditional independences (d-separations): for example, it may imply that once Steadiness of purpose is fixed, Reason makes no further difference to Attention to the present task. These are falsifiable against the text, which is the closest thing this app has to causal discovery's model-checking step.

**Recommend:** a read-only check, run on user request like the other engines, that lists the graph's implied independences between nearby nodes and asks Claude whether the corpus contradicts each one ("Does Marcus ever say Reason affects Attention directly, not through steadiness?"). A contradicted independence becomes an ordinary proposed patch (add the missing edge) with the quote. This extends the Refactorer's detectors; it is not a new role.

### 10. Paths stitch together beliefs from different passages

Reason → Steadiness comes from Book I, 8; Steadiness → Attention → Dignity from Book II, 5. The chained conclusion "reason leads to dignity" is the model's inference, not something Marcus wrote, and the `reason_to_dignity` probe rewards it as if it were his. Chaining is legitimate, but the UI should say when an effect is inferred across passages.

**Recommend:** in the Change tab and the Oracle, label each reported effect as *stated* (one quote covers the whole path) or *inferred* (stitched from N passages), with the passages listed.

### 11. The Evolver's probes risk overfitting

Six hand-written probes carry 30% of fitness, and the search optimises against them directly, so passing them stops being evidence that the design is good (Goodhart).

**Recommend:** split probes by book into a set the Evolver optimises and a held-out set it never sees, and report both scores on each proposed design. Probes should be drawn from passages the extractor has not ingested yet where possible, which also makes them a genuine out-of-sample test of the extraction.

### 12. Add counterfactuals to the Change tab (requested by Android K)

**Where the app is now.** Causal reasoning has three levels: association ("these go together"), intervention ("what if I raise X?") and counterfactual ("given what actually happened, what would have happened if X had been different?"). The Change tab answers only the middle question. It also answers it in shifts ("+0.34 Dignity"), with no actual state to start from. A counterfactual needs a starting state: a specific moment in the author's life.

**One honest caveat first.** In the current model (linear, additive, unbounded), a counterfactual shift is always identical to the interventional shift, whatever happened in the episode. So a counterfactual engine is only worth building if the model can respond differently depending on the starting state. That is three changes, not one:

1. **Episodes as the starting state.** A lot of the Meditations narrates or imagines specific situations, such as the morning with the busy-body or a court scene with an insult. The Cartographer already reads those passages. It should also extract an *episode*: an `event` node in the KG, plus observed levels on 0–1 for the causal nodes the passage mentions ("he was vexed: high; he cooperated: low"), each level with its quote. Episodes are ordinary patches the user approves; the engine itself never writes to the ontology.
2. **Bounded, saturating equations.** Give each node a level in 0–1 set by a logistic function of its weighted parents plus a node-specific noise term. With this, an effect depends on the starting state (raising Reason helps more when Reason starts low). It also gives "would it have crossed the line" questions a real answer.
3. **The standard three-step procedure.** *Abduction:* infer each node's noise term from the episode's observed levels, which captures what was particular about that moment. *Action:* apply do(X = x′). *Prediction:* recompute the descendants with the same noise terms. Nodes the episode didn't observe get their noise from the prior, so the answer becomes a distribution, not a single number.

**What the user sees.** The Change tab gets a "Starting from" picker. It defaults to "nothing in particular", which is today's intervention behavior. Picking an episode turns the question counterfactual: "In the Book II morning, had Marcus done his morning preparation, would he have cooperated?" The answer shows:
- the factual and counterfactual level for each affected node, side by side, with an 80% interval;
- **probability of necessity**: given that it happened, would the outcome have failed without the cause? This is the "but for" question;
- **probability of sufficiency**: would adding the cause have produced the outcome where it was absent?
- which edges carry the difference, and whether each one is *stated* or *inferred* (finding #10).

The Storyteller already retells interventions. For an episode it can retell the alternative version of that passage, staying faithful to the computed numbers as it does now.

**What makes it scientifically defensible.**
- **Model-dependence is labeled.** Unlike interventions, counterfactuals depend on the equations and noise model chosen, not only on the graph. Every counterfactual answer carries a fixed note ("depends on the model's assumed functional form") and a sensitivity check: rerun with linear and with threshold equations and report whether the answer changes direction. PN and PS in particular can only be bounded, not pinned down, without a monotonicity assumption; show the bounds when that assumption is off.
- **Edge uncertainty carries through.** Run the procedure inside the Monte Carlo from finding #1, so a weakly supported edge widens the interval instead of being silently trusted.
- **The default is interventionist.** "Had he been calmer" sets calmness directly and leaves its causes as they were (Pearl's convention). A second, optional mode, *backtracking*, answers "what would have had to be different upstream for him to be calmer?" by finding the smallest change to the noise terms. That second question is often the one a reader of the Meditations cares about, but it must be labeled as a different question.
- **It can be tested against the text.** Marcus sometimes states counterfactuals himself ("if thou hadst..."). Those make good held-out probes for the engine, in the split from finding #11.

**Where it lives.** A read-only `analysis` extension next to `propagate`/`intervene`, exposed through the existing intervene endpoint and the Change tab. Episode extraction goes in the Cartographer's schema. No new agent role, no automatic writes; episodes, like every other change, enter only through approved patches.

**Order.** It depends on #1 (uncertainty), #3 (direct vs overall effects, otherwise the abduction step double counts) and #4 (definitions can't be intervened on). Build it after those.

## What I'd do first

If you want a short list: **#2 (evidence aggregation)**, **#1 (separate uncertainty from effect)** and **#4 (causal vs constitutive)** fix the places where the current output is actively misleading. #7 and #8 make the numbers trustworthy. #9 to #11 are what make the claims defensible to an outside reader.

All of these fit inside the existing roles: #2, #5 and #6 live in the ontology core and Steward, #3, #4, #7 in the Cartographer/Causal Modeler prompt and schema, #5, #8, #9 in the Refactorer's detectors, #1, #10 and #12 in analysis and the Change tab, #11 in the Evolver's fitness. Every resulting change would still be a user-approved patch.

## Overlap with the descriptive-knowledge review

#4 (constitutive claims moving to the KG) and #5 (person/place nodes leaving the DAG) change what the knowledge graph receives, so they should be agreed with the descriptive inference review before either is built.
