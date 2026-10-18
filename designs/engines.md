# The engines

Each engine reads the built Mirror. Any change it proposes is previewed and applied only when a person approves it. Transmutate's commands are in the README's command reference.

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
