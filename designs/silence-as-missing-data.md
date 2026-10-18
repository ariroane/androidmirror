# Silence as missing data

Brainstorm for the hidden-cause view (PAG) of the causal model. Ideas only: nothing here is built, and nothing in the PAG pull request depends on it.

## 1. What the current rule assumes

The Mirror's rule is: if the subject never connects two ideas, they are not connected. In the fitted model that is harmless, because the model only ever *adds* what the subject says. In a causal graph, though, a missing link is itself a claim: "some set of ideas screens these two off from each other". So the rule is a **closed-world assumption**: silence is read as a statement of separation.

Almost every graph method (PC, FCI, PAGs) works this way, because with data you can test separation. We have no data to test it with, only what the subject says. Silence there is better described as **missing**: the subject did not say, which is different from saying "no".

## 2. Where the "pairs nothing can pull apart" come from

These are not pairs where we lack evidence. They follow from the two-way links themselves, once each link is read as a hidden common cause.

The usual shape is this. Fate drives Anger. Anger and Calm are joined by a two-way link, and Anger also leads to Calm by a chain. Holding Anger fixed blocks Fate's chain to Calm, but it opens the route Fate → Anger ← hidden cause → Calm. Leaving Anger free leaves the chain open. No choice of ideas to hold fixed separates Fate from Calm.

In your three built Mirrors (staged from your machine, read-only):

| Mirror | Two-way links | Pairs nothing can pull apart | ...already joined by a chain | ...not otherwise connected |
|---|---|---|---|---|
| Meditations | 79 | 122 | 86 | 36 |
| FOMC | 12 | 84 | 82 | 2 |
| Meditations + FOMC | 1 | 0 | 0 | 0 |

So about 80% of these pairs are already connected: one leads to the other through the model's own arrows. Linking them adds no new cause-and-effect claim, only "can't be screened off". The rest (36 in the Meditations) would be genuinely new connections, between ideas the model otherwise treats as unrelated.

There is a third way to see it. Many two-way links come from **circles** (the subject says A drives B, and the stronger claims say B leads to A). Those links are the subject's feedback loops or contradictions, not hidden causes. If a two-way link meant only "these claims clash", with no causal reading, there would be no inseparable pairs at all. But then the PAG wouldn't apply to it either. That choice is worth making on purpose. Splitting the links by origin would help: "would make a circle" links read as feedback, and "goes against a stronger no-difference claim" links read as hidden causes.

## 3. Ways to treat silence as missing

### A. Three-valued pairs: what every completion agrees on (recommended first)

Give every pair of ideas one of three statuses: **linked** (the subject connects them), **separated** (a held "makes no difference" claim), or **unstated**. Then ask which statements hold in *every* complete graph consistent with the stated ones, and report each unstated pair as:

- **forced linked**: every consistent graph links it, as with the inseparable pairs above;
- **forced unlinked**: no consistent graph can link it;
- **free**: the subject's words don't decide it.

This is exact reasoning over constraints, with no tuning. It's the approach of SAT and answer-set causal discovery (Hyttinen, Eberhardt & Järvisalo 2014; Triantafillou & Tsamardinos 2015, "COmbINE"), which work from partial and overlapping evidence by design. The current closed-world PAG becomes one extreme of it: every free pair set to unlinked. The current fit already does a small version of this, since it adds no arrow it isn't forced to.

What the user would see: on the hidden-cause view, unstated pairs that matter shown in three styles. A toggle compares "silence means no link" with "silence means not said", and lists the marks that change between them.

Cost: a solver call over at most a few hundred ideas. That's fine for the subgraph around a two-way link; the whole Meditations model may need to be split by district (ideas joined by two-way links).

### B. Each passage as its own small dataset

Each passage talks about a handful of ideas. That makes the corpus a set of **overlapping variable sets**: two ideas never mentioned together were never *measured* together, which is not the same as being independent. ION and IOD (Tillman, Danks & Glymour 2008; Tillman & Spirtes 2011) build every graph over all the ideas that is consistent with each passage's local picture. This is A with the evidence kept per passage, so we can say which passages force a link.

### C. Why is the subject silent? (the missingness mechanism)

Silence is not random. People skip what is obvious, what they don't care about, and what the topic doesn't call for. That makes it *missing not at random*. Missingness graphs (Mohan & Pearl) model this by adding an "was this said?" indicator per pair, with its own causes (salience, topic, how often each idea comes up).

A practical version is **opportunity-weighted silence**. Silence counts as weak evidence of "no link" only where the subject had occasion to say it: both ideas appear in the same passages, and those passages discuss what drives one of them, yet the other is never named as a cause. Silence about two ideas that never meet counts as nothing. This also gives a principled middle ground between "silence means no" and "silence means unknown".

### D. Probabilities instead of a rule

Put a prior on each unstated pair: a sparsity prior, perhaps raised by opportunity (C). Then sample complete graphs, compute the PAG for each, and report how often each mark holds. The closed-world rule is the limit of a very strong sparsity prior. This is the most flexible approach, but it introduces a number we'd have to defend, and the user asked for the Mirror to stay with what the subject says.

### E. Ask about the gaps that matter

Of the free pairs (A), some change many marks and most change none. The Question engine can rank free pairs by how many marks or traced effects their answer would settle (value of information), and ask about those first. Answers become pinned constraints through the existing approval flow, so nothing changes without the user's approval.

## 4. Suggested order

1. **Decide what a two-way link means**, by origin. "Would make a circle" links might be read as feedback rather than hidden causes. This decides whether most inseparable pairs exist at all.
2. **A (three-valued pairs)** on the subgraphs around two-way links. It's exact, uses only the subject's words, and replaces the "add them or not" toggle with "forced, free, or impossible" per pair.
3. **E (ask)** on the free pairs A finds.
4. **C (opportunity-weighted silence)** as a later refinement, if free pairs turn out to be too many to ask about.

B is A with per-passage bookkeeping, and is worth doing only if we want to show which passages force a link. D is held back unless the user wants probabilities.

## References

- Richardson & Spirtes (2002), *Ancestral graph Markov models*: MAGs, inducing paths.
- Zhang (2008), *On the completeness of orientation rules for causal discovery in the presence of latent confounders and selection bias*: PAG orientation rules.
- Hyttinen, Eberhardt & Järvisalo (2014), *Constraint-based causal discovery: conflict resolution with answer set programming*.
- Triantafillou & Tsamardinos (2015), *Constraint-based causal discovery from multiple interventions over overlapping variable sets*.
- Tillman, Danks & Glymour (2008), *Integrating locally learned causal structures with overlapping variables*; Tillman & Spirtes (2011), *Learning equivalence classes of acyclic models with latent and selection variables from multiple datasets with overlapping variables*.
- Mohan, Pearl & Tian (2013), *Graphical models for inference with missing data*.
