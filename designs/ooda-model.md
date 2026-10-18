# The OODA model

A third model of the subject beside the knowledge graph and the causal model: the subject's ideas placed in the four stages of John Boyd's OODA loop (Observe, Orient, Decide, Act) and linked by the subject's own causal claims. It is built from the two graphs only, with no Claude calls, and it changes neither of them. Code: `androidmirror/mirror/ooda.py`.

## Placing an idea in a stage

Every idea in the Mirror collects points toward each stage, and goes to the stage with the most. Each point keeps its reason, which the OODA tab shows.

| Evidence | Stage | Points |
|---|---|---|
| Share of its mentions in passages grouped by stage (the [OODA interview](../README.md#the-ooda-interview) groups each answer by its question's stage) | that stage | 3 × share |
| Kind: agent, event, object, place | Observe | 1 |
| Kind: state | Observe and Orient | 0.5 each |
| Kind: belief, disposition, emotion, faculty | Orient | 1 |
| Kind: concept, class | Orient | 0.5 |
| Kind: practice | Act | 1.5 |
| The subject practices or embodies it | Act | 1 |
| The subject values or rejects it, or calls it good or bad | Decide | 1 |
| The subject learned from it (a person or tradition) | Orient | 1 |
| The subject urges more or less of it | Decide | 1 |
| In the causal model it causes something and nothing causes it (an input) | Observe | 0.75 |

A tie goes to Orient, the stage Boyd put at the centre. A placement decided by less than half a point is marked as barely decided. You can move any idea to another stage in the OODA tab; that is saved in `ooda_overrides.json` in the Mirror's folder, survives a rebuild, and can be set back to automatic. It changes only this model.

Without stage groups (Marcus, the FOMC, the DGN guidelines, Epictetus) the placement rests on kinds, facts, values and causal position alone, so it is coarser: read the reasons before trusting a stage.

## Links between stages

- **Causal links**: the causal model's arrows between placed ideas.
- **Loop links**: claims the subject makes that the causal model holds aside because they would close a cycle. The causal model must be acyclic; an OODA loop is a cycle by definition, so these are kept here and drawn dashed.
- **Facts**: stated facts between ideas in different stages, drawn dotted, as context.

Each link is classified by the stages it joins: to the next stage (forward), within a stage, skipping a stage (Orient straight to Act is Boyd's "implicit guidance and control"), from Act back to an earlier stage (feedback, which closes the loop), or backward.

The tab sums the passages behind causal and loop links into a stage-to-stage table, and states what the shape shows: stages nothing reaches, whether the loop closes, shortcuts from Orient or Observe straight to Act, and the one stage-to-stage flow that dominates, if one does.

## Asking it questions

The Ask tab reads the OODA model as a third source, cited as [O1], [O2] and so on: the stage of each idea named in the question and why, the stage links that touch it, and what the loop shows. A question that names a stage ("what does Marcus observe?", "how does the FOMC decide?") adds that stage's ideas and its links; one about the loop, OODA or stages adds every stage and the passages behind the links between them.

## What it is not

The stages are a reading of the Mirror, not something the subject said. A heuristic places them, and nobody has validated it against people's own view of their loop.
