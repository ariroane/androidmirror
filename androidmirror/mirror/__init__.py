"""Android Mirror, Prototype 3.

The Mirror is built from a whole corpus in one pass, with no per-patch review:

1. `corpus`: split the text into passages.
2. `coding`: Claude codes each passage once (concept mentions, descriptive
   statements with modality and attribution).
3. `codebook`: the mentions are consolidated into one fixed set of concepts.
4. `coding` again: Claude reads the causal judgments each passage makes over
   the codebook's concepts, and what it urges more or less of.
5. `kg`: the statements become a knowledge graph with a schema and rules, so
   it supports taxonomic and rule-based inference with proofs back to quotes.
6. `judgments` and `causal`: the judgments become constraints, weighted by the
   passages that make them, and the subject's causal model is fitted to them.

Every node and edge keeps the passages it came from, so the source text is one
click away. The Prototype 1 engines (refactor, breaker, chainer, evolver) are
paused and not wired into this app.
"""
