"""Android Mirror, Prototype 2.

The Mirror is built from a whole corpus in one pass, with no per-patch review:

1. `corpus`: split the text into passages (the units of observation).
2. `coding`: Claude codes each passage once (concept mentions with their level,
   descriptive statements with modality and attribution, stated causal claims).
3. `codebook`: the mentions are consolidated into one fixed set of concepts.
4. `kg`: the statements become a knowledge graph with a schema and rules, so
   it supports taxonomic and rule-based inference with proofs back to quotes.
5. `causal`: the passages become an observational dataset over a selected set
   of variables; PC finds its Markov equivalence class (a CPDAG), and the
   subject's own stated causal claims are applied as background knowledge.

Every node and edge keeps the passages it came from, so the source text is one
click away. The Prototype 1 engines (refactor, breaker, chainer, evolver) are
paused and not wired into this app.
"""
