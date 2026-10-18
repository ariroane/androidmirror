# The descriptive ontology: built for inference

Following the two families of inference in Stanford CS520's notes ([What are some inference algorithms?](https://web.stanford.edu/class/cs520/2020/notes/What_Are_Some_Inference_Algorithms.html)):

- **Taxonomic.** Every node has a declared *kind* (agent, disposition, emotion, practice, belief, state, faculty, concept, class, ...). `is_a` is transitive, so membership and evaluation are inherited: *justice is a virtue* and *virtue is good* gives *justice is good*. The evaluations `good`, `bad` and `indifferent` are built-in classes that form a partition, so calling something both good and bad is a detectable inconsistency.
- **Rule-based.** Thirteen relations with declared domains, ranges and properties: `is_a`, `part_of`, `requires` and `same_as` are transitive; `opposes` and `same_as` are symmetric; `opposes`, `requires` and `accords_with` pass from a class to its members; `values` and `rejects` pass from a class to its members on the object side (*valuing the good* means *valuing justice*). The rules run bottom-up to a fixpoint, and every derived fact keeps the rule and premises that produced it.
- **Only what the subject holds is reasoned over.** Its own assertions and self-prescriptions count; quoted views of others, questions and hypotheses are kept as reported speech and never enter inference.
- **Evidence accumulates.** The tenth passage that says the same thing adds to that fact's support instead of being dropped or overwriting it.
- **Checks.** Disjoint evaluations, values-and-rejects, a stated denial of something that follows, taxonomy cycles, and domain/range violations are listed in the Knowledge tab with their proofs.
- **Open world.** A fact that is neither stated nor derived is "not found", never false.
