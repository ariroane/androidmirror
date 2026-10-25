# Merging Mirrors through a formal ontology

Research and brainstorm: which open formal ontologies a Mirror could import, and how aligning to one could replace Transmutate's shared-node merging. Ideas only: nothing here is built.

The standard for a merge stays the one in the knowledge-layer note: **would each subject endorse the merged item on reflection?** An imported ontology is never evidence of what a subject holds. It is a shared vocabulary to meet in, a source of candidate mappings, and a source of checks.

## 1. What Transmutate does today, and where it runs out

`androidmirror/mirror/transmutate.py` aligns two Mirrors concept by concept. A pair comes from the same label or alias, from Claude, or from a person's `--pair` / `--opposite` / `--drop`. A pair is either *the same* or *opposite* (polarity −1), and each concept is paired at most once. Paired concepts become one node; every other concept keeps its Mirror's prefix (`epictetus:prohairesis`). Then evidence is pooled, the causal model is refitted, and fitness trimming removes what a reader of half the corpus would not come away with.

That works when the two subjects use the same words for the same things. It runs out in five ways:

1. **Only two answers.** A pair is identical or opposite. There is no "narrower than", "broader than", "overlaps with" or "related to". Marcus's *the ruling part* (hegemonikon) and Epictetus's *prohairesis* are neither the same concept nor unrelated, so today they either merge wrongly or float apart as two islands.
2. **Islands.** Everything unaligned is prefixed and has no structural link to the other Mirror. A combined Mirror is two graphs joined at a few nodes.
3. **Homonyms pass.** Same name means same concept. *Pleasure* in a Stoic corpus (often a passion, bad) and in an interview (often a plain good) merge by name, and the clash then shows up as a "tension" when it is really two senses.
4. **Pairwise.** Each pair of Mirrors is aligned from scratch. Five corpora need ten alignments; peer-to-peer search (see the peer-to-peer note) would need one per pair of peers.
5. **Disagreement can only be won or kept.** Where the two disagree, the fit takes the better-supported side and reports the other as a tension. It has no way to say "they agree at a coarser level" (both say *passions* disturb tranquillity, though one names *anger* and the other *grief*).

A structured ontology addresses all five: richer mapping relations (1), a hierarchy that connects islands (2), word senses and disjointness that separate homonyms (3), a pivot that each Mirror aligns to once (4), and a hierarchy to generalise along when two claims clash (5).

## 2. Survey: open ontologies that can be imported

Licences below were checked on the publisher's page where a link is given; the rest should be checked on the file header at import time. "Fit" is fit for a Mirror's kinds of concept (dispositions, emotions, practices, beliefs, states, faculties, abstract concepts, values) and for the corpora (Marcus, Epictetus, FOMC, DGN Parkinson's guidelines, OODA interviews).

### Lexical ontologies: word senses and their relations

| Resource | Licence | Formats | Fit |
|---|---|---|---|
| [Open English WordNet](https://en-word.net/downloads) (2025 edition) | CC BY 4.0 | WN-LMF XML, RDF/Turtle, JSON, WNDB | **Best first pivot.** ~120k synsets with glosses; hypernymy (broader/narrower), antonymy (opposite), meronymy (part of), derivation. Covers everyday mental vocabulary (anger, grief, tranquillity, judgment, habit) well, Stoic technical terms poorly. |
| Collaborative Interlingual Index (CILI) | CC BY 4.0 (check) | TSV/RDF | Language-neutral ids that Open English WordNet synsets map to; would let a Greek or German corpus share a pivot later. |
| [ConceptNet 5](https://arxiv.org/pdf/1612.03975) | CC BY-SA 4.0 (share-alike) | CSV, JSON-LD | Commonsense relations including `Causes`, `Desires`, `MotivatedByGoal`, `Antonym`. Useful as candidate generator only: noisy, and share-alike means it should be fetched at build time, never bundled. |
| FrameNet | CC BY (check version) | XML | Frames such as *Emotion_directed*, *Causation*, *Judgment*. Better for reading passages than for naming concepts. |
| [SUMO](https://github.com/ontologyportal/sumo) + WordNet mappings | GPL | SUO-KIF (OWL exports exist) | A large upper ontology with every WordNet synset mapped to a SUMO term. Rich, but GPL and KIF make it heavy to import; use the mapping files as a reference, fetched at build time. |

### Upper (foundational) ontologies: what kind of thing a concept is

| Resource | Licence | Formats | Fit |
|---|---|---|---|
| [gUFO](https://arxiv.org/pdf/2603.20948) (lightweight UFO) | CC BY 4.0 | OWL 2 DL, Turtle, SHACL shapes | Kinds, roles, phases, qualities, *modes* (a belief or an admiration is a mode inhering in a person), relators, situations. The kind/phase distinction fits Mirror's `state` vs `disposition`. Mental states are modes but there is no ready taxonomy of them. |
| [DOLCE / DUL](https://www.loa.istc.cnr.it/dolce/overview.html) (DOLCE+DnS Ultralite) | Creative Commons (check file header) | FOL, OWL (DUL), Common Logic | "Descriptive Ontology for Linguistic and Cognitive Engineering": categories follow how people talk and think, not physics, which suits subjective systems. DUL's *Descriptions and Situations* pattern models a viewpoint (a Description) that classifies Situations; that is close to what a Mirror is. ISO/IEC 21838-3. Framester and ValueNet are aligned to it. |
| BFO (Basic Formal Ontology) | CC BY 4.0 | OWL | Realist, built for science (ISO/IEC 21838-2). Needed only because the emotion, mental-functioning and behaviour-change ontologies below are built on it. |
| Common Core Ontologies | BSD-3 (check) | OWL | Mid-level BFO extension for agents, actions, information. Good for FOMC and OODA *Act*, less so for inner life. |

### Mind, emotion, values and practice

| Resource | Licence | Formats | Fit |
|---|---|---|---|
| [Emotion Ontology (MFOEM)](https://obofoundry.org/ontology/mfoem) | CC BY 3.0 | OWL (OBO) | Emotions, moods, appraisals, subjective feelings, as processes with appraisal and feeling parts. The appraisal part suits Stoic cognitivism (a passion as an assent to an impression). |
| Mental Functioning Ontology (MF) | CC BY 3.0 (check) | OWL (OBO) | Parent of MFOEM: mental processes, dispositions, cognitive representations (beliefs, desires). Maps onto Mirror's `faculty`, `belief`, `disposition`. |
| ValueNet (Framester) ([paper](https://arxiv.org/pdf/2303.00632)) | not stated in the paper; check repo | OWL | Schwartz's basic human values (10 to 19) and Moral Foundations Theory (care/harm, fairness/cheating, loyalty, authority, sanctity, liberty) in OWL, plus 300+ folk values, aligned to DOLCE. Gives the values layer and the evaluation classes a shared target. |
| Behaviour Change Intervention Ontology and its Mechanism of Action ontology ([BioPortal](https://bioportal.bioontology.org/ontologies/BCIO)) | CC BY 4.0 (check) | OWL | Behaviour change techniques (practices such as self-monitoring, rehearsal, reframing) and the mechanisms they act through (beliefs about consequences, emotion, self-image, habit). The closest open ontology to "a practice changes a state through a belief", which is the shape of many Stoic causal claims and of OODA. |
| Cognitive Atlas | CC BY-SA (check) | JSON API, OWL dumps | Cognitive concepts and tasks from neuroscience; useful for the interview corpora, thin on ethics. |
| InPhO (Internet Philosophy Ontology) | unclear; check before use | JSON API | Philosophical ideas and thinkers, built from the Stanford Encyclopedia. Would cover *Stoicism*, *virtue*, *providence*; maintenance status unclear. |

### Domain ontologies for the other corpora

| Corpus | Resource | Licence | Fit |
|---|---|---|---|
| DGN Parkinson's | Mondo (diseases), Human Phenotype Ontology (symptoms), ChEBI (drugs and chemicals), DrOn | Mondo and ChEBI CC BY 4.0; HPO has its own attribution licence | Lets two guideline Mirrors (or a guideline and a patient interview) meet on *bradykinesia*, *levodopa*, *motor fluctuation* without string luck. |
| FOMC | FIBO (Financial Industry Business Ontology) | MIT | Instruments and institutions; macro concepts (*inflation expectations*, *labor-market slack*) are thin, so WordNet plus a small hand list may do more. |
| any | Wikidata | CC0 | Entities: people, places, events, works (*Rusticus*, *Carnuntum*, *the Discourses*). The obvious target for Mirror's `agent`, `place`, `event`, `object` kinds. |

### Relations and mappings

| Resource | Licence | Use |
|---|---|---|
| SKOS mapping properties (W3C) | W3C | `exactMatch`, `closeMatch`, `broadMatch`, `narrowMatch`, `relatedMatch`: the five mapping relations section 1 says Transmutate lacks. |
| SSSOM (Simple Standard for Sharing Ontology Mappings) | CC BY 4.0 | A TSV of mappings, one row each: subject, predicate (a SKOS property), object, confidence, justification (lexical, manual, LLM), author. A ready file format for a Mirror's anchors. |
| OBO Relation Ontology (RO) | CC0 | Typed causal relations: *causally upstream of, positive effect / negative effect*, *regulates*, *has part*. A target for the causal layer's raises/lowers and for `part_of`, `requires`. |

**Licence summary.** Open English WordNet, CILI, gUFO, BFO, RO, SSSOM, Mondo, ChEBI and Wikidata can be bundled with attribution. ConceptNet (share-alike) and SUMO (GPL) should be fetched at build time, the way the Epictetus loader fetches Standard Ebooks, and kept out of the repo. ValueNet, InPhO, Cognitive Atlas and the DUL file need their licence confirmed before anything is shipped.

## 3. The core idea: merge in a pivot, not node to node

Give each Mirror an **anchor file**: for each codebook concept, zero or more mappings to pivot concepts, each with a SKOS predicate, a polarity, a confidence, the passages that justify it, and who proposed and who approved it. Store it as SSSOM next to the Mirror (`data/mirror/<name>/anchors.sssom.tsv`). Anchoring is done **once per Mirror**, not once per pair, and the Question engine can ask about it.

Transmutate then merges two Mirrors through their anchors:

| A's concept → pivot ← B's concept | What the combined Mirror does |
|---|---|
| both `exactMatch` the same pivot concept | one node (today's merge) |
| `exactMatch` a pivot concept and its antonym | one node, polarity −1 (today's `--opposite`) |
| A's maps to a pivot concept that is broader than B's | **both nodes kept**, with a scaffold `is_a` from B's to A's. Islands get connected. |
| both `closeMatch` the same pivot concept | a merge *candidate*, shown with both glosses, never applied without a person |
| pivot concepts are siblings (share a parent) | no merge; the parent is offered as a meeting point (see 4.2) |
| same name, but pivot concepts disjoint or different senses | **never merged**, flagged as a homonym. Mutate's split is the remedy inside one Mirror; here it is the default |
| `noMatch` (*prohairesis*, *eupatheia*) | stays Mirror-local, as today. The pivot is a help, not a requirement |

Three properties follow:

- **Scaffold facts are not the subject's facts.** An `is_a` that comes from WordNet (*anger is a kind of emotion*) is stored with source `pivot`, drawn differently, and does not enter inference as something either subject holds. It may connect the graph for reading and for proposing questions. If one of the subjects says it, it becomes theirs with their passage, as now.
- **Merges are previewed and approved, as today.** The anchors only change which merges are proposed and with what relation.
- **Anchors are checkable.** A mapping is a claim about what the subject means. If X maps to pivot P and the subject holds facts about X that contradict what P is (P is an emotion and the subject calls X a faculty; P's antonym is something the subject says X *requires*), the anchor is suspect. The existing checks (kind/domain violations, disjoint evaluations) become checks on anchors.

## 4. Ways to merge in the pivot, from simplest to most formal

### 4.1 Anchor and merge (recommended first)

Exactly the table above, with Open English WordNet as the only pivot and the Mirror's own kinds checked against a tiny hand-made map to gUFO or DUL categories (emotion → mode/process, disposition → disposition, practice → action/process). Anchors are proposed lexically (label and aliases against WordNet lemmas), then by Claude choosing among the candidate synsets' glosses with the concept's own gloss and two quotes in view (a choice among listed senses, not free generation), then by a person. This replaces `candidates` and `claude_pairs` without changing `combine`, `fitness` or `approve` much.

### 4.2 Meeting at the lowest common concept

When A has *anger* and B has *grief* and neither has the other, today they stay apart. In the pivot both sit under *passion / negative emotion*. The combined Mirror can add the lowest common subsumer as a scaffold hub node. Nothing is asserted of the hub itself, but readers and the Question engine see that both subjects say a passion disturbs tranquillity, and a question can be put: "Marcus says anger disturbs tranquillity and Epictetus says grief does. Would both say every passion does?" If the subject (or proxy) says yes, the generalisation becomes a held fact with that answer as its source.

### 4.3 Merging as a pushout (the formal version)

In the algebraic view of ontology merging (Goguen's institutions, the OMG's Distributed Ontology Language and its tool Hets), combining two ontologies A and B along what they share is a *pushout*: take the shared part V with maps V → A and V → B, glue A and B along V, keep everything else apart. Today's Transmutate is a pushout where V is the set of aligned pairs and nothing more. With a pivot, V becomes the fragment of the pivot both Mirrors anchor into, *with its hierarchy and disjointness*, and the glue respects it: a broader/narrower pair is glued as an `is_a`, a disjoint pair can never be glued. Framing it this way gives a correctness condition to test against: the combined Mirror must contain an image of each source (nothing either said is lost, only trimmed later on evidence) and nothing that is not forced by A, B and V.

### 4.4 Amalgams: generalise instead of choosing a side

Ontañón and Plaza's *amalgams* (and the conceptual-blending work of Fauconnier and Turner, formalised by Goguen and the COINVENT project) combine two descriptions that clash by generalising each just enough, up an ontology, that they become consistent, then unifying. Applied here: when A says *pleasure is bad* and B says *joy is good*, and the pivot says both are kinds of *positive feeling*, today's fit picks a side. An amalgam keeps both claims at their own level and refuses to merge the two concepts, because merging them would be the inconsistency. When A says *anger lowers tranquillity* and B says *anger raises tranquillity*, no generalisation removes the clash, and it stays a tension. This is a principled reading of "combine consistently": merge as specifically as possible without contradiction, and let the ontology say where "as possible" stops. "Transmutate" in the old alchemical sense is close to what blending theory calls a blend: a new space with structure from both inputs that neither had alone.

### 4.5 Off-the-shelf ontology matching

The Ontology Alignment Evaluation Initiative (OAEI) benchmarks matchers each year. LogMap and AgreementMakerLight do lexical matching plus *repair* (removing mappings that make the merged ontology incoherent); BERTMap and recent LLM-based matchers add embeddings. Two ideas worth borrowing even without the tools: repair (drop the cheapest set of anchors that cause a contradiction, instead of letting the contradiction through as a tension) and the output format (SSSOM), which lets alignments be shared between peers.

### 4.6 No external ontology: induce one

A fallback if imported ontologies impose too much: build the pivot from the Mirrors themselves. Formal Concept Analysis over each concept's relation profile (what it is a kind of, what it opposes, what it raises and lowers, how it is evaluated) gives a lattice in which concepts with the same role sit together. This is what Mutate's "same or mirror-image causal role" detector does inside one Mirror, extended across two. It avoids anachronism but loses the shared vocabulary that makes pivots worth having.

## 5. What it means for the causal layer

Causal nodes are a subset of knowledge-graph nodes, so whatever the pivot does to concepts it does to causal variables. Points to decide:

1. **Granularity mismatch is causal abstraction.** If A's variable is *anger* and B's is *passion*, combining them is a question of causal abstraction (Rubenstein et al. 2017; Beckers and Halpern 2019): when is a model over coarse variables a faithful summary of one over fine ones? The pivot's broader/narrower links give candidate abstraction maps for free. The safe rule: never move a causal sign down `is_a` automatically. *Passion lowers tranquillity* does not imply *anger lowers tranquillity* (a subtype can behave differently), and summing fine variables into a coarse one can create or hide effects. Show "same direction at two levels" as agreement, and turn every downward step into a Question-engine question.
2. **Merging at a coarser level is marginalising.** Gluing at a hub (4.2) is the same move as dropping the fine variables and keeping their sum. In the MAG/PAG terms of the causal V&V note, this can turn arrows into bidirected edges. Run the existing clash-to-hidden-cause logic after the merge, not before.
3. **Names for the hidden cause.** Today a clash is drawn as a two-way link, "a cause neither subject names". If both endpoints' pivot concepts have a common parent or a pivot relation (WordNet derivation, ConceptNet `Causes`, MoA mechanism) to a concept neither Mirror has, that concept is a candidate name. It is offered as a question ("Is it fear of loss that drives both?"), never added to the causal model by the pivot.
4. **Typed causal relations.** Map raises/lowers to RO's *causally upstream of, positive / negative effect*, and the five kinds of judgment to RO's finer relations where they fit. This makes the causal layer exportable and lets a DGN Mirror be checked against domain ontologies (does the guideline Mirror's *levodopa raises motor fluctuation* agree with what ChEBI/RO-based resources say?). That comparison is information about the world, not about the subject, and must be shown as such.
5. **Mechanisms as mediators.** The Mechanism of Action ontology lists the intermediate states practices typically work through. When A says *practice X lowers distress* and B says *practice X changes beliefs about consequences, which lowers distress*, the pivot shows the two are the same claim at two resolutions (direct vs through a mediator), which the causal layer's "directly" / "only through" judgments can already express.
6. **Polarity extends.** Antonymy in the pivot gives more opposite pairs than aliases do, and each flips signs through the causal layer exactly as `--opposite` does now.
7. **Fitness trimming at the right level.** A fine claim with one passage may be trimmed while its coarse version, pooled from both Mirrors, survives. Fitness could be computed per pivot level, so trimming removes the detail and keeps the agreement.

## 6. Keeping the standard: the subject's endorsement on reflection

An anchor says "when this subject says X, they mean pivot concept P". A merge says "both subjects would accept this one node as what they mean". Both are tested by the proxies the knowledge V&V note ranks:

- **Ask the subject.** For interview corpora, every anchor and every non-trivial merge is a question: "When you say *calm*, do you mean (a) free from agitation, (b) free from passion, (c) something else?", with the pivot glosses as the options. The Question engine's machinery already does this.
- **Unseen own words.** For Marcus and Epictetus, hold out passages: does the anchored meaning predict how the word is used in passages the anchoring never saw?
- **Consistency with what they hold.** An anchor whose pivot axioms contradict the subject's held facts is wrong more often than the subject is.
- **A test with a known answer.** Split one corpus in two, rename half its vocabulary in the second half with synonyms, build two Mirrors, and merge. The right merge is known: the original. Compare shared-node merging against anchored merging on how much of the original they recover and how many wrong merges they make.

Two risks to keep in view:

- **Imposition.** A modern lexicon reads modern categories into ancient words. WordNet's *passion* is not Stoic *pathos* (a false judgment). That is why anchors carry `closeMatch` vs `exactMatch`, why `noMatch` is allowed and common for technical terms, and why scaffold facts never count as held. A Mirror that comes out more like WordNet and less like Marcus has failed the standard, and the comparison in A3 of the knowledge V&V note (transplant tests) should detect it.
- **Claude's taste in anchoring.** Every anchor records whether it was lexical, Claude-chosen or person-approved, so a merge that rests on Claude's choices alone can be shown as such.

## 7. Suggested first step

1. Add an anchor file per Mirror, proposed against Open English WordNet only (lexical match, then Claude picks among listed senses), approved like answers.
2. In Transmutate, read anchors when present: `exactMatch` and antonym pairs as today, broader/narrower as scaffold `is_a` links, disjoint senses as blocked merges, `closeMatch` as candidates.
3. Run it on Marcus + Epictetus and on the split-half test, and compare with the current alignment: concepts connected, homonyms caught, held-out fitness of the combined Mirror, and a blind reading of which combined Mirror each author's text supports better.
4. Only if that helps: add MFOEM for emotions and ValueNet for values, then the lowest-common-concept hubs (4.2), then amalgam-style generalisation (4.4).

The same anchors would serve `compare.py` (differences measured in a shared space), the Question engine (questions about meaning), and peer-to-peer search (a peer publishes its anchors, and "anger" meets "wrath" through the pivot instead of by string luck).

## References

- Open English WordNet downloads: https://en-word.net/downloads
- Emotion Ontology (MFOEM), OBO Foundry: https://obofoundry.org/ontology/mfoem
- DOLCE overview, Laboratory for Applied Ontology: https://www.loa.istc.cnr.it/dolce/overview.html
- gUFO: A Gentle Foundational Ontology for Semantic Web Knowledge Graphs: https://arxiv.org/pdf/2603.20948
- ValueNet and the values knowledge graph: https://arxiv.org/pdf/2303.00632
- SUMO repository: https://github.com/ontologyportal/sumo
- ConceptNet 5.5: https://arxiv.org/pdf/1612.03975
- Behaviour Change Intervention Ontology, BioPortal: https://bioportal.bioontology.org/ontologies/BCIO
- Rubenstein, Weichwald et al., "Causal Consistency of Structural Equation Models" (UAI 2017); Beckers and Halpern, "Abstracting Causal Models" (AAAI 2019)
- Ontañón and Plaza, "Amalgams: A Formal Approach for Combining Multiple Case Solutions" (ICCBR 2010); Goguen, "What is a Concept?" (ICCS 2005); Fauconnier and Turner, *The Way We Think* (2002)
- Mossakowski, Codescu, Kutz et al., the Distributed Ontology, Model and Specification Language (DOL), OMG standard
