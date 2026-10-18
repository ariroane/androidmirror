# The causal model: the subject's own judgments

The causal layer is not a model of the world, and not Claude's model of the subject. It is the model the subject's own causal judgments state. An earlier prototype tried to find it in how concepts co-occur across passages; a skeptical evaluation showed that a reflective text is not a sample of situations, so that approach mostly measured the author's habits of mind. Prototype 3 reads the judgments themselves.

**Five kinds of judgment**, each read with a sign (+1 raises, −1 lowers, 0 makes no difference), a scope (direct, overall, or unsaid), a register (describes, urges, entertains, asks) and an attribution (the subject, or someone they quote):

| Kind | Example | Constraint on the graph |
|---|---|---|
| causes | "anger comes from opinion" | x reaches y with that sign; an arrow if it says "directly" |
| intervention | "take away the opinion, and the hurt is gone" | x reaches y (sign 0: no path) |
| means to an end | "do this so that you are free" | x reaches y |
| not this but that | "not things but our judgments disturb us" | x acts on z only through y; y reaches z |
| makes no difference | "whether or not they praise you, you are the same" | no path from x to y |

**Only what the subject holds is fitted.** A judgment counts when it is the subject's own and describes or urges. Views it reports from others, hypotheses and questions are kept and shown, never fitted.

**Weight is passages, not repetitions.** A constraint's weight is the number of distinct passages that make it; saying it three times in one passage counts once. Repetitions are only a tie-breaker.

**Fitting.** Constraints are taken strongest first, a denial before a claim on a tie:
- a claim the graph already implies by a path of the same sign is marked *follows* and adds no arrow;
- a claim the subject does not call direct waits until all the others are in, so an overall claim never becomes a shortcut arrow when the subject also gives the chain;
- a claim that would close a loop, or break a stronger denial, is put *in tension*, with the constraints it ran into, and its two ends are joined by a two-way link (x ↔ y): something the subject never names drives both. That is the one reading under which the claim and what it ran into can both stand, and a two-way link adds no directed path, so it never makes a loop or breaks a denial. The Question engine can settle it (one side holds, or neither drives the other, naming the shared cause when the model already has it driving both), Mutate can split an end used in two senses, and Transmutate lists the links combining two Mirrors resolves or creates;
- an arrow from an overall claim that the rest of the graph also explains is drawn dashed: the subject may mean only that.

The Causal tab lists every constraint with its status, the tensions, the pairs said in both directions, and the subject's values.

**What it can answer.** *Trace* in the Causal tab answers "what does more x do to y, in the subject's model?" with the paths and the passages behind each step, or "makes no difference (the subject says so)", or "not said".

**Does the rest of the corpus predict a passage?** The model is fitted with one contiguous block of passages held out at a time (5 blocks), and each held-out claim is checked: said elsewhere too, not said but follows from the rest, nothing bears on it, opposite sign, or contradicted. Claims that *follow* without being said are the evidence that the judgments form one model rather than a list.

## Benchmark

`python -m androidmirror benchmark` plants a 12-concept causal model in a synthetic subject who states pieces of it (direct claims, overall claims, thought experiments, "not this but that", denials) with Zipf-like repetition, some views of other people, and a share of misread judgments. Offline, 300 passages, 5 seeds, share of drawn arrows that are arrows of the planted model:

| Reading errors | Fitted model | Every claim an arrow |
|---|---|---|
| 0% | 1.00 | 0.43 |
| 10% | 0.85 | 0.28 |
| 20% | 0.68 | 0.22 |

`benchmark --claude` has Claude write the passages in the subject's voice and runs the real reading pipeline on them, so it also measures how accurately Claude reads judgments. It costs about as much as building a corpus of that size.
