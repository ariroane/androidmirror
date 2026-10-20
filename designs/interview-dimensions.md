# Interview dimensions: what else can fill a person's graphs

Brainstorm, 2026-10-20. Sections 1, 2, 3, 5 and 13 (the recommended first set) are built as the **Get to Know You!** questionnaire, `data/interview_know.json`, 16 questions each; the rest are candidates.

## What the current interview covers, and what it leaves out

The OODA interview (106 questions) asks about **process**: how the person observes, orients, decides and acts. It says little about **content**: what they love, fear, believe is sacred, think a good life is, or what their inner world looks like. The knowledge graph and causal model both get richer from content, because content supplies the nodes (dispositions, emotions, states, practices, beliefs) that causal claims run between.

The dimensions below are sections of questionnaires on /interview, all writing to the same `interview-<name>` corpus. No new model: answers go through the existing coder into the same KG and causal layer.

## What each graph can take in

**Knowledge graph** relations (schema.py): `is_a` (including good / bad / indifferent), `part_of`, `same_as`, `opposes`, `requires`, `accords_with`, `expresses`, `exemplifies`, `practices`, `learned_from`, `values`, `rejects`.

**Causal layer** fits only the subject's own causal claims:

| Judgment | Question shape that draws it out |
|---|---|
| causes | "What leads to X?" "What does X do to you?" |
| intervention | "When you want more/less X, what do you do?" |
| means_end | "What is X for?" "How do you get to X?" |
| screens_off ("not x but y") | "People say X causes Y. Is it really X, or something else?" |
| independent | "What do people think matters for X that, in your view, doesn't?" |

Rule of thumb: a question that asks **what something is** feeds the KG. A question that asks **what it does, what feeds it, or what would change it** feeds the causal layer. The best questions ask both.

Independence and "not x but y" claims are the scarcest in every corpus so far (the convention is no evidence = no connection), so dimensions that elicit them are worth more per answer.

## Candidate dimensions

Rated for KG yield and causal yield (high / medium / low).

### 1. The sacred and the ultimate (spiritual)  KG high, causal medium-high
What the person treats as beyond price, what they think the universe is like, death, fate, grace. Fills `concept` and `class` nodes and evaluations; the "what is up to you" questions yield independence claims, Stoic-style.
- What, if anything, do you hold sacred? What would it mean to betray it?
- Is there an order or intelligence to the world, or is it indifferent? How does your answer change how you live?
- What is up to you, and what is not up to you at all?
- What do you think happens at death, and how does that shape what you do now?
- Have you ever felt grace, luck or providence act on you? What brought it about?
- What does your conscience tell you, and where do you think its voice comes from?

### 2. Metaphor and inner image  KG high, causal low unless followed up
The person's pictures of themselves and their life. Metaphors yield `same_as`, `part_of`, `expresses` richly but rarely causal claims, so each image question should carry a follow-up asking what moves what inside the picture.
- If your mind were a place, what kind of place would it be? Who or what lives there, and what changes the weather?
- What animal, element or landscape are you most like? What makes it thrive, and what wounds it?
- If your life were a journey, where are you on the road? What got you here, and what blocks the way?
- What object would you keep as a symbol of who you are? What would losing it mean?
- Describe the voice in your head when things go wrong. What sets it off, and what quiets it?
- If your fear had a shape, what would it be? What feeds it?

### 3. Emotions and their triggers  KG medium, causal high
The most reliable source of `causes` and `intervention` claims, because people already think about feelings causally.
- What reliably makes you angry? What happens after?
- What brings you calm, and how long does it last?
- When you feel shame, what usually came just before it?
- Which feelings do you try to change, and what do you do to change them?
- Which feelings lead you to act well, and which lead you astray?

### 4. Values and ends  KG high, causal high (means_end)
What the person aims at and what serves what. Directly feeds `values`/`rejects` and means_end chains.
- What do you want most from your life? What is that for?
- Which of your goals are stepping stones, and which are ends in themselves?
- What would you give up to get what you want most? What would you never give up?
- What do you admire in others that you don't have yet? How would you get it?

### 5. What doesn't matter (independence probes)  KG low, causal very high
Asks directly for the claims the causal layer rarely gets: independence and "not x but y".
- What do most people think matters for happiness that you think doesn't?
- People often blame X for Y. Where do you think they have the cause wrong, and what is the real cause?
- What used to seem important to you that you now think has no effect on how your life goes?
- If you removed money (or status, or praise) from your life, what would stay exactly the same?

### 6. Character: virtues, vices, temptations  KG high, causal medium
Dispositions and their evaluations, plus what strengthens or erodes them.
- Which of your traits do you count as virtues, and which as faults?
- What tempts you most, and what usually leads you into it?
- What has made you a better person over the years? What has made you worse?
- Which of your good qualities could become a fault if pushed too far?

### 7. Body, health and energy  KG medium, causal high
States that people treat as both causes and effects.
- What does your body need to keep you at your best? What happens when it doesn't get it?
- How does sleep (or food, or movement) change how you think and feel?
- When your energy drops, what usually caused it?

### 8. People and lineage  KG high (agents, learned_from, exemplifies), causal medium
- Who shaped you most, and what did they give you?
- Whom do you try to be like, and whom do you try not to be like?
- Which relationships make you better, and how?
- Who do you need to forgive, and what would forgiving them change?

### 9. Turning points and counterfactuals  KG medium (events), causal high
"If not for X" is a direct causal claim, and often a "not x but y" one.
- What were the turning points of your life? What caused each one?
- If one event had not happened, how would you be different?
- What looked like a disaster at the time but turned out well? What turned it?

### 10. Practices and disciplines  KG medium, causal high (intervention)
What the person does to change themselves, which is an intervention claim by definition.
- What daily practices do you keep, and what does each one do for you?
- When you want to become more patient (or brave, or kind), what do you actually do?
- Which practices did you drop, and what changed when you did?

### 11. How the world works  KG medium, causal medium
Beliefs about society, history, money, nature: concept-level causal claims outside the self.
- What do you think really drives how society changes?
- Why do some people flourish and others don't?
- What is wrong with the world, and what would fix it?

### 12. Dreams, longing and the ideal self  KG high, causal low-medium
- Describe the person you hope to be in ten years. What stands between you and them?
- What do you long for that you can't name easily?
- Describe a dream (asleep or awake) that has stayed with you. What do you think it means?

### 13. Good and bad (valuations)  KG very high, causal medium
What the person counts as good, bad or indifferent, and where the line falls. Feeds the evaluation partition directly (`is_a good / bad / indifferent`, `values`, `rejects`), and because that partition is disjoint, the KG can flag anything they call both good and bad as a tension for the Question engine. Asking why something is good or what turns it bad brings in causal claims too.
- Name three things you think are truly good, and three truly bad. What makes each one so?
- What do most people call good that you think is bad, or bad that you think is good?
- What do you think is neither good nor bad, though people get worked up about it?
- What is good in small amounts but bad in large ones? What tips it over?
- Can something bad lead to something good? Give an example and say how.
- When you call an action wrong, is it because of what it causes, what it shows about the person, or something else?
- What good thing would you refuse if getting it meant doing something you think is wrong?
- Has something you once thought was bad come to look good, or the reverse? What changed?

## Recommended first set

A mix weighted toward your spiritual and metaphorical idea, with two causal-heavy buttons to balance it:

1. **The sacred and the ultimate** (spiritual; also gives "up to you / not up to you" independence claims)
2. **Metaphor and inner image** (metaphorical; every image question carries a "what moves what" follow-up so the causal layer is fed too)
3. **Emotions and their triggers** (the most reliable causes/intervention yield)
4. **What doesn't matter** (the scarcest claims: independence and "not x but y")
5. **Good and bad** (fills the evaluation partition and surfaces contradictions)

Built as one questionnaire, Get to Know You!, 80 questions. A second wave could add Values and ends, Turning points, and Practices.

## How it is built

- Each questionnaire is a bank in the same format (stages with ids and questions): `data/interview_ooda.json` and `data/interview_know.json`. One interview of a person holds the answers to both, and the page switches between them.
- Write the corpus writes all answers, from both, into the one `interview-<name>` corpus. Passages keep the "The interviewer asked: Q / <subject> answered: A" form, so only the answer is attributed to the subject. OODA passages keep their file names (stages 1 to 4), so existing builds stay cached; Get to Know You! sections follow as 5 to 9.
- The OODA model reads only the four OODA stage groups, so Get to Know You! answers place their concepts through the graphs, not through their section.
- Open: metaphor answers will test the coder. "My mind is a lake" should not become a literal `is_a`. If builds show that, a codebook note could tell the coder to keep figurative identity as `expresses` or `related_to` with the original wording.
