# Peer-to-peer Mirrors

Brainstorm: using Mirrors in place of parts of the ordinary internet, with each Mirror a peer. Ideas only: nothing here is built.

## 1. The idea in one paragraph

Most of what the web does for a person (search, feeds, recommendations, follows, reputation, moderation, messaging when you're away) is done by a platform holding a hidden model of you and of everyone else. A Mirror is the opposite kind of object: a model of a subjective system that its owner holds, can read, and can correct, built from that system's own words, with every item traced to the passages behind it. If Mirrors could talk to each other directly, several of those platform jobs could be done between peers, each peer answering from its own Mirror, with no central model in the middle.

## 2. What a Mirror already has that a peer needs

| A peer needs | The Mirror already has |
|---|---|
| A thing to publish | The built Mirror (knowledge graph, causal model, OODA model, values) as files under `data/mirror/<name>/` |
| A fingerprint | Transmutate already hashes `mirror.json` and refuses to combine a source that changed |
| A way to answer strangers without exposing raw text | The Ask tab answers from the structures only, never from passages; "as the subject" answers in the system's voice |
| A way to find who is close to you | `compare.py`: how two Mirrors differ in knowledge, causes and OODA loop |
| A way to merge and fork | Transmutate combines two Mirrors; the planned Breaker (open problem 3) would split one |
| A way to ask a peer what you can't infer | The Question engine, whose answers are proposals approved by name |
| A record of who changed what | Approved answers and mutations, each with an approver, replayed on every rebuild |
| A way to bridge vocabularies | Transmutate's alignment: same name or alias, Claude, or a person's `--pair` / `--opposite` |

What's missing is the network: an address, a signature, a message format, and rules for what a peer may see.

## 3. Internet functions a Mirror could replace

Ordered roughly by how directly the existing code supports them.

### A. Asking someone who isn't there (recommended first)

Today, if you want to know what someone thinks, you message them and wait, or search their old posts. A peer could instead put a question to their Mirror: "As the subject" when you want their voice, the cited Oracle when you want the structure behind the answer, Trace for "in your view, what does more x do to y?". The owner sets what is exposed and sees a log of what was asked. The answer is always labelled as the Mirror's, never the person's.

This is the smallest step from what exists: the Ask routes already take a corpus name, so a "peer" can be a second server with its own Mirrors.

### B. Search by worldview, not by keyword

A search engine ranks pages. A Mirror network could answer "who holds that x lowers y?" or "who urges more of z?" by sending the question to the peers whose codebooks contain x, y or z, and returning each peer's Trace with its support. Disagreement is kept, not averaged: a peer that holds the opposite sign is a result too.

Routing needs an index from concepts to peers. Each peer publishes only its codebook (concept names and aliases), which a distributed hash table can key on, the way BitTorrent finds who has a file. The hard part is that "anger" in one Mirror and "wrath" in another must meet; see section 5.

### C. A feed you own

A recommendation feed guesses what will keep you scrolling. A Mirror feed would score each incoming item (a post, an article, another peer's new claim) by what it would touch in your own model: does it bear on something you urge more or less of, does it contradict a held fact, does it fill a gap the Question engine has flagged, does it rest on what Derange calls your weakest basis? The ranking rule is yours, readable, and editable, and it can deliberately favour what challenges you as well as what agrees.

### D. Follows and friends from overlap and difference

Instead of a follower graph, peers find each other through `compare.py`: Mirrors that share most of their concepts but differ on a few causal claims are the most informative people to talk to, because the differences are where one of you could learn something. A "people like you" list and a "people who disagree with you about exactly one thing" list come from the same comparison.

### E. Finding common ground: Transmutate as negotiation

Two peers (two people, two teams, a community and a newcomer) Transmutate their Mirrors without approving the result. The proposal shows what they share, which links the combination resolves, and which tensions it creates. The tensions are the cruxes, and the Question engine can turn each one into a question for both sides. Nobody's Mirror changes unless its owner approves.

### F. Community rules as a Mirror

A forum's moderation policy is a hidden model too. A community could keep its rules as a Mirror built from its own charter and past rulings (the FOMC corpus is already an institution's Mirror). A ruling is then an Ask against that Mirror, with citations, and a change in the rules is an approved answer with a name on it. A community that splits is a Breaker; two that join are a Transmutate.

### G. Reputation from consistency, not popularity

Stars and likes measure popularity. A Mirror can measure other things about the peer that published it: whether its held-out check passes (do its claims follow from each other), how many inconsistencies the Knowledge tab lists, how much of what matters most rests on one passage (Derange), and whether other peers endorse its facts. These say whether a worldview is coherent and well supported, not whether it's right, and the note on reputation should say so plainly.

### H. A commons without edit wars

A shared reference work could be the Transmutate of many Mirrors, with disagreements kept as tensions and two-way links instead of being settled by whoever edits last. Each claim keeps the peers who make it, so "who holds this" is part of the entry.

## 4. A protocol sketch

Enough to build a first test; none of it is decided.

- **Identity.** Each peer has a key pair. A Mirror bundle is signed, and its hash is its address. A new build or an approved change makes a new bundle that points to the previous hash, so a Mirror's history is an append-only chain (as in Secure Scuttlebutt or git), which also gives open problem 5 the traceable lineage it needs.
- **Layers a peer can expose**, each separately:
    1. the card: name, Face, OODA shape, size;
    2. the codebook: concepts and aliases, for routing;
    3. the structures: knowledge graph, causal model, OODA model, values, without passages;
    4. the passages.
    Today's Ask rule (structures, never passages) is a natural default: layers 1 to 3 shared on request, layer 4 kept at home.
- **Messages.** A small set of typed requests: `trace x y`, `follows? fact`, `ask "question" [--as-subject | --cited]`, `compare`, `propose-answer` (a peer suggests an answer to one of your Question-engine questions; it waits for your approval like any other).
- **Who runs the model.** Either the asker sends the question and the owner's machine answers (the owner pays the Claude call and the structures never leave), or the owner sends layer 3 and the asker runs it (the asker pays, the structures travel). Trace, "does it follow?" and compare need no Claude call at all, so they can always run on the owner's side cheaply.
- **Transport.** Existing peer-to-peer systems already carry signed append-only logs: Secure Scuttlebutt, Hypercore, Nostr relays, IPFS for the bundles, ActivityPub if federation is enough. The Mirror shouldn't invent its own; it needs a message format that rides on one of them.

## 5. The hard problem: concepts don't share names

Two Mirrors only talk if their concepts meet. Transmutate's alignment does this for one pair at a time, with a person checking. A network needs something like DNS for concepts: a shared, versioned list of concept identifiers that each codebook maps onto, while keeping its own words. Options:

- **No shared list.** Align each pair on demand (by name, alias, then Claude). Simple, but every query pays for it, and errors are silent.
- **Anchor on an existing vocabulary** (Wikidata items, WordNet synsets). Broad, but a subject's "opinion" is rarely the dictionary's.
- **Grow one from the network.** Each approved alignment between two peers is published as a signed mapping; mappings that many pairs agree on become shared identifiers. This mirrors the earlier decision that the relation vocabulary must be derived from the material, not hand-picked.

The third fits the project best, and it can start as the first.

## 6. Risks, which are larger than for an ordinary profile

- **A Mirror is a map of how to move someone.** The "Ways to use this" step already names a lever, a route and an outcome. Published, that is a manipulation guide. Layer 3 should not be public by default, a peer should see who asked what, and "Ways to use this" should run only for the owner.
- **Impersonation.** "As the subject" speaks in your voice. Every answer must be marked as coming from a Mirror, with its hash and date, and a peer must never be able to pass one off as the person.
- **Consent.** Marcus and the FOMC are public; a living person's Mirror built from their posts without consent is not acceptable on the network. A bundle should be publishable only by the key of the subject it describes, or, for public institutions and historical texts, marked as made by someone else.
- **Faithfulness is unproven.** Whether a Mirror matches its subject is still the open question of the knowledge V&V and causal V&V work. A network would make a wrong Mirror travel further. Until that is settled, every answer should carry the support behind it.
- **Sybils and spam.** Keys are free, so reputation from section 3G can be gamed with many fake Mirrors endorsing each other. A web of trust (endorsements count only through peers you already trust) limits it; nothing removes it.
- **Staleness.** A Mirror is a snapshot of a corpus. Answers should say how old it is.
- **Cost.** Every "as the subject" answer is a Claude call. The free operations (Trace, follows, compare, codebook routing) should do most of the network's work.

## 7. Small experiments, all possible on one computer

1. **Two local peers.** Run two servers, one with Marcus and one with Epictetus, and add a route that sends an Ask, Trace or compare to the other server and shows the answer beside your own. Tests section 3A and the message format with no new infrastructure.
2. **Signed bundles.** Export a Mirror as one signed, hashed file with a pointer to its previous version; import it read-only on another peer. Tests identity and history.
3. **Codebook routing.** With the four existing corpora as four peers, publish only codebooks and route "who holds that x lowers y?" to the peers that have x and y. Measure how often name and alias matching finds the right peer before Claude has to align.
4. **An owned feed.** Score a day of RSS items against one Mirror's values and weakest basis (section 3C) and see whether the ranking is one a person would endorse.
5. **Negotiation.** Transmutate Marcus with Epictetus, and the FOMC with the DGN guidelines, without approving, and check whether the tensions it lists are the real points of disagreement.

## 8. Questions for whoever picks this up

- Which internet function matters most to replace first? Section 3A is the cheapest to test; 3B and 3C change the most.
- Does a Mirror travel, or stay home and answer? It decides the privacy model and who pays.
- Who may publish a Mirror of whom?
