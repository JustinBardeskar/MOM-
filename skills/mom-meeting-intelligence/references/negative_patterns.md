# Negative Patterns & Rejected Conversational Chatter

All MOM Intelligence Agents must reject the following conversational patterns. Never treat these as deliverables, decisions, or risks.

---

## 1. Meta-Commentary & Copula Grammar Statements
Statements where participants analyze, critique, or discuss words, definitions, or labels rather than assigning work:
- `"Manage is even less descriptive than that"` -> REJECT (Not a task)
- `"Track is a better word to use here"` -> REJECT (Discussion of vocabulary)
- `"This sounds like a feature rather than a bug"` -> REJECT (Categorization opinion)
- `"Is is not the right phrasing"` -> REJECT

## 2. Trailing Incomplete Fragments
Sentences that end abruptly on prepositions, conjunctions, or conversational hesitations:
- `"Track people down. So I think if if this team can take the mission to like"` -> REJECT (Incomplete fragment ending in "to like")
- `"We should probably look into that and like"` -> REJECT
- `"Let's see if we can do something with that or"` -> REJECT
- `"To make sure that we"` -> REJECT

## 3. Conversational Filler & Idle Thoughts
Spoken stream-of-consciousness remarks containing no commitment:
- `"Kind of the interplay here"` -> REJECT
- `"Yadda, yadda, yadda"` -> REJECT
- `"You know what I mean"` -> REJECT
- `"Sort of like that"` -> REJECT
- `"I don't know, maybe we could"` -> REJECT

## 4. Spoken Stutters & Speech Disfluencies
Verbatim verbal hiccups must never appear in extracted tasks:
- `"if if"`
- `"we we"`
- `"to to"`
- `"that that"`

## 5. Casual Agreement Banter (Not Formal Decisions)
Spoken reactions that do not constitute a ratified business or technical decision:
- `"That's crazy. Yeah, and this is something we don't do, which I agree."` -> REJECT
- `"Yeah, totally agree on that one."` -> REJECT
- `"Some bodies one or somebody's three"` -> REJECT
- `"Sounds good to me"` -> REJECT (Unless tied to a specific proposal)

## 6. Fake Boilerplate Hallucinations
Never output these hardcoded placeholders:
- Fake Jira items: `"Lack of documented engineering sign-offs on tracked Jira backlog issues"`
- Fake Okta configs: `"Standardize enterprise authentication on Okta SAML 2.0 architecture"`
- Fake database plans: `"Approve PostgreSQL 16 production database upgrade"`
- Fake budget allocations: `"$60,000 Phase 1 budget allocation"`
