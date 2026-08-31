---
name: mom-meeting-intelligence
description: >
  Use this skill whenever analyzing, summarizing, or extracting Minutes of Meeting
  (MOM) from meeting audio, video, or spoken transcripts. Activate this skill to extract
  executive summaries, action items with owners and deadlines, formal decisions,
  risks with mitigations, requirements, or agenda topics — even if the user simply provides
  a raw conversation and asks "what was discussed?", "who does what?", or "generate notes"
  without explicitly mentioning "MOM". Do NOT use for general non-meeting text summaries
  or raw audio file format conversions.
---

# Minutes of Meeting (MOM) Enterprise Intelligence Engine

This skill defines the authoritative operating procedures, behavioral constraints, and quality standards for extracting enterprise-grade Minutes of Meeting (MOM) from spoken transcripts.

---

## 1. Core Universal Mandates (Mandatory for ALL Agents)

Every specialist agent MUST adhere to these three foundational laws:

1. **Zero Hallucination & Strict Grounding**:
   - Every extractable item must be strictly anchored in what participants actually said.
   - **NEVER** invent fictitious Jira backlog tickets, Okta configurations, PostgreSQL database plans, or phantom budgets.
   - If an attribute (such as deadline or owner) was not explicitly agreed upon or mentioned, set it to `null` or omit it. Do NOT guess.

2. **Anti-Copy-Paste Synthesis (Reframe Real Work)**:
   - **NEVER** echo verbatim conversational rambling, fillers, hesitations, or disfluencies as deliverables.
   - Distinguish strictly between **What needs to be done** (synthesized into concise imperative corporate English) and **What was said** (preserved as supporting evidence).
   - Filter out spoken stutters (`"if if"`, `"we we"`), filler words (`"yadda yadda"`, `"kind of"`, `"like you know"`), and unfinished trailing fragments (`"to like"`, `"and that"`).

3. **Conservative Extraction over Fabrication**:
   - If a 45-minute meeting had only 2 concrete decisions, return exactly 2 decisions.
   - If no valid items exist for a category (e.g., no risks were discussed), return an empty array `[]`. **Never** generate placeholder items to make the report look full.

---

## 2. Gotchas & Anti-Patterns (Pre-Emptive Error Prevention)

Consult `references/negative_patterns.md` for a comprehensive catalogue of rejected speech patterns.

- **Grammar/Vocabulary Discussions are NOT Tasks**:
  Statements analyzing words (e.g., *"Manage is even less descriptive than that"*, *"Track is a better word"*) are linguistic comments, NOT action items.
- **Incomplete Thoughts are NOT Tasks**:
  Sentences ending in prepositions or hesitations (e.g., *"Track people down. So I think if if this team can take the mission to like"*) are incomplete speech fragments and must be rejected.
- **Agreement Banter is NOT a Formal Decision**:
  Do NOT prepend `"Approved:"` to casual phrases like *"That's crazy. Yeah, and this is something we don't do, which I agree."* A decision must be a concrete ratified architectural choice, budget approval, or operational consensus.
- **Whole-Word Matching for Classifications**:
  When classifying meeting types, check `references/meeting_taxonomy.md`. Never classify a meeting as `HR` because of substring matches in words like *"three"* or *"throughout"*.

---

## 3. Specialist Agent Operating Standards

### Agent 1: Action Extraction Agent (`AgentName.ACTION`)
- **Primary Goal**: Extract discrete, executable, imperative task commitments.
- **Schema**:
  ```json
  {
    "task": "Active imperative verb + concise deliverable (e.g., 'Prepare technical architecture documentation')",
    "owner": "Accountable stakeholder name or null",
    "deadline": "Normalized deadline (e.g., 'Next Friday') or null",
    "priority": "High | Medium | Low",
    "evidence": "Verbatim transcript sentence proving commitment"
  }
  ```
- **Task Phrasing Rules**:
  - Must begin with an active imperative verb: `Prepare`, `Review`, `Deploy`, `Investigate`, `Finalize`, `Audit`, `Configure`, `Create`, `Update`.
  - Strip conversational preamble (*"I think I will"*, *"Can someone please"*).

### Agent 2: Decision & Governance Agent (`AgentName.DECISION`)
- **Primary Goal**: Extract formal organizational, technical, and business resolutions agreed upon by stakeholders.
- **Schema**:
  ```json
  {
    "description": "Clear declarative statement of the agreed resolution",
    "approved_by": ["List of approving stakeholders"],
    "impact": "Direct operational or technical consequence of the decision",
    "evidence_quote": "Verbatim dialogue quote showing consensus"
  }
  ```

### Agent 3: Executive Summary Agent (`AgentName.SUMMARY`)
- **Primary Goal**: Produce an executive boardroom brief adhering to the Minto-Pyramid principle.
- **Structure**:
  - **Paragraph 1 (Context & Intent)**: Strategic context, kickoff objectives, and attending stakeholders.
  - **Paragraph 2 (Debates & Consensus)**: Core architecture choices, trade-offs evaluated, and key resolutions.
  - **Paragraph 3 (Forward Trajectory)**: Primary operational deliverables, sprint commitments, and milestone targets.
  - **Key Points**: 3 to 5 punchy bullet points (under 12 words each) highlighting major breakthroughs.

### Agent 4: Meeting Understanding Agent (`AgentName.MEETING_UNDERSTANDING`)
- **Primary Goal**: Classify the session into its canonical domain archetype using `references/meeting_taxonomy.md`.
- **Allowed Types**: `technical`, `scrum`, `product`, `marketing`, `sales`, `client`, `strategy`, `executive`, `interview`, `hr`, `general`.
- **Rationale**: Must cite specific domain terminology used in the discussion to justify the classification.

### Agent 5: Risk Control & Mitigation Agent (`AgentName.RISK`)
- **Primary Goal**: Surface genuine operational bottlenecks, security vulnerabilities, or dependencies discussed.
- **Schema**:
  ```json
  {
    "description": "Concise statement of the technical or operational risk",
    "severity": "high | medium | low",
    "mitigation": "Concrete action or safeguard discussed to alleviate risk",
    "owner": "Accountable lead"
  }
  ```

### Agent 6: Requirements Specification Agent (`AgentName.REQUIREMENT`)
- **Primary Goal**: Extract explicit functional and technical specifications agreed upon.
- **Syntax**: `[Functional Requirement]` or `[Technical Requirement]` followed by precise capabilities.

### Agent 7: Agenda & Topic Breakdown Agent (`AgentName.TOPIC`)
- **Primary Goal**: Cluster discussions into structured business workstreams.
- **Naming Rule**: Must be formal, concise corporate topic names (e.g., *"Q3 Marketing Campaign & Messaging Framework"*). Never output truncated raw subtitle strings like `"Topic: it can record. And we don't have a..."`.

### Agents 8–10: Deadlines, Open Questions & Follow-Ups
- **Deadlines**: Absolute calendar dates or explicit relative milestones (*"by end of sprint"*, *"by Friday EOD"*).
- **Questions**: Real unresolved technical or business dilemmas requiring investigation.
- **Follow-Ups**: Discrete forward-looking milestone syncs or demonstration sessions.

---

## 4. Self-Audit Checklist Before Outputting JSON

Before returning the final JSON payload, run this internal verification:
- [ ] Are all action items formulated as active, imperative instructions?
- [ ] Are all raw quotes quarantined strictly inside `evidence` or `evidence_quote` fields?
- [ ] Did you reject all conversational fillers, stutters, and trailing fragments?
- [ ] Did you verify that no invented technologies, Jira numbers, or budgets are present?
- [ ] Is the output valid, well-formed JSON matching the exact schema?
