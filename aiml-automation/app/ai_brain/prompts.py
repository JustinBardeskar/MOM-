import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from app.ai_brain.models import AgentName, MeetingType


class SkillPromptLoader:
    """Dynamically loads and injects enterprise rules from the skills/ folder into agent prompts."""

    @classmethod
    def _find_skill_dir(cls) -> Path | None:
        candidates = [
            Path(__file__).resolve().parent.parent.parent / "skills" / "mom-meeting-intelligence",
            Path.cwd() / "skills" / "mom-meeting-intelligence",
            Path.cwd().parent / "skills" / "mom-meeting-intelligence",
        ]
        for c in candidates:
            if c.exists() and (c / "SKILL.md").exists():
                return c
        return None

    @classmethod
    def load_agent_skill(cls, agent: AgentName) -> str:
        """Extracts targeted instructions, mandates, and anti-patterns for the given agent."""
        skill_dir = cls._find_skill_dir()
        if not skill_dir:
            return ""

        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            return ""

        try:
            content = skill_file.read_text(encoding="utf-8")
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    content = parts[2].strip()

            # 1. Extract Core Universal Mandates & Gotchas
            mandates_match = re.search(r"(## 1\. Core Universal Mandates.*?)(?=## 3\.)", content, re.DOTALL)
            universal_text = mandates_match.group(1).strip() if mandates_match else ""

            # 2. Extract Agent-Specific Section
            agent_patterns = {
                AgentName.ACTION: r"(### Agent 1: Action Extraction Agent.*?)(?=### Agent|\Z)",
                AgentName.DECISION: r"(### Agent 2: Decision & Governance Agent.*?)(?=### Agent|\Z)",
                AgentName.SUMMARY: r"(### Agent 3: Executive Summary Agent.*?)(?=### Agent|\Z)",
                AgentName.MEETING_UNDERSTANDING: r"(### Agent 4: Meeting Understanding Agent.*?)(?=### Agent|\Z)",
                AgentName.RISK: r"(### Agent 5: Risk Control & Mitigation Agent.*?)(?=### Agent|\Z)",
                AgentName.REQUIREMENT: r"(### Agent 6: Requirements Specification Agent.*?)(?=### Agent|\Z)",
                AgentName.TOPIC: r"(### Agent 7: Agenda & Topic Breakdown Agent.*?)(?=### Agent|\Z)",
                AgentName.DEADLINE: r"(### Agents 8–10: Deadlines, Open Questions & Follow-Ups.*?)(?=## 4\.|\Z)",
                AgentName.QUESTION: r"(### Agents 8–10: Deadlines, Open Questions & Follow-Ups.*?)(?=## 4\.|\Z)",
                AgentName.FOLLOW_UP: r"(### Agents 8–10: Deadlines, Open Questions & Follow-Ups.*?)(?=## 4\.|\Z)",
            }
            spec_pat = agent_patterns.get(agent)
            spec_text = ""
            if spec_pat:
                m = re.search(spec_pat, content, re.DOTALL)
                if m:
                    spec_text = m.group(1).strip()

            # 3. Extract Self-Audit Checklist
            audit_match = re.search(r"(## 4\. Self-Audit Checklist.*)", content, re.DOTALL)
            audit_text = audit_match.group(1).strip() if audit_match else ""

            # 4. Load Negative Patterns reference if available
            neg_file = skill_dir / "references" / "negative_patterns.md"
            neg_text = ""
            if neg_file.exists():
                try:
                    neg_text = neg_file.read_text(encoding="utf-8").strip()
                except Exception:
                    pass

            sections = [
                "-----------------------------------------------",
                "ENTERPRISE SKILL SPECIFICATION: mom-meeting-intelligence",
                "-----------------------------------------------",
            ]
            if universal_text:
                sections.append(universal_text)
            if spec_text:
                sections.append(spec_text)
            if neg_text:
                sections.append("FORBIDDEN ANTI-PATTERNS & CHATTER TO REJECT:\n" + neg_text)
            if audit_text:
                sections.append(audit_text)

            return "\n\n".join(sections)
        except Exception:
            return ""


@dataclass(frozen=True)
class PromptTemplate:
    agent: AgentName
    version: str
    objective: str
    rules: list[str]


class PromptManager:
    """Owns versioned prompts; agents never embed provider-specific instructions."""

    def __init__(self, version: str) -> None:
        self._templates = self._build_templates(version)

    @classmethod
    def _get_concise_schema(cls, agent: AgentName, response_model: type[BaseModel]) -> str:
        schemas = {
            AgentName.MEETING_UNDERSTANDING: '{"meeting_type": "technical|scrum|product|marketing|sales|client|strategy|executive|board|interview|hr|general", "rationale": "1-2 sentence executive explanation", "confidence": 0.95}',
            AgentName.SUMMARY: '{"executive_summary": "2-3 polished executive paragraphs", "key_points": ["Milestone 1", "Milestone 2", "Milestone 3"], "confidence": 0.95}',
            AgentName.ACTION: '{"actions": [{"task": "Imperative task description", "owner": "Assigned person", "deadline": "Due date/time", "priority": "High|Medium|Low", "evidence": "Verbatim quote"}], "confidence": 0.95}',
            AgentName.DECISION: '{"decisions": [{"description": "Formal agreed decision", "rationale": "Reason for decision", "approved_by": ["Owner name"], "confidence": 0.95}], "confidence": 0.95}',
            AgentName.RISK: '{"risks": [{"description": "Identified risk", "severity": "low|medium|high", "mitigation": "Mitigation strategy", "owner": "Owner name", "confidence": 0.95}], "blockers": [], "confidence": 0.95}',
            AgentName.TOPIC: '{"topics": [{"name": "Topic Name", "summary": "Key discussion takeaway", "keywords": ["kw1", "kw2"], "duration_percent": 25.0}], "confidence": 0.95}',
            AgentName.REQUIREMENT: '{"requirements": [{"description": "Technical or product requirement", "category": "functional|non-functional", "priority": "Must|Should|Could", "requested_by": "Name"}], "confidence": 0.95}',
            AgentName.DEADLINE: '{"deadlines": [{"source_text": "Mentioned timeframe", "normalized_date": "YYYY-MM-DD", "deadline_text": "Timeframe", "owner": "Name", "related_action": "Task"}], "confidence": 0.95}',
            AgentName.QUESTION: '{"open_questions": [{"question": "Unresolved question", "raised_by": "Name", "status": "open", "assigned_to": "Name"}], "confidence": 0.95}',
            AgentName.FOLLOW_UP: '{"follow_up_tasks": [{"description": "Follow-up deliverable", "owner": "Name", "suggested_timeframe": "Timeframe"}], "confidence": 0.95}',
            AgentName.SENTIMENT: '{"overall": "Positive|Neutral|Constructive", "client_mood": "Mood description", "team_mood": "Mood description", "evidence": [], "confidence": 0.95}',
        }
        return schemas.get(agent) or json.dumps(response_model.model_json_schema(), separators=(",", ":"))

    def render(
        self,
        agent: AgentName,
        response_model: type[BaseModel],
        context: str,
        memory: str,
        meeting_type: MeetingType | None,
        validation_feedback: str | None = None,
    ) -> tuple[str, str]:
        template = self._templates[agent]
        schema = self._get_concise_schema(agent, response_model)
        rules_text = "\n".join(f"- {rule}" for rule in template.rules)
        skill_instructions = SkillPromptLoader.load_agent_skill(agent)
        
        if agent == AgentName.ACTION:
            system = (
                "You are the ACTION EXTRACTION AGENT in the MOM system.\n\n"
                "Your ONLY responsibility is to identify concrete, actionable tasks from the meeting transcript.\n\n"
                "═══════════════════════════════════════════════\n"
                "CRITICAL ANTI-COPY-PASTE MANDATE\n"
                "═══════════════════════════════════════════════\n"
                "NEVER copy-paste or parrot raw conversational dialogue from the transcript.\n"
                "NEVER use the exact spoken words as the task description.\n"
                "You MUST SYNTHESIZE and REFRAME every action into a clear, professional, imperative work instruction.\n"
                "The 'task' field must describe WHAT SPECIFIC WORK needs to be done — NOT what someone said.\n\n"
                "═══════════════════════════════════════════════\n"
                "TASK FRAMING EXAMPLES (MANDATORY PATTERN)\n"
                "═══════════════════════════════════════════════\n\n"
                "Example 1:\n"
                "  TRANSCRIPT: \"Rahul: Yeah so I'll take a look at the configuration stuff, the SSO thing, and like try to get that sorted out\"\n"
                "  CORRECT task: \"Investigate and resolve SSO configuration issues\"\n"
                "  WRONG task: \"Take a look at the configuration stuff the SSO thing and try to get that sorted out\"\n\n"
                "Example 2:\n"
                "  TRANSCRIPT: \"We need to make sure the deployment thing gets done before the end of the sprint\"\n"
                "  CORRECT task: \"Complete the pending deployment\"\n"
                "  WRONG task: \"Make sure the deployment thing gets done\"\n\n"
                "Example 3:\n"
                "  TRANSCRIPT: \"Priya mentioned she'll handle the client feedback from last week's demo and get back to them\"\n"
                "  CORRECT task: \"Address and incorporate client feedback from the demo session\"\n"
                "  WRONG task: \"Handle the client feedback from last week's demo and get back to them\"\n\n"
                "Example 4:\n"
                "  TRANSCRIPT: \"Amit: I will fix the login timeout issue by Friday\"\n"
                "  CORRECT task: \"Fix the login timeout issue\"\n"
                "  WRONG task: \"I will fix the login timeout issue by Friday\"\n\n"
                "Example 5:\n"
                "  TRANSCRIPT: \"So what we agreed is that the security audit report needs to be finalized and sent to compliance\"\n"
                "  CORRECT task: \"Finalize the security audit report and submit to compliance team\"\n"
                "  WRONG task: \"The security audit report needs to be finalized and sent to compliance\"\n\n"
                "═══════════════════════════════════════════════\n"
                "OUTPUT RULES\n"
                "═══════════════════════════════════════════════\n"
                "For every valid action item, determine:\n"
                "1. task: A clear, imperative work instruction (e.g. 'Fix the login timeout issue', 'Deploy Redis caching cluster', 'Prepare Q3 budget report'). Start with an action verb.\n"
                "2. owner: WHO is responsible? Use the person's name. Return null if not explicitly stated.\n"
                "3. deadline: WHEN must it be completed? Return null if not explicitly stated.\n"
                "4. evidence: The verbatim quote from the transcript that supports this action item.\n\n"
                "You are NOT a Summary Agent, Decision Agent, Discussion Agent, or Recommendation Agent.\n"
                "Do not summarize the meeting. Do not convert opinions into actions. Do not invent information.\n\n"
                "REJECTED VAGUE & CONVERSATIONAL NON-ACTIONS (MUST NEVER BE RETURNED):\n"
                "- Descriptive statements about words (e.g. 'Manage is even less descriptive than that') — REJECT\n"
                "- Conversational rambles / trailing fragments (e.g. 'Track people down. So I think if if this team can take the mission to like') — REJECT\n"
                "- Vague phrases: 'Improve things', 'Address security and quality', 'Work on the project', 'Follow up', 'Take necessary action', 'Handle it', 'Do the needful', 'Make things better'\n\n"
                "If the transcript does not contain enough information to identify a concrete action, return an empty actions array.\n\n"
                "⚠️ CRITICAL OUTPUT FORMAT MANDATE ⚠️\n"
                "Your response MUST be EXACTLY ONE valid JSON object.\n"
                "DO NOT add any text, explanation, preamble, or commentary before or after the JSON.\n"
                "DO NOT wrap the JSON in markdown code fences (no ```json or ```).\n"
                "DO NOT include <think> tags, reasoning, or any non-JSON content.\n"
                "START your response with { and END with }.\n"
                "Return ONLY a JSON object conforming strictly to the required JSON Schema."
            )
        else:
            system = (
                "You are a Tier-1 Principal Enterprise Executive Intelligence AI Agent specializing in corporate meeting synthesis.\n"
                "COGNITIVE ANALYSIS & ANTI-COPY-PASTE MANDATE:\n"
                "1. UNDERSTAND FIRST: Analyze the transcript's technical domain, organizational context, operational dependencies, and business stakes.\n"
                "2. SYNTHESIZE & ELEVATE: NEVER copy-paste or parrot raw conversational dialogue, fragmented speech, or spoken filler. Every output item must be logically synthesized into consultant-grade, publication-ready executive language tailored to your specialist role.\n"
                "3. FACTUAL GROUNDING: Anchor your synthesis strictly in the transcript reality. Do not invent commitments, budgets, or stakeholders not discussed.\n"
                "4. CONCRETE SPECIFICITY: Avoid vague statements (e.g. avoid 'work on system'). Use precise domain deliverables (e.g. 'Deploy distributed Redis caching cluster', 'Ratify Tier-1 competitor positioning collateral').\n"
                "5. ⚠️ CRITICAL OUTPUT FORMAT MANDATE ⚠️ Your response MUST be EXACTLY ONE valid JSON object. DO NOT add any preamble, explanation, reasoning, or text before or after the JSON. DO NOT use markdown code fences (no ```json). DO NOT include <think> tags. START your response with { and END with }.\n\n"
                f"SPECIALIST ROLE & OBJECTIVE: {template.objective}\n"
                f"ROLE-SPECIFIC SYNTHESIS RULES:\n{rules_text}\n"
                f"Prompt Engine: v{template.version}"
            )

        if skill_instructions:
            system = f"{system}\n\n{skill_instructions}"
            
        user_parts = [
            f"Meeting Type: {(meeting_type or MeetingType.GENERAL).value.upper()}",
            f"═══ CURRENT MEETING TRANSCRIPT (PRIMARY SOURCE — EXTRACT FROM THIS ONLY) ═══\n\"\"\"\n{context}\n\"\"\"",
        ]

        if agent in [AgentName.ACTION, AgentName.DECISION]:
            try:
                from app.ai_brain.quality import NLPCommitmentAnchorExtractor
                anchors = NLPCommitmentAnchorExtractor.extract_anchors(context)
                if anchors:
                    anchor_lines = [
                        f"• Speaker: '{a.speaker or 'Speaker'}' | Quote: \"{a.cue_text}\" | Candidate Task: \"{a.inferred_task}\" (Due: {a.target_deadline or 'N/A'})"
                        for a in anchors[:6]
                    ]
                    user_parts.append(
                        "═══ DETECTED VERBAL COMMITMENTS & SEED ANCHORS (NLP ACCELERATOR) ═══\n"
                        "The NLP commitment detector identified these candidate verbal anchors in the transcript.\n"
                        "Verify each against the transcript, discard non-commitments, and reframe confirmed tasks into SMART imperative actions:\n"
                        + "\n".join(anchor_lines)
                    )
            except Exception as e:
                pass
        
        if memory and memory.strip() and memory != "No previous meeting context":
            user_parts.append(
                f"═══ HISTORICAL ORGANIZATIONAL MEMORY (PAST MEETINGS — REFERENCE ONLY) ═══\n"
                f"⚠️ STRICT ISOLATION MANDATE:\n"
                f"The items below are past records retrieved from MongoDB memory.\n"
                f"DO NOT extract or report any of these past items as new actions, decisions, or risks for the current meeting.\n"
                f"Only extract items that are explicitly discussed and agreed upon in the CURRENT MEETING TRANSCRIPT above.\n"
                f"\"\"\"\n{memory}\n\"\"\""
            )

        if validation_feedback:
            user_parts.append(
                f"CRITICAL RETRY INSTRUCTION — PREVIOUS OUTPUT FAILED QUALITY AUDIT:\n"
                f"{validation_feedback}\n"
                f"You MUST fix these specific defects. Output strictly valid JSON matching the schema."
            )
        user_parts.append(f"REQUIRED JSON SCHEMA:\n{schema}")
        user = "\n\n".join(user_parts)
        return system, user

    def render_turbo_deliverables(
        self,
        transcript: str,
        schema: str,
        meeting_type: MeetingType | None = None,
    ) -> tuple[str, str]:
        """Renders unified high-speed prompt for Boardroom Deliverables (Summary + Actions + Decisions + Risks)."""
        system = (
            "You are the Lead Executive Deliverables AI Agent for enterprise boardroom Minutes of Meeting.\n"
            "Analyze the meeting transcript and synthesize:\n"
            "1. 'meeting_title': A formal, descriptive 4-7 word title reflecting the core topic/achievement of this session.\n"
            "2. 'executive_summary': A polished 2-3 paragraph executive brief.\n"
            "3. 'key_points': 2-4 concise 1-line milestone takeaways.\n"
            "4. 'action_summary': A single 1-sentence executive overview summarizing the primary post-meeting commitments.\n"
            "5. 'action_items': Extract ONLY genuine, actionable post-meeting work deliverables and commitments.\n"
            "   For each action item provide:\n"
            "   - 'task': Crisp, professional imperative task statement (e.g. 'Run proposed stage name by David DeSanto for feedback').\n"
            "   - 'assigner': Who said / requested / assigned this action (e.g. 'Wayne (Meeting Chair)' or 'Alex'). If self-committed, specify the speaker.\n"
            "   - 'owner': Who is responsible / assigned to complete the work (e.g. 'Wayne' or 'David DeSanto' or 'Team Members').\n"
            "   - 'recipient': Target collaborator or recipient if mentioned (e.g. 'David DeSanto' or 'Compliance Team').\n"
            "   - 'deadline': Explicit deadline mentioned in speech (e.g. 'Tomorrow', 'By Friday', 'Next Sprint') or 'Not specified'.\n"
            "   - 'priority': 'High', 'Medium', or 'Low'.\n"
            "   - 'evidence': Exact quote from the transcript.\n"
            "   🚫 STRICT NEGATIVE MANDATES FOR ACTION ITEMS:\n"
            "   - NEVER extract in-meeting real-time speech/chatter (e.g. 'I was going to mention I will type this in here', 'Verbalize and then I will finish writing', 'Put a note in there', 'Say it now').\n"
            "   - NEVER extract conversational rambling, thoughts, or fragmented phrases (e.g. 'Figure it out', 'At least to me I will admit this', 'The end of the quarter Tiago I cannot remember', 'Link in another issue that is').\n"
            "   - NEVER extract screen-sharing or meeting administrative transitions (e.g. 'Share one thing', 'Show my screen', 'Take a look').\n"
            "   - Ensure every action item is a concrete, professional imperative task statement with an assigned owner, target deadline, and priority.\n"
            "6. 'decisions': Extract ALL ratified agreements, architectural choices, consensus conclusions, policy approvals, budget allocations, scope sign-offs, or resolved approaches established during the meeting.\n"
            "   For each decision provide:\n"
            "   - 'description': Clear, definitive statement of the agreed decision/policy (e.g. 'Adopt proposed stage name Enrichment across security and compliance workflows').\n"
            "   - 'rationale': Why this decision was made / core justification discussed.\n"
            "   - 'approved_by': List of key stakeholders or leaders who approved/ratified it (e.g. ['Wayne (Meeting Chair)', 'David DeSanto'] or ['Alex', 'Team Consensus']).\n"
            "   - 'impact': Expected operational, technical, or strategic impact.\n"
            "   - 'evidence_quote': Verbatim transcript statement showing consensus or approval.\n"
            "7. 'risks': Identified operational/technical risks paired with actionable mitigations and severity.\n\n"
            "⚠️ CRITICAL OUTPUT FORMAT MANDATE: Return ONLY a valid JSON object matching the schema. No markdown fences, no preamble."
        )
        user_parts = [
            f"═══ CURRENT MEETING TRANSCRIPT ═══\n\"\"\"\n{transcript}\n\"\"\"",
        ]
        try:
            from app.ai_brain.quality import NLPCommitmentAnchorExtractor
            anchors = NLPCommitmentAnchorExtractor.extract_anchors(transcript)
            if anchors:
                anchor_lines = [
                    f"• Speaker: '{a.speaker or 'Speaker'}' | Inferred Task: \"{a.inferred_task}\" (Due: {a.target_deadline or 'N/A'})"
                    for a in anchors[:5]
                ]
                user_parts.append(
                    "═══ DETECTED VERBAL COMMITMENT ANCHORS (NLP ACCELERATOR) ═══\n"
                    + "\n".join(anchor_lines)
                )
        except Exception:
            pass

        user_parts.append(f"REQUIRED JSON SCHEMA:\n{schema}")
        return system, "\n\n".join(user_parts)

    def render_turbo_intelligence(
        self,
        transcript: str,
        schema: str,
    ) -> tuple[str, str]:
        """Renders unified high-speed prompt for Context Dynamics & Intelligence (Meeting Type + Sentiment + Topics + Questions)."""
        system = (
            "You are the Intelligence & Context Dynamics AI Agent for enterprise Minutes of Meeting.\n"
            "Analyze the transcript and extract:\n"
            "1. 'meeting_type': One of 'technical', 'scrum', 'product', 'marketing', 'sales', 'client', 'strategy', 'executive', 'board', 'interview', 'hr', 'general'.\n"
            "2. 'rationale': 1-2 sentence executive explanation of WHY this classification was chosen.\n"
            "3. 'sentiment': Multi-dimensional sentiment intelligence with overall tone, engagement_level ('High'/'Medium'/'Low'), polarity_score (-1.0 to 1.0), friction_points, alignment_signals, and per-speaker sentiments.\n"
            "4. 'topics': 2-5 structured agenda topics with name, description, time_spent_percent, sentiment, and key_speakers.\n"
            "5. 'requirements': Functional or technical specifications discussed.\n"
            "6. 'open_questions': Unresolved questions needing future clarification.\n"
            "7. 'follow_up_tasks': Next meeting action items and 'next_meeting_agenda' bullet points.\n\n"
            "⚠️ CRITICAL OUTPUT FORMAT MANDATE: Return ONLY a valid JSON object matching the schema. No markdown fences, no preamble."
        )
        user = f"═══ CURRENT MEETING TRANSCRIPT ═══\n\"\"\"\n{transcript}\n\"\"\"\n\nREQUIRED JSON SCHEMA:\n{schema}"
        return system, user

    @staticmethod
    def _build_templates(version: str) -> dict[AgentName, PromptTemplate]:
        return {
            AgentName.MEETING_UNDERSTANDING: PromptTemplate(
                agent=AgentName.MEETING_UNDERSTANDING,
                version=version,
                objective="Classify the meeting type, extract the central theme, and synthesize a formal meeting title based on transcript content.",
                rules=[
                    "Analyze the discussion content, terminology, participant roles, and conversational goals to classify into EXACTLY one of: 'technical', 'scrum', 'product', 'marketing', 'sales', 'client', 'strategy', 'executive', 'board', 'interview', 'hr', or 'general'.",
                    "Classification Taxonomy Guidelines:",
                    "  • 'technical': System architecture, APIs, databases (Postgres/Redis), DevOps, code, infrastructure, penetration testing, security, bug remediation.",
                    "  • 'scrum': Sprint updates, daily standups, Jira tickets, story points, sprint backlog, blockers, sprint planning.",
                    "  • 'product': Product feature roadmap, wireframes, Figma, user experience, onboarding funnel, release notes, user journey.",
                    "  • 'marketing': Go-to-market (GTM), campaigns, LinkedIn, product launch, brand messaging, announcements, press releases.",
                    "  • 'sales': Enterprise pricing, subscription tiers, customer contracts, sales enablement, deals, pipeline, ARR/MRR.",
                    "  • 'client': External customer sync, client requirements, delivery SLA, customer milestone review, SOW review.",
                    "  • 'strategy': Business strategy, market positioning, expansion, organizational realignment, funding, M&A.",
                    "  • 'executive': C-level leadership meeting, VP alignment, board prep, corporate performance.",
                    "  • 'board': Board of directors, fiduciary governance, investor voting, equity/dividends.",
                    "  • 'interview': Candidate interview, technical hiring screen, HR screening.",
                    "  • 'hr': People operations, performance reviews, employee policy, benefits, headcount, onboarding.",
                    "  • 'general': Cross-functional catch-up, informational sharing, team sync without a single domain focus.",
                    "In 'meeting_title', synthesize a formal, executive-grade 4-7 word title reflecting the ACTUAL specific topic and achievement of this session (e.g. 'PostgreSQL Optimization & Redis Architecture Review', 'Q3 User Onboarding & Figma Wireframe Review'). DO NOT output generic titles like 'Meeting', 'Sync', or 'Direct Transcript Sync'.",
                    "In 'theme', provide a 1-sentence executive summary of the central mission/topic of this session.",
                    "In 'rationale', provide a concise 1-2 sentence executive explanation of WHY this classification was chosen based on evidence from the transcript.",
                    "If the audio is silence or background noise with no intelligible speech, classify as 'general' with confidence 0.1.",
                ],
            ),
            AgentName.SUMMARY: PromptTemplate(
                agent=AgentName.SUMMARY,
                version=version,
                objective="Synthesize an executive-ready business summary and key outcome bullet points based STRICTLY on the current meeting transcript.",
                rules=[
                    "The 'meeting_summary' MUST summarize ONLY the discussion and decisions that occurred in the CURRENT transcript provided above.",
                    "NEVER mention, echo, or blend in past meeting summaries or historical context.",
                    "Paragraph 1: Strategic context, primary meeting objectives, and core agenda.",
                    "Paragraph 2: Key debates, technical/business consensus, architecture choices, and milestone agreements.",
                    "Paragraph 3: Forward-looking next steps, operational commitments, and ownership trajectory.",
                    "In 'suggested_title', provide a professional 4-6 word title summarizing what was accomplished in this meeting.",
                    "Do NOT summarize every conversation turn or re-narrate meeting dialogue. State strictly the core business takeaway and finalized outcome.",
                    "The 'key_points' list must contain 2-4 ultra-short bullet points (strictly 1 line each, max 10 words per point).",
                    "Eliminate all filler, transitional phrases, preamble, and conversational fluff.",
                ],
            ),
            AgentName.ACTION: PromptTemplate(
                agent=AgentName.ACTION,
                version="3.1.0",
                objective="Identify, reframe, and extract concrete post-meeting tasks and synthesize an executive 1-line action summary.",
                rules=[
                    "Extract ONLY real, meaningful post-meeting business deliverables, engineering tasks, or operational assignments.",
                    "STRICT ANTI-FILLER MANDATE: NEVER extract in-meeting conversational speech transitions as tasks (e.g. DO NOT extract 'Share one thing', 'Show one thing', 'Let me share my screen', 'Point out something', 'Mention one thing', 'Say a few words', 'Give an update', 'Take a look').",
                    "Determine WHO is responsible (owner), WHAT exactly needs to be done (task), WHEN it must be done (deadline), and supporting evidence.",
                    "Reframe each task into a clear, imperative statement starting with a strong verb (e.g. 'Deploy Redis cluster', 'Submit security audit report', 'Audit database connection pooling').",
                    "In 'action_summary', provide a single concise 1-sentence executive overview summarizing the primary commitments made across the meeting (e.g. 'Engineering team committed to deploying Redis caching and auditing connection pools ahead of Friday cutoff.').",
                    "If owner is not explicitly identifiable, return null.",
                    "If deadline is not explicitly identifiable, return null.",
                    "Reject vague actions ('Improve things', 'Address security', 'Work on project', 'Follow up', 'Handle it', 'Do the needful').",
                    "If the conversation is casual discussion with no post-meeting work commitments, return an empty actions array: 'action_items': [] and 'action_summary': 'No pending post-meeting deliverables were assigned in this discussion.'",
                ],
            ),
            AgentName.DECISION: PromptTemplate(
                agent=AgentName.DECISION,
                version=version,
                objective="Extract final decisions, architectural approvals, budget approvals, and consensus.",
                rules=[
                    "STRICT RELEVANCE MANDATE: Extract ONLY genuine agreed decisions, ratified resolutions, or budget/architectural approvals.",
                    "DO NOT extract ongoing discussions, ideas, questions, proposals, or general status updates as decisions.",
                    "Keep each decision 'description' short, sharp, and imperative (maximum 1 sentence, 6-15 words, e.g. 'Approved $50K Q3 infrastructure budget', 'Adopted PostgreSQL 16 as primary database', 'Finalized launch date for September 15').",
                    "NEVER copy informal conversational speech (e.g. DO NOT write 'I agree', 'We discussed', 'Yeah we agreed').",
                    "If no formal decision was finalized in the meeting, return an empty list: 'decisions': [].",
                ],
            ),
            AgentName.REQUIREMENT: PromptTemplate(
                agent=AgentName.REQUIREMENT,
                version=version,
                objective="Extract functional and non-functional requirements discussed.",
                rules=[
                    "State requirements clearly in formal specification language (e.g. 'The system must support sub-50ms query latency under simulated peak concurrent load').",
                    "Categorize requirements as 'functional' or 'technical'.",
                    "Assign priority 'High', 'Medium', or 'Low' based on urgency and criticality.",
                ],
            ),
            AgentName.RISK: PromptTemplate(
                agent=AgentName.RISK,
                version=version,
                objective="Extract risks, blockers, mitigations, and owners.",
                rules=[
                    "Formulate risks as professional risk statements (e.g. 'Query latency degradation under peak concurrent load', 'Communication misalignment between Slack discussions and Jira issue tracking').",
                    "NEVER echo raw spoken filler (e.g. DO NOT write 'etc etc etc', 'On slack only'). Formulate the underlying operational or technical vulnerability.",
                    "Provide a concrete, actionable mitigation strategy and identify the accountable owner.",
                ],
            ),
            AgentName.SENTIMENT: PromptTemplate(
                agent=AgentName.SENTIMENT,
                version=version,
                objective="Assess client and team mood, tension points, alignment signals, and chronological emotional shifts.",
                rules=[
                    "Evaluate 'overall' tone: e.g. 'Constructive & Professional', 'Collaborative & High-Morale', 'Cautious with Initial Skepticism', or 'Tense with Resolving Alignment'.",
                    "Determine 'client_mood' and 'team_mood' based on conversational dynamics and commitment confidence.",
                    "Calculate 'polarity_score' from -1.0 (extremely negative/conflict-ridden) to +1.0 (enthusiastic/fully aligned).",
                    "Assess 'engagement_level' as 'High', 'Moderate', or 'Low'.",
                    "Identify 'friction_points': Concrete concerns, timeline pushbacks, skepticism, or operational bottlenecks raised by participants.",
                    "Identify 'alignment_signals': Unanimous agreements, enthusiastic approvals, and shared commitments.",
                    "Map 'speaker_sentiments': For each active participant, provide a 2-4 word executive tone summary (e.g. {'Sarah': 'Confident & Solution-Oriented', 'David': 'Analytical & Methodical'}).",
                    "Trace 'chronological_shifts': Summarize emotional and alignment progression across Opening, Mid-discussion, and Closing phases.",
                    "Provide specific quotes and verbatim transcript 'evidence' supporting the assessment.",
                ],
            ),
            AgentName.TOPIC: PromptTemplate(
                agent=AgentName.TOPIC,
                version=version,
                objective="Classify discussion topics and summarize each workstream.",
                rules=[
                    "Group dialogue into 3-5 structured agenda topics with formal titles and executive summaries.",
                ],
            ),
            AgentName.DEADLINE: PromptTemplate(
                agent=AgentName.DEADLINE,
                version=version,
                objective="Extract and normalize relative dates and calendar deadlines.",
                rules=[
                    "Capture both relative delivery dates ('next Friday') and standardized calendar dates ('November 15th').",
                ],
            ),
            AgentName.QUESTION: PromptTemplate(
                agent=AgentName.QUESTION,
                version=version,
                objective="Extract unresolved questions and pending inquiries.",
                rules=[
                    "State questions in clear, formal business/technical inquiry language.",
                ],
            ),
            AgentName.FOLLOW_UP: PromptTemplate(
                agent=AgentName.FOLLOW_UP,
                version=version,
                objective="Extract follow-up tasks and next meeting agenda.",
                rules=[
                    "Formulate follow-up tasks as clear operational deliverables and list proposed next meeting topics.",
                ],
            ),
        }
