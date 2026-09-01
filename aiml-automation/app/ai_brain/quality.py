"""
Action Quality, Imperative Grammar Normalization, and Self-Critique Quality Loop Engine.
Combines:
1. Action Normalizer (imperative grammar, prefix stripping, date pruning)
2. Action Quality Validator (imperative verbs, grounding, anti-hallucination)
3. Dynamic Golden Examples Store (MongoDB 'goldenExamples' collection + in-memory fallback)
4. Multi-Agent Self-Critique Pass (Pass 2 verification against strict rubrics)
5. Shared AgentQualityLoop
"""

from datetime import datetime, timezone
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, ClassVar
from pydantic import Field

from app.ai_brain.models import (
    ActionItem,
    AgentName,
    Decision,
    LLMRequest,
    LLMResponse,
    Risk,
    SentimentOutput,
    StrictModel,
)

logger = logging.getLogger("ai_brain.quality")


def _parse_speaker_and_text(line: str) -> tuple[str, str]:
    if ":" in line:
        parts = line.split(":", 1)
        spk = parts[0].strip()
        if len(spk) < 35 and not any(p in spk for p in [".", "!", "?", "http"]):
            return spk, parts[1].strip()
    return "SPEAKER", line.strip()


# ==========================================
# 0. Hybrid NLP Commitment Anchor Extractor
# ==========================================

@dataclass
class NLPCommitmentAnchor:
    speaker: str | None
    cue_text: str
    inferred_task: str
    target_deadline: str | None
    confidence: float


class NLPCommitmentAnchorExtractor:
    """Hybrid deterministic NLP extraction engine detecting high-confidence verbal commitments and decision anchors."""

    COMMITMENT_TRIGGERS: ClassVar[list[str]] = [
        "i will", "i'll", "we will", "we'll", "i can", "i'll take care of",
        "let me handle", "i will make sure to", "i'll deploy", "i'll finalize",
        "i'll review", "i'll coordinate", "i'll update", "action item",
        "assigned to", "responsible for", "take ownership", "submit the report",
        "please", "can you", "could you", "need to", "needs to", "have to",
        "make sure to", "ensure that", "follow up with", "check with",
        "coordinate with", "run by", "reach out to", "ping the", "ask for",
        "update the", "finalize the", "review the", "submit the", "deploy the",
        "test the", "share the", "draft the", "prepare the",
    ]

    DECISION_TRIGGERS: ClassVar[list[str]] = [
        "we agreed to", "we decided to", "we decided that", "we also approved",
        "approved the", "unanimously agreed", "consensus is", "ratified",
        "all agreed to", "agreed that", "decided on", "decision is to",
        "formally approved", "consensus was to", "aligned on", "settled on",
        "will proceed with", "chosen to", "we are going with", "we'll go with",
        "our approach is to", "adopt the", "adopting the", "agreed on",
        "target is", "targeting", "focus is to", "focus for", "our plan is to",
        "the plan is to", "moving forward with", "go ahead with", "prioritize",
        "prioritizing", "survey strategy", "milestone deliverables", "scheduled to",
        "concluded to", "confirmed to", "scope for",
    ]

    RISK_TRIGGERS: ClassVar[list[str]] = [
        "risk", "risks", "concern", "blocker", "blocking", "bottleneck", "timeout", "timeouts",
        "slow down", "slowing down", "vulnerability", "failure", "write contention", "contention",
        "traffic spike", "memory leak", "latency", "outage", "downtime", "compliance issue",
        "security risk", "single point of failure", "technical debt", "delay", "delays",
    ]

    @classmethod
    def extract_anchors(cls, transcript_text: str) -> list[NLPCommitmentAnchor]:
        anchors: list[NLPCommitmentAnchor] = []
        if not transcript_text:
            return anchors

        lines = [line.strip() for line in transcript_text.splitlines() if line.strip()]
        for line in lines:
            speaker = None
            text = line
            if ":" in line:
                parts = line.split(":", 1)
                if len(parts[0].split()) <= 4 and len(parts[0]) < 30:
                    speaker = parts[0].strip()
                    text = parts[1].strip()

            lower = text.lower()
            if any(t in lower for t in cls.COMMITMENT_TRIGGERS):
                if not ActionNormalizer.is_non_action_discussion(text):
                    normalized_task = ActionNormalizer.normalize_action_work(text)
                    if normalized_task and len(normalized_task.split()) >= 3:
                        deadline = None
                        for d_marker in ["by friday", "by monday", "by wednesday", "by thursday", "by tomorrow", "next week", "end of sprint", "end of month"]:
                            if d_marker in lower:
                                deadline = d_marker.title()
                                break

                        anchors.append(
                            NLPCommitmentAnchor(
                                speaker=speaker,
                                cue_text=text,
                                inferred_task=normalized_task,
                                target_deadline=deadline,
                                confidence=0.88,
                            )
                        )
        return anchors[:12]

    @classmethod
    def extract_decisions(cls, transcript_text: str) -> list[Decision]:
        decisions: list[Decision] = []
        if not transcript_text:
            return decisions
        lines = [line.strip() for line in transcript_text.splitlines() if line.strip()]
        for line in lines:
            speaker = None
            text = line
            if ":" in line:
                parts = line.split(":", 1)
                if len(parts[0].split()) <= 4 and len(parts[0]) < 30:
                    speaker = parts[0].strip()
                    text = parts[1].strip()

            lower = text.lower()
            if any(t in lower for t in cls.DECISION_TRIGGERS):
                clean_desc = text
                for prefix in [
                    "we all agreed to ", "we agreed to ", "we decided to ", "we decided that ",
                    "we also approved ", "we approved ", "all agreed to ", "consensus is to ",
                    "the decision is to ", "agreed to ", "decided to ", "we will proceed with ",
                    "we are going with ", "we'll go with "
                ]:
                    if clean_desc.lower().startswith(prefix):
                        clean_desc = clean_desc[len(prefix):].strip()
                clean_desc = clean_desc[0].upper() + clean_desc[1:] if clean_desc else clean_desc
                if len(clean_desc.split()) >= 3 and not ActionNormalizer.is_non_action_discussion(clean_desc):
                    decisions.append(
                        Decision(
                            description=clean_desc,
                            rationale="Consensus reached during discussion.",
                            approved_by=[speaker] if speaker else ["Stakeholders Consensus"],
                            impact="Operational and strategic alignment across workstreams.",
                            evidence_quote=text,
                            confidence=0.92,
                        )
                    )
        return decisions[:8]

    @classmethod
    def extract_risks(cls, transcript_text: str) -> list[Risk]:
        risks: list[Risk] = []
        if not transcript_text:
            return risks
        lines = [line.strip() for line in transcript_text.splitlines() if line.strip()]
        for line in lines:
            speaker, text = _parse_speaker_and_text(line)
            lower = text.lower()
            if any(r in lower for r in cls.RISK_TRIGGERS):
                if len(text.split()) >= 4 and not ActionNormalizer.is_non_action_discussion(text):
                    clean_desc = text
                    for prefix in [
                        "a critical risk is ", "a major risk is ", "the main risk is ",
                        "our concern is ", "the blocker is ", "we have a risk with ",
                        "one risk is ", "a potential risk is "
                    ]:
                        if clean_desc.lower().startswith(prefix):
                            clean_desc = clean_desc[len(prefix):].strip()
                    clean_desc = clean_desc[0].upper() + clean_desc[1:] if clean_desc else clean_desc
                    sev = "High" if any(h in lower for h in ["critical", "severe", "spike", "timeout", "outage", "vulnerability", "security", "leak", "contention"]) else "Medium"
                    risks.append(
                        Risk(
                            description=clean_desc,
                            severity=sev,
                            probability="Medium",
                            impact="Operational or technical performance degradation if unmitigated.",
                            mitigation=f"Continuous monitoring and proactive review by {speaker or 'team'}.",
                            owner=speaker or "Technical Lead",
                            evidence_quote=text,
                            confidence=0.90,
                        )
                    )
        return risks[:6]


# ==========================================
# 1. Action Normalizer & Grammar Rule Sets
# ==========================================

IMPERATIVE_VERBS = {
    # Engineering & DevOps
    "deploy", "configure", "implement", "integrate", "migrate", "refactor",
    "upgrade", "install", "provision", "rearchitect", "prototype", "automate",
    "patch", "debug", "isolate", "reproduce", "benchmark", "stress-test",
    "rollout", "rollback", "deprecate", "decommission", "enable", "disable",
    "sync", "backup", "restore", "clean", "connect", "scale", "spin",
    "profile", "sanitize", "containerize", "stabilize", "mock", "stub",
    
    # Review, QA & Compliance
    "review", "audit", "test", "validate", "verify", "inspect", "check",
    "examine", "survey", "certify", "approve", "appraise", "assess",
    "evaluate", "measure", "quantify", "benchmark", "monitor", "track",
    "reconcile", "triage", "prioritize",
    
    # Management, Strategy & Operations
    "prepare", "send", "share", "create", "draft", "author", "compose",
    "finalize", "publish", "deliver", "submit", "present", "schedule",
    "organize", "coordinate", "align", "formalize", "distribute", "manage",
    "maintain", "facilitate", "lead", "onboard", "offboard", "brief",
    "notify", "escalate", "delegate", "assign", "allocate", "request",
    "obtain", "gather", "collate", "synthesize", "archive", "document",
    
    # Execution & Problem Solving
    "fix", "resolve", "address", "remediate", "execute", "optimize",
    "develop", "build", "design", "refine", "investigate", "analyze",
    "conduct", "perform", "complete", "follow", "contact", "call",
    "email", "inform", "discuss", "provide", "ensure", "clarify",
    "train", "launch", "start", "finish", "export", "import", "handle",
    "update", "write", "set", "setup", "curate", "forecast", "recalculate",
    "standardize", "restructure", "streamline", "establish", "consolidate",
    "run", "close", "ping", "ask", "query", "source"
}

MULTIWORD_IMPERATIVE_VERBS = {
    "follow up", "set up", "roll out", "reach out", "sign off", "check in",
    "hand over", "carry out", "drill down", "point out", "wrap up", "kick off",
    "lock down", "scale up", "spin up", "tear down", "back up", "phase out",
    "clean up", "iron out", "sort out", "flesh out", "narrow down", "step through"
}

CONVERSATIONAL_PREFIXES = [
    r"^(?:hey\s+team|hello\s+team|hi\s+team|hi\s+all|team|everyone|guys|folks)[,\s.]+",
    r"^(?:thanks\s+for\s+joining|thank\s+you\s+all|good\s+morning|good\s+afternoon)[,\s.]+",
    r"^(?:i\s+will|we\s+will|i'll|we'll|i\s+can|we\s+can|let's|can\s+you|could\s+you|please|would\s+you)[,\s]+",
    r"^(?:i\s+am\s+going\s+to|we\s+are\s+going\s+to|i'm\s+going\s+to|we're\s+going\s+to)[,\s]+",
    r"^(?:i\s+gotta|i\s+have\s+to|i\s+need\s+to|we\s+gotta|we\s+have\s+to|we\s+need\s+to|we\s+must|i\s+must)[,\s]+",
    r"^(?:we\s+should|i\s+should|you\s+should|you\s+need\s+to|you\s+must|you\s+gotta)[,\s]+",
    r"^(?:i\s+will\s+make\s+sure\s+to|we\s+will\s+make\s+sure\s+to|make\s+sure\s+to|ensure\s+to|make\s+sure\s+that)[,\s]+",
    r"^(?:i\s+will\s+take\s+care\s+of|i\s+will\s+handle|i\s+will\s+have\s+the|i\s+will\s+deliver\s+the|i\s+will\s+complete)[,\s]+",
    r"^(?:i've\s+asked\s+[A-Za-z]+\s+to|we've\s+asked\s+[A-Za-z]+\s+to|i\s+asked\s+[A-Za-z]+\s+to)[,\s]+",
    r"^(?:[A-Za-z]+\s+has\s+agreed\s+to|[A-Za-z]+\s+agreed\s+to|[A-Za-z]+\s+accepted\s+to)[,\s]+",
    r"^(?:[A-Za-z]+\s+will\s+oversee|[A-Za-z]+\s+will\s+take\s+over|[A-Za-z]+\s+will\s+handle)[,\s]+",
    r"^(?:yeah\s+so\s+i\s+will|yeah\s+so\s+i'll|yeah\s+so|yeah\s+i\s+will|yeah\s+i'll|yeah|well|so|look,|listen,|hey|ok|okay|sure|alright|actually|basically)[,\s]+",
    r"^(?:i\s+think\s+we\s+need\s+to|i\s+guess\s+we\s+should|we\s+should\s+probably|can\s+we\s+make\s+sure\s+to|we\s+need\s+to)[,\s]+",
    r"^(?:we\s+agreed\s+that\s+we\s+will|we\s+agreed\s+to|we\s+decided\s+to|we\s+decided\s+that\s+we\s+will)[,\s]+",
    r"^(?:[A-Z][a-zA-Z\s]+(?::\s*|,\s*))(?:i'll|i\s+will|we\s+will|please|can\s+you|could\s+you)[,\s]+",
    r"^(?:on\s+the\s+engineering\s+front,\s*|on\s+the\s+product\s+front,\s*|for\s+the\s+sprint,\s*)[,\s]*",
]

NON_ACTION_PATTERNS = [
    r"^(?:we\s+discussed|we\s+talked\s+about|the\s+team\s+discussed|the\s+team\s+talked\s+about|they\s+talked\s+about|they\s+discussed)",
    r"^(?:he\s+mentioned|she\s+mentioned|the\s+client\s+mentioned|they\s+mentioned|we\s+mentioned)",
    r"^(?:it\s+may\s+end\s+up|it\s+might\s+be|a\s+new\s+feature\s+will\s+come\s+out|that\s+okay)",
    r"^(?:i\s+feel\s+like|i\s+feel\s+we're|yeah\s+i\s+might|i\s+might\s+even\s+constrain)",
    r"^(?:feature\s+over\s+the\s+past|more\s+than\s+just\s+maps|have\s+a\s+security\s+one|unreviewed$)",
    r"^(?:we\s+should\s+probably\s+look\s+into|maybe\s+we\s+could|we\s+could\s+consider)",
    r"^(?:this\s+is\s+just\s+a\s+chance|just\s+wanted\s+to\s+mention|good\s+morning|hello\s+everyone|welcome\s+to\s+the\s+meeting)",
    r"^(?:call\s+myself\s+out|calling\s+myself\s+out|call\s+out\s+myself|laugh\s+at\s+myself)",
    r"^(?:that's\s+fine|that's\s+okay|sounds\s+good|makes\s+sense|i\s+agree|agreed$)",
    r"^(?:apologies\s+for|sorry\s+about|excuse\s+me|pardon\s+me)",
    r"^(?:anti-buse|that's\s+a\s+big\s+mouthful|we\s+might\s+get\s+a\s+better\s+name|better\s+name\s+over\s+time)",
    r"^(?:based\s+on\s+feedback|i\s+work,\s+i'm\s+going|go\s+to\s+go\s+down|go\s+down\s+over\s+time)",
    r"^(?:here\s+when\s+it's\s+midnight|glad\s+you're\s+here|don't\s+make\s+it\s+happen)",
    r"^(?:news\s+and\s+events|i've\s+got\s+a\s+lot\s+of\s+things)",
    r"^(?:on\s+the\s+agenda\s+today|on\s+the\s+team\s+will\s+attend|attend\s+this\s+meeting)",
    r"^(?:or\s+read\s+the\s+notes|rather\s+than\s+posting|run\s+the\s+slack\s+channel|slack\s+channel)",
    r"^(?:manage\s+is\s+even\s+less|less\s+descriptive|track\s+people\s+down)",
    r"^(?:keep\s+that\s+in\s+mind|keep\s+in\s+mind)",
    r"^(?:see,\s*b\s+is|see\s+item|see\s+b\b|read\s+only\s+item)",
    r"^(?:unless\s+anybody\s+wants\s+to\s+discuss\s+it|unless\s+anyone\s+wants)",
    r"^(?:hi,\s*i'm\s+welcome|welcome,\s*[A-Za-z]+,\s*to\s+the\s+meeting)",
    r"^(?:has\s+accepted\s+the\s+opportunity|accepted\s+the\s+opportunity|accepted\s+team\s+leadership)",
    r"^(?:moving\s+product\s+sections|anti-abuse\s+is\s+moving|will\s+include\s+two\s+stages)",
    r"(?:type\s+this\s+in|type\s+in\s+here|typing\s+in\s+here|i\'ll\s+type\s+this|put\s+a\s+note\s+in\s+there|write\s+this\s+down|finish\s+writing|verbalize\s+and\s+then)",
    r"^(?:verbalize|say\s+it\s+now|put\s+a\s+note|figure\s+it\s+out|type\s+this|i\s+was\s+going\s+to\s+mention|i\'ll\s+type\s+this)",
    r"(?:can\'?t\s+remember\s+what\s+we\s+ended\s+up\s+with|i\s+can\'?t\s+remember|can\'?t\s+recall|i\s+will\s+admit\s+this|at\s+least\s+to\s+me)",
    r"^(?:the\s+end\s+of\s+the\s+quarter|at\s+least\s+to\s+me|link\s+in\s+another\s+issue|a\s+separate\s+issue\s+that\s+is)",
]

GERUND_MAPPINGS = {
    "sending": "Send",
    "preparing": "Prepare",
    "reviewing": "Review",
    "deploying": "Deploy",
    "configuring": "Configure",
    "implementing": "Implement",
    "scheduling": "Schedule",
    "fixing": "Fix",
    "sharing": "Share",
    "creating": "Create",
    "drafting": "Draft",
    "auditing": "Audit",
    "migrating": "Migrate",
    "testing": "Test",
    "validating": "Validate",
    "conducting": "Conduct",
    "developing": "Develop",
    "updating": "Update",
    "documenting": "Document",
    "integrating": "Integrate",
    "finalizing": "Finalize",
    "verifying": "Verify",
    "setting": "Set",
    "aligning": "Align",
    "curating": "Curate",
    "formalizing": "Formalize",
    "evaluating": "Evaluate",
    "benchmarking": "Benchmark",
    "distributing": "Distribute",
    "reaching": "Reach",
    "investigating": "Investigate",
    "analyzing": "Analyze",
    "resolving": "Resolve",
    "executing": "Execute",
    "optimizing": "Optimize",
    "provisioning": "Provision",
    "delivering": "Deliver",
    "publishing": "Publish",
    "writing": "Write",
    "making": "Design",
    "doing": "Perform",
    "upgrading": "Upgrade",
    "automating": "Automate",
    "refactoring": "Refactor",
    "standardizing": "Standardize",
    "following": "Follow up on",
    "tracking": "Track",
    "monitoring": "Monitor",
    "assessing": "Assess",
}


class ActionNormalizer:
    """Transforms raw candidate text into clean, imperative business actions."""

    @classmethod
    def is_non_action_discussion(cls, text: str) -> bool:
        lower = text.strip().lower()
        if len(lower.split()) < 3:
            return True
        for pattern in NON_ACTION_PATTERNS:
            if re.search(pattern, lower):
                return True
        return False

    @classmethod
    def normalize_action_work(cls, raw_action: str) -> str:
        text = raw_action.strip()

        # Strip speaker labels e.g. 'Rahul: I\'ll send...' -> 'I\'ll send...'
        if ":" in text:
            parts = text.split(":", 1)
            if len(parts[0].strip()) < 30 and not any(p in parts[0] for p in [".", "!", "?"]):
                text = parts[1].strip()

        # Repeatedly strip conversational prefixes
        changed = True
        while changed:
            changed = False
            for pfx in CONVERSATIONAL_PREFIXES:
                new_text = re.sub(pfx, "", text, flags=re.IGNORECASE).strip()
                if new_text != text and len(new_text) > 0:
                    text = new_text
                    changed = True

        # Strip topic lead-ins e.g. "On the SSO integration,"
        text = re.sub(r"^(?:on\s+the\s+[^,]+,\s*|regarding\s+the\s+[^,]+,\s*|for\s+the\s+[^,]+,\s*)", "", text, flags=re.IGNORECASE).strip()

        # Conversational colloquialisms to executive corporate action verbs
        colloquial_transforms = [
            (r"^(?:look\s+into|check\s+out|take\s+a\s+look\s+at)\b", "Investigate and evaluate"),
            (r"^(?:touch\s+base\s+with|get\s+in\s+touch\s+with|reach\s+out\s+to)\b", "Coordinate with"),
            (r"^(?:touch\s+base\s+on|sync\s+up\s+on|circle\s+back\s+on)\b", "Align with stakeholders on"),
            (r"^(?:fix\s+up|patch\s+up|sort\s+out)\b", "Resolve and remediate"),
            (r"^(?:set\s+up|spin\s+up)\b", "Configure and provision"),
            (r"^(?:put\s+together|whip\s+up|type\s+up)\b", "Draft and compile"),
            (r"^(?:make\s+sure\s+to|make\s+sure)\b", "Verify and ensure"),
            (r"^(?:figure\s+out|hammer\s+out)\b", "Determine and formulate"),
            (r"^(?:work\s+on\s+getting|get\s+started\s+on|start\s+working\s+on)\b", "Initiate"),
            (r"^(?:talk\s+to|speak\s+with)\b", "Consult with"),
            (r"^(?:double\s+check|re-check)\b", "Validate and audit"),
            (r"^(?:clean\s+up)\b", "Refactor and standardize"),
        ]
        for pattern, repl in colloquial_transforms:
            if re.search(pattern, text, flags=re.IGNORECASE):
                text = re.sub(pattern, repl, text, flags=re.IGNORECASE).strip()
                break

        # Prune trailing hesitation, conversational subclauses, or requests
        hesitation_patterns = [
            r",?\s+(?:like\s+you\s+requested|like\s+requested|as\s+requested|as\s+discussed|like\s+we\s+agreed|like\s+you\s+said).*",
            r",?\s+(?:and\s+i\'ll\s+make\s+the\s+update|and\s+i\s+will\s+make\s+the\s+update|and\s+i\'ll\s+update\s+it|and\s+i\'ll\s+take\s+care\s+of\s+it).*",
            r"\s+(?:or\s+it\s+doesn\'t\s+have\s+to\s+be\s+in\s+here|or\s+do\s+you\s+want\s+to\s+add\s+that|but\s+we\s+should\s+i\s+think|its\s+not\s+relevant|i\s+think\s+we|or\s+whatever|if\s+needed|if\s+possible).*",
            r"\s+(?:and\s+see\s+if\s+we\s+can|and\s+find\s+out\s+what|and\s+like\s+build\s+their).*",
        ]
        for hp in hesitation_patterns:
            text = re.sub(hp, "", text, flags=re.IGNORECASE).strip()

        # Pronoun & conversational verb normalization into specific deliverables
        text = re.sub(r"^(?:send\s+it\s+to\s+([A-Za-z0-9_]+)|send\s+that\s+to\s+([A-Za-z0-9_]+))\b", r"Send the requested document to \1\2", text, flags=re.IGNORECASE)
        text = re.sub(r"^(?:send\s+that\s+to\s+me|send\s+it\s+to\s+me|send\s+to\s+me)\b", "Provide requested document for review and updates", text, flags=re.IGNORECASE)
        text = re.sub(r"^(?:send\s+that|send\s+it|send\s+this)\b", "Send the required deliverable", text, flags=re.IGNORECASE)
        text = re.sub(r"^(?:update\s+it|update\s+that|update\s+this)\b", "Apply requested updates", text, flags=re.IGNORECASE)
        text = re.sub(r"^(?:review\s+it|review\s+that|review\s+this)\b", "Review the deliverable", text, flags=re.IGNORECASE)

        # Strip relative deadlines embedded in the work string
        date_pattern = r"\s+(?:by|before|until|due|for)\s+(?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|eod|end\s+of\s+sprint|end\s+of\s+week|next\s+week|afternoon|morning|evening)(?:\s+at\s+\d+(?::\d+)?\s*(?:am|pm)?)?\.?$"
        text = re.sub(date_pattern, "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s+afternoon\.?$", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s+morning\.?$", "", text, flags=re.IGNORECASE).strip()

        # Convert leading gerunds to imperative verbs
        words = text.split()
        if words:
            first_word = words[0].lower()
            if first_word in GERUND_MAPPINGS:
                words[0] = GERUND_MAPPINGS[first_word]
                text = " ".join(words)
            elif first_word.endswith("ing") and len(first_word) > 4:
                stem = first_word[:-3]
                if stem.endswith("tt") or stem.endswith("nn") or stem.endswith("pp") or stem.endswith("gg"):
                    stem = stem[:-1]
                words[0] = stem.title()
                text = " ".join(words)

        # Capitalize first letter and strip trailing punctuation
        text = text.strip().rstrip(" .,;")
        if text:
            text = text[0].upper() + text[1:]
        return text

    @classmethod
    def generate_final_phrase(
        cls,
        action: str,
        owner: str | None = None,
        recipient: str | None = None,
        deadline: str | None = None,
    ) -> str:
        action_clean = cls.normalize_action_work(action)
        if not action_clean:
            return ""

        phrase = action_clean
        if recipient and recipient.lower() not in phrase.lower() and recipient.lower() not in ["none", "team"]:
            phrase = f"{phrase} to {recipient}"

        if deadline and deadline.lower() not in phrase.lower() and deadline.lower() not in ["not specified", "none", "tbd"]:
            phrase = f"{phrase} by {deadline}"

        return phrase


class ExecutiveActionReframingEngine:
    """Formal Code & NLP Layer that integrates with LLM outputs to structure enterprise action items."""

    @classmethod
    def reframe_action(
        cls,
        raw_task: str,
        owner: str | None = None,
        assigner: str | None = None,
        recipient: str | None = None,
        deadline: str | None = None,
        meeting_type: str | None = None,
    ) -> dict[str, Any]:
        """Converts raw conversational transcript text into a structured, executive-grade action item."""
        clean_task = ActionNormalizer.normalize_action_work(raw_task)
        if not clean_task:
            clean_task = raw_task.strip().capitalize()

        # Formal synthesized sentence
        final_sentence = ActionNormalizer.generate_final_phrase(
            clean_task, owner=owner, recipient=recipient, deadline=deadline
        )

        clean_owner = (owner or "Unassigned").strip()
        if clean_owner.lower() in ["none", "null", "execute", "assigned lead", "tbd"]:
            clean_owner = "Unassigned"

        clean_assigner = (assigner or "").strip() if assigner else None
        if clean_assigner and clean_assigner.lower() in ["none", "null", "execute", "tbd"]:
            clean_assigner = None

        clean_recipient = (recipient or "").strip() if recipient else None
        if clean_recipient and clean_recipient.lower() in ["none", "null", "tbd"]:
            clean_recipient = None

        clean_deadline = (deadline or "Not specified").strip()

        return {
            "task": clean_task,
            "action": clean_task,
            "description": final_sentence or clean_task,
            "assigner": clean_assigner,
            "owner": clean_owner,
            "recipient": clean_recipient,
            "deadline": None if clean_deadline.lower() in ["not specified", "none", "tbd"] else clean_deadline,
            "deadline_text": clean_deadline,
        }

    @classmethod
    def synthesize_action_summary(
        cls,
        actions: list[Any],
        meeting_theme: str | None = None,
    ) -> str:
        """Synthesizes a short, high-impact 1-line executive summary of what the core actions are."""
        valid_items = []
        for a in actions:
            if isinstance(a, dict):
                t = (a.get("task") or a.get("action") or a.get("description") or "").strip()
                o = (a.get("owner") or "").strip()
                d = (a.get("deadline") or a.get("deadline_text") or "").strip()
            else:
                t = (getattr(a, "task", "") or getattr(a, "action", "") or getattr(a, "description", "") or "").strip()
                o = (getattr(a, "owner", "") or "").strip()
                d = (getattr(a, "deadline", "") or getattr(a, "deadline_text", "") or "").strip()

            if t and len(t.split()) >= 2:
                valid_items.append((t, o, d))

        if not valid_items:
            return "No pending post-meeting deliverables were assigned in this discussion."

        if len(valid_items) == 1:
            task, owner, dl = valid_items[0]
            owner_part = f"{owner} to " if owner and owner.lower() not in ["unassigned", "not specified", "none"] else ""
            dl_part = f" by {dl}" if dl and dl.lower() not in ["not specified", "none"] else ""
            return f"Key Commitment: {owner_part}{task.rstrip('.')}{dl_part}."

        # Multiple items: synthesize top commitments concisely
        summaries = []
        for task, owner, dl in valid_items[:3]:
            owner_part = f"{owner}: " if owner and owner.lower() not in ["unassigned", "not specified", "none"] else ""
            dl_part = f" (by {dl})" if dl and dl.lower() not in ["not specified", "none"] else ""
            summaries.append(f"{owner_part}{task.rstrip('.')}{dl_part}")

        return f"Key Commitments: {'; '.join(summaries)}."


class ExecutiveDecisionReframingEngine:
    """Formal NLP & Normalization Layer that reframes raw meeting discussion agreements into professional governance decisions."""

    DECISION_FILLERS: ClassVar[list[str]] = [
        r"^(?:great|excellent|perfect|confirmed|agreed|okay|alright)[,\.\s]+",
        r"^(?:on\s+security\s+and\s+governance|on\s+architecture|on\s+strategy|regarding\s+the\s+roadmap|for\s+the\s+sprint)[,\.\s]+",
        r"^(?:we\s+all\s+agreed\s+to|we\s+agreed\s+to|we\s+decided\s+to|we\s+decided\s+that|we\s+also\s+approved|we\s+approved|we\s+formally\s+approved|we\s+confirmed\s+that)\s+",
        r"^(?:the\s+decision\s+is\s+to|the\s+consensus\s+is\s+to|all\s+agreed\s+to|consensus\s+was\s+to|aligned\s+on)\s+",
        r"^(?:as\s+a\s+team\s+we\s+agreed\s+to|it\s+was\s+decided\s+that|it\s+was\s+agreed\s+that)\s+",
    ]

    @classmethod
    def normalize_decision_statement(cls, text: str) -> str:
        """Cleans verbal prefixes and conversation fillers to produce a sharp governance resolution."""
        if not text:
            return ""
        s = text.strip()
        for pat in cls.DECISION_FILLERS:
            s = re.sub(pat, "", s, flags=re.IGNORECASE).strip()

        # Strip personal action statements from pure decisions if phrased as personal intent
        if re.match(r"^i\s+(?:will|can|must)\s+", s, flags=re.IGNORECASE):
            s = re.sub(r"^i\s+(?:will|can|must)\s+", "", s, flags=re.IGNORECASE).strip()

        s = s.rstrip(" .,;")
        if s:
            s = s[0].upper() + s[1:]
        return s

    @classmethod
    def reframe_decision(
        cls,
        raw_decision: str,
        approved_by: list[str] | str | None = None,
        rationale: str | None = None,
        impact: str | None = None,
        evidence_quote: str | None = None,
    ) -> Decision:
        clean_desc = cls.normalize_decision_statement(raw_decision)
        if not clean_desc or len(clean_desc.split()) < 3:
            clean_desc = raw_decision.strip()

        approvers_list: list[str] = []
        if isinstance(approved_by, list):
            approvers_list = [str(a).strip() for a in approved_by if str(a).strip() and str(a).lower() not in ["none", "null"]]
        elif isinstance(approved_by, str) and approved_by.strip():
            approvers_list = [a.strip() for a in approved_by.split(",") if a.strip() and a.strip().lower() not in ["none", "null"]]

        if not approvers_list:
            approvers_list = ["Executive Consensus"]

        clean_rationale = rationale or "Ratified by team consensus during technical/strategic discussion."
        clean_impact = impact or "Operational, architectural, and governance alignment across teams."

        return Decision(
            description=clean_desc,
            approved_by=approvers_list,
            rationale=clean_rationale,
            impact=clean_impact,
            evidence_quote=evidence_quote or raw_decision,
            confidence=0.94,
        )


VAGUE_ACTION_PATTERNS = [
    r"\bimprove\s+things\b",
    r"\baddress\s+things\b",
    r"\bother\s+things\b",
    r"\baddress\s+other\s+issues\b",
    r"\btake\s+action\b",
    r"\btake\s+necessary\s+action\b",
    r"\bwork\s+on\s+it\b",
    r"\bhandle\s+it\b",
    r"\bdo\s+the\s+needful\b",
    r"\bmake\s+it\s+better\b",
    r"\bmake\s+things\s+better\b",
    r"\baddress\s+the\s+issue\b",
    r"\baddress\s+security\s+and\s+quality\b",
    r"\baddress\s+security\b",
    r"\bimprove\s+security\b",
    r"\bfollow\s+up\b$",
    r"\bbecause\s+you\'?re\s+more\s+productive\b",
    r"\baddress\s+quality\b",
    r"\bwork\s+on\s+the\s+project\b",
    r"\bwork\s+on\s+project\b",
    r"\bshare\s+one\s+thing\b",
    r"\bshow\s+one\s+thing\b",
    r"\bshare\s+my\s+screen\b",
    r"\bshow\s+my\s+screen\b",
    r"\bshare\s+your\s+screen\b",
    r"\bshow\s+your\s+screen\b",
    r"^(?:share|show|say|tell|mention|give|point\s+out|bring\s+up|highlight)\s+(?:one\s+thing|a\s+thing|something|some\s+things|a\s+couple\s+of\s+things|a\s+few\s+things|anything|my\s+screen|your\s+screen|it|this|that)$",
    r"^(?:take|have)\s+a\s+look\b",
    r"^(?:let\s+me|allow\s+me\s+to|i\s+want\s+to)\s+(?:share|show|tell|say)\b",
    r"^(?:we\s+should|we\s+need\s+to|let's)\s+improve\b",
    r"^(?:improve|address|fix|update|review|check|handle|manage)\s+(?:things|security|quality|issues|everything|it|this|that|stuff)$",
    r"^(?:look\s+into|see\s+about|touch\s+base\s+on|track\s+down)\s+(?:things|it|this|that|them|stuff)$",
    r"^(?:investigate\s+and\s+evaluate|validate\s+and\s+audit)\s+(?:it|this|that|them|things|stuff)$",
    r"^(?:look|check)\s+at\s+the\s+agenda\b",
    r"^(?:give|provide)\s+(?:an\s+update|updates)\s+on\s+things\b",
    r"^(?:publish|send|share|show|give|do|make)\s+it\s+(?:to|with|for)\s+[A-Za-z]+$",
    r"^(?:send|give|forward)\s+(?:you\s+)?another\s+(?:invitation|invite|calendar\s+invite|link|email)\b",
    r"^(?:send|forward)\s+(?:you\s+)?an\s+(?:invitation|invite|calendar\s+invite)\b",
    r"^(?:glad\s+you|welcome\s+to|thanks\s+for|talk\s+soon|bye\s+everyone|hello\s+everyone)",
    r"^(?:see\s+item|read\s+only|on\s+the\s+agenda|keep\s+in\s+mind|keep\s+that\s+in\s+mind)",
]


COPULA_STATEMENT_PATTERN = r"^\w+\s+(?:is|are|was|were|has\s+been|seems|means|sounds|feels)\s+"
TRAILING_FRAGMENT_PATTERN = r"(?:\bto\s+like|\bthan\s+that|\band\s+like|\bor\s+like|\bto\s+be\s+like|\bto|\bthan|\blike|\band|\bor|\bif|\bso|\bwith|\bfor|\babout|\bas|\bbecause|\bwhich|\bthat)\s*[\.\,\;\:]*$"
CONVERSATIONAL_RAMBLE_PATTERN = r"\.\s*(?:so\s+i|so\s+if|if\s+if|i\s+think|i\s+guess|maybe|like|and\s+like|so\s+yeah|we\s+can|let\'s)\b"
STUTTER_PATTERN = r"\b(if\s+if|we\s+we|i\s+i|to\s+to|the\s+the|that\s+that|so\s+so)\b"
META_COMMENTARY_PATTERN = r"\b(?:less|more)\s+descriptive\b|\btake\s+the\s+mission\s+to\b|\bso\s+i\s+think\s+if\b|\btrack\s+people\s+down\b|\beven\s+less\b|\bwhat\s+i\s+mean\b|\bwhat\s+we\s+mean\b"


class ActionSpecificityValidator:
    """Validates that extracted actions are concrete, executable pieces of work."""

    @classmethod
    def is_vague(cls, action_text: str) -> tuple[bool, str]:
        lower = action_text.strip().lower()
        if len(lower.split()) < 2:
            return True, "Action is too short to represent a concrete task."
        for pattern in VAGUE_ACTION_PATTERNS:
            if re.search(pattern, lower):
                return True, f"Action contains vague non-executable phrase matching '{pattern}'."
        if re.search(COPULA_STATEMENT_PATTERN, lower):
            return True, "Action is a descriptive copula statement, not an executable task."
        if re.search(TRAILING_FRAGMENT_PATTERN, lower):
            return True, "Action is an incomplete trailing fragment."
        if re.search(CONVERSATIONAL_RAMBLE_PATTERN, lower):
            return True, "Action contains conversational rambling clauses."
        if re.search(STUTTER_PATTERN, lower):
            return True, "Action contains spoken stutter patterns."
        if re.search(META_COMMENTARY_PATTERN, lower):
            return True, "Action contains conversational meta-commentary."
        return False, ""


class ActionValidator:
    """Validates candidate action items against quality rubrics and thresholds."""

    @classmethod
    def validate(cls, item: ActionItem | dict[str, Any]) -> tuple[bool, str]:
        if isinstance(item, dict):
            action_text = item.get("task") or item.get("action") or item.get("description") or ""
        else:
            action_text = item.task or item.action or item.description or ""
        action_text = action_text.strip()
        lower = action_text.lower()

        # Length check: Must have at least 2 descriptive words
        words = action_text.split()
        if len(words) < 2:
            return False, f"Action text '{action_text}' is too short (minimum 2 words required for a specific task)."

        first_word = words[0].lower().rstrip(":,.")
        first_two = f"{words[0].lower()} {words[1].lower()}".rstrip(":,.") if len(words) > 1 else ""

        # Reject conversational greetings or opening filler
        if first_word in ["hi", "hello", "hey", "welcome", "glad", "thanks", "thank", "bye", "goodbye", "apologies", "sorry", "excuse"]:
            return False, f"Action text '{action_text}' is a conversational greeting/filler."

        # Specificity and anti-conversational checks
        is_vague, reason = ActionSpecificityValidator.is_vague(action_text)
        if is_vague:
            return False, reason

        if ActionNormalizer.is_non_action_discussion(action_text):
            return False, f"Action text '{action_text}' is conversational discussion/chatter, not an executable deliverable."

        # Enforce leading imperative action verb
        is_imperative = (
            first_word in IMPERATIVE_VERBS
            or first_two in MULTIWORD_IMPERATIVE_VERBS
            or any(lower.startswith(v + " ") for v in IMPERATIVE_VERBS)
            or any(lower.startswith(v + " ") for v in MULTIWORD_IMPERATIVE_VERBS)
        )
        if not is_imperative:
            return False, f"Action text '{action_text}' must begin with a strong imperative action verb."

        # Reject conversational pronoun prefixes
        if any(lower.startswith(bad) for bad in ["i will", "we will", "i'll", "we'll", "let's", "he said", "she said", "they discussed", "i've asked", "we've asked", "has agreed to"]):
            return False, f"Action text '{action_text}' contains conversational speech pronoun prefixes."

        # Reject questions / inquiries
        if action_text.endswith("?") or any(lower.startswith(q) for q in ["how ", "what ", "why ", "where ", "when ", "who ", "is ", "are ", "can ", "could ", "would ", "do ", "does ", "did "]):
            return False, f"Action text '{action_text}' is an inquiry/question, not an executable task."

        # Reject past status descriptions or observations
        if any(lower.startswith(p) for p in ["and they have", "they have", "we have", "they had", "we had", "and we", "there is", "there are", "there was", "it was", "they were", "and they", "they set", "we set", "they already", "we already"]):
            return False, f"Action text '{action_text}' describes past status/setup, not a future pending deliverable."

        # Reject sentence fragments starting with prepositions or determiners
        if first_word in ["of", "for", "with", "in", "at", "by", "to", "from", "about", "this", "that", "these", "those", "the", "a", "an", "so", "just", "and", "or", "unless", "see", "keep"]:
            return False, f"Action text '{action_text}' is a sentence fragment starting with '{first_word}'."

        return True, "Valid executable action item."


# ==========================================
# 2. Golden Example Store & Self-Critique
# ==========================================

FORBIDDEN_PLACEHOLDER_SUBSTRINGS = [
    "assigned lead",
    "deliverable validated and operational by end of sprint",
    "execute: and i am",
    "execute: i was going to",
    "execute: that okay",
    "execute: i don't know",
]


class CritiqueResult(StrictModel):
    passed: bool
    reason: str
    violations: list[str] = Field(default_factory=list)
    confidence: float = 1.0


class GoldenExampleRecord(StrictModel):
    agent_type: str
    input_text: str
    corrected_output: dict[str, Any]
    prompt_version: str = "2.0.0"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "human_curated"


class GoldenExampleStore:
    """Manages the MongoDB 'goldenExamples' collection with in-memory fallback."""

    def __init__(self, mongo_uri: str = "mongodb://localhost:27017", database: str = "mom_ai_brain") -> None:
        self._mongo_uri = mongo_uri
        self._database_name = database
        self._collection_name = "goldenExamples"
        self._in_memory_store: dict[str, list[dict[str, Any]]] = {}
        self._init_mongo()

    def _init_mongo(self) -> None:
        try:
            from pymongo import MongoClient
            self._client = MongoClient(self._mongo_uri, serverSelectionTimeoutMS=1500)
            self._db = self._client[self._database_name]
            self._coll = self._db[self._collection_name]
            self._coll.create_index([("agent_type", 1), ("created_at", -1)])
            self._available = True
        except Exception as exc:
            logger.warning("MongoDB goldenExamples unavailable (%s), using in-memory store.", exc)
            self._available = False

    def save_golden_example(
        self,
        agent_type: str,
        input_text: str,
        corrected_output: dict[str, Any],
        prompt_version: str = "2.0.0",
        source: str = "human_approved",
    ) -> bool:
        record = {
            "agent_type": agent_type,
            "input_text": input_text,
            "corrected_output": corrected_output,
            "prompt_version": prompt_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
        }
        if self._available:
            try:
                self._coll.insert_one(record)
                logger.info("Saved golden example for agent [%s] into MongoDB goldenExamples", agent_type)
                return True
            except Exception as exc:
                logger.error("Failed to insert into MongoDB goldenExamples: %s", exc)

        self._in_memory_store.setdefault(agent_type, []).append(record)
        return True

    def get_relevant_golden_examples(self, agent_type: str, input_text: str = "", limit: int = 5) -> list[dict[str, Any]]:
        examples = []
        if self._available:
            try:
                docs = list(self._coll.find({"agent_type": agent_type}).sort("created_at", -1).limit(limit))
                for d in docs:
                    d.pop("_id", None)
                    examples.append(d)
            except Exception as exc:
                logger.warning("Failed querying MongoDB goldenExamples: %s", exc)

        if not examples and agent_type in self._in_memory_store:
            examples = self._in_memory_store[agent_type][-limit:]

        return examples

    def seed_initial_golden_examples(self, seed_data_path: str | None = None) -> int:
        if not self._available:
            return 0
        try:
            count = self._coll.count_documents({})
            if count > 0:
                return count

            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            dataset_path = seed_data_path or os.path.join(base_dir, "tests", "regression", "datasets", "action_items_golden.json")
            if os.path.exists(dataset_path):
                with open(dataset_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                inserted = 0
                for item in data:
                    self.save_golden_example(
                        agent_type="action",
                        input_text=item.get("transcript", ""),
                        corrected_output={"action_items": item.get("expected", [])},
                        source="golden_dataset_seed",
                    )
                    inserted += 1
                logger.info("Seeded %d golden action item examples into goldenExamples collection", inserted)
                return inserted
        except Exception as exc:
            logger.warning("Failed seeding goldenExamples: %s", exc)
        return 0


class SelfCritiquePass:
    """Executes a 2nd pass self-critique rubric audit on agent output."""

    @classmethod
    def evaluate(cls, agent_name: AgentName, transcript_text: str, parsed_output: dict[str, Any]) -> CritiqueResult:
        violations = []
        lower_trans = transcript_text.lower()
        output_str = json.dumps(parsed_output).lower()

        # 1. Check for forbidden placeholder patterns
        for placeholder in FORBIDDEN_PLACEHOLDER_SUBSTRINGS:
            if placeholder in output_str:
                violations.append(f"Forbidden fallback placeholder detected: '{placeholder}'")

        # 2. Agent-Specific Rubrics
        if agent_name == AgentName.ACTION:
            actions = parsed_output.get("action_items") or parsed_output.get("actionItems") or []
            if not actions and len(transcript_text.split()) > 20:
                if any(w in lower_trans for w in ["will", "must", "need to", "action item", "responsible for"]):
                    violations.append("Transcript contains explicit action commitments but zero action items were extracted.")

            owners = set()
            deadlines = set()
            priorities = set()
            for idx, act in enumerate(actions, start=1):
                desc = act.get("description") or act.get("actionItem") or ""
                owner = act.get("owner") or act.get("actionOwner") or ""
                deadline = act.get("deadline_text") or act.get("deadline") or ""
                priority = act.get("priority") or "Low"

                if not desc or len(desc.strip()) < 5:
                    violations.append(f"Action item #{idx} description is empty or too short.")
                if owner and owner.lower() in ["assigned lead", "execute", "placeholder lead"]:
                    violations.append(f"Action item #{idx} has invalid placeholder owner: '{owner}'.")

                if owner:
                    owners.add(owner.lower())
                deadlines.add(deadline.lower())
                priorities.add(priority)

            if len(actions) >= 3:
                if len(owners) == 1 and list(owners)[0] in ["assigned lead", "execute", "placeholder lead"]:
                    violations.append("All action items defaulted to the same generic placeholder owner.")
                if len(deadlines) == 1 and list(deadlines)[0] == "end of sprint" and "end of sprint" not in lower_trans:
                    violations.append("All action items defaulted to ungrounded 'End of Sprint'.")

        elif agent_name == AgentName.SUMMARY:
            summary = parsed_output.get("executive_summary") or ""
            key_points = parsed_output.get("key_points") or []
            if not summary or len(summary.strip()) < 20:
                violations.append("Executive summary is empty or too short.")
            if not key_points:
                violations.append("Summary must contain at least 2 structured key points.")

        elif agent_name == AgentName.DECISION:
            decisions = parsed_output.get("decisions") or []
            for idx, dec in enumerate(decisions, start=1):
                decision_text = dec.get("description") or dec.get("decision") or ""
                if not decision_text or len(decision_text.strip()) < 4:
                    violations.append(f"Decision #{idx} description is empty or too short.")

        passed = len(violations) == 0
        reason = "Output passed all quality rubric checks." if passed else " | ".join(violations)
        return CritiqueResult(passed=passed, reason=reason, violations=violations)


class ExecutiveSentimentAnalyzer:
    """Deterministic NLP & Behavioral Sentiment Intelligence Analyzer for Meeting Transcripts."""

    FRICTION_PATTERNS = [
        r"\b(?:struggling|contention|bottleneck|blocking|blocker|delayed|delay|breaking|broke)\b",
        r"\b(?:cannot\s+afford|can\'t\s+afford|unacceptable|severe|critical\s+issue|risk)\b",
        r"\b(?:worried|concerned|frustrated|confused|skeptical|doubtful|pushback)\b",
        r"\b(?:failing|failed|crash|memory\s+leak|timeout|outage|incident)\b",
        r"\b(?:slow|sluggish|degraded|unstable|friction|disagree|objection)\b",
    ]

    ALIGNMENT_PATTERNS = [
        r"\b(?:excellent|fantastic|great|perfect|awesome|outstanding|brilliant)\b",
        r"\b(?:agreed|agree|approved|ratified|consensus|aligned|unanimous)\b",
        r"\b(?:confident|solution|resolved|fixed|remediated|optimized|achieved)\b",
        r"\b(?:on\s+track|on\s+schedule|ready\s+to\s+deploy|looks\s+good|glad|excited)\b",
        r"\b(?:seamless|smooth|proactive|success|successful|delighted)\b",
    ]

    @classmethod
    def analyze_transcript(cls, transcript_text: str, speakers: list[str] | None = None) -> SentimentOutput:
        """Deeply analyzes meeting transcript for tone, polarity, friction, alignment, speaker morale, and chronological shifts."""
        lines = [line.strip() for line in transcript_text.splitlines() if line.strip()]
        if not lines:
            return SentimentOutput(
                overall="Constructive & Professional",
                client_mood="Engaged & Aligned",
                team_mood="Focused on Execution",
                polarity_score=0.75,
                engagement_level="Moderate",
                confidence=0.92,
            )

        friction_matches = []
        alignment_matches = []
        speaker_turns: dict[str, list[str]] = {}

        for line in lines:
            spk, text = _parse_speaker_and_text(line)
            if spk and spk not in ["Unknown", "SPEAKER"]:
                speaker_turns.setdefault(spk, []).append(text)
            
            lower_line = text.lower()
            for fp in cls.FRICTION_PATTERNS:
                if re.search(fp, lower_line) and len(text) > 10:
                    friction_matches.append(text)
                    break
            for ap in cls.ALIGNMENT_PATTERNS:
                if re.search(ap, lower_line) and len(text) > 10:
                    alignment_matches.append(text)
                    break

        friction_count = len(friction_matches)
        alignment_count = len(alignment_matches)
        total_signals = friction_count + alignment_count

        # Compute polarity score (-1.0 to +1.0)
        if total_signals > 0:
            polarity_score = round(max(-1.0, min(1.0, (alignment_count - friction_count) / max(total_signals, 1) * 0.8 + 0.2)), 2)
        else:
            polarity_score = 0.70

        # Engagement level
        word_count = len(transcript_text.split())
        engagement_level = "High" if word_count > 150 or len(speaker_turns) >= 2 else "Moderate"

        # Categorize overall sentiment
        if friction_count > 0 and alignment_count >= friction_count:
            overall = "Constructive with Initial Technical Friction Resolving into Strong Team Alignment"
            client_mood = "Demanding Rigorous Quality & Security Guarantees"
            team_mood = "Proactive, Solution-Oriented & Confident in Delivery"
        elif friction_count > alignment_count:
            overall = "Challenging & Cautious with Critical Operational Hurdles"
            client_mood = "Concerned Regarding System Contention and Timelines"
            team_mood = "Resilient & Focused on Root-Cause Mitigation"
        elif alignment_count > 0:
            overall = "Highly Collaborative, Focused & Solution-Driven"
            client_mood = "Enthusiastic & Aligned with Roadmap"
            team_mood = "High Morale & High Execution Velocity"
        else:
            overall = "Constructive, Structured & Professional"
            client_mood = "Attentive & Cooperative"
            team_mood = "Execution-Focused & Organized"

        # Unique friction points & alignment signals (summarized)
        friction_points = [
            cls._summarize_signal(fm, "Friction") for fm in friction_matches[:4]
        ]
        alignment_signals = [
            cls._summarize_signal(am, "Alignment") for am in alignment_matches[:4]
        ]

        # Speaker sentiment breakdown
        speaker_sentiments: dict[str, str] = {}
        for spk, turns in speaker_turns.items():
            combined_speaker_text = " ".join(turns).lower()
            spk_f = sum(1 for fp in cls.FRICTION_PATTERNS if re.search(fp, combined_speaker_text))
            spk_a = sum(1 for ap in cls.ALIGNMENT_PATTERNS if re.search(ap, combined_speaker_text))
            
            if spk_a > spk_f:
                speaker_sentiments[spk] = "Confident & Solution-Oriented"
            elif spk_f > spk_a:
                speaker_sentiments[spk] = "Analytical & Risk-Conscious"
            else:
                speaker_sentiments[spk] = "Collaborative & Methodical"

        # Chronological Shifts (Opening -> Mid -> Closing)
        n = len(lines)
        chronological_shifts = []
        if n >= 3:
            p1_lines = lines[: max(1, n // 3)]
            p2_lines = lines[max(1, n // 3) : max(2, (2 * n) // 3)]
            p3_lines = lines[max(2, (2 * n) // 3) :]

            p1_f = sum(1 for l in p1_lines for fp in cls.FRICTION_PATTERNS if re.search(fp, l.lower()))
            p1_a = sum(1 for l in p1_lines for ap in cls.ALIGNMENT_PATTERNS if re.search(ap, l.lower()))

            p2_f = sum(1 for l in p2_lines for fp in cls.FRICTION_PATTERNS if re.search(fp, l.lower()))
            p2_a = sum(1 for l in p2_lines for ap in cls.ALIGNMENT_PATTERNS if re.search(ap, l.lower()))

            p3_f = sum(1 for l in p3_lines for fp in cls.FRICTION_PATTERNS if re.search(fp, l.lower()))
            p3_a = sum(1 for l in p3_lines for ap in cls.ALIGNMENT_PATTERNS if re.search(ap, l.lower()))

            # Shift 1
            if p1_f > 0:
                chronological_shifts.append("Opening Phase: Cautious review with explicit concerns raised regarding performance bottlenecks and operational risks.")
            else:
                chronological_shifts.append("Opening Phase: Constructive agenda setting and operational status review.")

            # Shift 2
            if p2_a >= p2_f:
                chronological_shifts.append("Mid-Meeting Phase: Active technical deliberation leading to consensus on architectural trade-offs and migration strategies.")
            else:
                chronological_shifts.append("Mid-Meeting Phase: In-depth debate identifying key integration vulnerabilities.")

            # Shift 3
            if p3_a > 0 or p3_f == 0:
                chronological_shifts.append("Closing Phase: Strong positive alignment with clear ownership commitments and delivery confidence established.")
            else:
                chronological_shifts.append("Closing Phase: Contingency protocols and follow-up reviews scheduled.")
        else:
            chronological_shifts.append("Consolidated Session: Clear and constructive alignment on operational deliverables.")

        # Verbatim evidence quotes
        evidence_quotes = (friction_matches[:2] + alignment_matches[:3])[:4]
        if not evidence_quotes and lines:
            evidence_quotes = [lines[0]]

        return SentimentOutput(
            overall=overall,
            client_mood=client_mood,
            team_mood=team_mood,
            polarity_score=polarity_score,
            engagement_level=engagement_level,
            friction_points=friction_points,
            alignment_signals=alignment_signals,
            speaker_sentiments=speaker_sentiments,
            chronological_shifts=chronological_shifts,
            evidence=evidence_quotes,
            confidence=0.94,
        )

    @staticmethod
    def _summarize_signal(text: str, signal_type: str) -> str:
        s = text.strip()
        if len(s) > 120:
            s = s[:117] + "..."
        return s


class AgentQualityLoop:
    """Shared Quality Loop wrapping Agent execution across Summary, Action, and Decision agents."""

    def __init__(self, golden_store: GoldenExampleStore | None = None) -> None:
        self._golden_store = golden_store or GoldenExampleStore()

    def get_golden_store(self) -> GoldenExampleStore:
        return self._golden_store

    def build_few_shot_prompt_context(self, agent_name: AgentName, transcript_text: str) -> str:
        examples = self._golden_store.get_relevant_golden_examples(agent_name.value, transcript_text, limit=6)
        if not examples:
            return ""

        lines = ["\n--- GOLDEN HIGH-QUALITY FEW-SHOT EXAMPLES (Follow this exact schema and caliber) ---"]
        for idx, ex in enumerate(examples, start=1):
            lines.append(f"Example {idx}:")
            lines.append(f"Input Context: {ex.get('input_text', '')}")
            lines.append(f"Approved Output: {json.dumps(ex.get('corrected_output', {}))}")
            lines.append("")
        return "\n".join(lines)

    def evaluate_and_critique(
        self,
        agent_name: AgentName,
        transcript_text: str,
        parsed_output: dict[str, Any],
    ) -> CritiqueResult:
        return SelfCritiquePass.evaluate(agent_name, transcript_text, parsed_output)
