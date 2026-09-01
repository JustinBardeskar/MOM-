from datetime import date, datetime
from enum import StrEnum
from functools import lru_cache
import re
from typing import Any, Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.domain import JobRecord, MeetingReadyRequest, PreprocessedTranscript
from app.integration import ValidatedM2Output


# ==========================================
# Base Strict Model
# ==========================================

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


# ==========================================
# Core Enums
# ==========================================

class LLMProviderName(StrEnum):
    GROQ = "groq"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"


class AgentName(StrEnum):
    MEETING_UNDERSTANDING = "meeting_understanding"
    SUMMARY = "summary"
    ACTION = "action"
    DECISION = "decision"
    REQUIREMENT = "requirement"
    RISK = "risk"
    SENTIMENT = "sentiment"
    TOPIC = "topic"
    DEADLINE = "deadline"
    QUESTION = "question"
    FOLLOW_UP = "follow_up"


class MeetingType(StrEnum):
    CLIENT = "client"
    SCRUM = "scrum"
    SALES = "sales"
    INTERVIEW = "interview"
    HR = "hr"
    TECHNICAL = "technical"
    STRATEGY = "strategy"
    PRODUCT = "product"
    MARKETING = "marketing"
    EXECUTIVE = "executive"
    BOARD = "board"
    GENERAL = "general"


# ==========================================
# Exceptions Hierarchy
# ==========================================

class AIBrainError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class NoModelAvailableError(AIBrainError):
    pass


class LLMProviderError(AIBrainError):
    pass


class AgentExecutionError(AIBrainError):
    pass


class M3ContractValidationError(AIBrainError):
    pass


class M4ContractValidationError(AIBrainError):
    pass


class M5ContractValidationError(AIBrainError):
    pass


class M6ContractValidationError(AIBrainError):
    pass


# ==========================================
# LLM Invocations & Profiles
# ==========================================

class ModelProfile(StrictModel):
    provider: LLMProviderName
    model: str
    quality_rank: int = Field(ge=1, le=5)
    cost_rank: int = Field(ge=1, le=5)
    max_context_tokens: int = Field(ge=1)
    supports_json: bool = True
    input_cost_per_million: float = Field(default=0, ge=0)
    output_cost_per_million: float = Field(default=0, ge=0)


class LLMRequest(StrictModel):
    system_prompt: str
    user_prompt: str
    max_output_tokens: int = Field(default=2000, ge=1)
    temperature: float = Field(default=0, ge=0, le=2)
    reasoning_enabled: bool = False


class LLMUsage(StrictModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMResponse(StrictModel):
    text: str
    usage: LLMUsage = Field(default_factory=LLMUsage)
    provider_request_id: str | None = None


# ==========================================
# Protocols / Runner Ports
# ==========================================

class LLMProvider(Protocol):
    @property
    def profile(self) -> ModelProfile: ...

    @property
    def available(self) -> bool: ...

    async def complete(self, request: LLMRequest) -> LLMResponse: ...

    async def close(self) -> None: ...


class MemoryStore(Protocol):
    async def recall(
        self,
        client_id: str | None,
        limit: int,
    ) -> list["MemoryRecord"]: ...

    async def save(self, record: "MemoryRecord") -> None: ...


# ==========================================
# Agent Output Models
# ==========================================

class MeetingUnderstandingOutput(StrictModel):
    meeting_type: MeetingType = MeetingType.GENERAL
    meeting_title: str | None = None
    theme: str | None = None
    rationale: str = "Standard meeting review"
    confidence: float = Field(default=0.95, ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def _normalize_understanding(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "title" in data and "meeting_title" not in data:
                data["meeting_title"] = data.pop("title")
        return data


class SummaryOutput(StrictModel):
    executive_summary: str
    suggested_title: str | None = None
    key_points: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.95, ge=0, le=1)
    prompt_version: str = "2.0.0"
    raw_agent_response: str | None = None
    agent_status: str = "completed"

    @model_validator(mode="before")
    @classmethod
    def _normalize_summary(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "summary" in data and "executive_summary" not in data:
                data["executive_summary"] = data.pop("summary")
            if "title" in data and "suggested_title" not in data:
                data["suggested_title"] = data.pop("title")
            if "bullet_points" in data and "key_points" not in data:
                data["key_points"] = data.pop("bullet_points")
            if "key_takeaways" in data and "key_points" not in data:
                data["key_points"] = data.pop("key_takeaways")
        return data


class ActionItem(StrictModel):
    task: str | None = None
    action: str = ""
    description: str = ""
    owner: str | None = None
    deadline: str | None = None
    deadline_text: str | None = None
    priority: str | None = "Medium"
    source: str | None = None
    evidence: str | None = None
    evidence_quote: str | None = None
    recipient: str | None = None
    confidence: float = Field(default=0.95, ge=0, le=1)
    status: str | None = None
    success_criteria: str | None = None
    evidence_speaker: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_action(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Extract main task text
            task_val = data.get("task") or data.get("action") or data.get("description") or ""
            if task_val:
                data["task"] = str(task_val).strip()
                data["action"] = str(task_val).strip()
                data["description"] = str(task_val).strip()

            if "assignee" in data and not data.get("owner"):
                data["owner"] = data.pop("assignee")
            if "due_date" in data:
                if not data.get("deadline"):
                    data["deadline"] = data["due_date"]
                if not data.get("deadline_text"):
                    data["deadline_text"] = data["due_date"]
            if "deadline" in data and not data.get("deadline_text"):
                data["deadline_text"] = data["deadline"]
            elif "deadline_text" in data and not data.get("deadline"):
                data["deadline"] = data["deadline_text"]
            if "source" in data:
                if not data.get("evidence"):
                    data["evidence"] = data["source"]
                if not data.get("evidence_quote"):
                    data["evidence_quote"] = data["source"]
            elif "evidence" in data and not data.get("source"):
                data["source"] = data["evidence"]
            elif "evidence_quote" in data and not data.get("source"):
                data["source"] = data["evidence_quote"]
            if "evidence" in data and not data.get("evidence_quote"):
                data["evidence_quote"] = data["evidence"]
            elif "evidence_quote" in data and not data.get("evidence"):
                data["evidence"] = data["evidence_quote"]
        return data


class ActionOutput(StrictModel):
    actions: list[ActionItem] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    action_summary: str | None = None
    confidence: float = Field(default=0.95, ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def _normalize_actions(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "actions" in data and not data.get("action_items"):
                data["action_items"] = data["actions"]
            elif "action_items" in data and not data.get("actions"):
                data["actions"] = data["action_items"]
            if "summary" in data and not data.get("action_summary"):
                data["action_summary"] = data["summary"]
            elif "actions_summary" in data and not data.get("action_summary"):
                data["action_summary"] = data["actions_summary"]
        return data

    @model_validator(mode="after")
    def _sync_actions(self) -> "ActionOutput":
        if self.actions and not self.action_items:
            self.action_items = list(self.actions)
        elif self.action_items and not self.actions:
            self.actions = list(self.action_items)
        return self


ActionAgentOutput = ActionOutput


class Decision(StrictModel):
    description: str
    rationale: str | None = None
    approved_by: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.95, ge=0, le=1)
    evidence_quote: str | None = None
    impact: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_decision(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "decision" in data and "description" not in data:
                data["description"] = data.pop("decision")
            if "approvers" in data and "approved_by" not in data:
                data["approved_by"] = data.pop("approvers")
        return data


class DecisionOutput(StrictModel):
    decisions: list[Decision] = Field(default_factory=list)
    confidence: float = Field(default=0.95, ge=0, le=1)


class Requirement(StrictModel):
    description: str
    category: str = "Functional"
    priority: str = "Medium"
    confidence: float = Field(default=0.92, ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def _normalize_requirement(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "requirement" in data and "description" not in data:
                data["description"] = data.pop("requirement")
        return data


class RequirementOutput(StrictModel):
    requirements: list[Requirement] = Field(default_factory=list)
    confidence: float = Field(default=0.95, ge=0, le=1)


class Risk(StrictModel):
    description: str
    owner: str | None = None
    severity: str = "Medium"
    probability: str = "Medium"
    impact: str = "Medium"
    mitigation: str | None = None
    confidence: float = Field(default=0.90, ge=0, le=1)
    evidence_quote: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_risk(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "risk" in data and "description" not in data:
                data["description"] = data.pop("risk")
        return data


class RiskOutput(StrictModel):
    risks: list[Risk] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.95, ge=0, le=1)


class SentimentOutput(StrictModel):
    overall: str = "Constructive & Professional"
    client_mood: str = "Engaged & Aligned"
    team_mood: str = "Focused on Execution"
    polarity_score: float = Field(default=0.75, ge=-1.0, le=1.0)
    engagement_level: str = "High"
    friction_points: list[str] = Field(default_factory=list)
    alignment_signals: list[str] = Field(default_factory=list)
    speaker_sentiments: dict[str, str] = Field(default_factory=dict)
    chronological_shifts: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.92, ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def _normalize_sentiment(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "sentiment" in data and "overall" not in data:
                data["overall"] = data.pop("sentiment")
            if "score" in data and "polarity_score" not in data:
                try:
                    data["polarity_score"] = float(data.pop("score"))
                except Exception:
                    pass
        return data


class Topic(StrictModel):
    name: str
    summary: str
    duration_minutes: float | None = None
    confidence: float = Field(default=0.92, ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def _normalize_topic(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "topic" in data and "name" not in data:
                data["name"] = data.pop("topic")
            if "title" in data and "name" not in data:
                data["name"] = data.pop("title")
            if "description" in data and "summary" not in data:
                data["summary"] = data.pop("description")
        return data


class TopicOutput(StrictModel):
    topics: list[Topic] = Field(default_factory=list)
    confidence: float = Field(default=0.95, ge=0, le=1)


class Deadline(StrictModel):
    source_text: str
    normalized_date: str | date | None = None
    deadline_text: str | None = None
    owner: str | None = None
    related_action: str | None = None
    confidence: float = Field(default=0.90, ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def _normalize_deadline(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "task" in data and "source_text" not in data:
                data["source_text"] = data.pop("task")
            if "description" in data and "source_text" not in data:
                data["source_text"] = data.pop("description")
            if "date" in data and "normalized_date" not in data:
                data["normalized_date"] = data.pop("date")
        return data


class DeadlineOutput(StrictModel):
    deadlines: list[Deadline] = Field(default_factory=list)
    confidence: float = Field(default=0.95, ge=0, le=1)


class OpenQuestion(StrictModel):
    question: str
    owner: str | None = None
    status: str = "open"
    confidence: float = Field(default=0.92, ge=0, le=1)


class QuestionOutput(StrictModel):
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    confidence: float = Field(default=0.95, ge=0, le=1)


class FollowUpTask(StrictModel):
    description: str
    owner: str | None = None
    due_date: str | date | None = None
    agenda_item: str | None = None
    confidence: float = Field(default=0.92, ge=0, le=1)


class FollowUpOutput(StrictModel):
    follow_up_tasks: list[FollowUpTask] = Field(default_factory=list)
    next_meeting_agenda: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.95, ge=0, le=1)


class TurboDeliverablesOutput(StrictModel):
    meeting_title: str | None = None
    executive_summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    action_summary: str | None = None
    action_items: list[ActionItem] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_turbo_deliv(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "summary" in data and not data.get("executive_summary"):
                data["executive_summary"] = data["summary"]
            if "actions" in data and not data.get("action_items"):
                data["action_items"] = data["actions"]
            if "bullet_points" in data and not data.get("key_points"):
                data["key_points"] = data["bullet_points"]
        return data


class TurboIntelligenceOutput(StrictModel):
    meeting_type: MeetingType = MeetingType.GENERAL
    rationale: str = "Standard operational session review"
    sentiment: SentimentOutput = Field(default_factory=SentimentOutput)
    topics: list[Topic] = Field(default_factory=list)
    requirements: list[Requirement] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    follow_up_tasks: list[FollowUpTask] = Field(default_factory=list)
    next_meeting_agenda: list[str] = Field(default_factory=list)


class Conflict(StrictModel):
    category: str
    description: str
    severity: str


class MemoryFinding(StrictModel):
    category: str
    description: str
    related_meeting_id: str | None = None


class ValidationReport(StrictModel):
    schema_valid: bool
    missing_fields: list[str]
    duplicates_removed: dict[str, int]
    conflicts: list[Conflict]
    memory_findings: list[MemoryFinding]
    reliability_score: float = Field(ge=0, le=1)


class ConfidenceScores(StrictModel):
    overall: float = Field(ge=0, le=1)
    by_agent: dict[AgentName, float]


class ModelInvocation(StrictModel):
    agent: AgentName
    provider: LLMProviderName
    model: str
    cached: bool
    attempts: int = Field(ge=1)
    latency_ms: float = Field(ge=0)
    usage: LLMUsage


class CostSummary(StrictModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    estimated_cost: float = Field(ge=0)


class MeetingIntelligenceResult(StrictModel):
    version: str
    job_id: UUID
    meeting_id: str
    meeting_title: str | None = None
    generated_at: datetime
    meeting_summary: str
    key_points: list[str] = Field(default_factory=list)
    meeting_type: MeetingType
    participants: list[str]
    decisions: list[Decision]
    action_items: list[ActionItem]
    action_summary: str | None = None
    owners: list[str]
    deadlines: list[Deadline]
    risks: list[Risk]
    blockers: list[str]
    requirements: list[Requirement]
    topics: list[Topic]
    sentiment: SentimentOutput
    open_questions: list[OpenQuestion]
    follow_up_tasks: list[FollowUpTask]
    next_meeting_agenda: list[str]
    confidence_scores: ConfidenceScores
    validation: ValidationReport
    cost: CostSummary
    model_trace: list[ModelInvocation]


class MemoryRecord(StrictModel):
    meeting_id: str
    title: str = ""
    client_id: str | None = None
    occurred_at: datetime
    summary: str
    decisions: list[Decision] = Field(default_factory=list)
    pending_action_items: list[ActionItem] = Field(default_factory=list)
    participants: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: str | None = None


# ==========================================
# Milestone Boundary Contracts & Validators
# ==========================================

class M2ToM3Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["1.0"] = "1.0"
    job_id: UUID
    meeting: MeetingReadyRequest
    preprocessing: PreprocessedTranscript


class M3InputContractValidator:
    def validate(self, job: JobRecord) -> M2ToM3Contract:
        if job.preprocessed_transcript is None:
            raise M3ContractValidationError(
                "missing_m2_output",
                "Model 3 requires validated M2 preprocessing output",
            )
        try:
            validated = ValidatedM2Output(
                job_id=job.id,
                preprocessing=job.preprocessed_transcript,
            )
            return M2ToM3Contract(
                job_id=validated.job_id,
                meeting=job.request,
                preprocessing=validated.preprocessing,
            )
        except ValidationError as exc:
            raise M3ContractValidationError(
                "invalid_m2_output",
                "M2 output failed validation for Model 3 consumption",
            ) from exc


# ==========================================
# AI Brain Configuration & Settings
# ==========================================

class AIBrainSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AUTOMATION_AI_",
        case_sensitive=False,
        extra="ignore",
    )

    result_version: str = "1.0.0"
    prompt_version: str = "1.0.0"
    provider_priority: str = "openrouter,groq,openai,anthropic,gemini"
    request_timeout_seconds: float = Field(default=20.0, gt=0)
    max_parallel_agents: int = Field(default=5, ge=1, le=10)
    agent_max_attempts: int = Field(default=2, ge=1, le=5)
    retry_base_seconds: float = Field(default=0.5, ge=0, le=10)
    context_max_tokens: int = Field(default=32_000, ge=500)
    max_output_tokens: int = Field(default=1_500, ge=100)
    memory_meeting_limit: int = Field(default=5, ge=0, le=50)
    cache_ttl_seconds: int = Field(default=3600, ge=0)

    # MongoDB Enterprise Memory Store Configuration
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "mom_ai_brain"
    mongodb_enabled: bool = True

    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "qwen/qwen3.8-27b"
    groq_quality_rank: int = Field(default=5, ge=1, le=5)
    groq_cost_rank: int = Field(default=1, ge=1, le=5)
    groq_context_tokens: int = Field(default=128_000, ge=1)
    groq_input_cost_per_million: float = Field(default=0, ge=0)
    groq_output_cost_per_million: float = Field(default=0, ge=0)

    openai_api_key: SecretStr | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4.1-mini"
    openai_quality_rank: int = Field(default=4, ge=1, le=5)
    openai_cost_rank: int = Field(default=2, ge=1, le=5)
    openai_context_tokens: int = Field(default=128_000, ge=1)
    openai_input_cost_per_million: float = Field(default=0, ge=0)
    openai_output_cost_per_million: float = Field(default=0, ge=0)

    anthropic_api_key: SecretStr | None = None
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_quality_rank: int = Field(default=5, ge=1, le=5)
    anthropic_cost_rank: int = Field(default=4, ge=1, le=5)
    anthropic_context_tokens: int = Field(default=200_000, ge=1)
    anthropic_input_cost_per_million: float = Field(default=0, ge=0)
    anthropic_output_cost_per_million: float = Field(default=0, ge=0)

    gemini_api_key: SecretStr | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_model: str = "gemini-2.5-flash"
    gemini_quality_rank: int = Field(default=4, ge=1, le=5)
    gemini_cost_rank: int = Field(default=1, ge=1, le=5)
    gemini_context_tokens: int = Field(default=1_000_000, ge=1)
    gemini_input_cost_per_million: float = Field(default=0, ge=0)
    gemini_output_cost_per_million: float = Field(default=0, ge=0)

    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "nvidia/nemotron-3.5-lightning:free"
    openrouter_quality_rank: int = Field(default=4, ge=1, le=5)
    openrouter_cost_rank: int = Field(default=2, ge=1, le=5)
    openrouter_context_tokens: int = Field(default=200_000, ge=1)
    openrouter_input_cost_per_million: float = Field(default=0, ge=0)
    openrouter_output_cost_per_million: float = Field(default=0, ge=0)
    openrouter_http_referer: str | None = Field(default=None, max_length=500)
    openrouter_app_name: str | None = Field(default=None, max_length=100)

    allow_unauthenticated: bool = False

    @property
    def provider_priority_list(self) -> list[LLMProviderName]:
        return [LLMProviderName(value.strip()) for value in self.provider_priority.split(",")]

    def profile_for(self, provider: LLMProviderName) -> ModelProfile:
        prefix = provider.value
        return ModelProfile(
            provider=provider,
            model=str(getattr(self, f"{prefix}_model")),
            quality_rank=int(getattr(self, f"{prefix}_quality_rank")),
            cost_rank=int(getattr(self, f"{prefix}_cost_rank")),
            max_context_tokens=int(getattr(self, f"{prefix}_context_tokens")),
            input_cost_per_million=float(getattr(self, f"{prefix}_input_cost_per_million")),
            output_cost_per_million=float(getattr(self, f"{prefix}_output_cost_per_million")),
        )

    @model_validator(mode="after")
    def validate_configuration(self) -> "AIBrainSettings":
        priorities = [value.strip() for value in self.provider_priority.split(",")]
        if not priorities or any(not value for value in priorities):
            raise ValueError("provider_priority must contain at least one provider")
        try:
            providers = [LLMProviderName(value) for value in priorities]
        except ValueError as exc:
            raise ValueError("provider_priority contains an unsupported provider") from exc
        if len(set(providers)) != len(providers):
            raise ValueError("provider_priority cannot contain duplicates")
        configured_keys = [getattr(self, f"{provider.value}_api_key") for provider in providers]
        if not self.allow_unauthenticated and not any(
            key is not None and bool(key.get_secret_value().strip()) for key in configured_keys
        ):
            raise ValueError("At least one configured LLM provider API key is required")
        return self


@lru_cache
def get_ai_brain_settings() -> AIBrainSettings:
    try:
        return AIBrainSettings()
    except Exception:
        return AIBrainSettings(allow_unauthenticated=True)
