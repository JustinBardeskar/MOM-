import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from app.ai_brain.agents import (
    AgentOrchestrator,
    AgentRuntime,
    ConfidenceAgent,
    ConflictAgent,
    MemoryValidationAgent,
    ValidationLayer,
    ValidatorAgent,
)
from app.ai_brain.memory import InMemoryMemoryStore, MemoryManager
from app.ai_brain.models import (
    ActionItem,
    AgentName,
    AIBrainSettings,
    LLMProviderName,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    M3InputContractValidator,
    MemoryRecord,
    ModelProfile,
)
from app.ai_brain.pipeline import (
    AIBrainPipeline,
    M4InputContractValidator,
    M5InputContractValidator,
    M6InputContractValidator,
    Model3PipelineOrchestrator,
    Model4PipelineOrchestrator,
    Model5PipelineOrchestrator,
    Model6PipelineOrchestrator,
)
from app.ai_brain.prompts import PromptManager
from app.ai_brain.providers import (
    AIBrainMonitor,
    CacheManager,
    ContextManager,
    CostOptimizer,
    ModelRouter,
    build_providers,
)
from app.domain import (
    ContextBundle,
    JobRecord,
    JobStatus,
    MeetingReadyRequest,
    MilestoneName,
    Participant,
    PipelineStage,
    PreprocessedTranscript,
    PreprocessingStatistics,
    ProcessingPath,
    TranscriptChunk,
    TranscriptSegment,
    UnifiedTranscript,
)
from app.infrastructure import InMemoryJobRepository
from app.integration import (
    M1M2PipelineOrchestrator,
    M1ToM2Contract,
    MilestoneContractValidator,
    StageReporter,
)
from app.services import AutomationPipelineCoordinator


class DeterministicLLMProvider:
    def __init__(
        self,
        fail_first_agent: AgentName | None = None,
        profile: ModelProfile | None = None,
        event_log: list[str] | None = None,
    ) -> None:
        self._profile = profile or ModelProfile(
            provider=LLMProviderName.OPENAI,
            model="deterministic-test-model",
            quality_rank=5,
            cost_rank=1,
            max_context_tokens=100_000,
            input_cost_per_million=1,
            output_cost_per_million=2,
        )
        self.fail_first_agent = fail_first_agent
        self.event_log = event_log
        self.calls = 0
        self.calls_by_agent: dict[AgentName, int] = {}
        self.active = 0
        self.max_active = 0

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    @property
    def available(self) -> bool:
        return True

    async def complete(self, request: LLMRequest) -> LLMResponse:
        agent = self._agent_from_prompt(request.system_prompt)
        if self.event_log is not None:
            self.event_log.append(f"llm:{agent.value}")
        self.calls += 1
        self.calls_by_agent[agent] = self.calls_by_agent.get(agent, 0) + 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.02)
        self.active -= 1
        if self.fail_first_agent == agent and self.calls_by_agent[agent] == 1:
            return LLMResponse(text="not-json")
        return LLMResponse(
            text=json.dumps(self._payload(agent)),
            usage=LLMUsage(input_tokens=100, output_tokens=20),
            provider_request_id=f"request-{self.calls}",
        )

    async def close(self) -> None:
        return None

    @staticmethod
    def _agent_from_prompt(prompt: str) -> AgentName:
        objectives = {
            "Agent: action": AgentName.ACTION,
            "Agent: summary": AgentName.SUMMARY,
            "Agent: decision": AgentName.DECISION,
            "Agent: meeting_understanding": AgentName.MEETING_UNDERSTANDING,
            "Agent: requirement": AgentName.REQUIREMENT,
            "Agent: risk": AgentName.RISK,
            "Agent: sentiment": AgentName.SENTIMENT,
            "Agent: topic": AgentName.TOPIC,
            "Agent: deadline": AgentName.DEADLINE,
            "Agent: question": AgentName.QUESTION,
            "ACTION EXTRACTION AGENT": AgentName.ACTION,
            "actionable tasks": AgentName.ACTION,
            "explicit tasks": AgentName.ACTION,
            "Classify the meeting": AgentName.MEETING_UNDERSTANDING,
            "executive-ready business summary": AgentName.SUMMARY,
            "executive summary": AgentName.SUMMARY,
            "business summary": AgentName.SUMMARY,
            "final decisions": AgentName.DECISION,
            "architectural approvals": AgentName.DECISION,
            "functional and non-functional": AgentName.REQUIREMENT,
            "risks, blockers": AgentName.RISK,
            "risk": AgentName.RISK,
            "client and team mood": AgentName.SENTIMENT,
            "discussion topics": AgentName.TOPIC,
            "relative dates": AgentName.DEADLINE,
            "unresolved questions": AgentName.QUESTION,
            "follow-up tasks": AgentName.FOLLOW_UP,
        }
        for text, agent in objectives.items():
            if text in prompt:
                return agent
        return AgentName.GENERAL if hasattr(AgentName, "GENERAL") else AgentName.SUMMARY

    @staticmethod
    def _payload(agent: AgentName) -> dict[str, object]:
        payloads: dict[AgentName, dict[str, object]] = {
            AgentName.MEETING_UNDERSTANDING: {
                "meeting_type": "client",
                "rationale": "The team reviewed a client delivery.",
                "confidence": 0.95,
            },
            AgentName.SUMMARY: {
                "executive_summary": "The client delivery plan was approved.",
                "key_points": ["Delivery plan", "Testing"],
                "confidence": 0.9,
            },
            AgentName.ACTION: {
                "action_items": [
                    {
                        "description": "Send the test report",
                        "owner": "Maya",
                        "deadline_text": "tomorrow",
                        "status": "pending",
                        "confidence": 0.9,
                    },
                    {
                        "description": "Send the test report.",
                        "owner": "Maya",
                        "deadline_text": "tomorrow",
                        "status": "pending",
                        "confidence": 0.8,
                    },
                ],
                "confidence": 0.9,
            },
            AgentName.DECISION: {
                "decisions": [
                    {
                        "description": "Approve the delivery plan",
                        "approved_by": ["Client"],
                        "confidence": 0.9,
                    }
                ],
                "confidence": 0.9,
            },
            AgentName.REQUIREMENT: {
                "requirements": [
                    {
                        "description": "Provide the test report",
                        "category": "functional",
                        "priority": "high",
                        "confidence": 0.85,
                    }
                ],
                "confidence": 0.85,
            },
            AgentName.RISK: {
                "risks": [
                    {
                        "description": "Testing may be delayed",
                        "severity": "medium",
                        "mitigation": "Run tests in parallel",
                        "owner": "Maya",
                        "confidence": 0.8,
                    }
                ],
                "blockers": ["Pending environment access"],
                "confidence": 0.8,
            },
            AgentName.SENTIMENT: {
                "overall": "positive",
                "client_mood": "satisfied",
                "team_mood": "focused",
                "evidence": ["The plan was approved"],
                "confidence": 0.8,
            },
            AgentName.TOPIC: {
                "topics": [
                    {
                        "name": "Delivery",
                        "summary": "Delivery and testing plan",
                        "confidence": 0.85,
                    }
                ],
                "confidence": 0.85,
            },
            AgentName.DEADLINE: {
                "deadlines": [
                    {
                        "source_text": "tomorrow",
                        "normalized_date": "2026-08-08",
                        "owner": "Maya",
                        "related_action": "Send the test report",
                        "confidence": 0.9,
                    }
                ],
                "confidence": 0.9,
            },
            AgentName.QUESTION: {
                "open_questions": [
                    {
                        "question": "When will environment access be approved?",
                        "owner": None,
                        "status": "open",
                        "confidence": 0.8,
                    }
                ],
                "confidence": 0.8,
            },
            AgentName.FOLLOW_UP: {
                "follow_up_tasks": [
                    {
                        "description": "Review the test report",
                        "owner": "Client",
                        "due_date": "2026-08-09",
                        "agenda_item": "Testing status",
                        "confidence": 0.8,
                    }
                ],
                "next_meeting_agenda": ["Testing status"],
                "confidence": 0.8,
            },
        }
        return payloads[agent]


class TrackingRepository(InMemoryJobRepository):
    def __init__(self) -> None:
        super().__init__()
        self.stages: list[PipelineStage] = []

    async def save(self, job: JobRecord) -> None:
        self.stages.append(job.current_stage)
        await super().save(job)


class EndToEndM1:
    def __init__(self, event_log: list[str]) -> None:
        self._event_log = event_log

    async def execute(
        self,
        _: JobRecord,
        __: Path,
        ___: StageReporter,
    ) -> UnifiedTranscript:
        self._event_log.append("m1")
        return UnifiedTranscript(
            text="Maya will send the test report tomorrow.",
            language="en",
            duration_seconds=5,
            segments=[
                TranscriptSegment(
                    start_seconds=0,
                    end_seconds=5,
                    text="Maya will send the test report tomorrow.",
                )
            ],
            source_path=ProcessingPath.DIRECT_TRANSCRIPT,
        )


class EndToEndM2:
    def __init__(self, event_log: list[str]) -> None:
        self._event_log = event_log

    async def execute(
        self,
        _: M1ToM2Contract,
        __: StageReporter,
    ) -> PreprocessedTranscript:
        self._event_log.append("m2")
        return make_preprocessed()


def make_request() -> MeetingReadyRequest:
    return MeetingReadyRequest(
        event_id="m3-event",
        meeting_id="m3-meeting",
        provider="microsoft_teams",
        title="Client delivery review",
        ended_at="2026-08-07T10:00:00Z",
        transcript={
            "url": "https://storage.example.com/meeting.vtt",
            "content_type": "text/vtt",
        },
        participants=[Participant(display_name="Maya")],
        metadata={"client_id": "client-1"},
    )


def test_openrouter_provider_configuration() -> None:
    settings = AIBrainSettings(
        provider_priority="openrouter",
        openrouter_api_key=SecretStr("test-key"),
    )
    assert settings.provider_priority_list == [LLMProviderName.OPENROUTER]
    providers = build_providers(settings)
    assert len(providers) == 1
    assert providers[0].profile.provider == LLMProviderName.OPENROUTER
    assert providers[0].profile.model == settings.openrouter_model


def make_preprocessed(*, valid: bool = True) -> PreprocessedTranscript:
    segments = [
        TranscriptSegment(
            start_seconds=0,
            end_seconds=5,
            text="Maya will send the test report tomorrow.",
            speaker="SPEAKER_01",
        )
    ]
    chunks = (
        [
            TranscriptChunk(
                id="chunk-0",
                index=0,
                text=segments[0].text,
                start_seconds=0,
                end_seconds=5,
                speaker_ids=["SPEAKER_01"],
                token_count=10,
                source_segment_indexes=[0],
            )
        ]
        if valid
        else []
    )
    contexts = (
        [
            ContextBundle(
                id="context-0",
                chunk_id="chunk-0",
                meeting_id="m3-meeting",
                meeting_title="Client delivery review",
                provider="microsoft_teams",
                ended_at="2026-08-07T10:00:00Z",
                text=segments[0].text,
                speaker_ids=["SPEAKER_01"],
                start_seconds=0,
                end_seconds=5,
                token_count=10,
            )
        ]
        if valid
        else []
    )
    return PreprocessedTranscript(
        version="test",
        text=segments[0].text,
        language="en",
        duration_seconds=5,
        segments=segments,
        speakers=[],
        chunks=chunks,
        contexts=contexts,
        statistics=PreprocessingStatistics(
            original_characters=len(segments[0].text),
            cleaned_characters=len(segments[0].text),
            fillers_removed=0,
            noise_segments_removed=0,
            timestamps_corrected=0,
            speaker_count=0,
            chunk_count=len(chunks),
        ),
    )


async def make_job(repository: TrackingRepository, *, valid: bool = True) -> JobRecord:
    request = make_request()
    job = JobRecord(
        event_id=request.event_id,
        meeting_id=request.meeting_id,
        idempotency_key=request.event_id,
        status=JobStatus.AWAITING_ANALYSIS,
        selected_path=ProcessingPath.DIRECT_TRANSCRIPT,
        current_stage=PipelineStage.PREPROCESSED_TRANSCRIPT_READY,
        progress_percent=60,
        planned_steps=[],
        request=request,
        preprocessed_transcript=make_preprocessed(valid=valid),
    )
    await repository.create_or_get(job)
    return job


def make_model3(
    repository: TrackingRepository,
    provider: DeterministicLLMProvider,
    memory_store: InMemoryMemoryStore | None = None,
    max_attempts: int = 2,
) -> Model3PipelineOrchestrator:
    costs = CostOptimizer()
    monitor = AIBrainMonitor()
    memory = MemoryManager(memory_store or InMemoryMemoryStore(), 5)
    runtime = AgentRuntime(
        router=ModelRouter([provider], costs),
        prompts=PromptManager("test"),
        contexts=ContextManager(1000),
        cache=CacheManager(3600),
        costs=costs,
        monitor=monitor,
        max_output_tokens=1000,
    )
    agents = AgentOrchestrator(
        runtime=runtime,
        memory=memory,
        max_parallel_agents=5,
        max_attempts=max_attempts,
        retry_base_seconds=0,
    )
    validation = ValidationLayer(
        ValidatorAgent(),
        ConflictAgent(),
        ConfidenceAgent(),
        MemoryValidationAgent(),
    )
    brain = AIBrainPipeline("test", agents, validation, memory, costs, monitor)
    return Model3PipelineOrchestrator(
        repository,
        brain,
        M3InputContractValidator(),
    )


def test_model3_runs_full_agent_pipeline_and_persists_json() -> None:
    async def scenario() -> None:
        repository = TrackingRepository()
        job = await make_job(repository)
        provider = DeterministicLLMProvider()
        memory_store = InMemoryMemoryStore()
        await memory_store.save(
            MemoryRecord(
                meeting_id="previous-meeting",
                client_id="client-1",
                occurred_at=datetime(2026, 8, 1, tzinfo=UTC),
                summary="Testing was discussed.",
                pending_action_items=[
                    ActionItem(
                        description="Send the test report",
                        owner="Maya",
                        confidence=0.8,
                    )
                ],
            )
        )

        await make_model3(repository, provider, memory_store).run(job.id)

        stored = await repository.get(job.id)
        assert stored is not None
        assert stored.status == JobStatus.AWAITING_DELIVERY
        assert stored.current_stage == PipelineStage.FINAL_STRUCTURED_JSON_READY
        assert stored.failed_milestone is None
        assert stored.result is not None
        assert stored.result["meeting_type"] == "client"
        assert len(stored.result["action_items"]) == 1
        assert stored.result["validation"]["duplicates_removed"]["action_items"] == 1
        assert stored.result["validation"]["memory_findings"][0]["category"] == (
            "repeated_pending_action"
        )
        assert len(stored.result["model_trace"]) == 11
        assert provider.calls == 11
        assert provider.max_active > 1
        assert repository.stages.index(PipelineStage.MEETING_UNDERSTANDING) < (
            repository.stages.index(PipelineStage.PARALLEL_AGENT_ANALYSIS)
        )
        assert repository.stages.index(PipelineStage.VALIDATE_AGENT_OUTPUTS) < (
            repository.stages.index(PipelineStage.DETECT_AGENT_CONFLICTS)
        )

    asyncio.run(scenario())


def test_complete_m1_m2_model3_chain_executes_in_order(tmp_path: Path) -> None:
    async def scenario() -> None:
        repository = TrackingRepository()
        request = make_request()
        job = JobRecord(
            event_id=request.event_id,
            meeting_id=request.meeting_id,
            idempotency_key=request.event_id,
            selected_path=ProcessingPath.DIRECT_TRANSCRIPT,
            planned_steps=[],
            request=request,
        )
        await repository.create_or_get(job)
        event_log: list[str] = []
        m1_m2 = M1M2PipelineOrchestrator(
            repository=repository,
            milestone1=EndToEndM1(event_log),
            milestone2=EndToEndM2(event_log),
            validator=MilestoneContractValidator(),
            work_directory=tmp_path,
            keep_work_files=False,
        )
        provider = DeterministicLLMProvider(event_log=event_log)
        coordinator = AutomationPipelineCoordinator(
            m1_m2=m1_m2,
            model3=make_model3(repository, provider),
            model4=None,
            model5=None,
            model6=None,
        )

        await coordinator.run(job.id)

        stored = await repository.get(job.id)
        assert stored is not None
        assert event_log[:3] == ["m1", "m2", "llm:meeting_understanding"]
        assert stored.status == JobStatus.AWAITING_DELIVERY
        assert stored.result is not None
        assert stored.result["meeting_summary"] == ("The client delivery plan was approved.")

    asyncio.run(scenario())


def test_agent_retry_recovers_from_invalid_json() -> None:
    async def scenario() -> None:
        repository = TrackingRepository()
        job = await make_job(repository)
        provider = DeterministicLLMProvider(fail_first_agent=AgentName.ACTION)

        await make_model3(repository, provider).run(job.id)

        stored = await repository.get(job.id)
        assert stored is not None
        assert stored.status == JobStatus.AWAITING_DELIVERY
        assert provider.calls_by_agent[AgentName.ACTION] == 2
        assert stored.result is not None
        action_trace = next(
            trace for trace in stored.result["model_trace"] if trace["agent"] == AgentName.ACTION
        )
        assert action_trace["attempts"] == 2

    asyncio.run(scenario())


def test_agent_retry_exhaustion_falls_back_gracefully() -> None:
    """When max_attempts=1 and an agent fails all retries, the pipeline should
    succeed using structured fallback output rather than crashing the entire job.
    This tests the resilient asyncio.gather(return_exceptions=True) behavior."""
    async def scenario() -> None:
        repository = TrackingRepository()
        job = await make_job(repository)
        provider = DeterministicLLMProvider(fail_first_agent=AgentName.ACTION)

        await make_model3(repository, provider, max_attempts=1).run(job.id)

        stored = await repository.get(job.id)
        assert stored is not None
        # Pipeline should succeed using fallback, not fail the entire job
        assert stored.status == JobStatus.AWAITING_DELIVERY
        # Result should be present (with fallback action items)
        assert stored.result is not None

    asyncio.run(scenario())


def make_model4(repository: TrackingRepository) -> Model4PipelineOrchestrator:
    return Model4PipelineOrchestrator(
        repository=repository,
        input_validator=M4InputContractValidator(),
    )


def make_model5(repository: TrackingRepository) -> Model5PipelineOrchestrator:
    return Model5PipelineOrchestrator(
        repository=repository,
        input_validator=M5InputContractValidator(),
    )


def make_model6(repository: TrackingRepository) -> Model6PipelineOrchestrator:
    return Model6PipelineOrchestrator(
        repository=repository,
        input_validator=M6InputContractValidator(),
    )


def test_model4_runs_pass_through_m3_output() -> None:
    async def scenario() -> None:
        repository = TrackingRepository()
        job = await make_job(repository)
        provider = DeterministicLLMProvider()

        await make_model3(repository, provider).run(job.id)
        await make_model4(repository).run(job.id)

        stored = await repository.get(job.id)
        assert stored is not None
        assert stored.status == JobStatus.AWAITING_DELIVERY
        assert stored.current_stage == PipelineStage.FINAL_STRUCTURED_JSON_READY
        assert stored.result is not None
        assert stored.result["meeting_type"] == "client"

    asyncio.run(scenario())


def test_model5_runs_after_model4_and_preserves_result() -> None:
    async def scenario() -> None:
        repository = TrackingRepository()
        job = await make_job(repository)
        provider = DeterministicLLMProvider()

        await make_model3(repository, provider).run(job.id)
        await make_model4(repository).run(job.id)
        await make_model5(repository).run(job.id)

        stored = await repository.get(job.id)
        assert stored is not None
        assert stored.status == JobStatus.AWAITING_DELIVERY
        assert stored.current_stage == PipelineStage.M5_VALIDATION
        assert stored.result is not None
        assert stored.result["meeting_type"] == "client"
        assert stored.failed_milestone is None

    asyncio.run(scenario())


def test_model6_runs_after_model5_and_persists_final_json() -> None:
    async def scenario() -> None:
        repository = TrackingRepository()
        job = await make_job(repository)
        provider = DeterministicLLMProvider()

        await make_model3(repository, provider).run(job.id)
        await make_model4(repository).run(job.id)
        await make_model5(repository).run(job.id)
        await make_model6(repository).run(job.id)

        stored = await repository.get(job.id)
        assert stored is not None
        assert stored.status == JobStatus.AWAITING_DELIVERY
        assert stored.current_stage == PipelineStage.FINAL_STRUCTURED_JSON_READY
        assert stored.result is not None
        assert stored.result["meeting_type"] == "client"
        assert stored.failed_milestone is None

    asyncio.run(scenario())


def test_invalid_m3_output_prevents_m4_execution() -> None:
    async def scenario() -> None:
        repository = TrackingRepository()
        request = make_request()
        job = JobRecord(
            event_id=request.event_id,
            meeting_id=request.meeting_id,
            idempotency_key=request.event_id,
            status=JobStatus.AWAITING_DELIVERY,
            selected_path=ProcessingPath.DIRECT_TRANSCRIPT,
            current_stage=PipelineStage.FINAL_STRUCTURED_JSON_READY,
            progress_percent=95,
            planned_steps=[],
            request=request,
            result={"invalid": "payload"},
        )
        await repository.create_or_get(job)

        await make_model4(repository).run(job.id)

        stored = await repository.get(job.id)
        assert stored is not None
        assert stored.status == JobStatus.FAILED
        assert stored.failed_milestone == MilestoneName.M4_VALIDATION
        assert stored.error_code == "invalid_m3_output"

    asyncio.run(scenario())


def test_cache_manager_avoids_repeated_provider_payloads() -> None:
    async def scenario() -> None:
        cache = CacheManager(60)
        profile = ModelProfile(
            provider=LLMProviderName.OPENAI,
            model="cache-test",
            quality_rank=5,
            cost_rank=1,
            max_context_tokens=1000,
        )
        request = LLMRequest(system_prompt="system", user_prompt="user")
        key = cache.key(AgentName.SUMMARY, profile, request)
        response = LLMResponse(text='{"result": true}')

        assert await cache.get(key) is None
        await cache.set(key, response)
        assert await cache.get(key) == response

    asyncio.run(scenario())


def test_model_router_selects_cheapest_capable_provider() -> None:
    cheap = DeterministicLLMProvider()
    expensive = DeterministicLLMProvider(
        profile=ModelProfile(
            provider=LLMProviderName.ANTHROPIC,
            model="expensive",
            quality_rank=5,
            cost_rank=5,
            max_context_tokens=100_000,
        )
    )
    selected = ModelRouter([expensive, cheap], CostOptimizer()).select(
        AgentName.SUMMARY,
        100,
    )
    assert selected is cheap


def test_ai_brain_configuration_requires_key_and_masks_secret() -> None:
    with pytest.raises(ValidationError, match="provider API key"):
        AIBrainSettings(
            _env_file=None,
            provider_priority="openai",
            openai_api_key=None,
        )

    settings = AIBrainSettings(
        _env_file=None,
        provider_priority="openai",
        openai_api_key=SecretStr("secret-test-key"),
    )
    assert settings.openai_api_key is not None
    assert "secret-test-key" not in repr(settings)
    assert settings.provider_priority_list == [LLMProviderName.OPENAI]
