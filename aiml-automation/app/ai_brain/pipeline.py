from dataclasses import dataclass
from datetime import UTC, datetime
import logging
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, ValidationError

from app.ai_brain.agents import (
    AgentOrchestrator,
    AgentRuntime,
    ConfidenceAgent,
    ConflictAgent,
    MemoryValidationAgent,
    ValidatedAnalysis,
    ValidationLayer,
    ValidatorAgent,
)
from app.ai_brain.memory import (
    InMemoryMemoryStore,
    MemoryManager,
    MongoMemoryStore,
    SQLiteMemoryStore,
)
from app.ai_brain.models import (
    ActionOutput,
    AgentName,
    AIBrainError,
    AIBrainSettings,
    CostSummary,
    DeadlineOutput,
    DecisionOutput,
    FollowUpOutput,
    LLMProvider,
    M2ToM3Contract,
    M3ContractValidationError,
    M3InputContractValidator,
    M4ContractValidationError,
    M5ContractValidationError,
    M6ContractValidationError,
    MeetingIntelligenceResult,
    MemoryStore,
    ModelInvocation,
    QuestionOutput,
    RequirementOutput,
    RiskOutput,
    SentimentOutput,
    SummaryOutput,
    TopicOutput,
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
    JobRecord,
    JobRepository,
    JobStatus,
    MeetingReadyRequest,
    MilestoneName,
    PipelineStage,
)
from app.integration import StageReporter

logger = logging.getLogger("automation.ai_brain.pipeline")


# ==========================================
# 1. Pipeline Contracts & Validators
# ==========================================

class M3ToM4Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["1.0"] = "1.0"
    job_id: UUID
    meeting: MeetingReadyRequest
    analysis: MeetingIntelligenceResult


class M4ToM5Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["1.0"] = "1.0"
    job_id: UUID
    meeting: MeetingReadyRequest
    result: MeetingIntelligenceResult


class M5ToM6Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["1.0"] = "1.0"
    job_id: UUID
    meeting: MeetingReadyRequest
    result: MeetingIntelligenceResult


class M4InputContractValidator:
    def validate(self, job: JobRecord) -> M3ToM4Contract:
        if job.result is None:
            raise M4ContractValidationError(
                "missing_m3_output",
                "Model 4 requires completed Model 3 intelligence analysis",
            )
        try:
            return M3ToM4Contract(
                job_id=job.id,
                meeting=job.request,
                analysis=MeetingIntelligenceResult.model_validate(job.result),
            )
        except ValidationError as exc:
            raise M4ContractValidationError(
                "invalid_m3_output",
                "Model 3 output failed validation for Model 4 consumption",
            ) from exc


class M5InputContractValidator:
    def validate(self, job: JobRecord) -> M4ToM5Contract:
        if job.result is None:
            raise M5ContractValidationError(
                "missing_m4_output",
                "Model 5 requires completed Model 4 validated result",
            )
        try:
            return M4ToM5Contract(
                job_id=job.id,
                meeting=job.request,
                result=MeetingIntelligenceResult.model_validate(job.result),
            )
        except ValidationError as exc:
            raise M5ContractValidationError(
                "invalid_m4_output",
                "Model 4 output failed validation for Model 5 consumption",
            ) from exc


class M6InputContractValidator:
    def validate(self, job: JobRecord) -> M5ToM6Contract:
        if job.result is None:
            raise M6ContractValidationError(
                "missing_m5_output",
                "Model 6 requires completed Model 5 validated result",
            )
        try:
            return M5ToM6Contract(
                job_id=job.id,
                meeting=job.request,
                result=MeetingIntelligenceResult.model_validate(job.result),
            )
        except ValidationError as exc:
            raise M6ContractValidationError(
                "invalid_m5_output",
                "Model 5 output failed validation for Model 6 consumption",
            ) from exc


# ==========================================
# 2. Model 3 AI Brain Core Pipeline
# ==========================================

class AIBrainPipeline:
    """Runs the complete Model 3 agent and validation workflow."""

    def __init__(
        self,
        version: str,
        orchestrator: AgentOrchestrator,
        validation: ValidationLayer,
        memory: MemoryManager,
        costs: CostOptimizer,
        monitor: AIBrainMonitor,
    ) -> None:
        self._version = version
        self._orchestrator = orchestrator
        self._validation = validation
        self._memory = memory
        self._costs = costs
        self._monitor = monitor

    async def analyze(
        self,
        contract: M2ToM3Contract,
        report_stage: StageReporter,
    ) -> MeetingIntelligenceResult:
        logger.info(
            "Starting AI Brain Analysis for job %s (Meeting: '%s', participants: %d)",
            contract.job_id,
            contract.meeting.title,
            len(contract.meeting.participants),
        )
        await self._monitor.start_run(contract.job_id)
        await self._costs.start_run(contract.job_id)
        analysis = await self._orchestrator.execute(contract, report_stage)
        validated = await self._validation.validate(analysis, report_stage)
        cost_summary = await self._costs.summary(contract.job_id)
        invocations = await self._monitor.invocations(contract.job_id)
        result = self._assemble(
            contract,
            validated,
            cost_summary,
            invocations,
        )
        await self._memory.remember(contract, result)
        logger.info(
            "AI Brain Pipeline FINISHED for job %s: Total Tokens: %d in + %d out = %d total | Est. Cost: $%.6f USD | Model Calls: %d",
            contract.job_id,
            cost_summary.input_tokens,
            cost_summary.output_tokens,
            cost_summary.input_tokens + cost_summary.output_tokens,
            cost_summary.estimated_cost,
            len(invocations),
        )
        return result

    def _assemble(
        self,
        contract: M2ToM3Contract,
        validated: ValidatedAnalysis,
        cost: CostSummary,
        model_trace: list[ModelInvocation],
    ) -> MeetingIntelligenceResult:
        outputs = validated.analysis.outputs
        summary = ValidatorAgent.get_output(outputs, AgentName.SUMMARY, SummaryOutput)
        actions = ValidatorAgent.get_output(outputs, AgentName.ACTION, ActionOutput)
        decisions = ValidatorAgent.get_output(outputs, AgentName.DECISION, DecisionOutput)
        requirements = ValidatorAgent.get_output(
            outputs,
            AgentName.REQUIREMENT,
            RequirementOutput,
        )
        risks = ValidatorAgent.get_output(outputs, AgentName.RISK, RiskOutput)
        sentiment = ValidatorAgent.get_output(
            outputs,
            AgentName.SENTIMENT,
            SentimentOutput,
        )
        topics = ValidatorAgent.get_output(outputs, AgentName.TOPIC, TopicOutput)
        deadlines = ValidatorAgent.get_output(
            outputs,
            AgentName.DEADLINE,
            DeadlineOutput,
        )
        questions = ValidatorAgent.get_output(
            outputs,
            AgentName.QUESTION,
            QuestionOutput,
        )
        follow_up = ValidatorAgent.get_output(
            outputs,
            AgentName.FOLLOW_UP,
            FollowUpOutput,
        )
        owners = sorted(
            {
                item.owner.strip()
                for item in actions.action_items
                if item.owner and item.owner.strip()
            }
        )
        raw_lines = [seg.text for seg in getattr(contract.preprocessing, "segments", []) if getattr(seg, "text", "").strip()]
        transcript_text = getattr(contract.preprocessing, "text", "") or " ".join(raw_lines) or ""
        from app.ai_brain.consensus import CrossAgentConsensusEngine
        suggested_title = (
            getattr(validated.analysis.meeting_understanding, "meeting_title", None)
            or getattr(summary, "suggested_title", None)
        )
        meeting_type_val = getattr(validated.analysis.meeting_understanding.meeting_type, "value", None)
        input_title = getattr(contract.meeting, "title", None) or getattr(contract.meeting, "meeting_title", None)

        resolved_title = CrossAgentConsensusEngine.generate_dynamic_meeting_title(
            transcript_text=transcript_text,
            current_title=input_title,
            suggested_title=suggested_title,
            meeting_type=meeting_type_val,
        )

        return MeetingIntelligenceResult(
            version=self._version,
            job_id=contract.job_id,
            meeting_id=contract.meeting.meeting_id,
            meeting_title=resolved_title,
            generated_at=datetime.now(UTC),
            meeting_summary=summary.executive_summary,
            key_points=summary.key_points,
            meeting_type=validated.analysis.meeting_understanding.meeting_type,
            participants=[
                participant.display_name for participant in contract.meeting.participants
            ],
            decisions=decisions.decisions,
            action_items=actions.action_items,
            owners=owners,
            deadlines=deadlines.deadlines,
            risks=risks.risks,
            blockers=risks.blockers,
            requirements=requirements.requirements,
            topics=topics.topics,
            sentiment=sentiment,
            open_questions=questions.open_questions,
            follow_up_tasks=follow_up.follow_up_tasks,
            next_meeting_agenda=follow_up.next_meeting_agenda,
            confidence_scores=validated.confidence,
            validation=validated.report,
            cost=cost,
            model_trace=model_trace,
        )


# ==========================================
# 3. Milestone Stage Orchestrators (M3, M4, M5, M6)
# ==========================================

class Model3PipelineOrchestrator:
    """Coordinates Model 3 contract validation and intelligence execution."""

    def __init__(
        self,
        repository: JobRepository,
        pipeline: AIBrainPipeline,
        validator: M3InputContractValidator | None = None,
        input_validator: M3InputContractValidator | None = None,
    ) -> None:
        self._repository = repository
        self._pipeline = pipeline
        self._validator = validator or input_validator or M3InputContractValidator()

    async def run(self, job_id: UUID) -> None:
        job = await self._repository.get(job_id)
        if job is None:
            return

        async def report_stage(stage: PipelineStage, progress_pct: int) -> None:
            current = await self._repository.get(job_id)
            if current is not None:
                updated = current.model_copy(
                    update={
                        "current_stage": stage,
                        "progress_percent": progress_pct,
                        "updated_at": datetime.now(UTC),
                    }
                )
                await self._repository.save(updated)

        try:
            await report_stage(PipelineStage.M2_TO_M3_HANDOFF, 60)
            contract = self._validator.validate(job)
            await report_stage(PipelineStage.MEETING_UNDERSTANDING, 63)
            result = await self._pipeline.analyze(contract, report_stage)
            current = await self._repository.get(job_id)
            if current is not None:
                completed = current.model_copy(
                    update={
                        "status": JobStatus.AWAITING_DELIVERY,
                        "current_stage": PipelineStage.FINAL_STRUCTURED_JSON_READY,
                        "progress_percent": 92,
                        "result": result.model_dump(mode="json"),
                        "failed_milestone": None,
                        "updated_at": datetime.now(UTC),
                    }
                )
                await self._repository.save(completed)
        except M3ContractValidationError as exc:
            logger.error("M3 Contract Validation Error for job %s: %s", job_id, exc)
            current = await self._repository.get(job_id)
            if current is not None:
                await self._repository.save(
                    current.model_copy(
                        update={
                            "status": JobStatus.FAILED,
                            "failed_milestone": MilestoneName.M3_VALIDATION,
                            "error_code": exc.code,
                            "error_message": exc.message,
                            "updated_at": datetime.now(UTC),
                        }
                    )
                )
        except AIBrainError as exc:
            logger.error("AI Brain execution error for job %s: %s", job_id, exc)
            current = await self._repository.get(job_id)
            if current is not None:
                await self._repository.save(
                    current.model_copy(
                        update={
                            "status": JobStatus.FAILED,
                            "failed_milestone": MilestoneName.M3,
                            "error_code": exc.code,
                            "error_message": exc.message,
                            "updated_at": datetime.now(UTC),
                        }
                    )
                )
        except Exception as exc:
            logger.exception("Unexpected AI Brain failure for job %s", job_id)
            current = await self._repository.get(job_id)
            if current is not None:
                await self._repository.save(
                    current.model_copy(
                        update={
                            "status": JobStatus.FAILED,
                            "failed_milestone": MilestoneName.M3,
                            "error_code": "ai_brain_unexpected_failure",
                            "error_message": str(exc),
                            "updated_at": datetime.now(UTC),
                        }
                    )
                )


class Model4PipelineOrchestrator:
    """Milestone 4 orchestrator."""

    def __init__(
        self,
        repository: JobRepository,
        input_validator: M4InputContractValidator | None = None,
        validator: M4InputContractValidator | None = None,
    ) -> None:
        self._repository = repository
        self._validator = validator or input_validator or M4InputContractValidator()

    async def run(self, job_id: UUID) -> None:
        job = await self._repository.get(job_id)
        if job is None or job.status == JobStatus.FAILED:
            return
        try:
            job = job.model_copy(update={"current_stage": PipelineStage.M3_TO_M4_HANDOFF, "progress_percent": 93, "updated_at": datetime.now(UTC)})
            await self._repository.save(job)
            contract = self._validator.validate(job)
            job = job.model_copy(update={"current_stage": PipelineStage.M4_ANALYSIS, "progress_percent": 94, "updated_at": datetime.now(UTC)})
            await self._repository.save(job)
            job = job.model_copy(update={"current_stage": PipelineStage.VALIDATE_M4_OUTPUT, "progress_percent": 95, "updated_at": datetime.now(UTC)})
            await self._repository.save(job)
            job = job.model_copy(update={"status": JobStatus.AWAITING_DELIVERY, "current_stage": PipelineStage.FINAL_STRUCTURED_JSON_READY, "progress_percent": 95, "updated_at": datetime.now(UTC)})
            await self._repository.save(job)
        except M4ContractValidationError as exc:
            job = job.model_copy(update={"status": JobStatus.FAILED, "failed_milestone": MilestoneName.M4_VALIDATION, "error_code": exc.code, "error_message": exc.message, "updated_at": datetime.now(UTC)})
            await self._repository.save(job)
        except Exception as exc:
            job = job.model_copy(update={"status": JobStatus.FAILED, "failed_milestone": MilestoneName.M4_VALIDATION, "error_code": "m4_unexpected_failure", "error_message": str(exc), "updated_at": datetime.now(UTC)})
            await self._repository.save(job)


class Model5PipelineOrchestrator:
    """Milestone 5 orchestrator."""

    def __init__(
        self,
        repository: JobRepository,
        input_validator: M5InputContractValidator | None = None,
        validator: M5InputContractValidator | None = None,
    ) -> None:
        self._repository = repository
        self._validator = validator or input_validator or M5InputContractValidator()

    async def run(self, job_id: UUID) -> None:
        job = await self._repository.get(job_id)
        if job is None or job.status == JobStatus.FAILED:
            return
        try:
            job = job.model_copy(update={"current_stage": PipelineStage.M4_TO_M5_HANDOFF, "progress_percent": 96, "updated_at": datetime.now(UTC)})
            await self._repository.save(job)
            contract = self._validator.validate(job)
            job = job.model_copy(update={"status": JobStatus.AWAITING_DELIVERY, "current_stage": PipelineStage.M5_VALIDATION, "progress_percent": 97, "updated_at": datetime.now(UTC)})
            await self._repository.save(job)
        except M5ContractValidationError as exc:
            job = job.model_copy(update={"status": JobStatus.FAILED, "failed_milestone": MilestoneName.M5_VALIDATION, "error_code": exc.code, "error_message": exc.message, "updated_at": datetime.now(UTC)})
            await self._repository.save(job)
        except Exception as exc:
            job = job.model_copy(update={"status": JobStatus.FAILED, "failed_milestone": MilestoneName.M5_VALIDATION, "error_code": "m5_unexpected_failure", "error_message": str(exc), "updated_at": datetime.now(UTC)})
            await self._repository.save(job)


class Model6PipelineOrchestrator:
    """Milestone 6 orchestrator: finalizes deliverables and marks job awaiting_delivery."""

    def __init__(
        self,
        repository: JobRepository,
        input_validator: M6InputContractValidator | None = None,
        validator: M6InputContractValidator | None = None,
    ) -> None:
        self._repository = repository
        self._validator = validator or input_validator or M6InputContractValidator()

    async def run(self, job_id: UUID) -> None:
        job = await self._repository.get(job_id)
        if job is None or job.status == JobStatus.FAILED:
            return
        try:
            job = job.model_copy(update={"current_stage": PipelineStage.M5_TO_M6_HANDOFF, "progress_percent": 98, "updated_at": datetime.now(UTC)})
            await self._repository.save(job)
            contract = self._validator.validate(job)
            job = job.model_copy(update={"current_stage": PipelineStage.M6_FINALIZATION, "progress_percent": 99, "updated_at": datetime.now(UTC)})
            await self._repository.save(job)
            job = job.model_copy(update={"status": JobStatus.AWAITING_DELIVERY, "current_stage": PipelineStage.FINAL_STRUCTURED_JSON_READY, "progress_percent": 100, "updated_at": datetime.now(UTC)})
            await self._repository.save(job)
        except M6ContractValidationError as exc:
            job = job.model_copy(update={"status": JobStatus.FAILED, "failed_milestone": MilestoneName.M6_VALIDATION, "error_code": exc.code, "error_message": exc.message, "updated_at": datetime.now(UTC)})
            await self._repository.save(job)
        except Exception as exc:
            job = job.model_copy(update={"status": JobStatus.FAILED, "failed_milestone": MilestoneName.M6_VALIDATION, "error_code": "m6_unexpected_failure", "error_message": str(exc), "updated_at": datetime.now(UTC)})
            await self._repository.save(job)


# ==========================================
# 4. Dependency Injection Factory
# ==========================================

@dataclass
class AIBrainResources:
    orchestrator: Model3PipelineOrchestrator
    model4: Model4PipelineOrchestrator
    model5: Model5PipelineOrchestrator
    model6: Model6PipelineOrchestrator
    providers: list[LLMProvider]

    async def close(self) -> None:
        for provider in self.providers:
            await provider.close()


def build_ai_brain(
    repository: JobRepository,
    settings: AIBrainSettings,
    memory_store: MemoryStore | None = None,
) -> AIBrainResources:
    providers = build_providers(settings)
    costs = CostOptimizer()
    monitor = AIBrainMonitor()
    store = (
        memory_store
        if memory_store is not None
        else MongoMemoryStore(
            uri=settings.mongodb_uri,
            database=settings.mongodb_database,
        )
    )
    memory = MemoryManager(
        store,
        settings.memory_meeting_limit,
    )
    runtime = AgentRuntime(
        router=ModelRouter(providers, costs),
        prompts=PromptManager(settings.prompt_version),
        contexts=ContextManager(settings.context_max_tokens),
        cache=CacheManager(settings.cache_ttl_seconds),
        costs=costs,
        monitor=monitor,
        max_output_tokens=settings.max_output_tokens,
    )
    agent_orchestrator = AgentOrchestrator(
        runtime=runtime,
        memory=memory,
        max_parallel_agents=settings.max_parallel_agents,
        max_attempts=settings.agent_max_attempts,
        retry_base_seconds=settings.retry_base_seconds,
    )
    validation = ValidationLayer(
        validator=ValidatorAgent(),
        conflict_agent=ConflictAgent(),
        confidence_agent=ConfidenceAgent(),
        memory_validator=MemoryValidationAgent(),
    )
    brain = AIBrainPipeline(
        version=settings.result_version,
        orchestrator=agent_orchestrator,
        validation=validation,
        memory=memory,
        costs=costs,
        monitor=monitor,
    )
    return AIBrainResources(
        orchestrator=Model3PipelineOrchestrator(
            repository=repository,
            validator=M3InputContractValidator(),
            pipeline=brain,
        ),
        model4=Model4PipelineOrchestrator(
            repository=repository,
            validator=M4InputContractValidator(),
        ),
        model5=Model5PipelineOrchestrator(
            repository=repository,
            validator=M5InputContractValidator(),
        ),
        model6=Model6PipelineOrchestrator(
            repository=repository,
            validator=M6InputContractValidator(),
        ),
        providers=providers,
    )
