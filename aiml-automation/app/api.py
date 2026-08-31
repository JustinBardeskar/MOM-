"""
app.api
=======
FastAPI Dependency Injection, API Key Authentication & Complete REST API Routers.

This module provides:
- Health Router: Liveness, readiness, and health-check endpoints (/live, /ready, /health).
- API Router: Endpoints for meeting ingestion, jobs, transcripts, preprocessed chunks, media uploads, organizational memory, and AI brain patterns.
- Dependency Injection: get_settings, get_ingestion_service, require_api_key, Protected, IngestionService.
"""

from datetime import datetime, timezone
import hmac
import logging
from pathlib import Path
import shutil
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Header, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field, SecretStr

from app.ai_brain.models import AIBrainSettings, get_ai_brain_settings
from app.ai_brain.pipeline import build_ai_brain
from app.config import Settings, get_settings
from app.domain import (
    AssetReference,
    ContextBundle,
    InputPreference,
    JobAcceptedResponse,
    JobRecord,
    JobResultResponse,
    JobStatus,
    JobStatusResponse,
    JobSummaryResponse,
    MeetingProvider,
    MeetingReadyRequest,
    Participant,
    PipelineStage,
    PreprocessedTranscript,
    PreprocessedTranscriptResponse,
    PreprocessingStatistics,
    ProcessingPath,
    TranscriptChunk,
    TranscriptResponse,
    TranscriptSegment,
)
from app.export import IntegrationHub, generate_corporate_pdf
from app.infrastructure import InMemoryJobRepository, SQLiteJobRepository
from app.processing import FasterWhisperSpeechToText, FfmpegMediaProcessor
from app.services import MeetingIngestionService

logger = logging.getLogger("api")


# ==========================================
# FastAPI Dependency Injection
# ==========================================

def get_settings_dep(request: Request) -> Settings:
    """Retrieves cached application settings from FastAPI application state."""
    return request.app.state.settings  # type: ignore[no-any-return]


def get_ingestion_service(request: Request) -> MeetingIngestionService:
    """Retrieves singleton MeetingIngestionService from FastAPI application state."""
    return request.app.state.ingestion_service  # type: ignore[no-any-return]


async def require_api_key(
    settings: Annotated[Settings, Depends(get_settings_dep)],
    supplied_key: Annotated[str | None, Header(alias="X-Automation-Key")] = None,
) -> None:
    """Validates incoming X-Automation-Key using constant-time comparison."""
    configured = settings.api_key
    if configured is None or not configured.get_secret_value().strip():
        return
    if supplied_key is None or not hmac.compare_digest(
        supplied_key,
        configured.get_secret_value(),
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


Protected = Annotated[None, Depends(require_api_key)]
IngestionService = Annotated[MeetingIngestionService, Depends(get_ingestion_service)]


# ==========================================
# Routers and Request/Response Schemas
# ==========================================

health_router = APIRouter(tags=["Health"])
api_router = APIRouter()


class TranscriptProcessingRequest(BaseModel):
    transcript: str = Field(min_length=1, max_length=200_000)
    meeting_title: str = Field(default="Meeting transcript", min_length=1, max_length=300)
    meeting_id: str | None = Field(default=None, min_length=1, max_length=128)
    event_id: str | None = Field(default=None, min_length=1, max_length=128)
    provider: str = Field(default="offline")
    openrouter_api_key: str | None = Field(default=None, max_length=500)
    model: str | None = Field(default=None, max_length=200)


class TranscriptProcessingResponse(BaseModel):
    job_id: UUID
    meeting_id: str
    status: str
    result_url: str
    error_code: str | None = None
    error_message: str | None = None


# ==========================================
# Health Check Endpoints
# ==========================================

@health_router.get("/live")
async def live() -> dict[str, str]:
    return {"status": "live"}


@health_router.get("/ready")
async def ready() -> dict[str, str]:
    return {"status": "ready"}


@health_router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "service": "meeting-intelligence-automation"}


# ==========================================
# Meeting Ingestion & Job Status Endpoints
# ==========================================

@api_router.post(
    "/meetings/ready",
    response_model=JobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Meeting ingestion"],
)
async def meeting_ready(
    payload: MeetingReadyRequest,
    service: IngestionService,
    _: Protected,
    idempotency_key: Annotated[
        str | None,
        Header(alias="Idempotency-Key", min_length=1, max_length=128),
    ] = None,
) -> JobAcceptedResponse:
    return await service.accept(payload, idempotency_key or payload.event_id)


@api_router.get(
    "/jobs",
    response_model=list[JobSummaryResponse],
    tags=["Jobs"],
)
async def list_jobs(request: Request) -> list[JobSummaryResponse]:
    repository = request.app.state.job_repository
    jobs = await repository.list_all()
    return [JobSummaryResponse.from_record(job) for job in jobs]


@api_router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    tags=["Jobs"],
)
async def job_status(
    job_id: UUID,
    service: IngestionService,
    _: Protected,
) -> JobStatusResponse:
    return JobStatusResponse.from_record(await service.get_job(job_id))


@api_router.get(
    "/jobs/{job_id}/result",
    response_model=JobResultResponse,
    tags=["Jobs"],
)
async def job_result(
    job_id: UUID,
    service: IngestionService,
    _: Protected,
) -> JobResultResponse:
    job = await service.get_result(job_id)
    return JobResultResponse(job_id=job.id, meeting_id=job.meeting_id, result=job.result or {})


@api_router.get(
    "/jobs/{job_id}/transcript",
    response_model=TranscriptResponse,
    tags=["Jobs"],
)
async def job_transcript(
    job_id: UUID,
    service: IngestionService,
    _: Protected,
) -> TranscriptResponse:
    job = await service.get_transcript(job_id)
    assert job.unified_transcript is not None
    return TranscriptResponse(
        job_id=job.id,
        meeting_id=job.meeting_id,
        transcript=job.unified_transcript,
    )


@api_router.get(
    "/jobs/{job_id}/preprocessed",
    response_model=PreprocessedTranscriptResponse,
    tags=["Jobs"],
)
async def job_preprocessed_transcript(
    job_id: UUID,
    service: IngestionService,
    _: Protected,
) -> PreprocessedTranscriptResponse:
    job = await service.get_preprocessed_transcript(job_id)
    assert job.preprocessed_transcript is not None
    return PreprocessedTranscriptResponse(
        job_id=job.id,
        meeting_id=job.meeting_id,
        preprocessing=job.preprocessed_transcript,
    )


def _normalize_meeting_provider(val: str | None) -> MeetingProvider:
    if not val:
        return MeetingProvider.OFFLINE
    normalized = str(val).strip().lower()
    mapping = {
        "teams": MeetingProvider.TEAMS,
        "microsoft_teams": MeetingProvider.TEAMS,
        "meet": MeetingProvider.GOOGLE_MEET,
        "google_meet": MeetingProvider.GOOGLE_MEET,
        "zoom": MeetingProvider.ZOOM,
        "webex": MeetingProvider.WEBEX,
        "offline": MeetingProvider.OFFLINE,
    }
    return mapping.get(normalized, MeetingProvider.OFFLINE)


async def _execute_transcript_pipeline(
    job_id: UUID,
    repository: Any,
    ai_settings: AIBrainSettings,
) -> None:
    resources = None
    try:
        job = await repository.get(job_id)
        if job:
            job.current_stage = PipelineStage.PARALLEL_AGENT_ANALYSIS
            job.progress_percent = 70
            await repository.save(job)

        resources = build_ai_brain(repository, ai_settings)
        await resources.orchestrator.run(job_id)
        await resources.model4.run(job_id)
        await resources.model5.run(job_id)
        await resources.model6.run(job_id)

        saved_job = await repository.get(job_id)
        if saved_job:
            saved_job.current_stage = PipelineStage.FINAL_STRUCTURED_JSON_READY
            saved_job.progress_percent = 100
            await repository.save(saved_job)
            cost_info = saved_job.result.get("cost", {}) if (saved_job.result and isinstance(saved_job.result, dict)) else {}
            logger.info(
                "Completed transcript processing for job %s: status=%s | Tokens: %s in + %s out = %s total ($%s USD)",
                saved_job.id,
                saved_job.status.value,
                cost_info.get("input_tokens", "N/A"),
                cost_info.get("output_tokens", "N/A"),
                (cost_info.get("input_tokens", 0) + cost_info.get("output_tokens", 0)) or "N/A",
                cost_info.get("estimated_cost", 0),
            )
    except Exception as exc:
        logger.exception("Transcript async processing failed for job %s: %s", job_id, exc)
        job = await repository.get(job_id)
        if job:
            job.status = JobStatus.FAILED
            job.error_code = "transcript_pipeline_failed"
            job.error_message = str(exc)
            await repository.save(job)
    finally:
        if resources is not None:
            await resources.close()


@api_router.post(
    "/meetings/transcript",
    response_model=TranscriptProcessingResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Meeting ingestion"],
)
async def process_transcript(
    payload: TranscriptProcessingRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    service: IngestionService,
) -> TranscriptProcessingResponse:
    repository = service.repository
    meeting_id = payload.meeting_id or f"transcript-{uuid4().hex[:8]}"
    event_id = payload.event_id or meeting_id
    logger.info(
        "POST /api/v1/meetings/transcript: Received meeting '%s' (%d chars, provider: %s, meeting_id=%s)",
        payload.meeting_title,
        len(payload.transcript),
        payload.provider,
        meeting_id,
    )
    meeting_ready_request = MeetingReadyRequest(
        event_id=event_id,
        meeting_id=meeting_id,
        provider=_normalize_meeting_provider(payload.provider),
        title=payload.meeting_title,
        ended_at=datetime.now(timezone.utc).isoformat(),
        transcript=AssetReference(
            url="https://example.invalid/transcript.txt",
            content_type="text/plain",
        ),
        participants=[Participant(display_name="Transcriber")],
        input_preference=InputPreference.TRANSCRIPT,
    )
    preprocessed = _build_preprocessed_transcript(meeting_id, payload.meeting_title, payload.transcript)
    job = JobRecord(
        event_id=event_id,
        meeting_id=meeting_id,
        idempotency_key=event_id,
        status=JobStatus.PROCESSING,
        selected_path=ProcessingPath.DIRECT_TRANSCRIPT,
        current_stage=PipelineStage.PREPROCESSED_TRANSCRIPT_READY,
        progress_percent=60,
        planned_steps=[],
        request=meeting_ready_request,
        preprocessed_transcript=preprocessed,
    )
    job, _ = await repository.create_or_get(job)

    try:
        ai_settings = get_ai_brain_settings()
    except Exception:
        ai_settings = AIBrainSettings(allow_unauthenticated=True)

    effective_key = (
        payload.openrouter_api_key
        or request.headers.get("x-groq-key")
        or request.headers.get("x-openrouter-key")
        or request.headers.get("x-openai-key")
        or request.headers.get("x-gemini-key")
        or request.headers.get("x-anthropic-key")
        or ""
    ).strip()
    effective_model = (
        payload.model
        or request.headers.get("x-openrouter-model")
        or request.headers.get("x-model")
        or ""
    ).strip()

    updates = {}
    if effective_key:
        if effective_key.startswith("gsk_"):
            updates["groq_api_key"] = SecretStr(effective_key)
            updates["provider_priority"] = "groq,openrouter,openai,gemini"
            if effective_model: updates["groq_model"] = effective_model
        elif effective_key.startswith("AIzaSy"):
            updates["gemini_api_key"] = SecretStr(effective_key)
            updates["provider_priority"] = "gemini,openrouter,openai"
            if effective_model: updates["gemini_model"] = effective_model
        elif effective_key.startswith("sk-ant-"):
            updates["anthropic_api_key"] = SecretStr(effective_key)
            updates["provider_priority"] = "anthropic,openrouter,openai"
            if effective_model: updates["anthropic_model"] = effective_model
        elif effective_key.startswith("sk-proj-") or (effective_key.startswith("sk-") and not effective_key.startswith("sk-or-")):
            updates["openai_api_key"] = SecretStr(effective_key)
            updates["provider_priority"] = "openai,openrouter,gemini"
            if effective_model: updates["openai_model"] = effective_model
        else:
            updates["openrouter_api_key"] = SecretStr(effective_key)
            updates["provider_priority"] = "openrouter,openai,anthropic,gemini"
            if effective_model: updates["openrouter_model"] = effective_model
    elif effective_model:
        if "qwen" in effective_model or "groq" in effective_model:
            updates["groq_model"] = effective_model
            updates["provider_priority"] = "groq,openrouter,openai,gemini"
        else:
            updates["openrouter_model"] = effective_model

    if updates:
        ai_settings = ai_settings.model_copy(update=updates)

    background_tasks.add_task(
        _execute_transcript_pipeline,
        job.id,
        repository,
        ai_settings,
    )

    return TranscriptProcessingResponse(
        job_id=job.id,
        meeting_id=job.meeting_id,
        status="processing",
        result_url=f"/api/v1/jobs/{job.id}/result",
    )


async def _execute_media_pipeline(
    job_id: UUID,
    saved_media_path: Path,
    extracted_audio_path: Path,
    original_filename: str,
    meeting_id: str,
    meeting_title: str,
    provider: str,
    repository: Any,
    settings: Settings,
    ai_settings: AIBrainSettings,
    work_dir: Path,
) -> None:
    resources = None
    try:
        # Step 1: Extract 16kHz mono audio via FFmpeg
        job = await repository.get(job_id)
        if job:
            job.current_stage = PipelineStage.EXTRACT_AUDIO
            job.progress_percent = 25
            await repository.save(job)

        logger.info("Extracting audio from '%s' with FFmpeg for job %s...", saved_media_path.name, job_id)
        media_processor = FfmpegMediaProcessor(
            settings.ffmpeg_binary,
            settings.ffmpeg_timeout_seconds,
        )
        await media_processor.extract_audio(saved_media_path, extracted_audio_path)

        # Step 2: Transcribe via Faster-Whisper
        job = await repository.get(job_id)
        if job:
            job.current_stage = PipelineStage.SPEECH_TO_TEXT
            job.progress_percent = 50
            await repository.save(job)

        logger.info("Transcribing audio '%s' with Faster-Whisper for job %s...", extracted_audio_path.name, job_id)
        stt = FasterWhisperSpeechToText(
            settings.whisper_model,
            settings.whisper_device,
            settings.whisper_compute_type,
        )
        unified_transcript = await stt.transcribe(extracted_audio_path, language_hint=None)
        logger.info(
            "Speech transcription complete for job %s: %d chars, duration=%.1fs",
            job_id,
            len(unified_transcript.text),
            unified_transcript.duration_seconds,
        )

        # Step 3: Build Preprocessed Transcript
        preprocessed = _build_preprocessed_transcript(meeting_id, meeting_title, unified_transcript.text)

        job = await repository.get(job_id)
        if job:
            job.current_stage = PipelineStage.PARALLEL_AGENT_ANALYSIS
            job.progress_percent = 70
            job.unified_transcript = unified_transcript
            job.preprocessed_transcript = preprocessed
            await repository.save(job)

        # Step 4: Run AI Brain Swarm
        resources = build_ai_brain(repository, ai_settings)
        await resources.orchestrator.run(job_id)
        await resources.model4.run(job_id)
        await resources.model5.run(job_id)
        await resources.model6.run(job_id)

        saved_job = await repository.get(job_id)
        if saved_job:
            if saved_job.result is not None and isinstance(saved_job.result, dict):
                saved_job.result["transcript"] = unified_transcript.text
                saved_job.result["raw_transcript"] = unified_transcript.text
                saved_job.result["source_media"] = {
                    "filename": original_filename,
                    "duration_seconds": unified_transcript.duration_seconds,
                    "language": unified_transcript.language or "en",
                    "transcript_text": unified_transcript.text,
                    "segments": [
                        {
                            "start": s.start_seconds,
                            "end": s.end_seconds,
                            "text": s.text,
                        }
                        for s in unified_transcript.segments
                    ],
                }
            saved_job.current_stage = PipelineStage.FINAL_STRUCTURED_JSON_READY
            saved_job.progress_percent = 100
            await repository.save(saved_job)
            logger.info("Successfully finished media processing pipeline for job %s", job_id)

    except Exception as exc:
        logger.exception("Media upload async processing failed for job %s: %s", job_id, exc)
        job = await repository.get(job_id)
        if job:
            job.status = JobStatus.FAILED
            job.error_code = getattr(exc, "code", "media_pipeline_failed")
            job.error_message = getattr(exc, "message", None) or str(exc) or "Media pipeline execution failed"
            await repository.save(job)
    finally:
        if resources is not None:
            await resources.close()
        if not settings.keep_work_files:
            shutil.rmtree(work_dir, ignore_errors=True)


@api_router.post(
    "/meetings/upload",
    response_model=TranscriptProcessingResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Meeting ingestion"],
)
async def process_media_upload(
    request: Request,
    background_tasks: BackgroundTasks,
    service: IngestionService,
    file: UploadFile = File(...),
    meeting_title: str = Form("Uploaded Meeting Recording"),
    provider: str = Form("offline"),
    openrouter_api_key: str | None = Form(None),
    model: str | None = Form(None),
) -> TranscriptProcessingResponse:
    repository = service.repository
    settings = getattr(request.app.state, "settings", None) or get_settings()
    meeting_id = f"media-{uuid4().hex[:8]}"
    event_id = meeting_id

    original_filename = file.filename or "uploaded_recording.mp4"
    logger.info(
        "POST /api/v1/meetings/upload: Received media '%s' (%s, title='%s', provider=%s)",
        original_filename,
        file.content_type,
        meeting_title,
        provider,
    )

    work_dir = Path(settings.work_directory) / meeting_id
    work_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(original_filename).suffix or ".mp4"
    saved_media_path = work_dir / f"source_media{ext}"
    extracted_audio_path = work_dir / "extracted_audio.wav"

    await file.seek(0)
    with open(saved_media_path, "wb") as f_out:
        shutil.copyfileobj(file.file, f_out)

    meeting_ready_request = MeetingReadyRequest(
        event_id=event_id,
        meeting_id=meeting_id,
        provider=_normalize_meeting_provider(provider),
        title=meeting_title,
        ended_at=datetime.now(timezone.utc).isoformat(),
        recording=AssetReference(
            url=f"https://storage.local/media/{saved_media_path.name}",
            content_type=file.content_type or "video/mp4",
        ),
        participants=[Participant(display_name="Speaker")],
        input_preference=InputPreference.RECORDING,
    )

    job = JobRecord(
        event_id=event_id,
        meeting_id=meeting_id,
        idempotency_key=event_id,
        status=JobStatus.PROCESSING,
        selected_path=ProcessingPath.RECORDING_TO_TRANSCRIPT,
        current_stage=PipelineStage.MEETING_READY,
        progress_percent=10,
        planned_steps=[],
        request=meeting_ready_request,
    )
    job, _ = await repository.create_or_get(job)

    try:
        ai_settings = get_ai_brain_settings()
    except Exception:
        ai_settings = AIBrainSettings(allow_unauthenticated=True)

    effective_key = (
        openrouter_api_key
        or request.headers.get("x-groq-key")
        or request.headers.get("x-openrouter-key")
        or request.headers.get("x-openai-key")
        or request.headers.get("x-gemini-key")
        or request.headers.get("x-anthropic-key")
        or ""
    ).strip()
    effective_model = (
        model
        or request.headers.get("x-openrouter-model")
        or request.headers.get("x-model")
        or ""
    ).strip()

    updates = {}
    if effective_key:
        if effective_key.startswith("gsk_"):
            updates["groq_api_key"] = SecretStr(effective_key)
            updates["provider_priority"] = "groq,openrouter,openai,gemini"
            if effective_model: updates["groq_model"] = effective_model
        elif effective_key.startswith("AIzaSy"):
            updates["gemini_api_key"] = SecretStr(effective_key)
            updates["provider_priority"] = "gemini,openrouter,openai"
            if effective_model: updates["gemini_model"] = effective_model
        elif effective_key.startswith("sk-ant-"):
            updates["anthropic_api_key"] = SecretStr(effective_key)
            updates["provider_priority"] = "anthropic,openrouter,openai"
            if effective_model: updates["anthropic_model"] = effective_model
        elif effective_key.startswith("sk-proj-") or (effective_key.startswith("sk-") and not effective_key.startswith("sk-or-")):
            updates["openai_api_key"] = SecretStr(effective_key)
            updates["provider_priority"] = "openai,openrouter,gemini"
            if effective_model: updates["openai_model"] = effective_model
        else:
            updates["openrouter_api_key"] = SecretStr(effective_key)
            updates["provider_priority"] = "openrouter,openai,anthropic,gemini"
            if effective_model: updates["openrouter_model"] = effective_model
    elif effective_model:
        if "qwen" in effective_model or "groq" in effective_model or "gpt-oss" in effective_model:
            updates["groq_model"] = effective_model
            updates["provider_priority"] = "groq,openrouter,openai,gemini"
        else:
            updates["openrouter_model"] = effective_model

    if updates:
        ai_settings = ai_settings.model_copy(update=updates)

    background_tasks.add_task(
        _execute_media_pipeline,
        job.id,
        saved_media_path,
        extracted_audio_path,
        original_filename,
        meeting_id,
        meeting_title,
        provider,
        repository,
        settings,
        ai_settings,
        work_dir,
    )

    return TranscriptProcessingResponse(
        job_id=job.id,
        meeting_id=job.meeting_id,
        status="processing",
        result_url=f"/api/v1/jobs/{job.id}/result",
    )


def _build_preprocessed_transcript(
    meeting_id: str,
    meeting_title: str,
    transcript_text: str,
) -> PreprocessedTranscript:
    safe_text = transcript_text.strip() or "[Recorded meeting with no spoken dialogue detected]"
    words = safe_text.split()
    chunk_size = 250  # ~250 words per chunk ~= 320 tokens
    
    chunks = []
    segments = []
    contexts = []
    
    if not words:
        words = ["[Recorded meeting with no spoken dialogue detected]"]
        
    num_chunks = max(1, (len(words) + chunk_size - 1) // chunk_size)
    for i in range(num_chunks):
        chunk_words = words[i * chunk_size : (i + 1) * chunk_size]
        chunk_text = " ".join(chunk_words)
        start_sec = i * 60.0
        end_sec = (i + 1) * 60.0
        t_count = len(chunk_words)
        
        chunks.append(
            TranscriptChunk(
                id=f"chunk-{i}",
                index=i,
                text=chunk_text,
                start_seconds=start_sec,
                end_seconds=end_sec,
                speaker_ids=["SPEAKER_01"],
                token_count=t_count,
                source_segment_indexes=[i],
            )
        )
        segments.append(
            TranscriptSegment(
                start_seconds=start_sec,
                end_seconds=end_sec,
                text=chunk_text,
                speaker="SPEAKER_01",
            )
        )
        contexts.append(
            ContextBundle(
                id=f"context-{i}",
                chunk_id=f"chunk-{i}",
                meeting_id=meeting_id,
                meeting_title=meeting_title,
                provider=MeetingProvider.OFFLINE,
                ended_at=datetime.now(timezone.utc).isoformat(),
                language="en",
                text=chunk_text,
                speaker_ids=["SPEAKER_01"],
                start_seconds=start_sec,
                end_seconds=end_sec,
                token_count=t_count,
                metadata={},
            )
        )

    return PreprocessedTranscript(
        version="frontend-direct",
        text=safe_text,
        language="en",
        duration_seconds=max(1, len(safe_text.split()) // 2),
        segments=segments,
        speakers=[],
        chunks=chunks,
        contexts=contexts,
        statistics=PreprocessingStatistics(
            original_characters=len(safe_text),
            cleaned_characters=len(safe_text),
            fillers_removed=0,
            noise_segments_removed=0,
            timestamps_corrected=0,
            speaker_count=1,
            chunk_count=len(chunks),
        ),
    )


# ==========================================
# Organizational Memory & Pattern Endpoints
# ==========================================

@api_router.get(
    "/memory",
    tags=["Memory"],
)
async def get_organizational_memory(limit: int = 20) -> list[dict]:
    """Retrieves remembered meetings, past approved decisions, and ongoing workstreams from MongoDB."""
    from app.ai_brain.models import get_ai_brain_settings
    from app.ai_brain.memory import MongoMemoryStore
    try:
        settings = get_ai_brain_settings()
        store = MongoMemoryStore(uri=settings.mongodb_uri, database=settings.mongodb_database)
    except Exception:
        store = MongoMemoryStore()
    records = await store.recall(None, limit)
    return [rec.model_dump(mode="json") for rec in records]


@api_router.get(
    "/memory/status",
    tags=["Memory"],
)
async def get_memory_status() -> dict:
    """Returns the operational status of MongoDB enterprise memory storage."""
    from app.ai_brain.models import get_ai_brain_settings
    from app.ai_brain.memory import MongoMemoryStore
    try:
        settings = get_ai_brain_settings()
        store = MongoMemoryStore(uri=settings.mongodb_uri, database=settings.mongodb_database)
    except Exception:
        store = MongoMemoryStore()
    is_mongo = await store.is_available()
    records = await store.recall(None, 100)
    return {
        "engine": "MongoDB (Active)" if is_mongo else "SQLite (Fallback)",
        "mongodb_connected": is_mongo,
        "database": getattr(store, "database_name", "mom_ai_brain"),
        "total_meetings_remembered": len(records),
        "total_pending_action_items": sum(len(r.pending_action_items) for r in records),
        "total_approved_decisions": sum(len(r.decisions) for r in records),
    }


@api_router.get(
    "/ai-brain/patterns",
    tags=["AI Brain"],
)
async def list_ai_brain_patterns(agent_type: str | None = None, limit: int = 50) -> list[dict]:
    """Retrieves all learned patterns, rules, and taxonomies stored in MongoDB per agent."""
    from app.ai_brain.models import get_ai_brain_settings
    from app.ai_brain.memory import AIBrainPatternStore
    try:
        settings = get_ai_brain_settings()
        store = AIBrainPatternStore(uri=settings.mongodb_uri, database=settings.mongodb_database)
    except Exception:
        store = AIBrainPatternStore()
    return await store.list_patterns(agent_type=agent_type, limit=limit)


@api_router.post(
    "/ai-brain/patterns",
    tags=["AI Brain"],
)
async def create_ai_brain_pattern(payload: dict) -> dict:
    """Teaches and stores a new custom pattern/rule into MongoDB for a specialist agent."""
    from app.ai_brain.models import get_ai_brain_settings
    from app.ai_brain.memory import AIBrainPatternStore
    try:
        settings = get_ai_brain_settings()
        store = AIBrainPatternStore(uri=settings.mongodb_uri, database=settings.mongodb_database)
    except Exception:
        store = AIBrainPatternStore()
    return await store.save_pattern(payload)


@api_router.delete(
    "/ai-brain/patterns/{pattern_id}",
    tags=["AI Brain"],
)
async def delete_ai_brain_pattern(pattern_id: str) -> dict:
    """Deletes a learned pattern from MongoDB by ID."""
    from app.ai_brain.models import get_ai_brain_settings
    from app.ai_brain.memory import AIBrainPatternStore
    try:
        settings = get_ai_brain_settings()
        store = AIBrainPatternStore(uri=settings.mongodb_uri, database=settings.mongodb_database)
    except Exception:
        store = AIBrainPatternStore()
    success = await store.delete_pattern(pattern_id)
    return {"success": success, "pattern_id": pattern_id}


@api_router.get(
    "/ai-brain/stats",
    tags=["AI Brain"],
)
async def get_ai_brain_stats() -> dict:
    """Returns real-time intelligence metrics and patterns stored across all 10 agents in MongoDB."""
    from app.ai_brain.models import get_ai_brain_settings
    from app.ai_brain.memory import AIBrainPatternStore
    try:
        settings = get_ai_brain_settings()
        store = AIBrainPatternStore(uri=settings.mongodb_uri, database=settings.mongodb_database)
    except Exception:
        store = AIBrainPatternStore()
    return await store.get_stats()
