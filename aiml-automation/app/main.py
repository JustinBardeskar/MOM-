"""
app.main
========
FastAPI Application Entry Point, Lifecycle Initialization, and Dependency Wiring.

This module provides:
- create_app: Factory function instantiating FastAPI with CORS, structured logging, SQLite persistence,
  M1-M6 pipeline coordination, background worker dispatching, and REST endpoints.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.ai_brain.models import get_ai_brain_settings
from app.ai_brain.pipeline import AIBrainResources, build_ai_brain
from app.api import api_router, health_router
from app.config import (
    Settings,
    configure_logging,
    configure_system_trust_store,
    get_settings,
)
from app.domain import JobDispatcher
from app.infrastructure import (
    AsyncJobDispatcher,
    InMemoryJobDispatcher,
    InMemoryJobRepository,
    SQLiteJobRepository,
)
from app.integration import (
    M1M2PipelineOrchestrator,
    Milestone2Adapter,
    MilestoneContractValidator,
)
from app.processing import (
    ContextBundleBuilder,
    FasterWhisperSpeechToText,
    FfmpegMediaProcessor,
    FillerRemover,
    HttpAssetDownloader,
    Milestone1ProcessingPipeline,
    SpeakerNormalizer,
    TiktokenCounter,
    TimestampCleaner,
    TranscriptCleanerNormalizer,
    TranscriptNoiseRemover,
    TranscriptPreprocessingPipeline,
    TranscriptChunker,
    TxtVttTranscriptReader,
)
from app.recorder import recorder_router
from app.services import (
    AutomationPipelineCoordinator,
    JobNotFoundError,
    MeetingIngestionService,
    PipelinePlanner,
    PreprocessedTranscriptNotReadyError,
    ResultNotReadyError,
    TranscriptNotReadyError,
)

logger = logging.getLogger("automation.request")


# ==========================================
# Request Logging Middleware & Error Handlers
# ==========================================

class RequestContextMiddleware:
    """Attaches a unique X-Request-ID and logs completion latency in structured JSON."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        import time
        from uuid import uuid4

        headers = dict(scope.get("headers", []))
        request_id = headers.get(b"x-request-id", b"").decode() or str(uuid4())
        started_at = time.perf_counter()
        response_status = 500

        async def send_with_context(message: Message) -> None:
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = message["status"]
                response_headers = MutableHeaders(scope=message)
                response_headers["X-Request-ID"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_context)
        finally:
            logger.info(
                "request_completed",
                extra={
                    "request_id": request_id,
                    "method": scope["method"],
                    "endpoint": scope["path"],
                    "status": response_status,
                    "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                },
            )


def register_exception_handlers(app: FastAPI) -> None:
    """Registers standard REST JSON error handlers for domain exceptions."""

    def _err(status_code: int, code: str, msg: str, details: object = None) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content={"error": {"code": code, "message": msg, "details": details}},
        )

    @app.exception_handler(JobNotFoundError)
    async def _handle_not_found(_: Request, exc: JobNotFoundError) -> JSONResponse:
        return _err(404, "job_not_found", f"Job {exc} was not found")

    @app.exception_handler(ResultNotReadyError)
    async def _handle_result_not_ready(_: Request, exc: ResultNotReadyError) -> JSONResponse:
        return _err(409, "result_not_ready", f"Job {exc} has not completed")

    @app.exception_handler(TranscriptNotReadyError)
    async def _handle_transcript_not_ready(_: Request, exc: TranscriptNotReadyError) -> JSONResponse:
        return _err(409, "transcript_not_ready", f"Job {exc} has not produced a unified transcript")

    @app.exception_handler(PreprocessedTranscriptNotReadyError)
    async def _handle_preprocessed_not_ready(_: Request, exc: PreprocessedTranscriptNotReadyError) -> JSONResponse:
        return _err(409, "preprocessing_not_ready", f"Job {exc} has not completed transcript preprocessing")

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [{k: v for k, v in err.items() if k != "ctx"} for err in exc.errors()]
        return _err(422, "validation_error", "Request validation failed", details)


# ==========================================
# FastAPI Application Factory
# ==========================================

def create_app(settings: Settings | None = None) -> FastAPI:
    """Constructs and wires the complete FastAPI application with all subsystems."""
    configure_system_trust_store()
    app_settings = settings or get_settings()
    configure_logging(app_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Application lifecycle context managing database initialization and worker queues."""
        repository = (
            InMemoryJobRepository()
            if app_settings.environment == "testing"
            else SQLiteJobRepository()
        )
        downloader: HttpAssetDownloader | None = None
        async_dispatcher: AsyncJobDispatcher | None = None
        ai_brain_resources: AIBrainResources | None = None
        dispatcher: JobDispatcher

        if app_settings.worker_enabled:
            token_counter = TiktokenCounter(app_settings.token_encoding)

            # Build M2 Preprocessing Pipeline
            preprocessor = TranscriptPreprocessingPipeline(
                version=app_settings.preprocessing_version,
                fillers=FillerRemover(app_settings.filler_word_list),
                speakers=SpeakerNormalizer(),
                timestamps=TimestampCleaner(),
                noise=TranscriptNoiseRemover(),
                chunker=TranscriptChunker(
                    token_counter=token_counter,
                    target_tokens=app_settings.chunk_target_tokens,
                    max_tokens=app_settings.chunk_max_tokens,
                    overlap_tokens=app_settings.chunk_overlap_tokens,
                ),
                context_builder=ContextBundleBuilder(
                    token_counter=token_counter,
                    neighbor_tokens=app_settings.context_neighbor_tokens,
                ),
            )

            # Build M1 Ingestion & Transcription Pipeline
            downloader = HttpAssetDownloader(
                allowed_hosts=app_settings.asset_host_list,
                max_bytes=app_settings.max_download_bytes,
                timeout_seconds=app_settings.download_timeout_seconds,
            )
            milestone1 = Milestone1ProcessingPipeline(
                downloader=downloader,
                processor=FfmpegMediaProcessor(
                    app_settings.ffmpeg_binary,
                    app_settings.ffmpeg_timeout_seconds,
                ),
                speech_to_text=FasterWhisperSpeechToText(
                    app_settings.whisper_model,
                    app_settings.whisper_device,
                    app_settings.whisper_compute_type,
                ),
                reader=TxtVttTranscriptReader(),
                normalizer=TranscriptCleanerNormalizer(),
            )

            # Build M1+M2 Sequential Orchestrator
            m1_m2_orchestrator = M1M2PipelineOrchestrator(
                repository=repository,
                milestone1=milestone1,
                milestone2=Milestone2Adapter(preprocessor),
                validator=MilestoneContractValidator(),
                work_directory=app_settings.work_directory,
                keep_work_files=app_settings.keep_work_files,
            )

            # Build M3 AI Brain Multi-Agent Swarm
            if app_settings.ai_brain_enabled:
                ai_brain_resources = build_ai_brain(
                    repository,
                    get_ai_brain_settings(),
                )

            # Build End-to-End Pipeline Coordinator
            coordinator = AutomationPipelineCoordinator(
                m1_m2=m1_m2_orchestrator,
                model3=ai_brain_resources.orchestrator if ai_brain_resources else None,
                model4=ai_brain_resources.model4 if ai_brain_resources else None,
                model5=ai_brain_resources.model5 if ai_brain_resources else None,
                model6=ai_brain_resources.model6 if ai_brain_resources else None,
            )

            # Launch Background Async Task Queue
            async_dispatcher = AsyncJobDispatcher(
                coordinator.run,
                app_settings.worker_concurrency,
            )
            await async_dispatcher.start()
            dispatcher = async_dispatcher
        else:
            dispatcher = InMemoryJobDispatcher()

        # Bind Core Dependencies to Application State
        app.state.settings = app_settings
        app.state.job_repository = repository
        app.state.job_dispatcher = dispatcher
        app.state.ingestion_service = MeetingIngestionService(
            repository=repository,
            dispatcher=dispatcher,
            planner=PipelinePlanner(),
        )

        try:
            yield
        finally:
            if async_dispatcher is not None:
                await async_dispatcher.stop()
            if downloader is not None:
                await downloader.close()
            if ai_brain_resources is not None:
                await ai_brain_resources.close()

    # Instantiate FastAPI App
    app = FastAPI(
        title="Meeting Intelligence Automation API",
        description="Enterprise AI MOM Agent & Multi-Agent Swarm Platform",
        version="1.0.0",
        docs_url="/docs" if app_settings.docs_enabled else None,
        redoc_url=None,
        lifespan=lifespan,
    )

    # Middleware Setup
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Routers and Handlers
    app.include_router(health_router)
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(recorder_router)
    register_exception_handlers(app)

    # Resilient fallback state bindings
    app.state.settings = app_settings
    fallback_repo = (
        InMemoryJobRepository()
        if app_settings.environment == "testing"
        else SQLiteJobRepository()
    )
    app.state.job_repository = fallback_repo
    app.state.ingestion_service = MeetingIngestionService(
        repository=fallback_repo,
        dispatcher=InMemoryJobDispatcher(),
        planner=PipelinePlanner(),
    )

    return app


app = create_app()
