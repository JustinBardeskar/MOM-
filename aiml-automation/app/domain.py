"""
app.domain
==========
Core Domain Models, Enums, Schemas, and Abstract Port Interfaces.

This module encapsulates all enterprise entities for the AI Meeting Automation Platform:
- Lifecycle State Enums: JobStatus, PipelineStage, MilestoneName, MeetingProvider
- Input Schemas: MeetingReadyRequest, AssetReference, Participant
- Processing Artifacts: UnifiedTranscript, TranscriptSegment, PreprocessedTranscript
- Job Entity: JobRecord (the central aggregate root persisted throughout processing)
- Protocol Ports: JobRepository, JobDispatcher, AssetDownloader, MediaProcessor, SpeechToTextProvider
"""

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


# ==========================================
# Canonical SSOT Meeting Document State
# ==========================================

class MeetingDocumentState(BaseModel):
    """
    Canonical Single Source of Truth (SSOT) Document State.
    Shared across FastAPI backend, MongoDB memory, Streamlit UI, and ReportLab PDF compiler.
    """
    model_config = ConfigDict(extra="ignore")

    job_id: str = ""
    meeting_title: str = "Minutes of Meeting"
    meeting_type: str = "General"
    generated_at: str = ""
    duration_seconds: float = 0.0
    participants: list[str] = Field(default_factory=list)
    
    # Core Deliverables
    meeting_summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    action_items: list[dict[str, Any]] = Field(default_factory=list)
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    risks: list[dict[str, Any]] = Field(default_factory=list)
    topics: list[dict[str, Any]] = Field(default_factory=list)
    requirements: list[dict[str, Any]] = Field(default_factory=list)
    sentiment: dict[str, Any] = Field(default_factory=dict)
    open_questions: list[dict[str, Any]] = Field(default_factory=list)
    follow_up_tasks: list[dict[str, Any]] = Field(default_factory=list)
    
    # Raw speech transcript
    transcript: str = ""


# ==========================================
# Lifecycle & Categorization Enums
# ==========================================

class MeetingProvider(StrEnum):
    """Supported video conferencing and recording platforms."""
    TEAMS = "microsoft_teams"
    GOOGLE_MEET = "google_meet"
    ZOOM = "zoom"
    WEBEX = "webex"
    OFFLINE = "offline"


class InputPreference(StrEnum):
    """Client preference for input ingestion."""
    AUTO = "auto"
    RECORDING = "recording"
    TRANSCRIPT = "transcript"


class ProcessingPath(StrEnum):
    """Processing strategy chosen during job planning."""
    RECORDING_TO_TRANSCRIPT = "recording_to_transcript"
    DIRECT_TRANSCRIPT = "direct_transcript"


class JobStatus(StrEnum):
    """Execution status of a meeting processing job."""
    QUEUED = "queued"
    PROCESSING = "processing"
    AWAITING_ANALYSIS = "awaiting_analysis"
    AWAITING_DELIVERY = "awaiting_delivery"
    COMPLETED = "completed"
    FAILED = "failed"


class MilestoneName(StrEnum):
    """Milestone identifiers for tracking and error handling."""
    M1 = "m1"
    M1_VALIDATION = "m1_validation"
    M2 = "m2"
    M2_VALIDATION = "m2_validation"
    M3 = "m3"
    M3_VALIDATION = "m3_validation"
    M4 = "m4"
    M4_VALIDATION = "m4_validation"
    M5_VALIDATION = "m5_validation"
    M6_VALIDATION = "m6_validation"


class PipelineStage(StrEnum):
    """Granular stages across the end-to-end processing pipeline."""
    MEETING_READY = "meeting_ready"
    DOWNLOAD_RECORDING = "download_recording"
    EXTRACT_AUDIO = "extract_audio"
    SPEECH_TO_TEXT = "speech_to_text"
    NORMALIZE_TRANSCRIPT = "normalize_transcript"
    VALIDATE_M1_OUTPUT = "validate_m1_output"
    UNIFIED_TRANSCRIPT_READY = "unified_transcript_ready"
    M1_TO_M2_HANDOFF = "m1_to_m2_handoff"
    REMOVE_FILLERS = "remove_fillers"
    NORMALIZE_SPEAKERS = "normalize_speakers"
    CLEAN_TIMESTAMPS = "clean_timestamps"
    REMOVE_TRANSCRIPT_NOISE = "remove_transcript_noise"
    CHUNK_TRANSCRIPT = "chunk_transcript"
    BUILD_CONTEXT = "build_context"
    VALIDATE_M2_OUTPUT = "validate_m2_output"
    PREPROCESSED_TRANSCRIPT_READY = "preprocessed_transcript_ready"
    M2_TO_M3_HANDOFF = "m2_to_m3_handoff"
    MEETING_UNDERSTANDING = "meeting_understanding"
    PARALLEL_AGENT_ANALYSIS = "parallel_agent_analysis"
    VALIDATE_AGENT_OUTPUTS = "validate_agent_outputs"
    DETECT_AGENT_CONFLICTS = "detect_agent_conflicts"
    SCORE_CONFIDENCE = "score_confidence"
    VALIDATE_MEMORY = "validate_memory"
    M3_TO_M4_HANDOFF = "m3_to_m4_handoff"
    M4_ANALYSIS = "m4_analysis"
    VALIDATE_M4_OUTPUT = "validate_m4_output"
    M4_TO_M5_HANDOFF = "m4_to_m5_handoff"
    M5_VALIDATION = "m5_validation"
    M5_TO_M6_HANDOFF = "m5_to_m6_handoff"
    M6_FINALIZATION = "m6_finalization"
    FINAL_STRUCTURED_JSON_READY = "final_structured_json_ready"
    ANALYZE_MEETING = "analyze_meeting"
    VALIDATE_OUTPUT = "validate_output"
    DELIVER_RESULTS = "deliver_results"


# ==========================================
# Ingestion Models & Asset References
# ==========================================

class AssetReference(BaseModel):
    """External media or transcript asset reference."""
    model_config = ConfigDict(extra="forbid")

    url: HttpUrl
    content_type: str = Field(min_length=3, max_length=100)
    file_name: str | None = Field(default=None, max_length=255)
    size_bytes: int | None = Field(default=None, ge=1)
    checksum_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[a-fA-F0-9]{64}$",
    )
    expires_at: datetime | None = None


class Participant(BaseModel):
    """Meeting participant information."""
    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(min_length=1, max_length=200)
    email: str | None = Field(default=None, max_length=320)


class MeetingReadyRequest(BaseModel):
    """Payload received when a meeting is concluded and ready for processing."""
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1, max_length=128)
    meeting_id: str = Field(min_length=1, max_length=128)
    provider: MeetingProvider
    title: str = Field(min_length=1, max_length=300)
    ended_at: datetime
    organizer: Participant | None = None
    participants: list[Participant] = Field(default_factory=list, max_length=500)
    language_hint: str | None = Field(default=None, max_length=20)
    recording: AssetReference | None = None
    transcript: AssetReference | None = None
    input_preference: InputPreference = InputPreference.AUTO
    callback_url: HttpUrl | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_input_assets(self) -> "MeetingReadyRequest":
        """Ensures valid combination of recording/transcript according to input preference."""
        if self.recording is None and self.transcript is None:
            raise ValueError("At least one of recording or transcript must be provided")
        if self.input_preference == InputPreference.RECORDING and self.recording is None:
            raise ValueError("recording is required when input_preference is recording")
        if self.input_preference == InputPreference.TRANSCRIPT and self.transcript is None:
            raise ValueError("transcript is required when input_preference is transcript")
        return self


# ==========================================
# Processing & Transcript Artifacts
# ==========================================

class PipelineStep(BaseModel):
    """Planned pipeline execution step."""
    stage: PipelineStage
    required: bool = True


class TranscriptSegment(BaseModel):
    """Individual timestamped utterance by a participant."""
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    text: str
    speaker: str | None = None


class UnifiedTranscript(BaseModel):
    """Normalized transcript structure produced by Milestone 1."""
    text: str
    language: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    segments: list[TranscriptSegment] = Field(default_factory=list)
    source_path: ProcessingPath


class NormalizedSpeaker(BaseModel):
    """Deduplicated and resolved speaker identity."""
    id: str
    display_name: str
    aliases: list[str] = Field(default_factory=list)


class TranscriptChunk(BaseModel):
    """Token-bounded text window for LLM context processing."""
    id: str
    index: int = Field(ge=0)
    text: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    speaker_ids: list[str] = Field(default_factory=list)
    token_count: int = Field(ge=0)
    source_segment_indexes: list[int] = Field(default_factory=list)
    overlap_tokens: int = Field(default=0, ge=0)


class ContextBundle(BaseModel):
    """Complete context package for downstream intelligence agents."""
    id: str
    chunk_id: str
    meeting_id: str
    meeting_title: str
    provider: MeetingProvider
    ended_at: datetime
    language: str | None = None
    text: str
    previous_context: str | None = None
    next_context: str | None = None
    speaker_ids: list[str] = Field(default_factory=list)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    token_count: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PreprocessingStatistics(BaseModel):
    """Metrics recorded during Milestone 2 preprocessing."""
    original_characters: int = Field(ge=0)
    cleaned_characters: int = Field(ge=0)
    fillers_removed: int = Field(ge=0)
    noise_segments_removed: int = Field(ge=0)
    timestamps_corrected: int = Field(ge=0)
    speaker_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)


class PreprocessedTranscript(BaseModel):
    """Cleaned, chunked, and contextualized transcript ready for AI Brain."""
    version: str
    text: str
    language: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    segments: list[TranscriptSegment]
    speakers: list[NormalizedSpeaker]
    chunks: list[TranscriptChunk]
    contexts: list[ContextBundle]
    statistics: PreprocessingStatistics


# ==========================================
# Central Job Aggregate Root
# ==========================================

class JobRecord(BaseModel):
    """Central domain aggregate tracking full lifecycle and artifacts of a meeting."""
    id: UUID = Field(default_factory=uuid4)
    event_id: str
    meeting_id: str
    idempotency_key: str
    status: JobStatus = JobStatus.QUEUED
    selected_path: ProcessingPath
    current_stage: PipelineStage = PipelineStage.MEETING_READY
    progress_percent: int = Field(default=0, ge=0, le=100)
    planned_steps: list[PipelineStep]
    request: MeetingReadyRequest
    unified_transcript: UnifiedTranscript | None = None
    preprocessed_transcript: PreprocessedTranscript | None = None
    result: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    failed_milestone: MilestoneName | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ==========================================
# REST API Presentation DTOs
# ==========================================

class JobAcceptedResponse(BaseModel):
    """HTTP 202 response returned upon job creation."""
    job_id: UUID
    meeting_id: str
    status: JobStatus
    selected_path: ProcessingPath
    status_url: str
    transcript_url: str
    preprocessed_url: str
    result_url: str
    duplicate: bool = False


class JobStatusResponse(BaseModel):
    """Detailed status response for polling."""
    job_id: UUID
    meeting_id: str
    status: JobStatus
    selected_path: ProcessingPath
    current_stage: PipelineStage
    progress_percent: int
    planned_steps: list[PipelineStep]
    error_code: str | None
    error_message: str | None
    failed_milestone: MilestoneName | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_record(cls, job: JobRecord) -> "JobStatusResponse":
        return cls(
            job_id=job.id,
            meeting_id=job.meeting_id,
            status=job.status,
            selected_path=job.selected_path,
            current_stage=job.current_stage,
            progress_percent=job.progress_percent,
            planned_steps=job.planned_steps,
            error_code=job.error_code,
            error_message=job.error_message,
            failed_milestone=job.failed_milestone,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )


class JobResultResponse(BaseModel):
    """Final structured analysis deliverables."""
    job_id: UUID
    meeting_id: str
    result: dict[str, Any]


class JobSummaryResponse(BaseModel):
    """Lightweight summary response for job listing."""
    job_id: UUID
    meeting_id: str
    title: str
    status: JobStatus
    selected_path: ProcessingPath
    current_stage: PipelineStage
    progress_percent: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_record(cls, job: JobRecord) -> "JobSummaryResponse":
        return cls(
            job_id=job.id,
            meeting_id=job.meeting_id,
            title=job.request.title,
            status=job.status,
            selected_path=job.selected_path,
            current_stage=job.current_stage,
            progress_percent=job.progress_percent,
            error_message=job.error_message,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )


class TranscriptResponse(BaseModel):
    """M1 Transcript query response."""
    job_id: UUID
    meeting_id: str
    transcript: UnifiedTranscript


class PreprocessedTranscriptResponse(BaseModel):
    """M2 Preprocessed transcript query response."""
    job_id: UUID
    meeting_id: str
    preprocessing: PreprocessedTranscript


# ==========================================
# Abstract Port Interfaces (Hexagonal Architecture)
# ==========================================

class JobRepository(Protocol):
    """Persistence port for JobRecord aggregates."""
    async def create_or_get(self, job: JobRecord) -> tuple[JobRecord, bool]:
        """Creates a new job or returns existing if idempotency key matches."""
        ...

    async def get(self, job_id: UUID) -> JobRecord | None:
        """Fetches a job by unique ID."""
        ...

    async def save(self, job: JobRecord) -> None:
        """Persists updated job state."""
        ...

    async def list_all(self) -> list[JobRecord]:
        """Lists all active and historic jobs."""
        ...


class JobDispatcher(Protocol):
    """Asynchronous job execution trigger port."""
    async def publish(self, job_id: UUID) -> None:
        """Enqueues job for background execution."""
        ...


class AssetDownloader(Protocol):
    """Port for streaming media/transcript assets from remote URLs."""
    async def download(self, asset: AssetReference, destination: Path) -> Path: ...


class MediaProcessor(Protocol):
    """Port for FFmpeg audio extraction from video recordings."""
    async def extract_audio(self, recording: Path, destination: Path) -> Path: ...


class SpeechToTextProvider(Protocol):
    """Port for Whisper speech-to-text audio transcription."""
    async def transcribe(
        self,
        audio: Path,
        language_hint: str | None,
    ) -> UnifiedTranscript: ...


class TranscriptReader(Protocol):
    """Port for parsing raw VTT, SRT, or JSON transcript files."""
    async def read(
        self,
        transcript: Path,
        content_type: str,
        source_path: str,
    ) -> UnifiedTranscript: ...


class TranscriptNormalizer(Protocol):
    """Port for cleaning and standardizing transcripts."""
    async def normalize(self, transcript: UnifiedTranscript) -> UnifiedTranscript: ...
