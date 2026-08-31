import asyncio
import math
import struct
import wave
from pathlib import Path

from app.domain import (
    AssetReference,
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
from app.processing import (
    BasicTranscriptNormalizer,
    ContextBundleBuilder as ContextBuilder,
    FfmpegMediaProcessor,
    FillerRemover,
    PipelineProcessingError,
    SpeakerNormalizer,
    TiktokenCounter,
    TimestampCleaner,
    TranscriptChunker,
    TranscriptMilestone1Runner,
    TranscriptNoiseRemover,
    TranscriptPreprocessingPipeline,
    TxtVttTranscriptReader,
)
from app.integration import (
    M1M2PipelineOrchestrator,
    M1ToM2Contract,
    Milestone2Adapter,
    MilestoneContractValidator,
    StageReporter,
)


class FakeDownloader:
    def __init__(self, payload: str = '') -> None:
        self.payload = payload
        self.destinations: list[Path] = []

    async def download(self, _: AssetReference, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.payload, encoding='utf-8')
        self.destinations.append(destination)
        return destination


class FakeMediaProcessor:
    def __init__(self) -> None:
        self.called = False

    async def extract_audio(self, _: Path, destination: Path) -> Path:
        self.called = True
        destination.write_bytes(b'fake-wave')
        return destination


class FakeSpeechToText:
    def __init__(self) -> None:
        self.called = False

    async def transcribe(self, _: Path, __: str | None) -> UnifiedTranscript:
        self.called = True
        return UnifiedTranscript(
            text='  Welcome   to the meeting.  ',
            language='en',
            duration_seconds=2.0,
            segments=[
                TranscriptSegment(
                    start_seconds=0,
                    end_seconds=2,
                    text='  Welcome   to the meeting.  ',
                )
            ],
            source_path=ProcessingPath.RECORDING_TO_TRANSCRIPT,
        )


class StubMilestone1:
    def __init__(
        self,
        events: list[str],
        result: UnifiedTranscript | None = None,
        error: Exception | None = None,
    ) -> None:
        self._events = events
        self._result = result
        self._error = error
        self.calls = 0

    async def execute(
        self,
        _: JobRecord,
        __: Path,
        ___: StageReporter,
    ) -> UnifiedTranscript:
        self.calls += 1
        self._events.append('m1')
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


class StubMilestone2:
    def __init__(
        self,
        events: list[str],
        result: PreprocessedTranscript | None = None,
        error: Exception | None = None,
    ) -> None:
        self._events = events
        self._result = result
        self._error = error
        self.calls = 0
        self.received_contract: M1ToM2Contract | None = None

    async def execute(
        self,
        contract: M1ToM2Contract,
        _: StageReporter,
    ) -> PreprocessedTranscript:
        self.calls += 1
        self._events.append('m2')
        self.received_contract = contract
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


def make_preprocessor() -> TranscriptPreprocessingPipeline:
    tokens = TiktokenCounter('cl100k_base')
    return TranscriptPreprocessingPipeline(
        version='test',
        filler_remover=FillerRemover(['um', 'uh', 'erm', 'hmm']),
        speaker_normalizer=SpeakerNormalizer(),
        timestamp_cleaner=TimestampCleaner(),
        noise_remover=TranscriptNoiseRemover(),
        chunker=TranscriptChunker(tokens, 100, 120, 20),
        context_builder=ContextBuilder(tokens, 20),
    )


def make_request(*, transcript: bool) -> MeetingReadyRequest:
    asset = AssetReference(
        url='https://storage.example.com/source.vtt' if transcript else 'https://storage.example.com/source.mp4',
        content_type='text/vtt' if transcript else 'video/mp4',
    )
    return MeetingReadyRequest(
        event_id='evt-process',
        meeting_id='meeting-process',
        provider='zoom',
        title='Weekly sync',
        ended_at='2026-08-07T10:00:00Z',
        transcript=asset if transcript else None,
        recording=asset if not transcript else None,
    )


def write_test_wav(path: Path, duration_seconds: float = 0.25) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    total_samples = int(sample_rate * duration_seconds)
    frequency = 440.0
    with wave.open(str(path), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for idx in range(total_samples):
            value = int(32767.0 * 0.2 * math.sin(2.0 * math.pi * frequency * (idx / sample_rate)))
            frames.extend(struct.pack('<h', value))
        wav_file.writeframes(bytes(frames))


# ==========================================
# Audio & Reader Unit Tests
# ==========================================

def test_reader_parses_txt_and_vtt(tmp_path: Path) -> None:
    reader = TxtVttTranscriptReader()
    txt_file = tmp_path / "meeting.txt"
    txt_file.write_text("Alice: Hello team\nBob: Ready to start", encoding="utf-8")

    vtt_file = tmp_path / "meeting.vtt"
    vtt_file.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.500\n<v Alice>Hello team</v>\n\n"
        "00:00:02.500 --> 00:00:05.000\nBob: Ready to start\n",
        encoding="utf-8",
    )

    from_txt = reader.read(txt_file, "text/plain", ProcessingPath.DIRECT_TRANSCRIPT.value)
    from_vtt = reader.read(vtt_file, "text/vtt", ProcessingPath.DIRECT_TRANSCRIPT.value)

    assert len(from_txt.segments) == 2
    assert len(from_vtt.segments) == 2
    assert from_vtt.duration_seconds == 5.0


def test_ffmpeg_processor_validates_audio(tmp_path: Path) -> None:
    wav_path = tmp_path / "audio.wav"
    write_test_wav(wav_path, duration_seconds=0.3)
    processor = FfmpegMediaProcessor(require_ffmpeg=False)

    metadata = processor.probe_audio(wav_path)
    assert metadata.channels == 1
    assert metadata.sample_rate_hz == 16000
    assert metadata.duration_seconds >= 0.25


# ==========================================
# Pipeline Orchestration Tests
# ==========================================

def test_direct_vtt_path_produces_unified_transcript(tmp_path: Path) -> None:
    async def scenario() -> None:
        repository = InMemoryJobRepository()
        request = make_request(transcript=True)
        job = JobRecord(
            event_id=request.event_id,
            meeting_id=request.meeting_id,
            idempotency_key=request.event_id,
            selected_path=ProcessingPath.DIRECT_TRANSCRIPT,
            planned_steps=[],
            request=request,
        )
        await repository.create_or_get(job)
        downloader = FakeDownloader(
            "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n<v Maya>Hello   team</v>\n\n"
            "00:00:02.000 --> 00:00:04.000\nProject update\n"
        )
        media = FakeMediaProcessor()
        whisper = FakeSpeechToText()
        milestone1 = TranscriptMilestone1Runner(
            downloader=downloader,
            media_processor=media,
            speech_to_text=whisper,
            transcript_reader=TxtVttTranscriptReader(),
            normalizer=BasicTranscriptNormalizer(),
        )
        executor = M1M2PipelineOrchestrator(
            repository=repository,
            milestone1=milestone1,
            milestone2=Milestone2Adapter(make_preprocessor()),
            validator=MilestoneContractValidator(),
            work_directory=tmp_path,
            keep_work_files=False,
        )

        await executor.run(job.id)

        stored = await repository.get(job.id)
        assert stored is not None
        assert stored.status == JobStatus.AWAITING_ANALYSIS
        assert stored.current_stage == PipelineStage.PREPROCESSED_TRANSCRIPT_READY
        assert stored.unified_transcript is not None
        assert stored.unified_transcript.text == "Hello team\nProject update"
        assert stored.unified_transcript.source_path == ProcessingPath.DIRECT_TRANSCRIPT
        assert media.called is False
        assert whisper.called is False

    asyncio.run(scenario())


def test_recording_path_uses_media_and_whisper(tmp_path: Path) -> None:
    async def scenario() -> None:
        repository = InMemoryJobRepository()
        request = make_request(transcript=False)
        job = JobRecord(
            event_id=request.event_id,
            meeting_id=request.meeting_id,
            idempotency_key=request.event_id,
            selected_path=ProcessingPath.RECORDING_TO_TRANSCRIPT,
            planned_steps=[],
            request=request,
        )
        await repository.create_or_get(job)
        media = FakeMediaProcessor()
        whisper = FakeSpeechToText()
        milestone1 = TranscriptMilestone1Runner(
            downloader=FakeDownloader('fake-video'),
            media_processor=media,
            speech_to_text=whisper,
            transcript_reader=TxtVttTranscriptReader(),
            normalizer=BasicTranscriptNormalizer(),
        )
        executor = M1M2PipelineOrchestrator(
            repository=repository,
            milestone1=milestone1,
            milestone2=Milestone2Adapter(make_preprocessor()),
            validator=MilestoneContractValidator(),
            work_directory=tmp_path,
            keep_work_files=False,
        )

        await executor.run(job.id)

        stored = await repository.get(job.id)
        assert stored is not None
        assert stored.status == JobStatus.AWAITING_ANALYSIS
        assert stored.unified_transcript is not None
        assert stored.preprocessed_transcript is not None
        assert stored.unified_transcript.text == 'Welcome to the meeting.'
        assert media.called is True
        assert whisper.called is True

    asyncio.run(scenario())


# ==========================================
# Full Preprocessing Pipeline Tests
# ==========================================

def test_complete_preprocessing_pipeline() -> None:
    async def scenario() -> None:
        tokens = TiktokenCounter('cl100k_base')
        pipeline = TranscriptPreprocessingPipeline(
            version='1.0-test',
            filler_remover=FillerRemover(['um', 'uh', 'erm', 'hmm']),
            speaker_normalizer=SpeakerNormalizer(),
            timestamp_cleaner=TimestampCleaner(),
            noise_remover=TranscriptNoiseRemover(),
            chunker=TranscriptChunker(tokens, 24, 34, 6),
            context_builder=ContextBuilder(tokens, 6),
        )
        request = MeetingReadyRequest(
            event_id='preprocess-event',
            meeting_id='preprocess-meeting',
            provider='microsoft_teams',
            title='Implementation planning',
            ended_at='2026-08-07T10:00:00Z',
            transcript={
                'url': 'https://storage.example.com/meeting.vtt',
                'content_type': 'text/vtt',
            },
            participants=[
                Participant(display_name='Maya Chen'),
                Participant(display_name='Arjun Mehta'),
            ],
            metadata={'department': 'Delivery'},
        )
        transcript = UnifiedTranscript(
            text='source',
            language='en',
            duration_seconds=14,
            source_path=ProcessingPath.DIRECT_TRANSCRIPT,
            segments=[
                TranscriptSegment(
                    start_seconds=0,
                    end_seconds=2,
                    text='Maya Chen: Um, welcome welcome welcome everyone.',
                ),
                TranscriptSegment(
                    start_seconds=1.8,
                    end_seconds=4,
                    text='Maya Chen: We have a detailed implementation plan to review.',
                ),
                TranscriptSegment(start_seconds=4, end_seconds=5, text='[Music]'),
                TranscriptSegment(
                    start_seconds=5,
                    end_seconds=7,
                    text='Arjun Mehta: uh Project update is ready for the client.',
                ),
                TranscriptSegment(
                    start_seconds=7,
                    end_seconds=8,
                    text='Arjun Mehta: Project update is ready for the client.',
                ),
                TranscriptSegment(
                    start_seconds=8,
                    end_seconds=11,
                    text='Maya Chen: The engineering team will complete testing tomorrow.',
                ),
                TranscriptSegment(
                    start_seconds=11,
                    end_seconds=14,
                    text='Arjun Mehta: I will send the approval request this afternoon.',
                ),
            ],
        )
        reported_stages: list[PipelineStage] = []

        async def report(stage: PipelineStage, _: int) -> None:
            reported_stages.append(stage)

        result = await pipeline.process(transcript, request, report)

        assert result.version == '1.0-test'
        assert 'Um' not in result.text
        assert ' uh ' not in f' {result.text} '
        assert 'welcome welcome welcome' not in result.text
        assert '[Music]' not in result.text
        assert result.statistics.fillers_removed == 2
        assert result.statistics.noise_segments_removed == 2
        assert result.statistics.timestamps_corrected == 1
        assert result.statistics.speaker_count == 2
        assert [speaker.id for speaker in result.speakers] == ['SPEAKER_01', 'SPEAKER_02']
        assert result.segments[0].speaker == 'SPEAKER_01'
        assert result.segments[1].start_seconds == 2
        assert len(result.chunks) >= 2
        assert all(chunk.token_count <= 34 for chunk in result.chunks)
        assert result.contexts[0].previous_context is None
        assert result.contexts[0].next_context is not None
        assert result.contexts[1].previous_context is not None
        assert result.contexts[0].metadata == {'department': 'Delivery'}
        assert reported_stages == [
            PipelineStage.REMOVE_FILLERS,
            PipelineStage.NORMALIZE_SPEAKERS,
            PipelineStage.CLEAN_TIMESTAMPS,
            PipelineStage.REMOVE_TRANSCRIPT_NOISE,
            PipelineStage.CHUNK_TRANSCRIPT,
            PipelineStage.BUILD_CONTEXT,
        ]

    asyncio.run(scenario())


# ==========================================
# M1 / M2 Integration Contract Tests
# ==========================================

def test_m1_output_is_validated_and_passed_to_m2(tmp_path: Path) -> None:
    async def scenario() -> None:
        repository = InMemoryJobRepository()
        req = make_request(transcript=True)
        job = JobRecord(
            event_id=req.event_id,
            meeting_id=req.meeting_id,
            idempotency_key=req.event_id,
            selected_path=ProcessingPath.DIRECT_TRANSCRIPT,
            planned_steps=[],
            request=req,
        )
        await repository.create_or_get(job)
        events: list[str] = []
        seg = TranscriptSegment(start_seconds=0, end_seconds=2, text='Project update is ready.', speaker='Maya')
        m1_output = UnifiedTranscript(text=seg.text, language='en', duration_seconds=2, segments=[seg], source_path=ProcessingPath.DIRECT_TRANSCRIPT)
        m2_output = PreprocessedTranscript(
            version='test',
            text=seg.text,
            language='en',
            duration_seconds=2,
            segments=[seg],
            speakers=[],
            chunks=[TranscriptChunk(id='chunk-0', index=0, text=seg.text, start_seconds=0, end_seconds=2, speaker_ids=['SPEAKER_01'], token_count=5, source_segment_indexes=[0])],
            contexts=[ContextBundle(id='ctx-0', chunk_id='chunk-0', meeting_id='m', meeting_title='t', provider='offline', ended_at=job.created_at if hasattr(job, 'created_at') else '2026-08-07T10:00:00Z', language='en', text=seg.text, start_seconds=0, end_seconds=2, token_count=5)],
            statistics=PreprocessingStatistics(original_characters=len(seg.text), cleaned_characters=len(seg.text), fillers_removed=0, noise_segments_removed=0, timestamps_corrected=0, speaker_count=1, chunk_count=1),
        )
        milestone1 = StubMilestone1(events, result=m1_output)
        milestone2 = StubMilestone2(events, result=m2_output)

        orchestrator = M1M2PipelineOrchestrator(
            repository=repository,
            milestone1=milestone1,
            milestone2=milestone2,
            validator=MilestoneContractValidator(),
            work_directory=tmp_path,
            keep_work_files=False,
        )
        await orchestrator.run(job.id)

        stored = await repository.get(job.id)
        assert stored is not None
        assert events == ['m1', 'm2']
        assert stored.status == JobStatus.AWAITING_ANALYSIS
        assert stored.current_stage == PipelineStage.PREPROCESSED_TRANSCRIPT_READY

    asyncio.run(scenario())


def test_m1_failure_prevents_m2_execution(tmp_path: Path) -> None:
    async def scenario() -> None:
        repository = InMemoryJobRepository()
        req = make_request(transcript=True)
        job = JobRecord(
            event_id=req.event_id,
            meeting_id=req.meeting_id,
            idempotency_key=req.event_id,
            selected_path=ProcessingPath.DIRECT_TRANSCRIPT,
            planned_steps=[],
            request=req,
        )
        await repository.create_or_get(job)
        events: list[str] = []
        milestone1 = StubMilestone1(
            events,
            error=PipelineProcessingError('m1_source_failed', 'M1 source failed'),
        )
        milestone2 = StubMilestone2(events, result=None)

        orchestrator = M1M2PipelineOrchestrator(
            repository=repository,
            milestone1=milestone1,
            milestone2=milestone2,
            validator=MilestoneContractValidator(),
            work_directory=tmp_path,
            keep_work_files=False,
        )
        await orchestrator.run(job.id)

        stored = await repository.get(job.id)
        assert stored is not None
        assert events == ['m1']
        assert milestone2.calls == 0
        assert stored.status == JobStatus.FAILED
        assert stored.error_code == 'm1_source_failed'

    asyncio.run(scenario())
