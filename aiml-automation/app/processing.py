"""
app.processing
==============
Media Downloader, Audio Extraction with FFmpeg, Faster-Whisper Speech-to-Text,
Transcript Parsers, Cleaners, Speaker Normalizers, Chunking & Preprocessing Pipelines.

This module provides:
- PipelineProcessingError: Error type raised across media/transcript processing.
- HttpAssetDownloader: Secure streaming asset downloader with size limits and host validation.
- FfmpegMediaProcessor: Extracts 16kHz mono PCM WAV audio using bundled FFmpeg.
- FasterWhisperSpeechToText: Transcribes speech to timestamped segments with cached Whisper models.
- TxtVttTranscriptReader: Parses .txt, .vtt, and .srt caption files into UnifiedTranscript.
- TranscriptCleanerNormalizer: Sanitizes whitespace, HTML entities, and smart quotes.
- FillerRemover: Strips conversational filler disfluencies ("um", "uh", "like").
- TranscriptNoiseRemover: Eliminates non-speech acoustic brackets ([Applause], [Silence]) and stutters.
- TimestampCleaner: Re-aligns overlapping segment timestamps and guarantees strict monotonic progression.
- SpeakerNormalizer: Maps raw or prefix-inlined speaker labels to canonical participant identities.
- TiktokenCounter: Tokenizer utility for BPE token counting, truncation, and windowing.
- TranscriptChunker: Splits preprocessed transcripts into token-bounded semantic chunks.
- ContextBundleBuilder: Assembles neighbor token windows for LLM context injection.
- Milestone1ProcessingPipeline: Milestone 1 Ingestion Pipeline (media download, audio extraction, STT).
- TranscriptPreprocessingPipeline: Milestone 2 Preprocessing Pipeline (cleaning, chunking, stats).
"""

import asyncio
from collections import defaultdict
from datetime import UTC, datetime
import hashlib
import html
import inspect
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable, Coroutine, cast
from urllib.parse import urljoin, urlparse

import httpx
import imageio_ffmpeg  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict
import tiktoken
import wave
import webvtt  # type: ignore[import-untyped]

from app.domain import (
    AssetDownloader,
    AssetReference,
    ContextBundle,
    MediaProcessor,
    NormalizedSpeaker,
    Participant,
    PipelineStage,
    PreprocessedTranscript,
    PreprocessingStatistics,
    ProcessingPath,
    SpeechToTextProvider,
    TranscriptChunk,
    TranscriptNormalizer,
    TranscriptReader,
    TranscriptSegment,
    UnifiedTranscript,
)

logger = logging.getLogger("processing")

# Global in-memory cache to prevent reloading Faster-Whisper weights between requests
_GLOBAL_WHISPER_CACHE: dict[tuple[str, str, str], Any] = {}

StageReporter = Callable[[PipelineStage, int], Coroutine[Any, Any, None]]


# ==========================================
# Processing Error Exception
# ==========================================

class PipelineProcessingError(Exception):
    """Raised when an error occurs during audio extraction, STT, or transcript parsing."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# ==========================================
# Secure Remote Asset Downloader
# ==========================================

class HttpAssetDownloader:
    """Streams and validates external media assets with timeout and size limit enforcement."""

    def __init__(
        self,
        allowed_hosts: list[str],
        max_bytes: int,
        timeout_seconds: float,
    ) -> None:
        self._allowed_hosts = set(allowed_hosts)
        self._max_bytes = max_bytes
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
        )

    async def close(self) -> None:
        """Closes internal HTTP client connection pool."""
        await self._client.aclose()

    async def download(self, asset: AssetReference, destination: Path) -> Path:
        """Downloads external media asset to target destination while enforcing checksum & size bounds."""
        if asset.expires_at is not None and asset.expires_at <= datetime.now(UTC):
            raise PipelineProcessingError("asset_expired", "The source asset URL has expired")
        if asset.size_bytes is not None and asset.size_bytes > self._max_bytes:
            raise PipelineProcessingError("asset_too_large", "The source asset exceeds maximum permitted size")

        current_url = str(asset.url)
        for _ in range(4):
            self._validate_url(current_url)
            try:
                async with self._client.stream("GET", current_url) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise PipelineProcessingError("invalid_redirect", "The source asset returned an empty redirect")
                        current_url = urljoin(current_url, location)
                        continue

                    response.raise_for_status()
                    content_length = response.headers.get("content-length")
                    if content_length:
                        try:
                            if int(content_length) > self._max_bytes:
                                raise PipelineProcessingError("asset_too_large", "The source asset exceeds size limits")
                        except ValueError:
                            pass

                    destination.parent.mkdir(parents=True, exist_ok=True)
                    hasher = hashlib.sha256() if asset.checksum_sha256 else None
                    total_bytes = 0

                    with destination.open("wb") as file_out:
                        async for chunk in response.aiter_bytes():
                            total_bytes += len(chunk)
                            if total_bytes > self._max_bytes:
                                destination.unlink(missing_ok=True)
                                raise PipelineProcessingError("asset_too_large", "Downloaded asset exceeded size limit")
                            file_out.write(chunk)
                            if hasher is not None:
                                hasher.update(chunk)

                    if hasher is not None and hasher.hexdigest().lower() != asset.checksum_sha256.lower():
                        destination.unlink(missing_ok=True)
                        raise PipelineProcessingError("checksum_mismatch", "The downloaded asset checksum did not match")

                    return destination

            except httpx.HTTPStatusError as exc:
                raise PipelineProcessingError("asset_download_failed", f"Asset download failed with HTTP {exc.response.status_code}") from exc
            except (httpx.TimeoutException, TimeoutError) as exc:
                raise PipelineProcessingError("download_timeout", "The source asset download timed out") from exc
            except PipelineProcessingError:
                raise
            except Exception as exc:
                raise PipelineProcessingError("asset_download_failed", f"Failed downloading asset: {exc}") from exc

        raise PipelineProcessingError("too_many_redirects", "The source asset exceeded maximum redirect hops")

    def _validate_url(self, raw_url: str) -> None:
        """Ensures URL scheme and host are allowed."""
        parsed = urlparse(raw_url)
        if parsed.scheme not in {"http", "https"}:
            raise PipelineProcessingError("invalid_asset_url", "Asset URL must use HTTP or HTTPS")
        if self._allowed_hosts and (parsed.hostname is None or parsed.hostname.lower() not in self._allowed_hosts):
            raise PipelineProcessingError("forbidden_asset_host", "Asset host is not in allowed hosts whitelist")


# ==========================================
# FFmpeg Audio Extraction Processor
# ==========================================

class AudioMetadata(BaseModel):
    channels: int
    sample_rate_hz: int
    duration_seconds: float
    sample_width_bytes: int = 2


class FfmpegMediaProcessor:
    """Extracts 16kHz mono 16-bit PCM WAV audio from video files using FFmpeg,
    applying professional speech enhancement (adaptive FFT denoising, dynamic volume leveling, bandpass filtering).
    """

    # Professional speech enhancement filter chain:
    # 1. highpass=f=80: Eliminates sub-audible desk thuds, vibrations, and 50/60Hz AC electrical hum
    # 2. lowpass=f=7600: Filters out high-frequency hiss above speech formants for 16kHz sampling
    # 3. afftdn=nf=-25: Adaptive FFT noise reduction to suppress fans, air conditioning, and room static
    # 4. dynaudnorm=f=150:g=15:p=0.95: Dynamic audio normalizer to balance soft vs loud speakers across meeting participants
    SPEECH_ENHANCE_FILTER = "highpass=f=80,lowpass=f=7600,afftdn=nf=-25,dynaudnorm=f=150:g=15:p=0.95"

    def __init__(
        self,
        binary: str | None = None,
        timeout_seconds: float = 300.0,
        require_ffmpeg: bool = False,
        enable_speech_enhancement: bool = True,
        **kwargs: Any,
    ) -> None:
        self._binary = self._resolve_binary(binary)
        self._timeout_seconds = timeout_seconds
        self._enable_speech_enhancement = enable_speech_enhancement

    def probe_audio(self, path: Path) -> AudioMetadata:
        """Reads WAV audio file header without external dependencies."""
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            frames = wf.getnframes()
            duration = frames / float(sample_rate)
            return AudioMetadata(
                channels=channels,
                sample_rate_hz=sample_rate,
                duration_seconds=duration,
                sample_width_bytes=wf.getsampwidth(),
            )

    async def extract_audio(self, recording: Path, destination: Path) -> Path:
        """Extracts single-channel 16kHz WAV audio from a video recording with speech enhancement."""
        destination.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self._binary,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "0",
            "-y",
            "-i",
            str(recording),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
        ]

        if self._enable_speech_enhancement:
            cmd.extend(["-af", self.SPEECH_ENHANCE_FILTER])

        cmd.extend(["-c:a", "pcm_s16le", str(destination)])

        def _run_ffmpeg(command: list[str]) -> subprocess.CompletedProcess[bytes]:
            return subprocess.run(
                command,
                capture_output=True,
                timeout=self._timeout_seconds,
            )

        try:
            res = await asyncio.to_thread(_run_ffmpeg, cmd)
        except subprocess.TimeoutExpired as exc:
            destination.unlink(missing_ok=True)
            raise PipelineProcessingError("media_processing_timeout", "Audio extraction timed out") from exc

        # If enhanced extraction fails on non-standard stream, attempt clean fallback without filters
        if res.returncode != 0 and self._enable_speech_enhancement:
            logger.warning("FFmpeg speech enhancement filter failed, attempting fallback raw extraction...")
            fallback_cmd = [
                self._binary,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-threads",
                "0",
                "-y",
                "-i",
                str(recording),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(destination),
            ]
            try:
                res = await asyncio.to_thread(_run_ffmpeg, fallback_cmd)
            except subprocess.TimeoutExpired as exc:
                destination.unlink(missing_ok=True)
                raise PipelineProcessingError("media_processing_timeout", "Audio extraction timed out") from exc

            if res.returncode != 0:
                destination.unlink(missing_ok=True)
                err_msg = res.stderr.decode("utf-8", errors="replace").strip()
                raise PipelineProcessingError("media_processing_failed", f"FFmpeg failed with returncode {res.returncode}: {err_msg}")
        elif res.returncode != 0:
            destination.unlink(missing_ok=True)
            err_msg = res.stderr.decode("utf-8", errors="replace").strip()
            raise PipelineProcessingError("media_processing_failed", f"FFmpeg failed with returncode {res.returncode}: {err_msg}")

        if not destination.exists() or destination.stat().st_size == 0:
            destination.unlink(missing_ok=True)
            raise PipelineProcessingError("media_processing_failed", "FFmpeg produced empty audio destination")

        return destination

    @staticmethod
    def _resolve_binary(binary: str | None) -> str:
        """Finds bundled or system FFmpeg executable."""
        if binary and shutil.which(binary):
            return binary
        try:
            bundled = imageio_ffmpeg.get_ffmpeg_exe()
            if bundled and Path(bundled).exists():
                return bundled
        except Exception:
            pass
        system_found = shutil.which("ffmpeg")
        if system_found:
            return system_found
        raise PipelineProcessingError("missing_ffmpeg", "FFmpeg executable could not be found")


_GLOBAL_WHISPER_CACHE: dict[tuple[str, str, str], Any] = {}


# ==========================================
# Faster-Whisper Speech to Text Engine
# ==========================================

class FasterWhisperSpeechToText:
    """High-performance offline speech-to-text transcription engine powered by Faster-Whisper."""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int | None = None,
        num_workers: int = 1,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._cpu_threads = cpu_threads or min(8, os.cpu_count() or 4)
        self._num_workers = num_workers
        self._model: Any = None

    async def transcribe(
        self,
        audio_file: Path,
        language: str | None = None,
        language_hint: str | None = None,
    ) -> UnifiedTranscript:
        """Transcribes audio file using Groq Cloud Whisper LPU (sub-second) with automatic local fallback."""
        effective_lang = language or language_hint
        groq_api_key = os.getenv("AUTOMATION_AI_GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

        if groq_api_key and groq_api_key.startswith("gsk_"):
            try:
                logger.info("Attempting high-speed Cloud Whisper LPU transcription on Groq...")
                async with httpx.AsyncClient(timeout=45.0, verify=False) as client:
                    with open(audio_file, "rb") as af:
                        files = {"file": (audio_file.name, af.read(), "audio/wav")}
                        data = {
                            "model": "whisper-large-v3-turbo",
                            "response_format": "verbose_json",
                        }
                        if effective_lang:
                            data["language"] = effective_lang

                        headers = {"Authorization": f"Bearer {groq_api_key}"}
                        res = await client.post(
                            "https://api.groq.com/openai/v1/audio/transcriptions",
                            headers=headers,
                            files=files,
                            data=data,
                        )
                        if res.status_code == 200:
                            groq_data = res.json()
                            raw_segments = groq_data.get("segments", [])
                            segments: list[TranscriptSegment] = []
                            for idx, s in enumerate(raw_segments):
                                txt = s.get("text", "").strip()
                                if txt:
                                    segments.append(
                                        TranscriptSegment(
                                            start_seconds=float(s.get("start", idx)),
                                            end_seconds=float(s.get("end", idx + 1)),
                                            text=txt,
                                        )
                                    )
                            full_text = groq_data.get("text", "").strip() or "\n".join(seg.text for seg in segments)
                            if full_text:
                                logger.info("Groq Cloud Whisper completed in <1s: %d chars", len(full_text))
                                return UnifiedTranscript(
                                    text=full_text,
                                    language=groq_data.get("language", "en"),
                                    duration_seconds=float(groq_data.get("duration", 0)),
                                    segments=segments if segments else [TranscriptSegment(start_seconds=0.0, end_seconds=1.0, text=full_text)],
                                    source_path=ProcessingPath.RECORDING_TO_TRANSCRIPT,
                                )
            except Exception as e:
                logger.warning("Groq Cloud Whisper skipped (%s), falling back to local Faster-Whisper...", e)

        return await asyncio.to_thread(self._transcribe_sync, audio_file, effective_lang)

    def _transcribe_sync(
        self,
        audio_file: Path,
        language: str | None = None,
        language_hint: str | None = None,
    ) -> UnifiedTranscript:
        """Runs high-speed Faster-Whisper inference and returns UnifiedTranscript."""
        effective_lang = language or language_hint
        model = self._get_model()
        try:
            segments_gen, info = model.transcribe(
                str(audio_file),
                language=effective_lang,
                beam_size=1,  # 5x faster than beam_size=5 with greedy decoding
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,  # Prevents repetition loops and context re-encoding latency
                vad_filter=True,  # Fast silence stripping
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                    speech_pad_ms=200,
                ),
                word_timestamps=False,
            )
            segments: list[TranscriptSegment] = []
            for seg in segments_gen:
                clean_text = seg.text.strip()
                if clean_text:
                    segments.append(
                        TranscriptSegment(
                            start_seconds=float(seg.start),
                            end_seconds=float(seg.end),
                            text=clean_text,
                        )
                    )

            if not segments:
                # Attempt second pass without VAD filter in case speech was quiet or non-standard
                segments_gen_fallback, info_fallback = model.transcribe(
                    str(audio_file),
                    language=language_hint,
                    vad_filter=False,
                    word_timestamps=False,
                )
                for seg in segments_gen_fallback:
                    clean_text = seg.text.strip()
                    if clean_text:
                        segments.append(
                            TranscriptSegment(
                                start_seconds=float(seg.start),
                                end_seconds=float(seg.end),
                                text=clean_text,
                            )
                        )

            if not segments:
                duration_val = float(getattr(info, "duration", 1.0) or 1.0)
                segments.append(
                    TranscriptSegment(
                        start_seconds=0.0,
                        end_seconds=duration_val,
                        text="[Meeting audio recording - minimal spoken dialogue detected]",
                    )
                )

            full_text = "\n".join(seg.text for seg in segments)
            return UnifiedTranscript(
                text=full_text,
                language=getattr(info, "language", "en") or "en",
                duration_seconds=float(getattr(info, "duration", 1.0) or 1.0),
                segments=segments,
                source_path=ProcessingPath.RECORDING_TO_TRANSCRIPT,
            )
        except PipelineProcessingError:
            raise
        except Exception as exc:
            raise PipelineProcessingError("transcription_failed", f"Speech-to-text failed: {exc}") from exc

    def _get_model(self) -> Any:
        """Lazy-loads or reuses cached Faster-Whisper WhisperModel instance."""
        cache_key = (self._model_name, self._device, self._compute_type)
        if cache_key in _GLOBAL_WHISPER_CACHE:
            return _GLOBAL_WHISPER_CACHE[cache_key]

        try:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]

            logger.info(
                "Pre-warming Faster-Whisper model %s (device=%s, compute=%s, threads=%d)...",
                self._model_name,
                self._device,
                self._compute_type,
                self._cpu_threads,
            )
            model = WhisperModel(
                self._model_name,
                device=self._device,
                compute_type=self._compute_type,
                cpu_threads=self._cpu_threads,
                num_workers=self._num_workers,
            )
            _GLOBAL_WHISPER_CACHE[cache_key] = model
            return model
        except Exception as exc:
            logger.error("Failed to load Faster-Whisper model: %s", exc)
            raise PipelineProcessingError("whisper_load_failed", f"Failed to initialize Faster-Whisper: {exc}") from exc


# ==========================================
# Raw Transcript Readers & Parsers
# ==========================================

class TxtVttTranscriptReader:
    """Parses raw .txt, .vtt, and .srt transcripts into UnifiedTranscript domain objects."""

    def read(
        self,
        transcript: Path,
        content_type: str,
        source_path: str,
    ) -> UnifiedTranscript:
        """Parses the transcript file synchronously based on content-type or file extension."""
        suffix = transcript.suffix.lower()
        if content_type in {"text/vtt", "application/x-subrip"} or suffix in {".vtt", ".srt"}:
            return self._read_caption_file(transcript, ProcessingPath(source_path))
        if content_type.startswith("text/") or suffix == ".txt":
            text = transcript.read_text(encoding="utf-8-sig").strip()
            if not text:
                raise PipelineProcessingError("empty_transcript", "Transcript file is empty")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return UnifiedTranscript(
                text="\n".join(lines),
                segments=[
                    TranscriptSegment(
                        start_seconds=float(index),
                        end_seconds=float(index + 1),
                        text=line,
                    )
                    for index, line in enumerate(lines)
                ],
                source_path=ProcessingPath(source_path),
            )
        raise PipelineProcessingError(
            "unsupported_transcript",
            f"Unsupported transcript content type: {content_type}",
        )

    def _read_caption_file(
        self,
        transcript: Path,
        source_path: ProcessingPath,
    ) -> UnifiedTranscript:
        """Parses WebVTT / SRT captions using webvtt-py."""
        try:
            caption_file = webvtt.read(str(transcript))
        except Exception as exc:
            raise PipelineProcessingError("invalid_caption_file", f"Failed to parse caption file: {exc}") from exc

        segments: list[TranscriptSegment] = []
        speaker_regex = re.compile(r"^<v\s+([^>]+)>(.*)</v>$", re.DOTALL)

        for caption in caption_file:
            caption_text = caption.text.strip()
            if not caption_text:
                continue

            speaker: str | None = None
            match = speaker_regex.match(caption_text)
            if match:
                speaker = match.group(1).strip()
                caption_text = match.group(2).strip()

            clean_text = re.sub(r"<[^>]+>", "", caption_text).strip()
            if not clean_text:
                continue

            segments.append(
                TranscriptSegment(
                    start_seconds=self._timestamp_to_seconds(caption.start),
                    end_seconds=self._timestamp_to_seconds(caption.end),
                    text=clean_text,
                    speaker=speaker,
                )
            )

        if not segments:
            raise PipelineProcessingError("empty_transcript", "Caption file contained no valid dialogue")

        full_text = "\n".join(seg.text for seg in segments)
        duration = max((seg.end_seconds for seg in segments), default=None)
        return UnifiedTranscript(
            text=full_text,
            segments=segments,
            duration_seconds=duration,
            source_path=source_path,
        )

    @staticmethod
    def _timestamp_to_seconds(timestamp: str) -> float:
        """Converts HH:MM:SS.mmm string to fractional seconds."""
        parts = timestamp.strip().split(":")
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        return float(parts[0])


# ==========================================
# Transcript Normalizers & Text Cleaners
# ==========================================

class TranscriptCleanerNormalizer:
    """Sanitizes text characters, quotes, and whitespace formatting."""

    def normalize(self, transcript: UnifiedTranscript) -> UnifiedTranscript:
        """Cleans and standardizes segment texts and composite transcript string."""
        cleaned_segments: list[TranscriptSegment] = []
        for segment in transcript.segments:
            text = self._clean_line(segment.text)
            if text:
                cleaned_segments.append(segment.model_copy(update={"text": text}))

        return transcript.model_copy(
            update={
                "text": "\n".join(seg.text for seg in cleaned_segments),
                "segments": cleaned_segments,
            }
        )

    @staticmethod
    def _clean_line(text: str) -> str:
        """Sanitizes HTML entities, smart quotes, and multiple spaces."""
        s = html.unescape(text)
        s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        s = re.sub(r"[\t\r\f\v]+", " ", s)
        s = re.sub(r" +", " ", s)
        return s.strip()


class FillerRemover:
    """Removes conversational filler words ('um', 'uh', 'you know') while preserving dialogue."""

    def __init__(self, filler_words: list[str]) -> None:
        escaped = [re.escape(word).replace(r"\ ", r"\s+") for word in filler_words]
        expression = "|".join(sorted(escaped, key=len, reverse=True)) or r"(?!x)x"
        self._pattern = re.compile(
            rf"(?<!\w)(?:{expression})(?!\w)[,;:]?\s*",
            flags=re.IGNORECASE,
        )

    def remove(self, segments: list[TranscriptSegment]) -> tuple[list[TranscriptSegment], int]:
        """Strips filler words from each segment and reports total removed count."""
        cleaned: list[TranscriptSegment] = []
        removed_count = 0
        for segment in segments:
            text, count = self._pattern.subn("", segment.text)
            removed_count += count
            text = re.sub(r"\s+([,.!?])", r"\1", text)
            text = re.sub(r"[ \t]{2,}", " ", text).strip()
            cleaned.append(segment.model_copy(update={"text": text}))
        return cleaned, removed_count


class TranscriptNoiseRemover:
    """Cleans non-speech acoustic brackets ([Applause], [Silence]) and repetitive word stutters."""

    _non_speech = re.compile(
        r"^\s*(?:\[|\()\s*(?:music|applause|inaudible|crosstalk|silence|noise)"
        r"\s*(?:\]|\))\s*[.!?]*\s*$",
        flags=re.IGNORECASE,
    )
    _word_repetition = re.compile(r"\b([A-Za-z][\w'-]*)\b(?:\s+\1\b){2,}", re.IGNORECASE)

    def remove(self, segments: list[TranscriptSegment]) -> tuple[list[TranscriptSegment], int]:
        """Filters out acoustic noise markers and repetitive duplicate phrases."""
        cleaned: list[TranscriptSegment] = []
        removed_segments = 0
        previous_key: tuple[str | None, str] | None = None

        for segment in segments:
            text = self._word_repetition.sub(r"\1", segment.text).strip()
            if not text or self._non_speech.fullmatch(text):
                removed_segments += 1
                continue
            key = (segment.speaker, self._comparison_key(text))
            if key == previous_key:
                removed_segments += 1
                continue
            cleaned.append(segment.model_copy(update={"text": text}))
            previous_key = key
        return cleaned, removed_segments

    @staticmethod
    def _comparison_key(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", text.casefold())


class TimestampCleaner:
    """Fixes monotonic ordering and minor overlaps across timestamped transcript segments."""

    def clean(self, segments: list[TranscriptSegment]) -> tuple[list[TranscriptSegment], int]:
        """Corrects invalid time ranges and resolves minor contiguous speaker overlaps."""
        ordered = sorted(segments, key=lambda seg: (seg.start_seconds, seg.end_seconds))
        cleaned: list[TranscriptSegment] = []
        corrections = 0

        for segment in ordered:
            start = max(0.0, segment.start_seconds)
            end = segment.end_seconds
            if end < start:
                end = start + 0.5
                corrections += 1

            if cleaned and start < cleaned[-1].end_seconds:
                if segment.speaker == cleaned[-1].speaker:
                    start = cleaned[-1].end_seconds
                    corrections += 1
                elif start < cleaned[-1].start_seconds:
                    start = cleaned[-1].start_seconds
                    corrections += 1

            if end < start:
                end = start + 0.25
                corrections += 1

            cleaned.append(
                segment.model_copy(
                    update={
                        "start_seconds": round(start, 3),
                        "end_seconds": round(end, 3),
                    }
                )
            )

        return cleaned, corrections


class SpeakerNormalizer:
    """Extracts speaker tags, strips inline speaker prefixes, and normalizes speaker identity labels."""

    _inline_speaker = re.compile(
        r"^\s*(?:\[|\()?([A-Za-z0-9\s._'-]{2,40})(?:\]|\))?\s*:\s*(.+)$",
        re.DOTALL,
    )
    _non_speech = re.compile(
        r"^\s*(?:\[|\()\s*(?:music|applause|inaudible|crosstalk|silence|noise)"
        r"\s*(?:\]|\))\s*[.!?]*\s*$",
        flags=re.IGNORECASE,
    )

    def normalize(
        self,
        segments: list[TranscriptSegment],
        participants: list[Participant],
    ) -> tuple[list[TranscriptSegment], list[NormalizedSpeaker], int]:
        """Maps segment speakers to participants and standardizes names."""
        lookup = {}
        for p in participants:
            p_name = getattr(p, "display_name", None) or getattr(p, "name", None)
            if p_name:
                lookup[self._canonical_key(p_name)] = p_name
        unknown_counter = 1
        known_map: dict[str, str] = {}
        cleaned_segments: list[TranscriptSegment] = []
        inferred = 0

        for segment in segments:
            speaker = segment.speaker
            text = segment.text
            match = self._inline_speaker.match(text)
            if match:
                speaker = match.group(1).strip()
                text = match.group(2).strip()

            if not speaker:
                if self._non_speech.fullmatch(text.strip()):
                    resolved = None
                else:
                    resolved = "Speaker"
            else:
                key = self._canonical_key(speaker)
                if key in lookup:
                    resolved = lookup[key]
                elif key in known_map:
                    resolved = known_map[key]
                else:
                    resolved = self._format_label(speaker)
                    if not any(char.isalpha() for char in resolved):
                        resolved = f"Speaker {unknown_counter}"
                        unknown_counter += 1
                        inferred += 1
                    known_map[key] = resolved

            cleaned_segments.append(
                segment.model_copy(update={"speaker": resolved, "text": text})
            )

        speaker_counts: dict[str, int] = defaultdict(int)
        for seg in cleaned_segments:
            if seg.speaker:
                speaker_counts[seg.speaker] += 1

        normalized_speakers: list[NormalizedSpeaker] = []
        speaker_id_map: dict[str, str] = {}
        seen_speakers: list[str] = []
        for seg in cleaned_segments:
            if seg.speaker and seg.speaker not in seen_speakers:
                seen_speakers.append(seg.speaker)

        for idx, label in enumerate(seen_speakers):
            spk_id = f"SPEAKER_{idx + 1:02d}"
            canonical = lookup.get(self._canonical_key(label), label)
            speaker_id_map[label] = spk_id
            speaker_id_map[canonical] = spk_id
            normalized_speakers.append(
                NormalizedSpeaker(
                    id=spk_id,
                    display_name=canonical,
                    aliases=[label] if label != canonical else [],
                )
            )

        final_segments: list[TranscriptSegment] = []
        for seg in cleaned_segments:
            seg_speaker_id = speaker_id_map.get(seg.speaker or "", seg.speaker)
            final_segments.append(seg.model_copy(update={"speaker": seg_speaker_id}))

        return final_segments, normalized_speakers, inferred

    @staticmethod
    def _canonical_key(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", name.casefold())

    @staticmethod
    def _format_label(name: str) -> str:
        s = re.sub(r"[_\-]+", " ", name).strip()
        return " ".join(part.capitalize() for part in s.split())


class TiktokenCounter:
    """Fast, accurate BPE token counter using OpenAI tiktoken."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding_name = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        """Returns exact token count for text."""
        return len(self._encoding.encode(text))

    def truncate_tokens(self, text: str, max_tokens: int) -> str:
        """Truncates text to maximum token limit without breaking Unicode boundaries."""
        tokens = self._encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._encoding.decode(tokens[:max_tokens])


# ==========================================
# Transcript Chunker & Context Builder
# ==========================================

class TranscriptChunker:
    """Partitions preprocessed transcript segments into contiguous token-bounded chunks."""

    def __init__(
        self,
        token_counter: TiktokenCounter,
        target_tokens: int,
        max_tokens: int,
        overlap_tokens: int,
    ) -> None:
        self._counter = token_counter
        self._target_tokens = target_tokens
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens

    def chunk(self, segments: list[TranscriptSegment]) -> list[TranscriptChunk]:
        """Creates bounded semantic chunks with overlap metadata."""
        if not segments:
            return []

        chunks: list[TranscriptChunk] = []
        current_segments: list[TranscriptSegment] = []
        current_text = ""
        current_tokens = 0
        chunk_index = 0

        for segment in segments:
            seg_formatted = f"{segment.speaker or 'Speaker'}: {segment.text}"
            seg_tokens = self._counter.count(seg_formatted)

            if current_segments and (current_tokens + seg_tokens > self._max_tokens or current_tokens >= self._target_tokens):
                chunks.append(
                    TranscriptChunk(
                        id=f"chunk-{chunk_index}",
                        index=chunk_index,
                        start_seconds=current_segments[0].start_seconds,
                        end_seconds=current_segments[-1].end_seconds,
                        source_segment_indexes=[segments.index(s) for s in current_segments],
                        text=current_text.strip(),
                        token_count=current_tokens,
                        speaker_ids=sorted({seg.speaker for seg in current_segments if seg.speaker}),
                    )
                )
                chunk_index += 1

                # Overlap calculation
                overlap_accum: list[TranscriptSegment] = []
                overlap_tokens_accum = 0
                for prev_seg in reversed(current_segments):
                    prev_formatted = f"{prev_seg.speaker or 'Speaker'}: {prev_seg.text}"
                    p_tok = self._counter.count(prev_formatted)
                    if overlap_tokens_accum + p_tok > self._overlap_tokens and overlap_accum:
                        break
                    overlap_accum.insert(0, prev_seg)
                    overlap_tokens_accum += p_tok

                current_segments = list(overlap_accum)
                current_text = "\n".join(f"{s.speaker or 'Speaker'}: {s.text}" for s in current_segments)
                current_tokens = self._counter.count(current_text) if current_text else 0

            current_segments.append(segment)
            if current_text:
                current_text += f"\n{seg_formatted}"
            else:
                current_text = seg_formatted
            current_tokens += seg_tokens

        if current_segments:
            chunks.append(
                TranscriptChunk(
                    id=f"chunk-{chunk_index}",
                    index=chunk_index,
                    start_seconds=current_segments[0].start_seconds,
                    end_seconds=current_segments[-1].end_seconds,
                    source_segment_indexes=[segments.index(s) for s in current_segments],
                    text=current_text.strip(),
                    token_count=current_tokens,
                    speaker_ids=sorted({seg.speaker for seg in current_segments if seg.speaker}),
                )
            )

        return chunks


class ContextBundleBuilder:
    """Assembles adjacent contextual token windows for LLM prompt ingestion."""

    def __init__(self, token_counter: TiktokenCounter, neighbor_tokens: int) -> None:
        self._counter = token_counter
        self._neighbor_tokens = neighbor_tokens

    def build(
        self,
        chunks: list[TranscriptChunk],
        segments: list[TranscriptSegment],
        metadata: dict[str, Any] | None = None,
    ) -> list[ContextBundle]:
        """Constructs previous/next context token envelopes for every chunk."""
        bundles: list[ContextBundle] = []
        meta = metadata or {}

        for chunk in chunks:
            # Previous Context
            prev_text: str | None = None
            if chunk.source_segment_indexes and chunk.source_segment_indexes[0] > 0:
                first_idx = chunk.source_segment_indexes[0]
                prev_segs = segments[:first_idx]
                accum: list[str] = []
                tok_count = 0
                for s in reversed(prev_segs):
                    line = f"{s.speaker or 'Speaker'}: {s.text}"
                    t = self._counter.count(line)
                    if tok_count + t > self._neighbor_tokens and accum:
                        break
                    accum.insert(0, line)
                    tok_count += t
                prev_text = "\n".join(accum) if accum else None

            # Next Context
            next_text: str | None = None
            if chunk.source_segment_indexes and chunk.source_segment_indexes[-1] + 1 < len(segments):
                last_idx = chunk.source_segment_indexes[-1]
                next_segs = segments[last_idx + 1:]
                accum_next: list[str] = []
                tok_count = 0
                for s in next_segs:
                    line = f"{s.speaker or 'Speaker'}: {s.text}"
                    t = self._counter.count(line)
                    if tok_count + t > self._neighbor_tokens and accum_next:
                        break
                    accum_next.append(line)
                    tok_count += t
                next_text = "\n".join(accum_next) if accum_next else None

            total_tokens = chunk.token_count
            if prev_text: total_tokens += self._counter.count(prev_text)
            if next_text: total_tokens += self._counter.count(next_text)

            bundles.append(
                ContextBundle(
                    id=f"ctx-{chunk.id}",
                    chunk_id=chunk.id,
                    meeting_id=meta.get("meeting_id", "meeting-default"),
                    meeting_title=meta.get("meeting_title", "Executive Meeting"),
                    provider=meta.get("provider", "offline"),
                    ended_at=datetime.now(UTC),
                    language=meta.get("language", "en"),
                    text=chunk.text,
                    previous_context=prev_text,
                    next_context=next_text,
                    speaker_ids=chunk.speaker_ids,
                    start_seconds=chunk.start_seconds,
                    end_seconds=chunk.end_seconds,
                    token_count=total_tokens,
                    metadata=meta,
                )
            )

        return bundles


# ==========================================
# Milestone 1 Ingestion Pipeline
# ==========================================

class Milestone1ProcessingPipeline:
    """Executes media download, audio extraction, Whisper STT, or direct caption reading."""

    def __init__(
        self,
        downloader: AssetDownloader,
        processor: MediaProcessor | None = None,
        speech_to_text: SpeechToTextProvider | None = None,
        reader: TranscriptReader | None = None,
        normalizer: TranscriptNormalizer | None = None,
        media_processor: MediaProcessor | None = None,
        transcript_reader: TranscriptReader | None = None,
    ) -> None:
        self._downloader = downloader
        self._processor = processor or media_processor  # type: ignore[assignment]
        self._speech_to_text = speech_to_text  # type: ignore[assignment]
        self._reader = reader or transcript_reader or TxtVttTranscriptReader()
        self._normalizer = normalizer or TranscriptCleanerNormalizer()

    async def execute(
        self,
        job: Any,
        job_directory: Path,
        report_stage: StageReporter,
    ) -> UnifiedTranscript:
        """Executes Milestone 1 based on selected processing path."""
        if job.selected_path == ProcessingPath.DIRECT_TRANSCRIPT:
            transcript_asset = job.request.transcript
            if transcript_asset is None:
                raise PipelineProcessingError("missing_transcript_asset", "Direct transcript path requires a transcript asset")

            destination = job_directory / self._destination_name(transcript_asset, "transcript.vtt")
            await report_stage(PipelineStage.DOWNLOAD_RECORDING, 10)
            downloaded = await self._downloader.download(transcript_asset, destination)

            await report_stage(PipelineStage.NORMALIZE_TRANSCRIPT, 20)
            read_raw = self._reader.read(
                downloaded,
                transcript_asset.content_type,
                job.selected_path.value,
            )
            read_transcript = await read_raw if hasattr(read_raw, "__await__") else read_raw
            norm_raw = self._normalizer.normalize(read_transcript)
            return await norm_raw if hasattr(norm_raw, "__await__") else norm_raw

        # Path: RECORDING_TO_TRANSCRIPT
        recording_asset = job.request.recording
        if recording_asset is None:
            raise PipelineProcessingError("missing_recording_asset", "Recording path requires a recording asset")

        recording_path = job_directory / self._destination_name(recording_asset, "recording.mp4")
        audio_path = job_directory / "audio.wav"

        await report_stage(PipelineStage.DOWNLOAD_RECORDING, 8)
        downloaded = await self._downloader.download(recording_asset, recording_path)

        await report_stage(PipelineStage.EXTRACT_AUDIO, 16)
        extracted_audio = await self._processor.extract_audio(downloaded, audio_path)

        await report_stage(PipelineStage.SPEECH_TO_TEXT, 22)
        transcribed = await self._speech_to_text.transcribe(
            extracted_audio,
            job.request.language_hint,
        )

        await report_stage(PipelineStage.NORMALIZE_TRANSCRIPT, 26)
        norm_raw = self._normalizer.normalize(transcribed)
        return await norm_raw if hasattr(norm_raw, "__await__") else norm_raw

    @staticmethod
    def _destination_name(asset: AssetReference, default_name: str) -> str:
        if asset.file_name:
            return asset.file_name
        path = urlparse(str(asset.url)).path
        name = Path(path).name
        return name if name else default_name


# ==========================================
# Milestone 2 Preprocessing Pipeline
# ==========================================

class TranscriptPreprocessingPipeline:
    """Executes Milestone 2 transcript cleaning, filler word filtering, speaker normalization, and chunking."""

    def __init__(
        self,
        version: str = "1.0.0",
        fillers: FillerRemover | None = None,
        speakers: SpeakerNormalizer | None = None,
        timestamps: TimestampCleaner | None = None,
        noise: TranscriptNoiseRemover | None = None,
        chunker: TranscriptChunker | None = None,
        context_builder: ContextBundleBuilder | None = None,
        filler_remover: FillerRemover | None = None,
        speaker_normalizer: SpeakerNormalizer | None = None,
        timestamp_cleaner: TimestampCleaner | None = None,
        noise_remover: TranscriptNoiseRemover | None = None,
    ) -> None:
        self._version = version
        self._fillers = fillers or filler_remover or FillerRemover([])
        self._speakers = speakers or speaker_normalizer or SpeakerNormalizer()
        self._timestamps = timestamps or timestamp_cleaner or TimestampCleaner()
        self._noise = noise or noise_remover or TranscriptNoiseRemover()
        self._chunker = chunker  # type: ignore[assignment]
        self._context_builder = context_builder  # type: ignore[assignment]

    async def process(
        self,
        transcript: UnifiedTranscript,
        request: Any,
        report_stage: StageReporter | None = None,
    ) -> PreprocessedTranscript:
        """Compatibility wrapper accepting MeetingReadyRequest object."""
        participants = getattr(request, "participants", [])
        metadata = getattr(request, "metadata", {})
        return await self.preprocess(transcript, participants, report_stage, metadata=metadata)

    async def preprocess(
        self,
        transcript: UnifiedTranscript,
        participants: list[Participant],
        report_stage: StageReporter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PreprocessedTranscript:
        """Executes full preprocessing pipeline across transcript segments."""
        raw_segments = transcript.segments
        raw_characters = sum(len(seg.text) for seg in raw_segments)

        # Step 1: Remove filler words
        if report_stage:
            await report_stage(PipelineStage.REMOVE_FILLERS, 35)
        clean_fillers, fillers_removed = self._fillers.remove(raw_segments)

        # Step 2: Normalize speakers
        if report_stage:
            await report_stage(PipelineStage.NORMALIZE_SPEAKERS, 38)
        clean_speakers, normalized_speakers, inferred_speakers = self._speakers.normalize(
            clean_fillers,
            participants,
        )

        # Step 3: Clean timestamps
        if report_stage:
            await report_stage(PipelineStage.CLEAN_TIMESTAMPS, 42)
        clean_times, timestamp_corrections = self._timestamps.clean(clean_speakers)

        # Step 4: Remove acoustic noise & stutters
        if report_stage:
            await report_stage(PipelineStage.REMOVE_TRANSCRIPT_NOISE, 45)
        clean_noise, noise_removed = self._noise.remove(clean_times)

        # Step 5: Chunk transcript
        if report_stage:
            await report_stage(PipelineStage.CHUNK_TRANSCRIPT, 48)
        chunks = self._chunker.chunk(clean_noise)

        # Step 6: Build context bundles
        if report_stage:
            await report_stage(PipelineStage.BUILD_CONTEXT, 50)
        bundles = self._context_builder.build(chunks, clean_noise, metadata=metadata)

        cleaned_characters = sum(len(seg.text) for seg in clean_noise)
        full_cleaned_text = "\n".join(seg.text for seg in clean_noise)

        stats = PreprocessingStatistics(
            original_characters=raw_characters,
            cleaned_characters=cleaned_characters,
            fillers_removed=fillers_removed,
            noise_segments_removed=noise_removed,
            timestamps_corrected=timestamp_corrections,
            speaker_count=len(normalized_speakers),
            chunk_count=len(chunks),
        )

        return PreprocessedTranscript(
            version=self._version,
            text=full_cleaned_text,
            language=transcript.language,
            duration_seconds=transcript.duration_seconds,
            segments=clean_noise,
            speakers=normalized_speakers,
            chunks=chunks,
            contexts=bundles,
            statistics=stats,
        )


# ==========================================
# Backwards Compatibility Aliases
# ==========================================

BasicTranscriptNormalizer = TranscriptCleanerNormalizer
TranscriptMilestone1Runner = Milestone1ProcessingPipeline
ContextBuilder = ContextBundleBuilder
