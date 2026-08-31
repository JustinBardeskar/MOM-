"""
app.recorder
============
Live Screen and Audio Meeting Recorder Engine and Webhook Router.

This module provides:
- PipelineDispatcher: Transmits recorded MP4 video and session metadata to the preprocessing pipeline.
- FFmpegScreenRecorder: Captures screen and microphone audio using DirectShow/GDI on Windows or x11grab on Linux.
- recorder_router: FastAPI REST router managing /start, /stop, /status, and /upload endpoints for recording sessions.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, cast
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
import httpx
import imageio_ffmpeg  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

logger = logging.getLogger("recorder")


# ==========================================
# Background Pipeline Dispatcher Client
# ==========================================

class PipelineDispatcher:
    """Dispatches recorded MP4 video and session metadata to the existing preprocessing pipeline."""

    def __init__(self, base_url: str = "http://127.0.0.1:8100") -> None:
        self._base_url = base_url.rstrip("/")
        self._upload_endpoint = f"{self._base_url}/api/v1/meetings/upload"

    async def dispatch_recording_async(
        self,
        video_path: Path,
        metadata: dict[str, Any],
        openrouter_api_key: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Asynchronously transmits the MP4 file to /api/v1/meetings/upload via multipart/form-data."""
        if not video_path.exists():
            raise FileNotFoundError(f"Recorded video file does not exist: {video_path}")

        meeting_title = metadata.get("meeting_title") or video_path.stem
        provider = metadata.get("provider") or "teams"

        logger.info(
            "Dispatching recorded file '%s' (%d bytes) to existing pipeline at %s [title='%s', provider=%s]",
            video_path.name,
            video_path.stat().st_size,
            self._upload_endpoint,
            meeting_title,
            provider,
        )

        data: dict[str, str] = {
            "meeting_title": meeting_title,
            "provider": provider,
        }
        if openrouter_api_key and openrouter_api_key.strip():
            data["openrouter_api_key"] = openrouter_api_key.strip()
        if model and model.strip():
            data["model"] = model.strip()

        # Stream multipart upload to pipeline
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0), verify=False) as client:
            with open(video_path, "rb") as f:
                files = {"file": (video_path.name, f, "video/mp4")}
                response = await client.post(self._upload_endpoint, data=data, files=files)

        if response.status_code not in (200, 202):
            logger.error("Pipeline rejected recording upload [status=%d]: %s", response.status_code, response.text)
            raise RuntimeError(f"Pipeline error (HTTP {response.status_code}): {response.text}")

        res_json = response.json()
        logger.info("Successfully ingested recording into MOM pipeline: Job ID %s", res_json.get("job_id"))
        return cast(dict[str, Any], res_json)


# ==========================================
# FFmpeg Screen & Audio Recording Engine
# ==========================================

class FFmpegScreenRecorder:
    """Manages full screen and system/microphone audio recording using bundled FFmpeg."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self._output_dir = output_dir or Path("runtime/recordings")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._ffmpeg_binary = self._resolve_ffmpeg()
        self._process: subprocess.Popen[bytes] | None = None
        self._current_file: Path | None = None
        self._session_metadata: dict[str, Any] = {}
        self._start_time: float | None = None
        self._is_recording = False

    @property
    def is_recording(self) -> bool:
        """Returns True if the recorder process is actively capturing."""
        if self._process is not None and self._process.poll() is None:
            return True
        return False

    @staticmethod
    def _resolve_ffmpeg() -> str:
        """Locates FFmpeg binary executable."""
        try:
            return cast(str, imageio_ffmpeg.get_ffmpeg_exe())
        except Exception:
            return "ffmpeg"

    @classmethod
    def detect_audio_device(cls) -> str | None:
        """Detects available DirectShow audio devices on Windows."""
        binary = cls._resolve_ffmpeg()
        try:
            res = subprocess.run(
                [binary, "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
            output = res.stderr or res.stdout or ""
            audio_matches = re.findall(r'\"([^\"]+)\"\s+\(audio\)', output)
            if audio_matches:
                logger.info("Detected Windows DirectShow audio input: '%s'", audio_matches[0])
                return audio_matches[0]
        except Exception as exc:
            logger.debug("Failed detecting directshow devices: %s", exc)
        return None

    def start_recording(
        self,
        meeting_title: str = "Live Meeting Recording",
        provider: str = "teams",
        participants: list[str] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Spawns an asynchronous FFmpeg child process to record the desktop screen & audio."""
        if self.is_recording:
            raise RuntimeError(f"A recording session is already active: {self._current_file}")

        session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        sanitized_title = re.sub(r"[^\w\-_\. ]", "_", meeting_title)[:50].strip()
        filename = f"rec_{timestamp}_{sanitized_title}_{session_id}.mp4"
        self._current_file = self._output_dir / filename

        self._session_metadata = {
            "session_id": session_id,
            "meeting_title": meeting_title,
            "provider": provider,
            "participants": participants or [],
            "start_time_iso": datetime.now(timezone.utc).isoformat(),
            "output_path": str(self._current_file.absolute()),
            "extra_metadata": extra_metadata or {},
        }

        # Build OS-specific FFmpeg screen capture command
        cmd: list[str] = [self._ffmpeg_binary, "-y"]
        if os.name == "nt":
            audio_dev = self.detect_audio_device()
            cmd.extend([
                "-f", "gdigrab",
                "-framerate", "15",
                "-draw_mouse", "1",
                "-i", "desktop",
            ])
            if audio_dev:
                cmd.extend([
                    "-f", "dshow",
                    "-i", f"audio={audio_dev}",
                ])
        else:
            cmd.extend([
                "-f", "x11grab",
                "-framerate", "15",
                "-draw_mouse", "1",
                "-i", ":0.0",
            ])

        cmd.extend([
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            str(self._current_file),
        ])

        logger.info("Starting screen recording with command: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._start_time = time.time()
        self._is_recording = True

        return {
            "status": "recording_started",
            "session_id": session_id,
            "meeting_title": meeting_title,
            "file_name": filename,
            "file_path": str(self._current_file),
        }

    def stop_recording(self) -> tuple[Path, dict[str, Any]]:
        """Stops the active recording session and flushes the MP4 file."""
        if not self._is_recording or self._process is None or self._current_file is None:
            raise RuntimeError("No active recording session to stop.")

        logger.info("Stopping screen recording process...")
        try:
            if self._process.stdin:
                self._process.stdin.write(b"q\n")
                self._process.stdin.flush()
            self._process.wait(timeout=5)
        except Exception:
            self._process.terminate()
            self._process.wait(timeout=3)

        duration_sec = round(time.time() - (self._start_time or time.time()), 2)
        file_size = self._current_file.stat().st_size if self._current_file.exists() else 0

        metadata = dict(self._session_metadata)
        metadata.update({
            "end_time_iso": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": duration_sec,
            "file_size_bytes": file_size,
        })

        out_path = self._current_file
        self._process = None
        self._current_file = None
        self._is_recording = False
        self._start_time = None
        self._session_metadata = {}

        return out_path, metadata

    def get_status(self) -> dict[str, Any]:
        """Returns current recording session status."""
        active = self.is_recording
        duration = round(time.time() - self._start_time, 1) if active and self._start_time else 0.0
        return {
            "is_recording": active,
            "status": "recording" if active else "idle",
            "duration_seconds": duration,
            "elapsed_seconds": duration,
            "current_session": self._session_metadata if active else None,
        }


# ==========================================
# REST API Endpoints & Request Models
# ==========================================

recorder_router = APIRouter(prefix="/api/v1/recorder", tags=["Recorder Plugin"])
_recorder_instance = FFmpegScreenRecorder()
_dispatcher_instance = PipelineDispatcher()


class StartRecordingRequest(BaseModel):
    """Payload to start a screen recording session."""
    meeting_title: str = Field(default="Live Meeting Recording", description="Title of the meeting session")
    provider: str = Field(default="teams", description="Meeting platform: teams, meet, zoom, webex, skype, offline")
    participants: list[str] = Field(default_factory=list, description="List of participant names")
    openrouter_api_key: str | None = None
    model: str | None = None


class StopRecordingRequest(BaseModel):
    """Payload to stop an active recording session."""
    openrouter_api_key: str | None = None
    model: str | None = None


@recorder_router.post("/start", status_code=status.HTTP_200_OK)
async def start_recording(payload: StartRecordingRequest) -> dict[str, Any]:
    """Starts the FFmpeg screen & audio recording engine for a live meeting session."""
    try:
        return _recorder_instance.start_recording(
            meeting_title=payload.meeting_title,
            provider=payload.provider,
            participants=payload.participants,
            extra_metadata={
                "openrouter_api_key": payload.openrouter_api_key,
                "model": payload.model,
            },
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Failed to start recording: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@recorder_router.post("/stop", status_code=status.HTTP_202_ACCEPTED)
async def stop_recording(payload: StopRecordingRequest | None = None) -> dict[str, Any]:
    """Stops the active recording session and automatically dispatches to the preprocessing pipeline."""
    try:
        video_path, metadata = _recorder_instance.stop_recording()
        extra_meta = metadata.get("extra_metadata", {})
        api_key = (payload and payload.openrouter_api_key) or extra_meta.get("openrouter_api_key")
        model = (payload and payload.model) or extra_meta.get("model")

        pipeline_res = await _dispatcher_instance.dispatch_recording_async(
            video_path=video_path,
            metadata=metadata,
            openrouter_api_key=api_key,
            model=model,
        )
        return {
            "status": "recording_stopped_and_dispatched",
            "recording_metadata": metadata,
            "pipeline_response": pipeline_res,
            "job_id": pipeline_res.get("job_id"),
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed stopping and dispatching recording: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@recorder_router.get("/status", status_code=status.HTTP_200_OK)
async def get_recording_status() -> dict[str, Any]:
    """Returns the live recording state and duration."""
    return _recorder_instance.get_status()


@recorder_router.post("/submit-blob", status_code=status.HTTP_202_ACCEPTED)
async def submit_blob(
    file: UploadFile = File(...),
    meeting_title: str = Form("Live Meeting Recording"),
    provider: str = Form("teams"),
    openrouter_api_key: str | None = Form(None),
    model: str | None = Form(None),
) -> dict[str, Any]:
    """Accepts a recording blob streamed from client, saves to temp file, and dispatches to pipeline."""
    temp_dir = Path("runtime/temp_uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / f"blob_{uuid.uuid4().hex[:8]}_{file.filename}"
    try:
        with temp_file.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        metadata = {
            "meeting_title": meeting_title,
            "provider": provider,
            "recorded_via": "browser_blob",
        }
        res = await _dispatcher_instance.dispatch_recording_async(
            video_path=temp_file,
            metadata=metadata,
            openrouter_api_key=openrouter_api_key,
            model=model,
        )
        return {
            "status": "dispatched",
            "pipeline_response": res,
            "job_id": res.get("job_id"),
        }
    finally:
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)
