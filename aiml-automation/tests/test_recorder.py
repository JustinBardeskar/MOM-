import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.recorder import FFmpegScreenRecorder, PipelineDispatcher


@pytest.fixture
def client() -> TestClient:
    app = create_app(Settings(environment="testing", docs_enabled=True, worker_enabled=False))
    return TestClient(app)


def test_recorder_engine_init(tmp_path: Path) -> None:
    recorder = FFmpegScreenRecorder(output_dir=tmp_path)
    status = recorder.get_status()
    assert not status["is_recording"]
    assert status["status"] == "idle"
    assert status["elapsed_seconds"] == 0.0


def test_recorder_audio_detection() -> None:
    dev = FFmpegScreenRecorder.detect_audio_device()
    # May return a string device or None depending on host audio drivers
    assert dev is None or isinstance(dev, str)


def test_recorder_api_status(client: TestClient) -> None:
    res = client.get("/api/v1/recorder/status")
    assert res.status_code == 200
    data = res.json()
    assert "is_recording" in data
    assert "status" in data


def test_pipeline_dispatcher_error_on_missing_file() -> None:
    dispatcher = PipelineDispatcher()
    fake_path = Path("runtime/non_existent_file.mp4")
    with pytest.raises(FileNotFoundError):
        asyncio.run(dispatcher.dispatch_recording_async(fake_path, {"meeting_title": "Test"}))


def test_submit_blob_endpoint(client: TestClient, tmp_path: Path) -> None:
    # Create a small sample file to simulate browser-submitted recording blob
    sample_file = tmp_path / "sample_blob.mp4"
    sample_file.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")

    with patch.object(PipelineDispatcher, "dispatch_recording_async", new_callable=AsyncMock) as mock_dispatch:
        mock_dispatch.return_value = {
            "job_id": "99999999-9999-9999-9999-999999999999",
            "meeting_id": "rec-1234",
            "status": "queued",
            "result_url": "/api/v1/jobs/99999999-9999-9999-9999-999999999999/result",
        }

        with open(sample_file, "rb") as f:
            response = client.post(
                "/api/v1/recorder/submit-blob",
                files={"file": ("recording.mp4", f, "video/mp4")},
                data={"meeting_title": "Teams Sync Call", "provider": "teams"},
            )

        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "dispatched"
        assert body["job_id"] == "99999999-9999-9999-9999-999999999999"
        mock_dispatch.assert_called_once()
