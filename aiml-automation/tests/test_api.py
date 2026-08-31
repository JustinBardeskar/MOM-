import pytest
from fastapi.testclient import TestClient


def test_health_endpoints(client: TestClient) -> None:
    assert client.get("/live").json() == {"status": "live"}
    assert client.get("/ready").status_code == 200
    assert client.get("/health").json()["status"] == "healthy"


def test_recording_creates_recording_pipeline(
    client: TestClient,
    base_payload: dict[str, object],
) -> None:
    response = client.post("/api/v1/meetings/ready", json=base_payload)

    assert response.status_code == 202
    body = response.json()
    assert body["selected_path"] == "recording_to_transcript"
    status_response = client.get(body["status_url"])
    stages = [step["stage"] for step in status_response.json()["planned_steps"]]
    assert stages[:3] == ["download_recording", "extract_audio", "speech_to_text"]


def test_transcript_skips_media_processing(
    client: TestClient,
    base_payload: dict[str, object],
) -> None:
    base_payload["transcript"] = {
        "url": "https://storage.example.com/transcripts/meeting-001.vtt?signature=test",
        "content_type": "text/vtt",
    }
    response = client.post("/api/v1/meetings/ready", json=base_payload)

    assert response.status_code == 202
    body = response.json()
    assert body["selected_path"] == "direct_transcript"
    status_body = client.get(body["status_url"]).json()
    stages = [step["stage"] for step in status_body["planned_steps"]]
    assert "extract_audio" not in stages
    assert "speech_to_text" not in stages
    assert stages[0] == "normalize_transcript"


def test_pipeline_includes_m4_stages(client: TestClient, base_payload: dict[str, object]) -> None:
    base_payload["transcript"] = {
        "url": "https://storage.example.com/transcripts/meeting-001.vtt?signature=test",
        "content_type": "text/vtt",
    }
    response = client.post("/api/v1/meetings/ready", json=base_payload)
    assert response.status_code == 202

    status_body = client.get(response.json()["status_url"]).json()
    stages = [step["stage"] for step in status_body["planned_steps"]]

    assert "m3_to_m4_handoff" in stages
    assert "m4_analysis" in stages
    assert "validate_m4_output" in stages
    assert "m4_to_m5_handoff" in stages
    assert "m5_validation" in stages
    assert "m5_to_m6_handoff" in stages
    assert "m6_finalization" in stages
    assert stages.index("score_confidence") < stages.index("m3_to_m4_handoff")
    assert stages.index("m3_to_m4_handoff") < stages.index("m4_analysis")
    assert stages.index("m4_analysis") < stages.index("validate_m4_output")
    assert stages.index("validate_m4_output") < stages.index("m4_to_m5_handoff")
    assert stages.index("m4_to_m5_handoff") < stages.index("m5_validation")
    assert stages.index("m5_validation") < stages.index("m5_to_m6_handoff")
    assert stages.index("m5_to_m6_handoff") < stages.index("m6_finalization")
    assert stages.index("m6_finalization") < stages.index("final_structured_json_ready")


def test_idempotency_prevents_duplicate_jobs(
    client: TestClient,
    base_payload: dict[str, object],
) -> None:
    headers = {"Idempotency-Key": "stable-event-key"}
    first = client.post("/api/v1/meetings/ready", json=base_payload, headers=headers)
    second = client.post("/api/v1/meetings/ready", json=base_payload, headers=headers)

    assert first.json()["job_id"] == second.json()["job_id"]
    assert first.json()["duplicate"] is False
    assert second.json()["duplicate"] is True


def test_rejects_event_without_recording_or_transcript(
    client: TestClient,
    base_payload: dict[str, object],
) -> None:
    base_payload.pop("recording")
    response = client.post("/api/v1/meetings/ready", json=base_payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_result_is_not_available_before_worker_finishes(
    client: TestClient,
    base_payload: dict[str, object],
) -> None:
    accepted = client.post("/api/v1/meetings/ready", json=base_payload).json()
    response = client.get(accepted["result_url"])

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "result_not_ready"


def test_media_upload_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from pathlib import Path
    from app.domain import TranscriptSegment, UnifiedTranscript, ProcessingPath
    from app.processing import FfmpegMediaProcessor, FasterWhisperSpeechToText
    from app.ai_brain.providers import HttpLLMProvider
    from app.ai_brain.models import LLMResponse, LLMUsage

    async def fake_extract_audio(self: FfmpegMediaProcessor, recording: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"RIFF_FAKE_AUDIO_DATA")
        return destination

    async def fake_transcribe(self: FasterWhisperSpeechToText, audio: Path, language_hint: str | None) -> UnifiedTranscript:
        return UnifiedTranscript(
            text="Sarah: Let us finalize the architecture. David: Agreed.",
            language="en",
            duration_seconds=12.5,
            segments=[
                TranscriptSegment(start_seconds=0, end_seconds=6, text="Sarah: Let us finalize the architecture."),
                TranscriptSegment(start_seconds=6, end_seconds=12.5, text="David: Agreed."),
            ],
            source_path=ProcessingPath.RECORDING_TO_TRANSCRIPT,
        )

    async def fake_llm_complete(self: HttpLLMProvider, request: object) -> LLMResponse:
        user = getattr(request, "user_prompt", "")
        system = getattr(request, "system_prompt", "")
        prompt_full = f"{system}\n{user}"
        if "MeetingUnderstandingOutput" in prompt_full or "meeting_type" in prompt_full:
            payload = {"meeting_type": "client", "confidence": 0.95, "rationale": "Client sync"}
        elif "SummaryOutput" in prompt_full or "executive_summary" in prompt_full:
            payload = {"executive_summary": "Architecture reviewed.", "key_points": ["Arch finalized"], "confidence": 0.9}
        elif "ActionOutput" in prompt_full or "actions" in prompt_full:
            payload = {"actions": [{"task": "Finalize architecture blueprint", "owner": "David", "deadline": "Friday", "priority": "High", "confidence": 0.9}], "confidence": 0.9}
        elif "DecisionOutput" in prompt_full or "decisions" in prompt_full:
            payload = {"decisions": [{"description": "Agreed architecture", "approved_by": ["Sarah"], "confidence": 0.9}], "confidence": 0.9}
        elif "RiskOutput" in prompt_full or "risks" in prompt_full:
            payload = {"risks": [{"description": "Timeline risk", "severity": "low", "confidence": 0.8}], "blockers": [], "confidence": 0.8}
        elif "RequirementOutput" in prompt_full or "requirements" in prompt_full:
            payload = {"requirements": [{"description": "Fast video encoding", "category": "performance", "priority": "high", "confidence": 0.9}], "confidence": 0.9}
        elif "TopicOutput" in prompt_full or "topics" in prompt_full:
            payload = {"topics": [{"name": "Architecture", "summary": "Sync", "confidence": 0.9}], "confidence": 0.9}
        elif "DeadlineOutput" in prompt_full or "deadlines" in prompt_full:
            payload = {"deadlines": [{"source_text": "Thursday", "normalized_date": "2026-08-20", "confidence": 0.9}], "confidence": 0.9}
        elif "QuestionOutput" in prompt_full or "open_questions" in prompt_full:
            payload = {"open_questions": [{"question": "Who deploys?", "confidence": 0.8}], "confidence": 0.8}
        elif "FollowUpOutput" in prompt_full or "follow_up_tasks" in prompt_full:
            payload = {"follow_up_tasks": [{"description": "Deploy", "confidence": 0.8}], "next_meeting_agenda": ["Review"], "confidence": 0.8}
        elif "SentimentOutput" in prompt_full or "client_mood" in prompt_full:
            payload = {"overall": "positive", "client_mood": "happy", "team_mood": "collaborative", "confidence": 0.9}
        else:
            payload = {"result": True}

        return LLMResponse(
            text=json.dumps(payload),
            usage=LLMUsage(input_tokens=150, output_tokens=50),
            provider_request_id="test-req-123",
        )

    monkeypatch.setattr(FfmpegMediaProcessor, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(FasterWhisperSpeechToText, "transcribe", fake_transcribe)
    monkeypatch.setattr(HttpLLMProvider, "complete", fake_llm_complete)

    response = client.post(
        "/api/v1/meetings/upload",
        files={"file": ("meeting_review.mp4", b"FAKE_VIDEO_CONTENT", "video/mp4")},
        data={"meeting_title": "Architecture Sync", "provider": "teams", "openrouter_api_key": "mock-test-key"},
    )

    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] in ("processing", "awaiting_delivery", "completed")
    assert "/api/v1/jobs/" in data["result_url"]

    # Verify result has source_media metadata
    result_res = client.get(data["result_url"])
    assert result_res.status_code == 200
    res_data = result_res.json()["result"]
    assert "source_media" in res_data
    assert res_data["source_media"]["filename"] == "meeting_review.mp4"
    assert res_data["source_media"]["duration_seconds"] == 12.5
    assert len(res_data["source_media"]["segments"]) == 2
    assert res_data["meeting_summary"] == "Architecture reviewed."


