from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app(
        Settings(environment="testing", docs_enabled=True, worker_enabled=False)
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def base_payload() -> dict[str, object]:
    return {
        "event_id": "evt-teams-001",
        "meeting_id": "meeting-001",
        "provider": "microsoft_teams",
        "title": "Implementation review",
        "ended_at": "2026-08-06T14:30:00Z",
        "recording": {
            "url": "https://storage.example.com/recordings/meeting-001.mp4?signature=test",
            "content_type": "video/mp4",
            "file_name": "meeting-001.mp4",
        },
    }
