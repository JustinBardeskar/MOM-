import pytest
from app.ai_brain.models import ActionItem, SummaryOutput, MeetingUnderstandingOutput, MeetingType
from app.ai_brain.consensus import CrossAgentConsensusEngine

def test_dynamic_meeting_title_from_technical_discussion():
    transcript = """
    Alex: Our PostgreSQL database is experiencing critical write contention under peak load.
    Sarah: I will deploy a distributed Redis caching cluster by Friday to resolve the database bottleneck.
    David: Agreed. I will optimize the slow queries by Tuesday.
    """
    
    title = CrossAgentConsensusEngine.generate_dynamic_meeting_title(
        transcript_text=transcript,
        current_title="Direct Transcript Sync",
    )
    
    assert "Database" in title or "Redis" in title or "PostgreSQL" in title
    assert title != "Direct Transcript Sync"
    assert title != "Minutes of Meeting"


def test_dynamic_meeting_title_from_product_onboarding():
    transcript = """
    Elena: We need to scale user onboarding by 40% in Q3.
    Product: I will finalize the onboarding Figma wireframes by next Friday.
    """
    
    title = CrossAgentConsensusEngine.generate_dynamic_meeting_title(
        transcript_text=transcript,
        current_title="Custom Transcript",
    )
    
    assert "Onboarding" in title or "Wireframe" in title
    assert title != "Custom Transcript"


def test_dynamic_meeting_title_respects_custom_specific_title():
    transcript = "Alex: Let's review the sprint deliverables."
    custom_title = "Global Infrastructure Security Audit 2026"
    
    title = CrossAgentConsensusEngine.generate_dynamic_meeting_title(
        transcript_text=transcript,
        current_title=custom_title,
    )
    
    assert title == custom_title


def test_suggested_title_from_llm_is_prioritized():
    transcript = "Alex: Status update on payment retries."
    suggested = "Payment Webhook Retries & Dead-Letter Alerts"
    
    title = CrossAgentConsensusEngine.generate_dynamic_meeting_title(
        transcript_text=transcript,
        current_title="Direct Transcript Sync",
        suggested_title=suggested,
    )
    
    assert title == suggested
