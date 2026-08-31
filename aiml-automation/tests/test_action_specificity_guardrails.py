import pytest
from app.ai_brain.models import ActionItem
from app.ai_brain.quality import ActionValidator, ActionSpecificityValidator, ExecutiveActionReframingEngine

def test_concrete_specific_actions_pass():
    valid_tasks = [
        ("deploy a distributed Redis caching cluster", "Sarah", "Friday"),
        ("audit the PostgreSQL connection pooling and optimize slow queries", "David", "Tuesday"),
        ("profile Android 12 memory footprint and submit patch pull request", "Vikram", "Wednesday"),
        ("configure exponential backoff retry queue with dead-letter alerts", "Priya", "Friday"),
        ("coordinate with compliance team to submit vendor security audit report", "Sarah", "Monday"),
    ]
    
    for task, owner, deadline in valid_tasks:
        reframed = ExecutiveActionReframingEngine.reframe_action(
            raw_task=task, owner=owner, deadline=deadline
        )
        item = ActionItem(
            task=reframed["task"],
            action=reframed["action"],
            description=reframed["description"],
            owner=reframed["owner"],
            deadline=reframed["deadline"],
            deadline_text=reframed["deadline_text"],
        )
        is_valid, reason = ActionValidator.validate(item)
        assert is_valid, f"Expected '{item.task}' to be valid, but got rejected: {reason}"


def test_garbage_and_meta_conversation_rejected():
    garbage_tasks = [
        "look at the agenda",
        "glad you could join",
        "on slack only",
        "keep that in mind",
        "see item b",
        "manage is even less",
        "track people down",
        "we discussed the release",
        "thanks for joining everyone",
        "bye everyone",
        "read only item",
        "moving product sections",
        "call myself out",
    ]
    
    for g in garbage_tasks:
        item = ActionItem(
            task=g,
            action=g,
            description=g,
            owner="David",
            deadline="Friday",
        )
        is_valid, reason = ActionValidator.validate(item)
        assert not is_valid, f"Expected garbage task '{g}' to be rejected, but it passed!"


def test_vague_and_pronoun_tasks_rejected():
    vague_tasks = [
        "improve things",
        "work on it",
        "address quality",
        "do the needful",
        "handle it",
        "make it better",
        "investigate it",
        "fix that",
        "review this",
        "update them",
        "address security",
        "take necessary action",
        "follow up",
    ]
    
    for v in vague_tasks:
        item = ActionItem(
            task=v,
            action=v,
            description=v,
            owner="Sarah",
            deadline="Friday",
        )
        is_valid, reason = ActionValidator.validate(item)
        assert not is_valid, f"Expected vague task '{v}' to be rejected, but it passed!"
