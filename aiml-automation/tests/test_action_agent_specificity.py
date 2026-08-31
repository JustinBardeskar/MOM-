import pytest
from app.ai_brain.models import ActionItem, ActionOutput
from app.ai_brain.quality import ActionNormalizer, ActionValidator, IMPERATIVE_VERBS, MULTIWORD_IMPERATIVE_VERBS
from app.ai_brain.consensus import CrossAgentConsensusEngine

def test_imperative_verbs_catalog():
    assert "deploy" in IMPERATIVE_VERBS
    assert "configure" in IMPERATIVE_VERBS
    assert "update" in IMPERATIVE_VERBS
    assert "review" in IMPERATIVE_VERBS
    assert "deliver" in IMPERATIVE_VERBS
    assert "investigate" in IMPERATIVE_VERBS
    assert "follow up" in MULTIWORD_IMPERATIVE_VERBS
    assert "set up" in MULTIWORD_IMPERATIVE_VERBS

def test_action_normalizer_strips_conversational_prefixes():
    assert ActionNormalizer.normalize_action_work("Rahul: I will fix the login timeout issue by Friday") == "Fix the login timeout issue"
    assert ActionNormalizer.normalize_action_work("We need to update our analyzer versions for CVE container scanning") == "Update our analyzer versions for CVE container scanning"

def test_action_normalizer_converts_gerunds():
    assert ActionNormalizer.normalize_action_work("upgrading the container base image") == "Upgrade the container base image"
    assert ActionNormalizer.normalize_action_work("reviewing Daniel's PR for compatibility") == "Review Daniel's PR for compatibility"
    assert ActionNormalizer.normalize_action_work("configuring SSO authentication") == "Configure SSO authentication"

def test_action_validator_accepts_valid_actions():
    valid_items = [
        ActionItem(task="Update analyzer versions for CVE container scanning", owner="Daniel", deadline="Thursday EOD"),
        ActionItem(task="Review PR and verify UBI base image compatibility", owner="Priya", deadline="Friday 10 AM"),
        ActionItem(task="Deploy Redis caching cluster to staging environment", owner="Vikram", deadline="Next sprint"),
        ActionItem(task="Prepare Q3 enterprise security audit report", owner="Sarah", deadline="Monday"),
        ActionItem(task="Follow up with client on SSO requirements", owner="Amit", deadline="Tomorrow"),
    ]
    for item in valid_items:
        is_valid, reason = ActionValidator.validate(item)
        assert is_valid is True, f"Failed on valid item '{item.task}': {reason}"

def test_action_validator_rejects_conversational_noise():
    bad_items = [
        ActionItem(task="Keep that in mind", owner="Alex"),
        ActionItem(task="See, B is a read only item", owner="Alex"),
        ActionItem(task="Unless anybody wants to discuss it", owner="Alex"),
        ActionItem(task="Hi, I welcome Ellen to the meeting", owner="Alex"),
        ActionItem(task="Anti-buse team meeting. That is a big mouthful", owner="Alex"),
        ActionItem(task="Apologies for that, but I think it is", owner="Alex"),
        ActionItem(task="News and events, I have got a lot of things", owner="Alex"),
        ActionItem(task="Here when it is midnight? We can talk about it", owner="Alex"),
        ActionItem(task="Manage is even less descriptive than that", owner="Alex"),
        ActionItem(task="I will do something", owner="Alex"),
        ActionItem(task="Fix", owner="Alex"),
        ActionItem(task="What time is the meeting tomorrow?", owner="Alex"),
    ]
    for item in bad_items:
        is_valid, reason = ActionValidator.validate(item)
        assert is_valid is False, f"Expected rejection for '{item.task}', but it passed with: {reason}"

def test_cross_agent_consensus_standardization():
    transcript_lines = [
        "Alex: Let us discuss the security deliverables.",
        "Daniel: I will update our analyzer versions for CVE container scanning before Friday.",
        "Priya: I will review Daniels PR by Friday 10 AM.",
        "Alex: Keep that in mind. Anti-abuse is moving product sections.",
    ]
    raw_actions = [
        ActionItem(task="I will update our analyzer versions for CVE container scanning before Friday", owner="Daniel"),
        ActionItem(task="I will review Daniels PR by Friday 10 AM", owner="Priya"),
        ActionItem(task="Keep that in mind", owner="Alex"),
        ActionItem(task="Anti-abuse is moving product sections", owner="Alex"),
    ]
    standardized = CrossAgentConsensusEngine._ground_and_standardize_actions(
        actions=raw_actions,
        lines=transcript_lines,
        decisions=[],
        risks=[],
    )
    assert len(standardized) == 2
    tasks = [a.task for a in standardized]
    assert any("Update our analyzer versions" in t or "Update analyzer versions" in t for t in tasks)
    assert any("Review Daniels PR" in t or "Review PR" in t for t in tasks)
    assert not any("Keep that in mind" in t for t in tasks)
    assert not any("Anti-abuse" in t for t in tasks)
