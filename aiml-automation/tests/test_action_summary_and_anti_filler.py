import pytest
from app.ai_brain.models import ActionItem
from app.ai_brain.quality import ActionValidator, ActionNormalizer, ExecutiveActionReframingEngine

def test_conversational_filler_rejected():
    fillers = [
        'Share one thing',
        'Show one thing',
        'Share my screen',
        'Show my screen',
        'Mention one thing',
        'Take a look',
        'Show something',
        'Give an update on things',
    ]
    for filler in fillers:
        item = ActionItem(
            task=filler,
            action=filler,
            description=filler,
            owner='Alex',
            deadline='Friday',
        )
        is_valid, reason = ActionValidator.validate(item)
        assert is_valid is False, f'Filler \"{filler}\" should have been rejected, but passed: {reason}'


def test_real_engineering_actions_accepted():
    actions = [
        ('Deploy Redis caching cluster to offload session traffic', 'Sarah', 'Friday'),
        ('Audit database connection pooling and optimize slow queries', 'David', 'Tuesday'),
        ('Submit vendor security compliance audit report', 'Elena', 'Monday'),
    ]
    for task, owner, dl in actions:
        item = ActionItem(
            task=task,
            action=task,
            description=f'{task} by {dl}',
            owner=owner,
            deadline=dl,
        )
        is_valid, reason = ActionValidator.validate(item)
        assert is_valid is True, f'Valid task \"{task}\" was rejected: {reason}'


def test_synthesize_action_summary_single():
    items = [
        ActionItem(task='Deploy Redis caching cluster', owner='Sarah', deadline='Friday')
    ]
    summary = ExecutiveActionReframingEngine.synthesize_action_summary(items)
    assert 'Sarah to Deploy Redis caching cluster by Friday' in summary


def test_synthesize_action_summary_multiple():
    items = [
        ActionItem(task='Deploy Redis caching cluster', owner='Sarah', deadline='Friday'),
        ActionItem(task='Audit connection pooling', owner='David', deadline='Tuesday'),
    ]
    summary = ExecutiveActionReframingEngine.synthesize_action_summary(items)
    assert 'Sarah: Deploy Redis caching cluster (by Friday)' in summary
    assert 'David: Audit connection pooling (by Tuesday)' in summary


def test_synthesize_action_summary_empty():
    summary = ExecutiveActionReframingEngine.synthesize_action_summary([])
    assert 'No pending post-meeting deliverables' in summary
