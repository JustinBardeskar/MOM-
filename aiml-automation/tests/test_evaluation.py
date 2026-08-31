import os
import json
import pytest
from app.ai_brain.quality import ActionNormalizer, ActionValidator
from app.ai_brain.models import ActionItem


def test_positive_action_normalization():
    """Validates that conversational commitments are transformed into imperative work."""
    with open("tests/datasets/positive_cases.json", "r", encoding="utf-8") as f:
        cases = json.load(f)

    for case in cases:
        raw_input = case["input"]
        expected = case["expected"]
        
        # Test work extraction & normalization
        normalized_work = ActionNormalizer.normalize_action_work(raw_input)
        assert normalized_work.startswith(tuple(ActionNormalizer.normalize_action_work(expected["action"]).split()[:1])), \
            f"Expected imperative verb start for '{raw_input}', got '{normalized_work}'"

        # Test validation passes
        item = ActionItem(
            action=normalized_work,
            description=normalized_work,
            owner=expected["owner"],
            deadline=expected["deadline"],
            status=expected["status"],
            evidence=raw_input,
        )
        is_valid, reason = ActionValidator.validate(item)
        assert is_valid is True, f"Valid action was rejected: {reason}"


def test_negative_cases_rejected():
    """Validates that non-action discussions and casual chatter are rejected."""
    with open("tests/datasets/negative_cases.json", "r", encoding="utf-8") as f:
        cases = json.load(f)

    for case in cases:
        raw_input = case["input"]
        
        # Check non-action filter
        is_non_action = ActionNormalizer.is_non_action_discussion(raw_input)
        assert is_non_action is True, f"Negative case '{raw_input}' was not detected as non-action discussion."

        # Check validator rejects if attempted
        item = ActionItem(
            description=raw_input,
            action=raw_input,
            owner="Unassigned",
        )
        is_valid, _ = ActionValidator.validate(item)
        assert is_valid is False, f"Negative case '{raw_input}' passed validation when it should fail."


def test_ambiguous_cases_rejected():
    """Validates that vague aspirational ideas without commitment are rejected."""
    with open("tests/datasets/ambiguous_cases.json", "r", encoding="utf-8") as f:
        cases = json.load(f)

    for case in cases:
        raw_input = case["input"]
        is_non_action = ActionNormalizer.is_non_action_discussion(raw_input)
        assert is_non_action is True, f"Ambiguous case '{raw_input}' was not flagged as non-action."


def test_action_item_contract_separation():
    """Validates that ActionItem separates pure work from final phrase and metadata."""
    item = ActionItem(
        action="Send the updated proposal",
        owner="Rahul",
        recipient="Sarah",
        deadline="2026-08-26",
        status="assigned",
        evidence="Rahul: I'll send the updated proposal to Sarah by tomorrow.",
        confidence=0.96,
    )
    assert item.action == "Send the updated proposal"
    assert item.owner == "Rahul"
    assert item.recipient == "Sarah"
    assert item.deadline == "2026-08-26"
    assert item.status == "assigned"
    assert item.evidence.startswith("Rahul:")

    # Generate final phrase
    final_phrase = ActionNormalizer.generate_final_phrase(
        item.action, item.owner, item.recipient, item.deadline
    )
    assert "Send the updated proposal to Sarah by 2026-08-26" in final_phrase


"""
Pytest regression test suite enforcing quality loop standards.
"""

import pytest
from app.ai_brain.models import AgentName
from app.ai_brain.quality import SelfCritiquePass, GoldenExampleStore
from tests.runner import RegressionSuiteRunner


def test_golden_store_seeding():
    store = GoldenExampleStore()
    count = store.seed_initial_golden_examples()
    assert count >= 0
    examples = store.get_relevant_golden_examples("action", limit=5)
    assert isinstance(examples, list)


def test_self_critique_catches_placeholders():
    bad_output = {
        "action_items": [
            {
                "description": "Execute: and I am going to be assuming that people leaders will attend",
                "owner": "Assigned Lead",
                "deadline_text": "End of Sprint",
                "priority": "Low",
                "success_criteria": "Deliverable validated and operational by End of Sprint"
            }
        ]
    }
    critique = SelfCritiquePass.evaluate(AgentName.ACTION, "Sample transcript", bad_output)
    assert not critique.passed
    assert any("placeholder" in v.lower() or "assigned lead" in v.lower() or "end of sprint" in v.lower() for v in critique.violations)


def test_self_critique_passes_valid_golden_item():
    good_output = {
        "action_items": [
            {
                "description": "Finalize the Q3 marketing budget and circulate to finance for sign-off",
                "owner": "Marketing Lead",
                "deadline_text": "By Friday, Aug 28",
                "priority": "High",
                "success_criteria": "Budget document approved and signed off by finance"
            }
        ]
    }
    critique = SelfCritiquePass.evaluate(AgentName.ACTION, "Alex: Let's finalize the Q3 marketing budget by Friday.", good_output)
    assert critique.passed
    assert len(critique.violations) == 0


def test_regression_dataset_loads_all_55_cases():
    runner = RegressionSuiteRunner()
    dataset = runner.load_dataset("action_items_golden.json")
    assert len(dataset) >= 55
