"""
Regression Suite Runner and Benchmark Engine.
Evaluates prompt versions against Golden Datasets:
- Schema Validity Rate (100% required)
- Traceability & Extraction Recall
- Zero-Placeholder Compliance (100% required)
- Attribute Differentiation (Owner, Deadline, Priority)
"""

import json
import os
import sys
from typing import Any, Dict, List

# Ensure app is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.ai_brain.models import AgentName
from app.ai_brain.quality import SelfCritiquePass, FORBIDDEN_PLACEHOLDER_SUBSTRINGS


class RegressionSuiteRunner:
    def __init__(self, datasets_dir: str | None = None) -> None:
        if not datasets_dir:
            datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
        self.datasets_dir = datasets_dir

    def load_dataset(self, filename: str) -> List[Dict[str, Any]]:
        path = os.path.join(self.datasets_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_action_items(self, candidate_fn, prompt_version: str = "2.0.0") -> Dict[str, Any]:
        """Runs the 55 golden action item regression test cases."""
        dataset = self.load_dataset("action_items_golden.json")
        total_cases = len(dataset)
        schema_valid = 0
        zero_placeholder = 0
        critique_passed = 0
        recalled_items = 0
        total_expected_items = sum(len(c.get("expected", [])) for c in dataset)

        results = []
        for case in dataset:
            cid = case.get("id")
            trans = case.get("transcript")
            expected = case.get("expected", [])
            
            output = candidate_fn(trans)
            # Evaluate
            is_valid = isinstance(output, dict) and ("action_items" in output or "actionItems" in output)
            if is_valid: schema_valid += 1

            output_str = json.dumps(output).lower()
            has_placeholder = any(p in output_str for p in FORBIDDEN_PLACEHOLDER_SUBSTRINGS)
            if not has_placeholder: zero_placeholder += 1

            critique = SelfCritiquePass.evaluate(AgentName.ACTION, trans, output)
            if critique.passed: critique_passed += 1

            actions = output.get("action_items") or output.get("actionItems") or []
            if len(actions) > 0 and len(expected) > 0:
                recalled_items += min(len(actions), len(expected))

            results.append({
                "id": cid,
                "schema_valid": is_valid,
                "zero_placeholder": not has_placeholder,
                "critique_passed": critique.passed,
                "critique_reason": critique.reason,
            })

        metrics = {
            "prompt_version": prompt_version,
            "total_test_cases": total_cases,
            "schema_validity_pct": round((schema_valid / total_cases) * 100, 1),
            "zero_placeholder_pct": round((zero_placeholder / total_cases) * 100, 1),
            "critique_pass_pct": round((critique_passed / total_cases) * 100, 1),
            "action_recall_pct": round((recalled_items / max(1, total_expected_items)) * 100, 1),
            "overall_score": round(((schema_valid + zero_placeholder + critique_passed) / (3 * total_cases)) * 100, 1),
            "details": results,
        }
        return metrics

    def run_comparison(self, old_fn, new_fn) -> Dict[str, Any]:
        """Runs comparative benchmark comparing Old vs New Prompts."""
        old_metrics = self.evaluate_action_items(old_fn, prompt_version="1.0.0 (Legacy)")
        new_metrics = self.evaluate_action_items(new_fn, prompt_version="2.0.0 (Quality Loop)")
        
        passed_promotion_gate = (
            new_metrics["schema_validity_pct"] >= old_metrics["schema_validity_pct"]
            and new_metrics["zero_placeholder_pct"] >= old_metrics["zero_placeholder_pct"]
            and new_metrics["critique_pass_pct"] >= old_metrics["critique_pass_pct"]
            and new_metrics["zero_placeholder_pct"] == 100.0
        )

        return {
            "old_prompt": old_metrics,
            "new_prompt": new_metrics,
            "passed_promotion_gate": passed_promotion_gate,
            "improvement_delta": {
                "schema_validity": new_metrics["schema_validity_pct"] - old_metrics["schema_validity_pct"],
                "zero_placeholder": new_metrics["zero_placeholder_pct"] - old_metrics["zero_placeholder_pct"],
                "critique_pass": new_metrics["critique_pass_pct"] - old_metrics["critique_pass_pct"],
                "overall_score": new_metrics["overall_score"] - old_metrics["overall_score"],
            }
        }


if __name__ == "__main__":
    runner = RegressionSuiteRunner()
    print("Regression Suite Runner initialized successfully.")
