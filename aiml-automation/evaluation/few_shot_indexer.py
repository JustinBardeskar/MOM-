"""
Dynamic Few-Shot Indexer and In-Context Example Selector.
Indexes GoldenMeetingSamples into a lightweight vector/lexical store to inject domain-specific few-shot examples into agent prompts.
"""

import os
import json
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional
from evaluation.schema import GoldenMeetingSample


class DynamicFewShotStore:
    """Manages gold-standard in-context examples for dynamic prompt injection."""

    def __init__(self, examples_file: str = "evaluation/data/golden_few_shots.json") -> None:
        self.examples_file = examples_file
        self.samples: List[GoldenMeetingSample] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.examples_file):
            try:
                with open(self.examples_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.samples = [GoldenMeetingSample.model_validate(x) for x in data]
            except Exception as e:
                print(f"Warning loading few-shot store: {e}")

    def save_samples(self, new_samples: List[GoldenMeetingSample]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.examples_file)), exist_ok=True)
        self.samples.extend(new_samples)
        # Deduplicate by ID
        unique = {s.id: s for s in self.samples}
        self.samples = list(unique.values())
        with open(self.examples_file, "w", encoding="utf-8") as f:
            json.dump([s.model_dump() for s in self.samples], f, indent=2)
        print(f" Stored {len(self.samples)} golden few-shot examples in {self.examples_file}")

    def get_best_examples(
        self, transcript: str, agent_type: str = "action", top_k: int = 2
    ) -> List[Dict[str, Any]]:
        """Finds the top-K most semantically similar golden examples for a given meeting transcript."""
        if not self.samples:
            return []

        scored = []
        trans_clean = transcript.lower()[:300]
        for s in self.samples:
            s_clean = s.transcript.lower()[:300]
            sim = SequenceMatcher(None, trans_clean, s_clean).ratio()
            scored.append((sim, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_samples = [item[1] for item in scored[:top_k]]

        examples = []
        for s in top_samples:
            if agent_type == "action" and s.expected_actions:
                examples.append({
                    "input_transcript": s.transcript,
                    "ideal_output": {"action_items": [a.model_dump() for a in s.expected_actions]},
                })
            elif agent_type == "summary":
                examples.append({
                    "input_transcript": s.transcript,
                    "ideal_output": s.expected_summary.model_dump(),
                })
            elif agent_type == "decision" and s.expected_decisions:
                examples.append({
                    "input_transcript": s.transcript,
                    "ideal_output": {"decisions": [d.model_dump() for d in s.expected_decisions]},
                })

        return examples
