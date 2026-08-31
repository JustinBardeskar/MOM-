"""
Dataset Ingestion, Normalization, and Validation Engine.
Loads multi-format meeting datasets (JSON, JSONL, or legacy test sets) into verified GoldenMeetingSample objects.
"""

import json
import os
import glob
from typing import List, Dict, Any
from evaluation.schema import GoldenMeetingSample, GoldenActionItem, GoldenDecision, GoldenRisk, GoldenSummary


class DatasetLoader:
    """Loads, validates, and indexes meeting dataset samples for evaluation and fine-tuning."""

    @classmethod
    def load_from_file(cls, filepath: str) -> List[GoldenMeetingSample]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        samples: List[GoldenMeetingSample] = []
        if filepath.endswith(".jsonl"):
            with open(filepath, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        raw = json.loads(line)
                        samples.append(GoldenMeetingSample.model_validate(raw))
                    except Exception as ex:
                        print(f"Warning: Line {line_idx} in {filepath} failed validation: {ex}")
        elif filepath.endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                if isinstance(raw_data, list):
                    for idx, item in enumerate(raw_data):
                        try:
                            # Handle legacy format or unified format
                            sample = cls._normalize_sample_dict(item, f"sample_{idx}")
                            samples.append(GoldenMeetingSample.model_validate(sample))
                        except Exception as ex:
                            print(f"Warning: Item {idx} in {filepath} failed validation: {ex}")
                elif isinstance(raw_data, dict):
                    sample = cls._normalize_sample_dict(raw_data, "sample_0")
                    samples.append(GoldenMeetingSample.model_validate(sample))

        print(f" Successfully loaded and validated {len(samples)} meeting samples from {filepath}")
        return samples

    @classmethod
    def load_all_from_directory(cls, dirpath: str) -> List[GoldenMeetingSample]:
        all_samples: List[GoldenMeetingSample] = []
        files = glob.glob(os.path.join(dirpath, "*.json*"))
        for f in files:
            try:
                samples = cls.load_from_file(f)
                all_samples.extend(samples)
            except Exception as e:
                print(f"Error reading {f}: {e}")
        return all_samples

    @classmethod
    def _normalize_sample_dict(cls, data: Dict[str, Any], fallback_id: str) -> Dict[str, Any]:
        """Normalizes heterogeneous input formats into the unified GoldenMeetingSample schema."""
        sample_id = data.get("id") or fallback_id
        transcript = data.get("transcript") or data.get("text") or data.get("dialogue") or ""
        title = data.get("meeting_title") or data.get("title") or "Executive Meeting Review"
        m_type = data.get("meeting_type") or data.get("type") or "general"

        # Summary
        summary_raw = data.get("expected_summary") or data.get("summary") or data.get("expected", {})
        if isinstance(summary_raw, str):
            expected_summary = {"executive_summary": summary_raw, "key_points": []}
        elif isinstance(summary_raw, dict):
            expected_summary = {
                "executive_summary": summary_raw.get("executive_summary") or summary_raw.get("summary") or "Executive meeting review",
                "key_points": summary_raw.get("key_points") or summary_raw.get("bullet_points") or [],
            }
        else:
            expected_summary = {"executive_summary": "Executive review", "key_points": []}

        # Actions
        actions_raw = data.get("expected_actions") or data.get("action_items") or data.get("actions") or []
        normalized_actions = []
        for a in actions_raw:
            if isinstance(a, str):
                normalized_actions.append({"task": a, "owner": "Unassigned"})
            elif isinstance(a, dict):
                normalized_actions.append({
                    "task": a.get("task") or a.get("action") or a.get("description") or "",
                    "owner": a.get("owner") or "Unassigned",
                    "deadline": a.get("deadline") or a.get("deadline_text"),
                    "priority": a.get("priority") or "Medium",
                    "evidence": a.get("evidence") or a.get("source"),
                })

        # Decisions
        decisions_raw = data.get("expected_decisions") or data.get("decisions") or []
        normalized_decisions = []
        for d in decisions_raw:
            if isinstance(d, str):
                normalized_decisions.append({"decision": d})
            elif isinstance(d, dict):
                normalized_decisions.append({
                    "decision": d.get("decision") or d.get("description") or "",
                    "approved_by": d.get("approved_by") or d.get("approvers") or [],
                    "rationale": d.get("rationale"),
                })

        # Risks
        risks_raw = data.get("expected_risks") or data.get("risks") or []
        normalized_risks = []
        for r in risks_raw:
            if isinstance(r, str):
                normalized_risks.append({"risk": r})
            elif isinstance(r, dict):
                normalized_risks.append({
                    "risk": r.get("risk") or r.get("description") or "",
                    "severity": r.get("severity") or "Medium",
                    "mitigation": r.get("mitigation"),
                })

        return {
            "id": sample_id,
            "meeting_title": title,
            "meeting_type": m_type,
            "transcript": transcript,
            "expected_summary": expected_summary,
            "expected_actions": normalized_actions,
            "expected_decisions": normalized_decisions,
            "expected_risks": normalized_risks,
            "expected_topics": data.get("expected_topics") or data.get("topics") or [],
            "domain_tags": data.get("domain_tags") or [],
            "metadata": data.get("metadata") or {},
        }
