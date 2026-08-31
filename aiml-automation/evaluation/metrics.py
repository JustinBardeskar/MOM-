"""
Quantitative Evaluation Metrics for Minutes of Meeting AI Swarm.
Computes Precision, Recall, F1, Attribute Exact-Match, and ROUGE-L semantic overlap.
"""

import re
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple


class MeetingMetricsEvaluator:
    """Evaluates agent predictions against golden ground-truth samples."""

    @staticmethod
    def compute_string_similarity(str1: str, str2: str) -> float:
        """Computes normalized character/token similarity ratio."""
        s1 = re.sub(r"[^\w\s]", "", str1.lower()).strip()
        s2 = re.sub(r"[^\w\s]", "", str2.lower()).strip()
        return SequenceMatcher(None, s1, s2).ratio()

    @classmethod
    def evaluate_action_items(
        cls,
        predicted: List[Dict[str, Any]],
        expected: List[Dict[str, Any]],
        match_threshold: float = 0.65,
    ) -> Dict[str, float]:
        """
        Computes Action Precision, Recall, F1, and Attribute Accuracies (Owner, Deadline).
        """
        if not expected and not predicted:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "owner_acc": 1.0, "deadline_acc": 1.0}
        if not predicted or not expected:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "owner_acc": 0.0, "deadline_acc": 0.0}

        tp = 0
        owner_matches = 0
        deadline_matches = 0
        matched_expected = set()

        for pred in predicted:
            pred_task = pred.get("task") or pred.get("action") or pred.get("description") or ""
            pred_owner = (pred.get("owner") or "").strip().lower()
            pred_dl = (pred.get("deadline") or pred.get("deadline_text") or "").strip().lower()

            best_sim = 0.0
            best_exp_idx = -1

            for exp_idx, exp in enumerate(expected):
                if exp_idx in matched_expected:
                    continue
                exp_task = exp.get("task") or exp.get("action") or ""
                sim = cls.compute_string_similarity(pred_task, exp_task)
                if sim > best_sim:
                    best_sim = sim
                    best_exp_idx = exp_idx

            if best_sim >= match_threshold and best_exp_idx >= 0:
                tp += 1
                matched_expected.add(best_exp_idx)
                exp = expected[best_exp_idx]
                exp_owner = (exp.get("owner") or "").strip().lower()
                exp_dl = (exp.get("deadline") or "").strip().lower()

                # Owner match check
                if pred_owner and (pred_owner in exp_owner or exp_owner in pred_owner):
                    owner_matches += 1

                # Deadline match check
                if exp_dl and pred_dl and (pred_dl in exp_dl or exp_dl in pred_dl):
                    deadline_matches += 1

        precision = tp / len(predicted) if predicted else 0.0
        recall = tp / len(expected) if expected else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        owner_acc = (owner_matches / tp) if tp > 0 else 0.0
        deadline_acc = (deadline_matches / tp) if tp > 0 else 0.0

        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "owner_accuracy": round(owner_acc, 3),
            "deadline_accuracy": round(deadline_acc, 3),
        }

    @classmethod
    def evaluate_summary(cls, predicted_summary: str, expected_summary: str) -> Dict[str, float]:
        """Computes unigram, bigram, and Longest Common Subsequence (ROUGE-L approximation)."""
        if not predicted_summary or not expected_summary:
            return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}

        pred_tokens = re.findall(r"\w+", predicted_summary.lower())
        exp_tokens = re.findall(r"\w+", expected_summary.lower())

        if not pred_tokens or not exp_tokens:
            return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}

        # ROUGE-1
        common_unigrams = set(pred_tokens) & set(exp_tokens)
        r1_recall = len(common_unigrams) / len(set(exp_tokens))
        r1_prec = len(common_unigrams) / len(set(pred_tokens))
        r1_f1 = (2 * r1_prec * r1_recall) / (r1_prec + r1_recall) if (r1_prec + r1_recall) > 0 else 0.0

        # ROUGE-L (LCS ratio)
        matcher = SequenceMatcher(None, pred_tokens, exp_tokens)
        match_len = sum(triple.size for triple in matcher.get_matching_blocks())
        rl_f1 = (2.0 * match_len) / (len(pred_tokens) + len(exp_tokens))

        return {
            "rouge_1": round(r1_f1, 3),
            "rouge_l": round(rl_f1, 3),
        }

    @classmethod
    def evaluate_decisions(
        cls, predicted: List[Dict[str, Any]], expected: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Computes Decision Precision and Recall."""
        if not expected and not predicted:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        if not predicted or not expected:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        tp = 0
        for pred in predicted:
            pred_desc = pred.get("description") or pred.get("decision") or ""
            for exp in expected:
                exp_desc = exp.get("decision") or exp.get("description") or ""
                if cls.compute_string_similarity(pred_desc, exp_desc) >= 0.60:
                    tp += 1
                    break

        prec = tp / len(predicted) if predicted else 0.0
        rec = tp / len(expected) if expected else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}
