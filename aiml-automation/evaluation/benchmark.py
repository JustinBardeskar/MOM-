"""
Autonomous Multi-Agent Benchmark Runner.
Executes the MOM AI Agent Swarm against gold-standard datasets and outputs quantitative scores.
"""

import asyncio
import os
import sys
import json
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.loader import DatasetLoader
from evaluation.metrics import MeetingMetricsEvaluator
from evaluation.schema import GoldenMeetingSample
from app.ai_brain.agents import AgentOrchestrator, ValidatorAgent
from app.ai_brain.models import (
    M2ToM3Contract,
    MeetingMetadata,
    PreprocessedTranscript,
    TranscriptSegment,
    AgentName,
    ActionOutput,
    SummaryOutput,
    DecisionOutput,
)


class BenchmarkRunner:
    """Executes automated end-to-end evaluation runs."""

    @classmethod
    async def evaluate_dataset(
        cls, dataset_path: str, max_samples: int = 50
    ) -> Dict[str, Any]:
        samples = DatasetLoader.load_from_file(dataset_path)[:max_samples]
        if not samples:
            print(" No valid samples found to benchmark.")
            return {}

        orchestrator = AgentOrchestrator()
        action_metrics_list = []
        summary_metrics_list = []
        decision_metrics_list = []

        print("\n" + "=" * 80)
        print(f" STARTING BENCHMARK: {len(samples)} Meeting Samples from {os.path.basename(dataset_path)}")
        print("=" * 80)

        for idx, sample in enumerate(samples, 1):
            print(f"[{idx}/{len(samples)}] Evaluating: '{sample.meeting_title}' ({sample.id})...")
            
            # Create synthetic contract
            contract = M2ToM3Contract(
                job_id=f"eval_{sample.id}",
                meeting=MeetingMetadata(
                    meeting_id=f"m_{sample.id}",
                    title=sample.meeting_title,
                    participants=[],
                ),
                preprocessing=PreprocessedTranscript(
                    text=sample.transcript,
                    segments=[TranscriptSegment(start=0.0, end=10.0, text=sample.transcript, speaker="Speaker")],
                ),
            )

            # Dummy stage reporter
            async def _dummy_stage(stage, pct): pass

            try:
                analysis = await orchestrator.execute(contract, _dummy_stage)
                outputs = analysis.outputs

                # Extract Predictions
                act_out = ValidatorAgent.get_output(outputs, AgentName.ACTION, ActionOutput)
                sum_out = ValidatorAgent.get_output(outputs, AgentName.SUMMARY, SummaryOutput)
                dec_out = ValidatorAgent.get_output(outputs, AgentName.DECISION, DecisionOutput)

                pred_actions = [a.model_dump() for a in act_out.action_items]
                exp_actions = [a.model_dump() for a in sample.expected_actions]

                # Evaluate Actions
                a_metric = MeetingMetricsEvaluator.evaluate_action_items(pred_actions, exp_actions)
                action_metrics_list.append(a_metric)

                # Evaluate Summary
                s_metric = MeetingMetricsEvaluator.evaluate_summary(
                    sum_out.executive_summary, sample.expected_summary.executive_summary
                )
                summary_metrics_list.append(s_metric)

                # Evaluate Decisions
                pred_decs = [d.model_dump() for d in dec_out.decisions]
                exp_decs = [d.model_dump() for d in sample.expected_decisions]
                d_metric = MeetingMetricsEvaluator.evaluate_decisions(pred_decs, exp_decs)
                decision_metrics_list.append(d_metric)

                print(f"   -> Action F1: {a_metric['f1']*100:.1f}% | Owner Acc: {a_metric['owner_accuracy']*100:.1f}% | Summary ROUGE-L: {s_metric['rouge_l']*100:.1f}%")

            except Exception as e:
                print(f"    Error on sample {sample.id}: {e}")

        # Aggregate Scores
        avg_act_f1 = sum(m["f1"] for m in action_metrics_list) / max(1, len(action_metrics_list))
        avg_owner_acc = sum(m["owner_accuracy"] for m in action_metrics_list) / max(1, len(action_metrics_list))
        avg_dl_acc = sum(m["deadline_accuracy"] for m in action_metrics_list) / max(1, len(action_metrics_list))
        avg_sum_rl = sum(m["rouge_l"] for m in summary_metrics_list) / max(1, len(summary_metrics_list))
        avg_dec_f1 = sum(m["f1"] for m in decision_metrics_list) / max(1, len(decision_metrics_list))

        overall_score = round((avg_act_f1 * 0.4 + avg_owner_acc * 0.15 + avg_sum_rl * 0.25 + avg_dec_f1 * 0.2) * 100, 1)

        print("\n" + "=" * 80)
        print(" BENCHMARK SCORECARD SUMMARY")
        print("=" * 80)
        print(f" Overall AI Swarm Accuracy Score: {overall_score}%")
        print(f" Action Item Extraction F1:     {avg_act_f1*100:.1f}%")
        print(f" Action Owner Assignment Acc:   {avg_owner_acc*100:.1f}%")
        print(f" Action Deadline Precision:     {avg_dl_acc*100:.1f}%")
        print(f" Executive Summary ROUGE-L:     {avg_sum_rl*100:.1f}%")
        print(f" Ratified Decision Recall F1:   {avg_dec_f1*100:.1f}%")
        print("=" * 80)

        return {
            "overall_score": overall_score,
            "action_f1": round(avg_act_f1, 3),
            "owner_accuracy": round(avg_owner_acc, 3),
            "deadline_accuracy": round(avg_dl_acc, 3),
            "summary_rouge_l": round(avg_sum_rl, 3),
            "decision_f1": round(avg_dec_f1, 3),
        }


if __name__ == "__main__":
    test_file = "tests/datasets/summary_golden.json"
    if os.path.exists(test_file):
        asyncio.run(BenchmarkRunner.evaluate_dataset(test_file, max_samples=5))
