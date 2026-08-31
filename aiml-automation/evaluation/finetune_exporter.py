"""
Fine-Tuning Dataset Exporter.
Converts verified GoldenMeetingSample datasets into production training formats:
- OpenAI / Groq JSONL SFT Format
- HuggingFace / Unsloth Instruction Format (Alpaca / ChatML / Qwen)
- Specialist Agent Task-Specific Fine-Tuning Sets
"""

import json
import os
from typing import List
from evaluation.schema import GoldenMeetingSample


class FineTuneDatasetExporter:
    """Exports structured meeting datasets to fine-tuning formats."""

    @classmethod
    def export_openai_chat_format(
        cls, samples: List[GoldenMeetingSample], output_path: str, agent_target: str = "all"
    ) -> str:
        """
        Exports to OpenAI/Groq JSONL format:
        {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for s in samples:
                if agent_target in ["all", "action"]:
                    sys_prompt = "You are an enterprise Action Extraction Agent. Extract concrete, imperative, SMART post-meeting action items in strict JSON."
                    user_content = f"Transcript:\n{s.transcript}"
                    assistant_content = json.dumps({"action_items": [a.model_dump() for a in s.expected_actions]}, indent=2)
                    entry = {"messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}]}
                    f.write(json.dumps(entry) + "\n")
                    count += 1

                if agent_target in ["all", "summary"]:
                    sys_prompt = "You are an executive Summary Synthesis Agent. Output a polished executive brief and concise bullet points in strict JSON."
                    user_content = f"Transcript:\n{s.transcript}"
                    assistant_content = json.dumps(s.expected_summary.model_dump(), indent=2)
                    entry = {"messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}]}
                    f.write(json.dumps(entry) + "\n")
                    count += 1

                if agent_target in ["all", "decision"]:
                    sys_prompt = "You are a Decision Extraction Agent. Extract ratified business decisions and approved choices in strict JSON."
                    user_content = f"Transcript:\n{s.transcript}"
                    assistant_content = json.dumps({"decisions": [d.model_dump() for d in s.expected_decisions]}, indent=2)
                    entry = {"messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}]}
                    f.write(json.dumps(entry) + "\n")
                    count += 1

        print(f" Exported {count} fine-tuning training records to {output_path}")
        return output_path

    @classmethod
    def export_qwen_chatml_format(
        cls, samples: List[GoldenMeetingSample], output_path: str
    ) -> str:
        """
        Exports to Qwen / ChatML format for Unsloth / HuggingFace fine-tuning.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for s in samples:
                payload = {
                    "meeting_title": s.meeting_title,
                    "summary": s.expected_summary.executive_summary,
                    "key_points": s.expected_summary.key_points,
                    "action_items": [a.model_dump() for a in s.expected_actions],
                    "decisions": [d.model_dump() for d in s.expected_decisions],
                    "risks": [r.model_dump() for r in s.expected_risks],
                }
                record = {
                    "instruction": "You are an enterprise AI Minutes of Meeting intelligence brain. Analyze the transcript and extract all structured parameters in JSON.",
                    "input": s.transcript,
                    "output": json.dumps(payload, indent=2),
                }
                f.write(json.dumps(record) + "\n")
                count += 1

        print(f" Exported {count} Qwen/Unsloth format samples to {output_path}")
        return output_path
