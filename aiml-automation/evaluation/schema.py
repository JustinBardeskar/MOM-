"""
Enterprise Meeting Dataset Schema.
Defines the standard data models for training, evaluation, few-shot injection, and fine-tuning.
"""

from typing import Any, List, Optional
from pydantic import BaseModel, Field


class GoldenActionItem(BaseModel):
    task: str = Field(description="Imperative, SMART task statement (e.g. 'Deploy Redis cluster')")
    owner: str = Field(default="Unassigned", description="Assigned individual or team")
    deadline: Optional[str] = Field(default=None, description="Formal deadline (e.g. 'Friday', '2026-09-15')")
    priority: str = Field(default="Medium", description="Priority level: High, Medium, Low")
    evidence: Optional[str] = Field(default=None, description="Exact spoken sentence in transcript")


class GoldenDecision(BaseModel):
    decision: str = Field(description="Ratified agreement or architecture choice")
    approved_by: List[str] = Field(default_factory=list, description="Approvers or stakeholders")
    rationale: Optional[str] = Field(default=None, description="Business rationale")


class GoldenRisk(BaseModel):
    risk: str = Field(description="Identified risk or bottleneck")
    severity: str = Field(default="Medium", description="Severity: High, Medium, Low")
    mitigation: Optional[str] = Field(default=None, description="Agreed mitigation strategy")


class GoldenSummary(BaseModel):
    executive_summary: str = Field(description="Polished 2-3 paragraph executive brief")
    key_points: List[str] = Field(default_factory=list, description="2-4 concise milestone takeaways")


class GoldenMeetingSample(BaseModel):
    id: str = Field(description="Unique sample identifier (e.g. 'sample_eng_001')")
    meeting_title: str = Field(description="Dynamic, descriptive meeting title")
    meeting_type: str = Field(default="general", description="Classification: technical, scrum, product, etc.")
    transcript: str = Field(description="Full spoken meeting transcript or diarized dialogue turns")
    expected_summary: GoldenSummary
    expected_actions: List[GoldenActionItem] = Field(default_factory=list)
    expected_decisions: List[GoldenDecision] = Field(default_factory=list)
    expected_risks: List[GoldenRisk] = Field(default_factory=list)
    expected_topics: List[str] = Field(default_factory=list)
    domain_tags: List[str] = Field(default_factory=list, description="Tags like ['backend', 'infra', 'fintech']")
    metadata: dict[str, Any] = Field(default_factory=dict)
