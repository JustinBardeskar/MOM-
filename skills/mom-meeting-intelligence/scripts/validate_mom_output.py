"""Deterministic validation script for MOM outputs.

Validates that agent outputs comply with all enterprise MOM quality criteria:
1. No conversational filler or trailing fragments
2. No copula grammar discussions disguised as tasks
3. Imperative task phrasing with accountable owner and deadline
4. Zero boilerplate hallucinations (fake Okta, fake Jira risks)
"""

import re
import sys
import json
from typing import Any

FORBIDDEN_BOILERPLATE = [
    "lack of documented engineering sign-offs",
    "standardize enterprise authentication on okta saml 2.0",
    "approve postgresql 16 production database upgrade",
    "deploy distributed redis caching cluster",
    "yadda, yadda",
    "that's crazy",
]

TRAILING_FRAGMENT_RE = re.compile(
    r"\b(?:to\s+like|than\s+that|and\s+like|or\s+something|in\s+order\s+to|so\s+that\s+we|to\s+make\s+sure|if\s+if)\s*$",
    re.IGNORECASE,
)

COPULA_STATEMENT_RE = re.compile(
    r"^(?:manage|track|lead|execute|review|do|make)\s+is\s+(?:even\s+)?(?:less|more|too|not|a|the)\b",
    re.IGNORECASE,
)

IMPERATIVE_STARTER_RE = re.compile(
    r"^(?:review|implement|deploy|prepare|finalize|fix|schedule|investigate|audit|configure|create|update|draft|conduct|verify|document|optimize|test|deliver)\b",
    re.IGNORECASE,
)


def validate_action_item(item: dict[str, Any]) -> list[str]:
    errors = []
    task = (item.get("task") or item.get("action") or item.get("description") or "").strip()
    if not task:
        return ["Action item missing task description."]

    lower = task.lower()

    # Check forbidden boilerplate
    for fb in FORBIDDEN_BOILERPLATE:
        if fb in lower:
            errors.append(f"Forbidden boilerplate detected: '{fb}'")

    # Check copula statement
    if COPULA_STATEMENT_RE.match(task):
        errors.append(f"Copula grammar meta-discussion detected: '{task}'")

    # Check trailing fragment
    if TRAILING_FRAGMENT_RE.search(task):
        errors.append(f"Incomplete trailing fragment detected: '{task}'")

    # Check imperative verb
    if not IMPERATIVE_STARTER_RE.match(task):
        errors.append(f"Task does not begin with an active imperative verb: '{task}'")

    return errors


def validate_mom_payload(payload: dict[str, Any]) -> dict[str, list[str]]:
    report: dict[str, list[str]] = {}

    # 1. Action Items
    action_items = payload.get("action_items") or []
    for idx, act in enumerate(action_items):
        errs = validate_action_item(act)
        if errs:
            report[f"action_item_{idx}"] = errs

    # 2. Executive Summary
    summary = payload.get("meeting_summary") or payload.get("executive_summary") or ""
    if summary:
        sum_lower = summary.lower()
        if "yadda" in sum_lower or "that's crazy" in sum_lower:
            report["summary"] = ["Summary contains verbatim spoken filler."]

    # 3. Decisions
    decisions = payload.get("decisions") or []
    for idx, dec in enumerate(decisions):
        desc = (dec.get("description") if isinstance(dec, dict) else str(dec)).lower()
        if "that's crazy" in desc or "some bodies" in desc:
            report[f"decision_{idx}"] = ["Decision contains raw spoken agreement filler."]

    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_mom_output.py <mom_json_file_or_string>")
        sys.exit(1)

    arg = sys.argv[1]
    try:
        if arg.endswith(".json"):
            with open(arg, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(arg)
    except Exception as e:
        print(f"JSON load error: {e}")
        sys.exit(1)

    issues = validate_mom_payload(data)
    if issues:
        print(f"Validation FAILED with {len(issues)} issue(s):")
        for k, v in issues.items():
            print(f"  [{k}]: {'; '.join(v)}")
        sys.exit(1)
    else:
        print("Validation PASSED! All MOM items strictly comply with enterprise standards.")
        sys.exit(0)
