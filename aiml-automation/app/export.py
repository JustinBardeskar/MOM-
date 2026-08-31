"""
app.export
==========
Enterprise Document Export Module and Third-Party Integrations.

This module provides:
- generate_corporate_pdf: Generates publication-grade, boardroom-ready PDF documents with tables, badges, and sign-off blocks using ReportLab.
- IntegrationHub: Generates integration payloads for enterprise tools:
  - Frappe / ERPNext Doctype payload
  - Jira / Linear sprint task issues
  - Slack Block Kit notifications
"""

from datetime import datetime
import html
import io
import re
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ==========================================
# Enterprise Third-Party Integrations Hub
# ==========================================

class IntegrationHub:
    """Formats structured MOM deliverables for enterprise project management tools."""

    @classmethod
    def to_frappe_payload(cls, mom_data: dict[str, Any]) -> dict[str, Any]:
        """Creates a validated Frappe / ERPNext Doctype payload."""
        meeting_title = mom_data.get("meeting_title", "Executive Meeting")
        summary_obj = mom_data.get("summary", {})
        exec_sum = summary_obj.get("executive_summary", "") if isinstance(summary_obj, dict) else str(summary_obj or "")

        tasks = []
        for act in mom_data.get("action_items", []):
            tasks.append({
                "doctype": "Task",
                "subject": act.get("description", ""),
                "allocated_to": act.get("owner", ""),
                "priority": act.get("priority", "Medium"),
                "expected_end_date": act.get("deadline_text", ""),
                "description": f"Success Criteria: {act.get('success_criteria', 'None')}\nEvidence: {act.get('evidence_quote', 'None')}",
                "status": "Open",
            })

        return {
            "doctype": "Meeting Minutes",
            "title": meeting_title,
            "meeting_type": mom_data.get("meeting_type", "General"),
            "minutes": exec_sum,
            "tasks": tasks,
            "decisions": [
                {"decision": d.get("description", ""), "approved_by": ", ".join(d.get("approved_by", []))}
                for d in mom_data.get("decisions", [])
            ],
            "risks": [
                {"risk": r.get("description", ""), "severity": r.get("severity", "Medium"), "mitigation": r.get("mitigation", "")}
                for r in mom_data.get("risks", [])
            ],
        }

    @classmethod
    def to_jira_tickets(cls, mom_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Generates Jira / Linear issue creation payloads for each action item."""
        meeting_title = mom_data.get("meeting_title", "Sprint Meeting")
        tickets = []
        for act in mom_data.get("action_items", []):
            tickets.append({
                "fields": {
                    "project": {"key": "MOM"},
                    "summary": f"[{meeting_title}] {act.get('description')}",
                    "description": (
                        f"*Action Item:* {act.get('description')}\n"
                        f"*Owner:* {act.get('owner')}\n"
                        f"*Target Date:* {act.get('deadline_text')}\n"
                        f"*Priority:* {act.get('priority')}\n"
                        f"*Success Criteria:* {act.get('success_criteria')}\n"
                        f"*Evidence:* {act.get('evidence_quote')}"
                    ),
                    "issuetype": {"name": "Task"},
                    "priority": {"name": str(act.get("priority", "Medium")).capitalize()},
                }
            })
        return tickets

    @classmethod
    def to_slack_blocks(cls, mom_data: dict[str, Any]) -> dict[str, Any]:
        """Generates rich interactive Slack Block Kit payload."""
        meeting_title = mom_data.get("meeting_title", "Executive Sync")
        summary_obj = mom_data.get("summary", {})
        exec_sum = summary_obj.get("executive_summary", "") if isinstance(summary_obj, dict) else str(summary_obj or "")
        actions = mom_data.get("action_items", [])
        decisions = mom_data.get("decisions", [])

        action_lines = "\n".join([
            f"• *{a.get('description')}* — <@{a.get('owner')}> (Due: `{a.get('deadline_text')}`)"
            for a in actions[:5]
        ]) or "_None registered._"

        decision_lines = "\n".join([
            f"• *{d.get('description')}* (Approved: {', '.join(d.get('approved_by', [])) or 'Consensus'})"
            for d in decisions[:4]
        ]) or "_None registered._"

        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": f"📋 MOM: {meeting_title}", "emoji": True}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Executive Summary:*\n{exec_sum}"}
                },
                {"type": "divider"},
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*⚡ Critical Action Items:*\n{action_lines}"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*🎯 Key Decisions Approved:*\n{decision_lines}"}
                },
                {
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": "Generated automatically by *IQSYS MOM Swarm AI Engine*"}
                    ]
                }
            ]
        }

    @classmethod
    def to_slack_card(cls, mom_data: dict[str, Any]) -> str:
        """Generates formatted Slack / Microsoft Teams card text."""
        meeting_title = mom_data.get("meeting_title", "Executive Sync")
        summary_obj = mom_data.get("summary", {})
        exec_sum = summary_obj.get("executive_summary", "") if isinstance(summary_obj, dict) else str(summary_obj or mom_data.get("meeting_summary", ""))
        actions = mom_data.get("action_items", [])
        decisions = mom_data.get("decisions", [])

        action_lines = "\n".join([
            f"• *{(a.get('action') or a.get('description', '')).strip().splitlines()[0]}* — <@{a.get('owner', 'Unassigned')}> (Due: `{a.get('deadline_text') or a.get('deadline') or 'Not specified'}`)"
            for a in actions[:5]
        ]) or "_None registered._"

        decision_lines = "\n".join([
            f"• *{d.get('description')}* (Approved: {', '.join(d.get('approved_by', [])) or 'Consensus'})"
            for d in decisions[:4]
        ]) or "_None registered._"

        return f"*📋 MOM: {meeting_title}*\n\n*Executive Summary:*\n{exec_sum}\n\n*⚡ Critical Action Items:*\n{action_lines}\n\n*🎯 Key Decisions:*\n{decision_lines}"

    @classmethod
    def to_markdown(cls, mom_data: dict[str, Any]) -> str:
        """Generates clean corporate Markdown document."""
        meeting_title = mom_data.get("meeting_title", "Executive Meeting")
        summary = mom_data.get("meeting_summary", "")
        actions = mom_data.get("action_items", [])
        decisions = mom_data.get("decisions", [])
        risks = mom_data.get("risks", [])

        md = f"# Minutes of Meeting: {meeting_title}\n\n## Executive Summary\n{summary}\n\n## Action Items\n"
        md += "\n".join([f"- **[{a.get('priority', 'Medium').upper()}] {a.get('description')}** (Owner: {a.get('owner')}, Deadline: {a.get('deadline_text')})" for a in actions])
        md += "\n\n## Decisions\n" + "\n".join([f"- **{d.get('description')}** (Approved: {', '.join(d.get('approved_by', []))})" for d in decisions])
        md += "\n\n## Risks\n" + "\n".join([f"- **[{r.get('severity', 'Medium').upper()}] {r.get('description')}** -> Mitigation: {r.get('mitigation')}" for r in risks])
        return md


# ==========================================
# Publication-Grade Corporate PDF Generation
# ==========================================

def _clean_pdf_text(text: Any) -> str:
    """Sanitizes text for ReportLab XML parser, strips emojis, and escapes XML entities."""
    if text is None:
        return ""
    s = str(text)
    s = re.sub(r"[\U00010000-\U0010ffff]", "", s)
    s = re.sub(r"[\u2600-\u27bf\u2300-\u23ff\u2b50-\u2b55\u200d\ufe0f]", "", s)
    placeholders = []

    def _save_tag(match: re.Match[str]) -> str:
        placeholders.append(match.group(0))
        return f"__TAG_PLACEHOLDER_{len(placeholders)-1}__"

    s = re.sub(r"<\/?(?:b|i|u|font|br)(?:\s+[^>]*)*\/?>", _save_tag, s, flags=re.IGNORECASE)
    s = html.escape(s, quote=False)
    for idx, tag in enumerate(placeholders):
        s = s.replace(f"__TAG_PLACEHOLDER_{idx}__", tag)
    return s.strip()


def generate_corporate_pdf(mom_data: dict[str, Any] | Any) -> bytes:
    """Renders a publication-grade, boardroom-ready executive PDF document from canonical SSOT state."""
    if hasattr(mom_data, "model_dump"):
        mom_data = mom_data.model_dump()
    elif not isinstance(mom_data, dict):
        mom_data = dict(mom_data)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=32,
        bottomMargin=32,
    )

    styles = getSampleStyleSheet()

    header_title_style = ParagraphStyle(
        "HeaderTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.white,
        spaceAfter=2,
    )

    header_sub_style = ParagraphStyle(
        "HeaderSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#93c5fd"),
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=8,
        spaceAfter=3,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10.5,
        textColor=colors.HexColor("#334155"),
    )

    summary_box_style = ParagraphStyle(
        "SummaryBoxText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=12,
        textColor=colors.HexColor("#0f172a"),
    )

    th_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=0,
    )

    td_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#1e293b"),
    )

    td_badge_high = ParagraphStyle(
        "BadgeHigh",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#dc2626"),
    )

    td_badge_medium = ParagraphStyle(
        "BadgeMedium",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#d97706"),
    )

    td_badge_low = ParagraphStyle(
        "BadgeLow",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#059669"),
    )

    kpi_num_style = ParagraphStyle(
        "KpiNum",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=15,
        textColor=colors.HexColor("#1e3a8a"),
        alignment=1,
    )

    kpi_label_style = ParagraphStyle(
        "KpiLabel",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.5,
        leading=8.5,
        textColor=colors.HexColor("#64748b"),
        alignment=1,
    )

    story = []

    # 1. Executive Top Navy Banner
    raw_title = mom_data.get("meeting_title") or "Executive Technical & Strategic Alignment Review"
    meeting_title = _clean_pdf_text(raw_title)
    meeting_type = _clean_pdf_text(str(mom_data.get("meeting_type", "General")).upper())
    gen_time = datetime.now().strftime("%B %d, %Y - %I:%M %p")
    participants = mom_data.get("participants", [])
    participants_str = _clean_pdf_text(", ".join(participants) if participants else "Executive Stakeholders")

    header_table = Table(
        [
            [
                Paragraph(f"<b>MINUTES OF MEETING:</b> {meeting_title}", header_title_style),
            ],
            [
                Paragraph(
                    f"<b>Classification:</b> {meeting_type} &nbsp;|&nbsp; <b>Date:</b> {gen_time} &nbsp;|&nbsp; <b>Attendees:</b> {participants_str}",
                    header_sub_style,
                )
            ],
        ],
        colWidths=[540],
    )
    header_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
            ("PADDING", (0, 0), (-1, -1), 8),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#1e3a8a")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(header_table)
    story.append(Spacer(1, 4))

    # 2. Executive KPI Dashboard Cards
    actions = mom_data.get("action_items", [])
    decisions = mom_data.get("decisions", [])
    risks = mom_data.get("risks", [])
    sentiment = mom_data.get("sentiment", {})
    tone_str = _clean_pdf_text(sentiment.get("overall", "Aligned"))

    kpi_table = Table(
        [
            [
                Paragraph(str(len(actions)), kpi_num_style),
                Paragraph(str(len(decisions)), kpi_num_style),
                Paragraph(str(len(risks)), kpi_num_style),
                Paragraph(tone_str, kpi_num_style),
            ],
            [
                Paragraph("ACTION ITEMS ASSIGNED", kpi_label_style),
                Paragraph("DECISIONS RATIFIED", kpi_label_style),
                Paragraph("RISKS IDENTIFIED", kpi_label_style),
                Paragraph("MEETING ALIGNMENT", kpi_label_style),
            ],
        ],
        colWidths=[135, 135, 135, 135],
    )
    kpi_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("PADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(kpi_table)
    story.append(Spacer(1, 4))

    # 3. Executive Summary & Core Milestones
    exec_summary = ""
    key_points = []
    summary_obj = mom_data.get("summary")
    if isinstance(summary_obj, dict):
        exec_summary = summary_obj.get("executive_summary", "") or ""
        key_points = summary_obj.get("key_points", []) or []
    elif hasattr(summary_obj, "executive_summary"):
        exec_summary = getattr(summary_obj, "executive_summary", "") or ""
        key_points = getattr(summary_obj, "key_points", []) or []
    elif isinstance(summary_obj, str) and summary_obj.strip():
        exec_summary = summary_obj.strip()

    if not exec_summary:
        exec_summary = str(mom_data.get("meeting_summary") or mom_data.get("executive_summary") or "").strip()
    if not key_points:
        key_points = mom_data.get("key_points", []) or []

    if not exec_summary or len(exec_summary.strip()) < 10:
        exec_summary = f"Executive discussion convened on '{raw_title}' to review milestones, ratify architectural and operational decisions, and prioritize upcoming deliverables."

    story.append(Paragraph("1. Executive Briefing & Strategic Summary", h2_style))
    clean_summary_p = _clean_pdf_text(exec_summary)
    summary_table = Table(
        [[Paragraph(f"<b>Executive Briefing:</b> {clean_summary_p}", summary_box_style)]],
        colWidths=[540],
    )
    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#93c5fd")),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(summary_table)

    if key_points:
        story.append(Spacer(1, 2))
        for pt in key_points:
            story.append(Paragraph(f"• <b>{_clean_pdf_text(pt)}</b>", body_style))

    # 4. SMART Action Items Matrix
    story.append(Spacer(1, 3))
    story.append(Paragraph("2. Action Items Matrix (Post-Meeting Tasks & Responsibilities)", h2_style))
    if actions:
        act_rows = [
            [
                Paragraph("#", th_style),
                Paragraph("Action Item (What needs to be done)", th_style),
                Paragraph("Owner", th_style),
                Paragraph("Priority", th_style),
                Paragraph("Target Deadline", th_style),
                Paragraph("Evidence Quote / Citation", th_style),
            ]
        ]

        for idx, act in enumerate(actions, 1):
            short_action = _clean_pdf_text((act.get("action") or act.get("description", "")).strip().splitlines()[0])
            pri = str(act.get("priority", "Medium")).capitalize()
            pri_style = td_badge_high if pri == "High" else (td_badge_medium if pri == "Medium" else td_badge_low)
            owner = _clean_pdf_text(act.get("owner", "Unassigned"))
            deadline = _clean_pdf_text(act.get("deadline_text") or act.get("deadline") or "Not specified")
            quote = _clean_pdf_text(act.get("evidence_quote") or act.get("evidence") or "")

            act_rows.append([
                Paragraph(str(idx), td_style),
                Paragraph(f"<b>{short_action}</b>", td_style),
                Paragraph(f"<code>{owner}</code>", td_style),
                Paragraph(pri, pri_style),
                Paragraph(deadline, td_style),
                Paragraph(f"<i>\"{quote}\"</i>" if quote else "-", td_style),
            ])

        act_table = Table(act_rows, colWidths=[16, 185, 75, 40, 75, 149])
        act_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("PADDING", (0, 0), (-1, -1), 3.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ])
        )
        story.append(act_table)
    else:
        story.append(Paragraph("<i>No explicit pending action items were registered in this session.</i>", body_style))

    # 5. Formal Decisions & Approvals Log
    story.append(Spacer(1, 3))
    story.append(Paragraph("3. Formal Decisions & Governance Approvals", h2_style))
    if decisions:
        dec_rows = [
            [
                Paragraph("#", th_style),
                Paragraph("Ratified Decision / Milestone", th_style),
                Paragraph("Approved By", th_style),
                Paragraph("Business / Technical Impact", th_style),
            ]
        ]
        for idx, dec in enumerate(decisions, 1):
            desc = _clean_pdf_text(dec.get("description", ""))
            approved_by = _clean_pdf_text(", ".join(dec.get("approved_by", [])) or "Executive Consensus")
            impact = _clean_pdf_text(dec.get("impact") or "Operational and architectural baseline approved.")
            quote = dec.get("evidence_quote")
            quote_text = f"<br/><i>Citation: \"{_clean_pdf_text(quote)}\"</i>" if quote else ""

            dec_rows.append([
                Paragraph(str(idx), td_style),
                Paragraph(f"<b>{desc}</b>", td_style),
                Paragraph(approved_by, td_style),
                Paragraph(f"{impact}{quote_text}", td_style),
            ])

        dec_table = Table(dec_rows, colWidths=[16, 220, 110, 194])
        dec_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("PADDING", (0, 0), (-1, -1), 3.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ])
        )
        story.append(dec_table)
    else:
        story.append(Paragraph("<i>No formal governance decisions were ratified in this session.</i>", body_style))

    # 6. Risk Matrix & Mitigation Plans
    if risks:
        story.append(Spacer(1, 3))
        story.append(Paragraph("4. Risk Control & Mitigation Matrix", h2_style))
        risk_rows = [
            [
                Paragraph("#", th_style),
                Paragraph("Identified Risk / Vulnerability", th_style),
                Paragraph("Severity", th_style),
                Paragraph("Prob / Impact", th_style),
                Paragraph("Actionable Mitigation Plan", th_style),
            ]
        ]
        for idx, r in enumerate(risks, 1):
            desc = _clean_pdf_text(r.get("description", ""))
            sev = str(r.get("severity", "Medium")).capitalize()
            sev_style = td_badge_high if sev == "High" else (td_badge_medium if sev == "Medium" else td_badge_low)
            prob = str(r.get("probability", "Medium")).capitalize()
            imp = str(r.get("impact", "Medium")).capitalize()
            mit = _clean_pdf_text(r.get("mitigation") or "Track mitigation in sprint planning and technical reviews.")

            risk_rows.append([
                Paragraph(str(idx), td_style),
                Paragraph(f"<b>{desc}</b>", td_style),
                Paragraph(sev, sev_style),
                Paragraph(f"{prob} / {imp}", td_style),
                Paragraph(mit, td_style),
            ])

        risk_table = Table(risk_rows, colWidths=[16, 175, 45, 80, 224])
        risk_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#991b1b")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fef2f2")]),
                ("PADDING", (0, 0), (-1, -1), 3.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ])
        )
        story.append(risk_table)

    # 7. Discussion Topics Breakdown
    topics = mom_data.get("topics", [])
    if topics:
        story.append(Spacer(1, 3))
        story.append(Paragraph("5. Discussion Topics & Agenda Breakdown", h2_style))
        topic_rows = [[Paragraph("Topic / Workstream", th_style), Paragraph("Discussion Summary & Technical Context", th_style)]]
        for t in topics:
            t_name = _clean_pdf_text(t.get("name", "Topic"))
            t_sum = _clean_pdf_text(t.get("summary", ""))
            topic_rows.append([
                Paragraph(f"<b>{t_name}</b>", td_style),
                Paragraph(t_sum, td_style),
            ])
        topic_table = Table(topic_rows, colWidths=[160, 380])
        topic_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("PADDING", (0, 0), (-1, -1), 3.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ])
        )
        story.append(topic_table)

    # 8. Requirements, Open Inquiries & Follow-Ups
    reqs = mom_data.get("requirements", [])
    questions = mom_data.get("open_questions", [])
    follow_ups = mom_data.get("follow_up_tasks", [])

    if reqs or questions or follow_ups:
        story.append(Spacer(1, 3))
        story.append(Paragraph("6. Requirements, Open Inquiries & Follow-Up Workstreams", h2_style))
        if reqs:
            for rq in reqs:
                rq_desc = _clean_pdf_text(rq.get("description", ""))
                rq_cat = str(rq.get("category", "Functional")).capitalize()
                story.append(Paragraph(f"• <b>[{rq_cat} Requirement]</b> {rq_desc}", body_style))
        if questions:
            for q in questions:
                q_text = _clean_pdf_text(q.get("question") if isinstance(q, dict) else str(q))
                q_asker = _clean_pdf_text(q.get("owner") or q.get("asked_by") or "Team" if isinstance(q, dict) else "Team")
                story.append(Paragraph(f"• <b>[Open Inquiry]</b> {q_text} <i>(Raised by: {q_asker})</i>", body_style))
        if follow_ups:
            for f in follow_ups:
                f_text = _clean_pdf_text(f.get("description") or f.get("task") if isinstance(f, dict) else str(f))
                f_owner = _clean_pdf_text(f.get("owner", "Team") if isinstance(f, dict) else "Team")
                story.append(Paragraph(f"• <b>[Follow-Up Workstream]</b> {f_text} <i>(Owner: {f_owner})</i>", body_style))

    # 9. Sentiment & Meeting Dynamics Analysis
    if sentiment:
        story.append(Spacer(1, 3))
        story.append(Paragraph("7. Meeting Tone & Dynamics Alignment", h2_style))
        tone = _clean_pdf_text(sentiment.get("overall", "Constructive & Professional"))
        c_mood = _clean_pdf_text(sentiment.get("client_mood", "Engaged & Aligned"))
        t_mood = _clean_pdf_text(sentiment.get("team_mood", "Focused on Execution"))
        evidence = sentiment.get("evidence", [])

        story.append(Paragraph(f"• <b>Overall Climate:</b> {tone} &nbsp;|&nbsp; <b>Client Posture:</b> {c_mood} &nbsp;|&nbsp; <b>Team Dynamics:</b> {t_mood}", body_style))
        if evidence:
            for ev in evidence[:2]:
                story.append(Paragraph(f"  <i>Supporting Evidence: \"{_clean_pdf_text(ev)}\"</i>", body_style))

    # 10. Formal Stakeholder Sign-Off Block
    story.append(Spacer(1, 4))
    story.append(
        KeepTogether([
            Paragraph("8. Stakeholder Verification & Formal Governance Sign-Off", h2_style),
            Spacer(1, 4),
            Table(
                [
                    [
                        Paragraph("<b>Meeting Chairperson:</b><br/><br/><br/>___________________________<br/>Signature & Date", body_style),
                        Paragraph("<b>Technical Lead:</b><br/><br/><br/>___________________________<br/>Signature & Date", body_style),
                        Paragraph("<b>Stakeholder / Client Lead:</b><br/><br/><br/>___________________________<br/>Signature & Date", body_style),
                    ]
                ],
                colWidths=[180, 180, 180],
            ),
        ])
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
