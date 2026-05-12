import os
import json
import tempfile
from dotenv import load_dotenv
from groq import Groq
from schema import Session

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_KEY"))

# ── Generate report JSON ────────────────────────────────────────────────────
async def generate_report(session: Session) -> dict:
    history_text = "\n".join(
        f"Q{i+1}: {q.question}\n"
        f"A: {q.answer}\n"
        f"Score: {q.technical_score}/10 | Depth: {q.depth} | "
        f"Missing: {', '.join(q.missing_concepts) if q.missing_concepts else 'none'}"
        for i, q in enumerate(session.question_history)
        if q.answer
    ) or "No answers recorded."

    prompt = f"""You are an expert career coach. Generate a structured interview feedback report.
Return ONLY valid JSON, no markdown, no backticks, no explanation.

Candidate skills: {", ".join(session.resume_data.skills)}
Final scores:
  - Technical:  {session.scores.technical:.0f}/100
  - Confidence: {session.scores.confidence:.0f}/100
  - Engagement: {session.scores.engagement:.0f}/100

Interview Q&A:
{history_text}

Return exactly this structure:
{{
  "executive_summary": "2-3 sentence overall assessment of the candidate",
  "technical_scorecard": [
    {{"topic": "Python", "score": 8, "comment": "Strong understanding, good examples"}},
    {{"topic": "System Design", "score": 5, "comment": "Basic knowledge, missed scalability"}}
  ],
  "communication": {{
    "confidence_score": 65,
    "key_moments": [
      "Hesitated when asked about database indexing",
      "Explained REST APIs very clearly"
    ]
  }},
  "next_steps": [
    "Practice system design on Excalidraw — draw out architectures",
    "Review CAP theorem and distributed systems basics",
    "Record yourself answering questions to improve speaking pace"
  ]
}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    text = response.choices[0].message.content.strip()

    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    try:
        return json.loads(text.strip())
    except Exception:
        return {
            "executive_summary": "Report generation encountered an issue. Please review raw scores.",
            "technical_scorecard": [],
            "communication": {"confidence_score": int(session.scores.confidence), "key_moments": []},
            "next_steps": ["Review your performance with your team."],
        }

# ── Generate PDF from report ────────────────────────────────────────────────
async def generate_pdf(session_id: str) -> str:
    from session_store import get_session
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    session = get_session(session_id)
    if not session:
        raise ValueError("Session not found")

    report = await generate_report(session)
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)

    # ── Page setup ──────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        tmp.name, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=14*mm, bottomMargin=14*mm
    )

    # ── Brand colors ────────────────────────────────────────────────────────
    PURPLE     = colors.HexColor("#6d28d9")
    PURPLE_LT  = colors.HexColor("#ede9fe")
    DARK       = colors.HexColor("#1e1b4b")
    GREY       = colors.HexColor("#6b7280")
    GREY_LT    = colors.HexColor("#f3f4f6")
    GREEN      = colors.HexColor("#059669")
    AMBER      = colors.HexColor("#d97706")
    RED        = colors.HexColor("#dc2626")
    WHITE      = colors.white

    # ── Custom styles ───────────────────────────────────────────────────────
    base = getSampleStyleSheet()

    def S(name, parent="Normal", **kw):
        return ParagraphStyle(name, parent=base[parent], **kw)

    sTitle    = S("sTitle",    fontSize=22, textColor=WHITE,      leading=28, spaceAfter=2,  fontName="Helvetica-Bold")
    sSub      = S("sSub",      fontSize=11, textColor=PURPLE_LT,  leading=16, spaceAfter=0,  fontName="Helvetica")
    sH2       = S("sH2",       fontSize=13, textColor=PURPLE,     leading=18, spaceBefore=10, spaceAfter=4, fontName="Helvetica-Bold")
    sBody     = S("sBody",     fontSize=10, textColor=DARK,       leading=15, spaceAfter=4)
    sBullet   = S("sBullet",   fontSize=10, textColor=DARK,       leading=15, leftIndent=12, spaceAfter=3)
    sLabel    = S("sLabel",    fontSize=8,  textColor=GREY,       leading=11, fontName="Helvetica-Bold", spaceAfter=1)
    sScore    = S("sScore",    fontSize=22, textColor=PURPLE,     leading=26, fontName="Helvetica-Bold", alignment=TA_CENTER)
    sScoreLbl = S("sScoreLbl", fontSize=9,  textColor=GREY,       leading=12, alignment=TA_CENTER)
    sTag      = S("sTag",      fontSize=9,  textColor=PURPLE,     leading=12, fontName="Helvetica-Bold")
    sSummary  = S("sSummary",  fontSize=10, textColor=DARK,       leading=16, leftIndent=8, rightIndent=8)

    def score_color(v):
        if v >= 70: return GREEN
        if v >= 45: return AMBER
        return RED

    def bar_table(value, color, width=120):
        filled = max(1, int(value / 100 * width))
        empty  = width - filled
        data = [[""]*filled + [""]*empty]
        style = TableStyle([
            ("BACKGROUND", (0,0), (filled-1,0), color),
            ("BACKGROUND", (filled,0), (-1,0), colors.HexColor("#e5e7eb")),
            ("ROWHEIGHT", (0,0), (-1,-1), 7),
            ("TOPPADDING",   (0,0),(-1,-1),0),
            ("BOTTOMPADDING",(0,0),(-1,-1),0),
            ("LEFTPADDING",  (0,0),(-1,-1),0),
            ("RIGHTPADDING", (0,0),(-1,-1),0),
        ])
        col_w = (width / (filled + empty)) * mm * 0.6
        return Table(data, colWidths=[col_w]*(filled+empty), style=style)

    elements = []

    # ── HEADER BANNER ───────────────────────────────────────────────────────
    role = (session.resume_data.inferred_roles[0]["role"]
            if session.resume_data.inferred_roles else "Software Engineer")
    candidate = session.resume_data.skills[:4]

    header_data = [[
        Paragraph("IntelliHire", sTitle),
        Paragraph(f"Interview Coaching Report<br/><font size=9>{role}</font>", sSub),
    ]]
    header_style = TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), PURPLE),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
        ("TOPPADDING",    (0,0), (-1,-1), 12),
        ("BOTTOMPADDING", (0,0), (-1,-1), 12),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ])
    elements.append(Table(header_data, colWidths=["40%","60%"], style=header_style))
    elements.append(Spacer(1, 8*mm))

    # ── SCORE CARDS (3 in a row) ─────────────────────────────────────────────
    def score_card(label, value):
        col = score_color(value)
        return [
            Paragraph(f"{value:.0f}", sScore),
            Paragraph("/100", sScoreLbl),
            Spacer(1, 2),
            bar_table(value, col, width=80),
            Spacer(1, 3),
            Paragraph(label, sScoreLbl),
        ]

    scores_data = [[score_card("Technical", session.scores.technical),
                    score_card("Confidence", session.scores.confidence),
                    score_card("Engagement", session.scores.engagement)]]

    scores_table = Table(scores_data, colWidths=["33%","33%","34%"],
        style=TableStyle([
            ("BOX",          (0,0),(-1,-1), 0.5, colors.HexColor("#e5e7eb")),
            ("INNERGRID",    (0,0),(-1,-1), 0.5, colors.HexColor("#e5e7eb")),
            ("BACKGROUND",   (0,0),(-1,-1), GREY_LT),
            ("TOPPADDING",   (0,0),(-1,-1), 10),
            ("BOTTOMPADDING",(0,0),(-1,-1), 10),
            ("ALIGN",        (0,0),(-1,-1), "CENTER"),
            ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
        ])
    )
    elements.append(scores_table)
    elements.append(Spacer(1, 6*mm))

    # ── EXECUTIVE SUMMARY ───────────────────────────────────────────────────
    elements.append(Paragraph("Executive Summary", sH2))
    elements.append(HRFlowable(width="100%", thickness=1, color=PURPLE_LT))
    elements.append(Spacer(1, 3))
    summary_box = Table(
        [[Paragraph(report.get("executive_summary", ""), sSummary)]],
        style=TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), PURPLE_LT),
            ("BOX",           (0,0),(-1,-1), 0.5, PURPLE),
            ("LEFTPADDING",   (0,0),(-1,-1), 10),
            ("RIGHTPADDING",  (0,0),(-1,-1), 10),
            ("TOPPADDING",    (0,0),(-1,-1), 8),
            ("BOTTOMPADDING", (0,0),(-1,-1), 8),
        ])
    )
    elements.append(summary_box)
    elements.append(Spacer(1, 5*mm))

    # ── TECHNICAL SCORECARD ─────────────────────────────────────────────────
    elements.append(Paragraph("Technical Scorecard", sH2))
    elements.append(HRFlowable(width="100%", thickness=1, color=PURPLE_LT))
    elements.append(Spacer(1, 3))

    scorecard = report.get("technical_scorecard", [])
    if scorecard:
        tc_data = [[ 
            Paragraph("Topic", sLabel),
            Paragraph("Score", sLabel),
            Paragraph("", sLabel),
            Paragraph("Comment", sLabel),
        ]]
        for item in scorecard:
            v = item.get("score", 0) * 10
            col = score_color(v)
            tc_data.append([
                Paragraph(f"<b>{item.get('topic','')}</b>", sBody),
                Paragraph(f"<font color='{col.hexval().replace('0x', '#')}'><b>{item.get('score',0)}/10</b></font>", sBody),
                bar_table(v, col, width=60),
                Paragraph(item.get("comment",""), sBody),
            ])
        tc_table = Table(tc_data, colWidths=["22%","10%","18%","50%"],
            style=TableStyle([
                ("BACKGROUND",    (0,0),(-1,0), colors.HexColor("#f5f3ff")),
                ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, GREY_LT]),
                ("BOX",           (0,0),(-1,-1), 0.5, colors.HexColor("#e5e7eb")),
                ("INNERGRID",     (0,0),(-1,-1), 0.3, colors.HexColor("#e5e7eb")),
                ("TOPPADDING",    (0,0),(-1,-1), 6),
                ("BOTTOMPADDING", (0,0),(-1,-1), 6),
                ("LEFTPADDING",   (0,0),(-1,-1), 6),
                ("RIGHTPADDING",  (0,0),(-1,-1), 6),
                ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
            ])
        )
        elements.append(tc_table)
    elements.append(Spacer(1, 5*mm))

    # ── COMMUNICATION ───────────────────────────────────────────────────────
    elements.append(Paragraph("Communication & Body Language", sH2))
    elements.append(HRFlowable(width="100%", thickness=1, color=PURPLE_LT))
    elements.append(Spacer(1, 3))
    comm = report.get("communication", {})
    conf_score = comm.get("confidence_score", int(session.scores.confidence))
    elements.append(Paragraph(f"Confidence score: <b>{conf_score}/100</b>", sBody))
    elements.append(bar_table(conf_score, score_color(conf_score), width=200))
    elements.append(Spacer(1, 4))
    for moment in comm.get("key_moments", []):
        elements.append(Paragraph(f"• {moment}", sBullet))
    elements.append(Spacer(1, 5*mm))

    # ── NEXT STEPS ──────────────────────────────────────────────────────────
    elements.append(Paragraph("Recommended Next Steps", sH2))
    elements.append(HRFlowable(width="100%", thickness=1, color=PURPLE_LT))
    elements.append(Spacer(1, 3))
    for i, step in enumerate(report.get("next_steps", []), 1):
        step_data = [[
            Paragraph(f"<b>{i}</b>", ParagraphStyle("Num", fontSize=11, textColor=WHITE, alignment=TA_CENTER)),
            Paragraph(step, sBody),
        ]]
        step_table = Table(step_data, colWidths=["8%","92%"],
            style=TableStyle([
                ("BACKGROUND",    (0,0),(0,0), PURPLE),
                ("BACKGROUND",    (1,0),(1,0), GREY_LT),
                ("TOPPADDING",    (0,0),(-1,-1), 6),
                ("BOTTOMPADDING", (0,0),(-1,-1), 6),
                ("LEFTPADDING",   (0,0),(-1,-1), 6),
                ("RIGHTPADDING",  (0,0),(-1,-1), 6),
                ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
                ("BOX",           (0,0),(-1,-1), 0.5, colors.HexColor("#e5e7eb")),
            ])
        )
        elements.append(step_table)
        elements.append(Spacer(1, 2*mm))

    # ── SKILLS ASSESSED ─────────────────────────────────────────────────────
    if session.resume_data.skills:
        elements.append(Spacer(1, 3*mm))
        elements.append(Paragraph("Skills Assessed", sH2))
        elements.append(HRFlowable(width="100%", thickness=1, color=PURPLE_LT))
        elements.append(Spacer(1, 3))
        skill_cols = 5
        skills = session.resume_data.skills[:15]
        rows = [skills[i:i+skill_cols] for i in range(0, len(skills), skill_cols)]
        for row in rows:
            while len(row) < skill_cols:
                row.append("")
            skill_data = [[Paragraph(s, sTag) if s else Paragraph("", sTag) for s in row]]
            skill_table = Table(skill_data, colWidths=["20%"]*skill_cols,
                style=TableStyle([
                    ("BACKGROUND",    (0,0),(-1,-1), PURPLE_LT),
                    ("BOX",           (0,0),(-1,-1), 0.3, PURPLE),
                    ("INNERGRID",     (0,0),(-1,-1), 0.3, PURPLE),
                    ("TOPPADDING",    (0,0),(-1,-1), 5),
                    ("BOTTOMPADDING", (0,0),(-1,-1), 5),
                    ("LEFTPADDING",   (0,0),(-1,-1), 6),
                    ("ALIGN",         (0,0),(-1,-1), "CENTER"),
                ])
            )
            elements.append(skill_table)
            elements.append(Spacer(1, 1*mm))

    # ── FOOTER ──────────────────────────────────────────────────────────────
    elements.append(Spacer(1, 8*mm))
    footer_data = [[Paragraph("Generated by <b>IntelliHire AI</b> · Confidential", 
        ParagraphStyle("Footer", fontSize=8, textColor=WHITE, alignment=TA_CENTER))]]
    footer = Table(footer_data, style=TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), PURPLE),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
    ]))
    elements.append(footer)

    doc.build(elements)
    return tmp.name