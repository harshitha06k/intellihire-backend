import os
import json
from dotenv import load_dotenv
from groq import Groq
from schema import Session, QuestionEntry
from evaluator import evaluate_answer
from session_store import sync_m2_scores

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_KEY"))

# ── State transition logic ──────────────────────────────────────────────────
def decide_next_state(session: Session) -> str:
    s = session.scores
    history_len = len(session.question_history)
    current = session.current_state

    if current == "WARM_UP" and history_len >= 2:
        return "CORE_SKILLS"

    if current == "CORE_SKILLS":
        if s.technical > 70 and s.confidence > 60:
            return "DEEP_DIVE"
        if s.technical < 30:
            return "CORE_SKILLS"

    if current == "DEEP_DIVE":
        if s.technical > 85 and s.confidence > 75:
            return "STRESS_TEST"

    if history_len >= 12:
        return "WRAP_UP"

    return current

def should_encourage(session: Session) -> bool:
    return session.scores.confidence < 40 and session.scores.technical > 60

# ── Infer confidence from answer quality when M2 not available ─────────────
def infer_confidence_from_answer(answer: str, tech_score: float) -> float:
    """
    Heuristic confidence score based on answer characteristics.
    Used when M2 audio analysis hasn't scored confidence yet (still 0).
    """
    word_count = len(answer.split())
    
    # Length signal: very short = low confidence
    if word_count < 10:
        length_score = 20.0
    elif word_count < 30:
        length_score = 45.0
    elif word_count < 80:
        length_score = 65.0
    else:
        length_score = 80.0

    # Technical quality also correlates with confidence
    tech_component = tech_score * 0.5  # 0-50 range

    # Combined
    raw = length_score * 0.5 + tech_component
    return round(min(raw + 20, 100), 1)  # floor at 20, max 100

# ── Main turn ───────────────────────────────────────────────────────────────
async def run_interview_turn(session: Session, answer: str) -> dict:
    # Pull latest M2 scores (confidence + engagement from audio/video)
    await sync_m2_scores(session.session_id)

    # 1. Evaluate answer
    if session.question_history and answer.strip():
        last_q = session.question_history[-1]
        last_q.answer = answer

        eval_result = await evaluate_answer(last_q.question, answer)
        tech_score = eval_result.get("score", 5)
        last_q.technical_score = tech_score
        last_q.depth = eval_result.get("depth", "surface")
        last_q.missing_concepts = eval_result.get("missing", [])

        # Update technical score (rolling average)
        session.scores.technical = round(
            session.scores.technical * 0.6 + tech_score * 10 * 0.4, 1
        )

        # ── Update confidence: use M2 if available, else infer from answer ──
        if session.scores.confidence < 5:
            # M2 hasn't scored yet — use heuristic
            session.scores.confidence = infer_confidence_from_answer(answer, tech_score)
        else:
            # M2 is active — blend M2 score with answer quality
            inferred = infer_confidence_from_answer(answer, tech_score)
            session.scores.confidence = round(
                session.scores.confidence * 0.7 + inferred * 0.3, 1
            )

        # ── Update engagement if M2 not active ───────────────────────────────
        if session.scores.engagement < 5:
            # Infer basic engagement from answer depth
            depth_map = {"surface": 40, "intermediate": 65, "deep": 85}
            session.scores.engagement = depth_map.get(last_q.depth, 50)

    # 2. Decide state
    prev_state = session.current_state
    new_state = decide_next_state(session)
    session.current_state = new_state

    # 3. Build prompt
    encourage_note = (
        "IMPORTANT: The candidate seems nervous. Ask the next question gently and encouragingly."
        if should_encourage(session) else ""
    )

    history_text = "\n".join(
        f"Q: {q.question}\nA: {q.answer}"
        for q in session.question_history[-3:]
        if q.answer
    ) or "No previous questions yet."

    skills_text = ", ".join(session.resume_data.skills[:8]) or "general software engineering"
    role = (
        session.resume_data.inferred_roles[0]["role"]
        if session.resume_data.inferred_roles
        else "Software Engineer"
    )

    state_guidance = {
        "WARM_UP":      "Ask a warm, easy question. Be conversational. Build rapport.",
        "CORE_SKILLS":  "Test core technical skills listed on their resume. Be direct but fair.",
        "DEEP_DIVE":    "Go deeper on something from a previous answer or their resume project.",
        "STRESS_TEST":  "Ask a system design or edge-case question. Push their limits.",
        "WRAP_UP":      "Wrap up the interview. Ask if they have any questions for you.",
    }

    prompt = f"""You are an expert technical interviewer conducting a real interview.

Candidate role: {role}
Candidate skills: {skills_text}
Interview phase: {new_state} — {state_guidance.get(new_state, "")}
Current scores — Technical: {session.scores.technical:.0f}/100, Confidence: {session.scores.confidence:.0f}/100

Recent conversation:
{history_text}

{encourage_note}

Generate the NEXT interview question.
- Reference their resume or a previous answer when possible
- One question only
- No preamble, no explanation — just the question itself"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    question_text = response.choices[0].message.content.strip()

    session.question_history.append(QuestionEntry(question=question_text))

    return {
        "question": question_text,
        "state": new_state,
        "adapted": new_state != prev_state,
        "adaptation_reason": (
            f"Moved to {new_state.replace('_', ' ').title()} based on performance"
            if new_state != prev_state else ""
        ),
        "scores": {
            "confidence": round(session.scores.confidence, 1),
            "technical":  round(session.scores.technical, 1),
            "engagement": round(session.scores.engagement, 1),
        },
    }