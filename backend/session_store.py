import httpx
import os
from dotenv import load_dotenv
from schema import Session

load_dotenv()

# Simple in-memory store — keyed by session_id
# Both M1 and M2 import this file
sessions: dict[str, Session] = {}

M2_BASE_URL = os.getenv("M2_URL", "http://localhost:8001")

def create_session() -> Session:
    s = Session()
    sessions[s.session_id] = s
    return s

def get_session(session_id: str) -> Session | None:
    return sessions.get(session_id)

def update_scores(session_id: str, confidence=None, technical=None, engagement=None):
    s = sessions.get(session_id)
    if not s:
        return
    if confidence is not None:
        s.scores.confidence = confidence
    if technical is not None:
        s.scores.technical = technical
    if engagement is not None:
        s.scores.engagement = engagement

async def sync_m2_scores(session_id: str):
    """Pull latest scores from M2 and write into M1 session."""
    session = get_session(session_id)
    if not session:
        return
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{M2_BASE_URL}/session/metrics")
            if resp.status_code == 200:
                m2 = resp.json()
                session.scores.confidence = m2.get("overall_confidence", session.scores.confidence)
                session.scores.engagement = m2.get("overall_engagement", session.scores.engagement)
                print(f"[M2 sync] confidence={session.scores.confidence} engagement={session.scores.engagement}")
    except Exception as e:
        print(f"[M2 sync] Could not reach M2 at {M2_BASE_URL}: {e}")
        # Don't crash — M1 keeps working with last known scores