from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

from resume_parser import parse_resume
from orchestrator import run_interview_turn
from report import generate_report, generate_pdf
from jobs import get_job_recommendations
from session_store import create_session, get_session

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

# Allow M3 frontend to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production, fine for hackathon
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/health")
async def health():
    return {"status": "ok", "module": "M1-Backend"}

# ── Resume parsing ──────────────────────────────────────────
@app.post("/parse-resume")
async def parse_resume_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    session = create_session()
    session.resume_data = await parse_resume(contents)
    return {
        "session_id": session.session_id,
        "resume_data": session.resume_data.__dict__,
    }

# ── Interview WebSocket ─────────────────────────────────────
@app.websocket("/interview")
async def interview_ws(websocket: WebSocket):
    await websocket.accept()
    session_id = await websocket.receive_text()   # first message = session_id
    session = get_session(session_id)
    if not session:
        await websocket.close(); return
    try:
        while True:
            answer = await websocket.receive_text()
            result = await run_interview_turn(session, answer)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass

# ── Job recommendations ─────────────────────────────────────
@app.get("/jobs")
async def jobs_endpoint(session_id: str):
    session = get_session(session_id)
    if not session:
        return {"error": "session not found"}
    return await get_job_recommendations(session.resume_data)

# ── Coaching report ─────────────────────────────────────────
@app.post("/report")
async def report_endpoint(body: dict):
    session = get_session(body["session_id"])
    if not session:
        return {"error": "session not found"}
    report = await generate_report(session)
    return report

@app.get("/report/pdf")
async def report_pdf(session_id: str):
    path = await generate_pdf(session_id)
    return FileResponse(path, media_type="application/pdf", filename="coaching_report.pdf")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
