import httpx
import os
import json
from dotenv import load_dotenv
from groq import Groq
from schema import ResumeData

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_KEY"))

ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_KEY    = os.getenv("ADZUNA_KEY")

async def get_job_recommendations(resume: ResumeData) -> list:
    # 1. Build search query from top skills + first inferred role
    role_query = resume.inferred_roles[0]["role"] if resume.inferred_roles else ""
    skills_query = " ".join(resume.skills[:3])
    query = f"{role_query} {skills_query}".strip()

    # 2. Fetch from Adzuna India
    jobs_raw = []
    try:
        url = "https://api.adzuna.com/v1/api/jobs/in/search/1"
        params = {
            "app_id":           ADZUNA_APP_ID,
            "app_key":          ADZUNA_KEY,
            "results_per_page": 20,
            "what":             query,
            "content-type":     "application/json",
        }
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.get(url, params=params)
            jobs_raw = resp.json().get("results", [])
    except Exception as e:
        print(f"Adzuna fetch error: {e}")
        jobs_raw = []

    if not jobs_raw:
        return []

    # 3. Score each job with Groq (limit to 10 to save quota)
    skills_text = ", ".join(resume.skills)
    scored_jobs = []

    for job in jobs_raw[:10]:
        title   = job.get("title", "Unknown Role")
        company = job.get("company", {}).get("display_name", "Unknown Company")
        desc    = job.get("description", "")[:400]
        url_job = job.get("redirect_url", "")

        prompt = f"""You are a job-fit analyser.
Return ONLY valid JSON, no markdown, no backticks.

Candidate skills: {skills_text}
Job title: {title}
Job description: {desc}

Return exactly:
{{
  "match_percent": 78,
  "why_fits": "One sentence explaining the strongest match reason",
  "skill_gaps": ["Kubernetes", "Go"]
}}

match_percent: integer 0-100
skill_gaps: up to 3 missing skills (empty list if none)"""

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            text = response.choices[0].message.content.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            match_data = json.loads(text.strip())
        except Exception:
            match_data = {
                "match_percent": 50,
                "why_fits":      "General skill overlap detected.",
                "skill_gaps":    [],
            }

        scored_jobs.append({
            "title":         title,
            "company":       company,
            "url":           url_job,
            "match_percent": match_data.get("match_percent", 50),
            "why_fits":      match_data.get("why_fits", ""),
            "skill_gaps":    match_data.get("skill_gaps", []),
        })

    # 4. Sort by match % descending
    scored_jobs.sort(key=lambda x: x["match_percent"], reverse=True)
    return scored_jobs