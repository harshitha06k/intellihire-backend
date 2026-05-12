import pdfplumber
import json
import os
from dotenv import load_dotenv
from schema import ResumeData
from groq import Groq
import io

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_KEY"))

async def parse_resume(pdf_bytes: bytes) -> ResumeData:
    # Extract text from PDF
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        raw_text = "\n".join(
            page.extract_text() for page in pdf.pages if page.extract_text()
        )

    prompt = f"""You are a resume parser. Return ONLY valid JSON, no markdown, no backticks, no explanation.

Resume text:
{raw_text}

Return exactly this structure:
{{
  "skills": ["Python", "React"],
  "experience": ["2 years at TCS as backend engineer"],
  "projects": ["Built e-commerce platform"],
  "seniority": "mid",
  "inferred_roles": [
    {{"role": "Backend Engineer", "confidence": 0.9}},
    {{"role": "Full Stack Developer", "confidence": 0.7}}
  ]
}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    text = response.choices[0].message.content.strip()
    
    # Strip markdown if model adds it
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    
    data = json.loads(text.strip())
    return ResumeData(
        skills=data.get("skills", []),
        experience=data.get("experience", []),
        projects=data.get("projects", []),
        seniority=data.get("seniority", "mid"),
        inferred_roles=data.get("inferred_roles", []),
    )