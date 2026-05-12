import json
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_KEY"))

async def evaluate_answer(question: str, answer: str) -> dict:
    prompt = f"""You are a senior technical interviewer evaluating a candidate's answer.
Return ONLY valid JSON, no markdown, no backticks, no explanation.

Question: {question}
Answer: {answer}

Return exactly:
{{
  "score": 7,
  "depth": "intermediate",
  "missing": ["load balancing", "caching strategy"]
}}

Rules:
- score: integer 0-10 (0=completely wrong, 5=partial, 10=excellent)
- depth: exactly one of "surface" or "intermediate" or "deep"
- missing: list of up to 3 key concepts the candidate missed (empty list if answer was complete)"""

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

    try:
        return json.loads(text.strip())
    except Exception:
        # Safe fallback if parsing fails
        return {"score": 5, "depth": "surface", "missing": []}