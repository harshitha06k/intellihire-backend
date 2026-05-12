from typing import List, Optional
from dataclasses import dataclass, field
import uuid

@dataclass
class Scores:
    confidence: float = 0.0      # M2 writes this (audio)
    technical: float = 0.0       # M1 writes this (evaluator)
    engagement: float = 0.0      # M2 writes this (video)

@dataclass
class ResumeData:
    skills: List[str] = field(default_factory=list)
    experience: List[str] = field(default_factory=list)
    projects: List[str] = field(default_factory=list)
    seniority: str = "mid"          # junior / mid / senior
    inferred_roles: List[dict] = field(default_factory=list)
    # inferred_roles format: [{"role": "Backend Engineer", "confidence": 0.87}]

@dataclass
class QuestionEntry:
    question: str = ""
    answer: str = ""
    technical_score: float = 0.0
    depth: str = "surface"          # surface / intermediate / deep
    missing_concepts: List[str] = field(default_factory=list)

@dataclass
class Session:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resume_data: ResumeData = field(default_factory=ResumeData)
    current_state: str = "WARM_UP"  # WARM_UP / CORE_SKILLS / DEEP_DIVE / STRESS_TEST / WRAP_UP
    question_history: List[QuestionEntry] = field(default_factory=list)
    scores: Scores = field(default_factory=Scores)
    candidate_name: str = ""
