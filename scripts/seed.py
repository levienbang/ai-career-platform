from hashlib import sha256

from sqlalchemy import select

from app.db.models import Company, Job, JobSkill, RequirementType, Skill, SkillAlias
from app.db.session import SessionLocal

SAMPLE_JOBS = [
    {
        "title": "AI Engineer Intern",
        "company": "Example AI Lab",
        "location": "Ho Chi Minh City",
        "employment_type": "Internship",
        "experience_years_min": 0,
        "description": "Build Python services and package them with Docker.",
        "source_url": "https://example.com/jobs/ai-engineer-intern",
        "skills": [("Python", "programming"), ("Docker", "platform")],
    },
    {
        "title": "Machine Learning Engineer",
        "company": "Example Data Company",
        "location": "Hanoi",
        "employment_type": "Full-time",
        "experience_years_min": 2,
        "description": "Develop machine-learning APIs using Python and FastAPI.",
        "source_url": "https://example.com/jobs/ml-engineer",
        "skills": [("Python", "programming"), ("FastAPI", "framework")],
    },
    {
        "title": "Computer Vision Intern",
        "company": "Vision Seed Studio",
        "location": "Ho Chi Minh City",
        "employment_type": "Internship",
        "experience_years_min": 0,
        "description": "Prototype image recognition and video analysis models with Python.",
        "source_url": "https://example.com/eval/computer-vision-intern",
        "skills": [
            ("Python", "programming"),
            ("Computer Vision", "domain"),
            ("Docker", "platform"),
        ],
    },
    {
        "title": "Junior Computer Vision Engineer",
        "company": "Camera Seed Labs",
        "location": "Hanoi",
        "employment_type": "Full-time",
        "experience_years_min": 0,
        "description": (
            "Build image detection services for candidates without production experience."
        ),
        "source_url": "https://example.com/eval/junior-computer-vision",
        "skills": [("Python", "programming"), ("Computer Vision", "domain")],
    },
    {
        "title": "NLP Engineer",
        "company": "Language Seed AI",
        "location": "Remote",
        "employment_type": "Full-time",
        "experience_years_min": 1,
        "description": (
            "Develop language-model retrieval applications and conversational AI pipelines."
        ),
        "source_url": "https://example.com/eval/nlp-engineer",
        "skills": [("Python", "programming"), ("LangChain", "framework")],
    },
    {
        "title": "AI Backend Engineer",
        "company": "Service Seed",
        "location": "Da Nang",
        "employment_type": "Full-time",
        "experience_years_min": 2,
        "description": "Build production APIs and relational data services for AI products.",
        "source_url": "https://example.com/eval/ai-backend-engineer",
        "skills": [
            ("Python", "programming"),
            ("FastAPI", "framework"),
            ("PostgreSQL", "database"),
        ],
    },
    {
        "title": "MLOps Engineer",
        "company": "Deploy Seed",
        "location": "Remote",
        "employment_type": "Full-time",
        "experience_years_min": 2,
        "description": "Package, deploy, and monitor machine-learning services in containers.",
        "source_url": "https://example.com/eval/mlops-engineer",
        "skills": [("Python", "programming"), ("Docker", "platform")],
    },
    {
        "title": "Data Scientist",
        "company": "Analytics Seed",
        "location": "Ho Chi Minh City",
        "employment_type": "Full-time",
        "experience_years_min": 1,
        "description": "Analyze product data, design experiments, and query relational datasets.",
        "source_url": "https://example.com/eval/data-scientist",
        "skills": [("Python", "programming"), ("PostgreSQL", "database")],
    },
    {
        "title": "AI Platform Engineer",
        "company": "Platform Seed",
        "location": "Ho Chi Minh City",
        "employment_type": "Full-time",
        "experience_years_min": 3,
        "description": "Operate containerized API platforms and databases for AI teams.",
        "source_url": "https://example.com/eval/ai-platform-engineer",
        "skills": [
            ("Docker", "platform"),
            ("FastAPI", "framework"),
            ("PostgreSQL", "database"),
        ],
    },
    {
        "title": "Machine Learning Research Intern",
        "company": "Research Seed Lab",
        "location": "Hanoi",
        "employment_type": "Internship",
        "experience_years_min": 0,
        "description": "Run machine-learning experiments and evaluate research prototypes.",
        "source_url": "https://example.com/eval/ml-research-intern",
        "skills": [("Python", "programming")],
    },
    {
        "title": "Backend Engineer Intern",
        "company": "Remote Seed Product",
        "location": "Remote",
        "employment_type": "Internship",
        "experience_years_min": 0,
        "description": "Develop Python web APIs for an early-stage AI product.",
        "source_url": "https://example.com/eval/backend-intern",
        "skills": [("Python", "programming"), ("FastAPI", "framework")],
    },
    {
        "title": "PostgreSQL Data Engineer",
        "company": "Database Seed",
        "location": "Hanoi",
        "employment_type": "Full-time",
        "experience_years_min": 2,
        "description": "Design reliable relational data pipelines and database-backed services.",
        "source_url": "https://example.com/eval/postgresql-data-engineer",
        "skills": [("Python", "programming"), ("PostgreSQL", "database")],
    },
]

TAXONOMY = {
    "Python": ("programming", ["python"]),
    "Docker": ("platform", ["docker"]),
    "FastAPI": ("framework", ["fast api", "fastapi"]),
    "PostgreSQL": ("database", ["postgres", "postgresql", "postgre sql"]),
    "LangChain": ("framework", ["lang chain", "langchain"]),
    "Computer Vision": ("domain", ["computer vision", "cv"]),
}


def seed() -> None:
    with SessionLocal.begin() as session:
        skills: dict[str, Skill] = {}
        for name, (category, _) in TAXONOMY.items():
            skill = session.scalar(select(Skill).where(Skill.canonical_name == name))
            if skill is None:
                skill = Skill(canonical_name=name, category=category)
                session.add(skill)
                session.flush()
            skills[name] = skill

        for canonical_name, (_, aliases) in TAXONOMY.items():
            for alias in aliases:
                if session.scalar(select(SkillAlias).where(SkillAlias.alias == alias)) is None:
                    session.add(SkillAlias(skill=skills[canonical_name], alias=alias))

        for item in SAMPLE_JOBS:
            content_hash = sha256(item["description"].encode()).hexdigest()
            if session.scalar(select(Job).where(Job.content_hash == content_hash)) is not None:
                continue
            company = session.scalar(select(Company).where(Company.name == item["company"]))
            if company is None:
                company = Company(name=item["company"], location=item["location"])
                session.add(company)
                session.flush()
            job = Job(
                title=item["title"],
                company=company,
                location=item["location"],
                employment_type=item["employment_type"],
                experience_years_min=item["experience_years_min"],
                description=item["description"],
                source_url=item["source_url"],
                content_hash=content_hash,
            )
            job.skills = [
                JobSkill(
                    skill=skills[name],
                    requirement_type=RequirementType.REQUIRED,
                    evidence_text=name,
                )
                for name, _ in item["skills"]
            ]
            session.add(job)


if __name__ == "__main__":
    seed()
    print("Sample data is ready.")
