import argparse
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.db.models import Company, Job, JobSkill, RequirementType, Skill, SkillAlias
from app.db.session import SessionLocal

SEED_DATA_FILE = Path(__file__).parents[1] / "data" / "seed_data.json"


def load_seed_data() -> dict[str, Any]:
    return json.loads(SEED_DATA_FILE.read_text(encoding="utf-8"))


def seed(*, taxonomy_only: bool = False) -> None:
    seed_data = load_seed_data()
    taxonomy = seed_data["taxonomy"]
    sample_jobs = [] if taxonomy_only else seed_data["jobs"]
    with SessionLocal.begin() as session:
        skills: dict[str, Skill] = {}
        for name, definition in taxonomy.items():
            skill = session.scalar(select(Skill).where(Skill.canonical_name == name))
            if skill is None:
                skill = Skill(
                    canonical_name=name, category=definition["category"], origin="curated"
                )
                session.add(skill)
                session.flush()
            else:
                skill.origin = "curated"
                skill.category = definition["category"]
            skills[name] = skill

        for canonical_name, definition in taxonomy.items():
            for alias in definition["aliases"]:
                if session.scalar(select(SkillAlias).where(SkillAlias.alias == alias)) is None:
                    session.add(SkillAlias(skill=skills[canonical_name], alias=alias))

        for item in sample_jobs:
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
                for name in item["skills"]
            ]
            session.add(job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed skills and optional demo jobs")
    parser.add_argument("--taxonomy-only", action="store_true", help="Seed skills and aliases only")
    args = parser.parse_args()
    seed(taxonomy_only=args.taxonomy_only)
    print("Taxonomy is ready." if args.taxonomy_only else "Sample data is ready.")
