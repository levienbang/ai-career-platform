import re
import unicodedata
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.db.models import RequirementType, Skill
from app.db.repositories import SkillRepository


class UnknownSkillsError(ValueError):
    def __init__(self, skills: list[str]) -> None:
        self.skills = skills
        super().__init__(f"Skills are not present in the database taxonomy: {', '.join(skills)}")


@dataclass(frozen=True)
class NormalizedJobSkill:
    skill: Skill
    requirement_type: RequirementType
    evidence_text: str


def skill_lookup_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", normalized).strip().casefold()


class SkillNormalizer:
    def __init__(self, session: Session) -> None:
        self._taxonomy: dict[str, Skill] = {}
        for skill in SkillRepository(session).list_with_aliases():
            self._taxonomy[skill_lookup_key(skill.canonical_name)] = skill
            for alias in skill.aliases:
                self._taxonomy[skill_lookup_key(alias.alias)] = skill

    def resolve(self, value: str) -> Skill | None:
        return self._taxonomy.get(skill_lookup_key(value))

    def evidence_mentions(self, skill: Skill, evidence: str) -> bool:
        for wording, candidate in self._taxonomy.items():
            if candidate.id == skill.id and re.search(
                rf"(?<!\w){re.escape(wording)}(?!\w)", skill_lookup_key(evidence)
            ):
                return True
        return False

    def normalize(
        self, required_skills: list[str], preferred_skills: list[str]
    ) -> list[NormalizedJobSkill]:
        normalized: list[NormalizedJobSkill] = []
        unknown: list[str] = []
        seen_skill_ids: set[int] = set()

        for requirement_type, values in (
            (RequirementType.REQUIRED, required_skills),
            (RequirementType.PREFERRED, preferred_skills),
        ):
            for evidence in values:
                skill = self._taxonomy.get(skill_lookup_key(evidence))
                if skill is None:
                    if evidence not in unknown:
                        unknown.append(evidence)
                    continue
                if skill.id in seen_skill_ids:
                    continue
                seen_skill_ids.add(skill.id)
                normalized.append(
                    NormalizedJobSkill(
                        skill=skill,
                        requirement_type=requirement_type,
                        evidence_text=evidence,
                    )
                )

        if unknown:
            raise UnknownSkillsError(unknown)
        return normalized
