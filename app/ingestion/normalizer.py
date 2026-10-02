import re
import unicodedata
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
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


def skill_match_key(value: str) -> str:
    return re.sub(r"[\s.\-_/]", "", skill_lookup_key(value))


class SkillNormalizer:
    def __init__(self, session: Session) -> None:
        self._session = session
        self._taxonomy: dict[str, Skill] = {}
        self._wordings: dict[int, set[str]] = {}
        for skill in SkillRepository(session).list_with_aliases():
            self._taxonomy.setdefault(skill_match_key(skill.canonical_name), skill)
            self._wordings[skill.id] = {skill_lookup_key(skill.canonical_name)}
            for alias in skill.aliases:
                self._taxonomy.setdefault(skill_match_key(alias.alias), skill)
                self._wordings[skill.id].add(skill_lookup_key(alias.alias))

    def resolve(self, value: str) -> Skill | None:
        return self._taxonomy.get(skill_match_key(value))

    def evidence_mentions(self, skill: Skill, evidence: str, *, name: str | None = None) -> bool:
        wordings = self._wordings.get(skill.id, set()).copy()
        # A resolved spelling still needs literal, word-boundary evidence in the source.
        if name is not None and self.resolve(name) is skill:
            wordings.add(skill_lookup_key(name))
        for wording in wordings:
            if re.search(rf"(?<!\w){re.escape(wording)}(?!\w)", skill_lookup_key(evidence)):
                return True
        return False

    def unknown_names(self, values: list[str], evidence_text: str) -> list[str]:
        """Only grounded, short names enter learning; retry undecided extracted skills."""
        names = []
        for value in values:
            name = " ".join(unicodedata.normalize("NFKC", value).split())
            skill = self.resolve(name)
            if skill is not None and skill.origin != "extracted":
                continue
            if not name or len(name) > 50 or len(name.split()) > 5:
                continue
            if re.search(
                rf"(?<!\w){re.escape(skill_lookup_key(name))}(?!\w)",
                skill_lookup_key(evidence_text),
            ):
                names.append(name)
        return names

    def normalize(
        self,
        required_skills: list[str],
        preferred_skills: list[str],
        *,
        evidence_text: str | None = None,
        create_missing: bool = True,
    ) -> list[NormalizedJobSkill]:
        normalized: list[NormalizedJobSkill] = []
        unknown: list[str] = []
        seen_skill_ids: set[int] = set()

        for requirement_type, values in (
            (RequirementType.REQUIRED, required_skills),
            (RequirementType.PREFERRED, preferred_skills),
        ):
            for evidence in values:
                name = " ".join(unicodedata.normalize("NFKC", evidence).split())
                if not name or len(name) > 100:
                    continue
                key = skill_lookup_key(name)
                skill = self.resolve(name)
                if skill is not None and evidence_text is not None:
                    if not self.evidence_mentions(skill, evidence_text, name=name):
                        continue
                if skill is None:
                    if len(name.split()) > 5 or len(name) > 50:
                        continue
                    if evidence_text is None or not re.search(
                        rf"(?<!\w){re.escape(key)}(?!\w)", skill_lookup_key(evidence_text)
                    ):
                        continue
                    if not create_missing:
                        if name not in unknown:
                            unknown.append(name)
                        continue
                    try:
                        with self._session.begin_nested():
                            skill = Skill(canonical_name=name, category=None, origin="extracted")
                            self._session.add(skill)
                            self._session.flush()
                    except IntegrityError:
                        skill = self._session.scalar(
                            select(Skill).where(func.lower(Skill.canonical_name) == name.lower())
                        )
                        if skill is None:
                            raise
                    self._taxonomy[skill_match_key(name)] = skill
                    self._wordings[skill.id] = {key}
                if skill.id in seen_skill_ids:
                    continue
                seen_skill_ids.add(skill.id)
                normalized.append(
                    NormalizedJobSkill(
                        skill=skill,
                        requirement_type=requirement_type,
                        evidence_text=name,
                    )
                )

        if unknown:
            raise UnknownSkillsError(unknown)
        return normalized
