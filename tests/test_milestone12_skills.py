import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from app.db.models import Job, JobSkill, RequirementType, Skill, SkillAlias
from app.ingestion.normalizer import SkillNormalizer, skill_match_key
from app.ingestion.pipeline import JobIngestionPipeline
from app.ingestion.structured import parse_skill_list
from app.schemas.cv import CVExtraction, CVSkillClaim
from app.services.skill_gap import analyze_skill_gap
from app.services.skill_taxonomy import (
    apply_skill_merges,
    load_skill_aliases,
    plan_skill_merges,
    seed_taxonomy,
)
from scripts import merge_skills as merge_module
from scripts import seed as seed_module


@pytest.mark.parametrize(
    "name,key",
    [
        ("Node.js", "nodejs"),
        ("NodeJS", "nodejs"),
        ("node js", "nodejs"),
        ("Power BI", "powerbi"),
        ("PowerBI", "powerbi"),
        ("PL/SQL", "plsql"),
        ("PLSQL", "plsql"),
        ("C++", "c++"),
        ("C#", "c#"),
        ("C", "c"),
        (".NET", "net"),
        ("Scikit-learn", "scikitlearn"),
        (" Ｐｏｗｅｒ＿ＢＩ\t", "powerbi"),
    ],
)
def test_match_key(name, key):
    assert skill_match_key(name) == key


def test_resolver_reuses_skills_but_requires_literal_evidence(db_session):
    node = Skill(canonical_name="Node.js", origin="curated")
    power = Skill(canonical_name="Power BI", origin="extracted")
    db_session.add_all([node, power])
    db_session.commit()
    normalizer = SkillNormalizer(db_session)
    assert normalizer.resolve("Nodejs") is node
    assert normalizer.resolve("node js") is node
    result = normalizer.normalize(["PowerBI"], [], evidence_text="Build PowerBI reports")
    assert result[0].skill.id == power.id
    assert db_session.scalar(select(func.count()).select_from(Skill)) == 2
    assert normalizer.normalize(["node js"], [], evidence_text="Python only") == []
    assert normalizer.normalize(["PowerBI"], [], evidence_text="PowerBIography") == []
    assert normalizer.normalize(["node js"], [], evidence_text="Use node js")


def test_alias_file_has_unique_owners():
    definitions = load_skill_aliases()
    assert 40 <= len(definitions) <= 60
    canonical_keys = [skill_match_key(name) for name in definitions]
    assert len(set(canonical_keys)) == len(canonical_keys)
    owners = {}
    for canonical, definition in definitions.items():
        assert definition["category"] is None or isinstance(definition["category"], str)
        for name in [canonical, *definition["aliases"]]:
            assert isinstance(name, str)
            key = skill_match_key(name)
            assert owners.setdefault(key, canonical) == canonical
            if key in {"r", "c", "go"}:
                assert name == canonical


def add_job(session, skills, number=1):
    job = Job(title="AI", description="Scikit-learn", content_hash=f"{number:064x}")
    job.skills = [JobSkill(skill=skill, requirement_type=kind) for skill, kind in skills]
    session.add(job)
    session.commit()
    return job


def test_merge_dry_run_apply_aliases_collisions_and_repeat(db_session, monkeypatch, capsys):
    canonical = Skill(canonical_name="Scikit-learn", origin="curated")
    sklearn = Skill(canonical_name="sklearn", origin="extracted")
    sklearn.aliases = [SkillAlias(alias="sklearn legacy")]
    node = Skill(canonical_name="Node.js", origin="curated")
    nodejs = Skill(canonical_name="Nodejs", origin="extracted")
    db_session.add_all([canonical, sklearn, node, nodejs])
    db_session.commit()
    job = add_job(
        db_session,
        [
            (canonical, RequirementType.PREFERRED),
            (sklearn, RequirementType.REQUIRED),
            (node, RequirementType.REQUIRED),
            (nodejs, RequirementType.REQUIRED),
        ],
    )
    second = add_job(db_session, [(sklearn, RequirementType.REQUIRED)], 2)
    node_id, canonical_id = node.id, canonical.id
    monkeypatch.setattr(merge_module, "SessionLocal", sessionmaker(bind=db_session.get_bind()))
    assert merge_module.merge_skills() == 2
    assert "Dry-run" in capsys.readouterr().out
    assert db_session.scalar(select(func.count()).select_from(Skill)) == 4
    assert db_session.scalar(select(func.count()).select_from(JobSkill)) == 5
    db_session.commit()
    assert merge_module.merge_skills(apply=True) == 2
    assert "scripts.index_jobs" in capsys.readouterr().out
    db_session.expire_all()
    assert db_session.scalar(select(func.count()).select_from(Skill)) == 2
    links = list(db_session.scalars(select(JobSkill).order_by(JobSkill.job_id, JobSkill.skill_id)))
    assert {(link.job_id, link.skill_id, link.requirement_type) for link in links} == {
        (job.id, canonical_id, RequirementType.REQUIRED),
        (job.id, node_id, RequirementType.REQUIRED),
        (second.id, canonical_id, RequirementType.REQUIRED),
    }
    aliases = {alias.alias: alias.skill_id for alias in db_session.scalars(select(SkillAlias))}
    assert aliases == {"sklearn": canonical_id, "sklearn legacy": canonical_id, "Nodejs": node_id}
    db_session.commit()
    assert merge_module.merge_skills(apply=True) == 0


def test_merge_survivor_priority_and_transaction_rollback(db_session):
    first = Skill(canonical_name="Other.Tool", origin="extracted")
    popular = Skill(canonical_name="OtherTool", origin="extracted")
    curated = Skill(canonical_name="other-tool", origin="curated")
    db_session.add_all([first, popular, curated])
    db_session.commit()
    add_job(db_session, [(popular, RequirementType.REQUIRED)])
    assert plan_skill_merges(db_session, {})[0].keep_id == curated.id
    curated.origin = "extracted"
    db_session.flush()
    assert plan_skill_merges(db_session, {})[0].keep_id == popular.id
    db_session.rollback()
    plan = plan_skill_merges(db_session, {})
    apply_skill_merges(db_session, plan)
    db_session.rollback()
    assert db_session.scalar(select(func.count()).select_from(Skill)) == 3
    assert db_session.scalar(select(func.count()).select_from(JobSkill)) == 1


def test_seed_taxonomy_merges_existing_extracted_and_is_idempotent(db_session, monkeypatch):
    extracted = Skill(canonical_name="sklearn", origin="extracted")
    db_session.add(extracted)
    db_session.commit()
    job = add_job(db_session, [(extracted, RequirementType.REQUIRED)])
    monkeypatch.setattr(seed_module, "SessionLocal", sessionmaker(bind=db_session.get_bind()))
    seed_module.seed(taxonomy_only=True)
    db_session.expire_all()
    normalizer = SkillNormalizer(db_session)
    skill = normalizer.resolve("sklearn")
    assert skill.canonical_name == "Scikit-learn" and skill.origin == "curated"
    assert job.skills[0].skill_id == skill.id
    assert normalizer.resolve("node js").canonical_name == "Node.js"
    first = (
        db_session.scalar(select(func.count()).select_from(Skill)),
        db_session.scalar(select(func.count()).select_from(SkillAlias)),
    )
    db_session.commit()
    seed_module.seed(taxonomy_only=True)
    assert first == (
        db_session.scalar(select(func.count()).select_from(Skill)),
        db_session.scalar(select(func.count()).select_from(SkillAlias)),
    )
    assert db_session.scalar(select(func.count()).select_from(Job)) == 1
    profile = CVExtraction(skills=[CVSkillClaim(name="Scikit-learn", evidence="Scikit-learn")])
    gap = analyze_skill_gap(db_session, profile, "Used Scikit-learn", [job.id])
    assert gap.jobs[0].matched_skills == ["Scikit-learn"]
    assert gap.jobs[0].missing_required_skills == []
    assert gap.jobs[0].fit_score == 1.0


@pytest.mark.parametrize("as_string", [False, True])
def test_parenthesized_skill_enumeration(as_string):
    phrase = "Thư viện học máy (Scikit-learn, TensorFlow, Keras, PyTorch)"
    value = phrase + "; Docker" if as_string else [phrase, "Docker", "Keras"]
    assert parse_skill_list(value) == ["Scikit-learn", "TensorFlow", "Keras", "PyTorch", "Docker"]
    assert parse_skill_list(["SQL (advanced)", "C++"]) == ["SQL (advanced)", "C++"]


def test_sentence_skills_dropped_without_rejecting_job(db_session, fake_extractor):
    seed_taxonomy(db_session, load_skill_aliases())
    db_session.commit()
    sentence = "one two three four five six"
    long = "x" * 51
    result = JobIngestionPipeline(db_session, fake_extractor).import_records(
        [
            {
                "title": "AI",
                "description": f"{sentence} {long}. Python.",
                "technical_skills": [sentence, long, "Python"],
            },
            {"title": "Empty skills", "description": sentence, "technical_skills": [sentence]},
        ]
    )
    assert result.inserted == 2 and result.failed == 0
    assert SkillNormalizer(db_session).resolve(sentence) is None
    assert SkillNormalizer(db_session).resolve(long) is None
    jobs = list(db_session.scalars(select(Job).order_by(Job.id)))
    assert [link.skill.canonical_name for link in jobs[0].skills] == ["Python"]
    assert jobs[1].skills == []
    curated_sentence = Skill(canonical_name=sentence, origin="curated")
    db_session.add(curated_sentence)
    db_session.commit()
    assert SkillNormalizer(db_session).normalize([sentence], [], evidence_text=sentence)


def test_extracted_length_boundaries_and_same_request_reuse(db_session):
    normalizer = SkillNormalizer(db_session)
    values = ["one two three four five", "x" * 50, "Fresh.Tool", "FreshTool"]
    result = normalizer.normalize(values, [], evidence_text=" ".join(values))
    assert len(result) == 3
    assert all(item.skill.origin == "extracted" for item in result)


def test_merge_canonical_priority_ties_and_preferred_duplicates(db_session):
    canonical = Skill(canonical_name="Scikit-learn", origin="extracted")
    alias = Skill(canonical_name="sklearn", origin="curated")
    db_session.add_all([canonical, alias])
    db_session.commit()
    first = add_job(
        db_session, [(canonical, RequirementType.PREFERRED), (alias, RequirementType.PREFERRED)]
    )
    second = add_job(db_session, [(alias, RequirementType.REQUIRED)], 2)
    plan = plan_skill_merges(db_session, load_skill_aliases())
    assert plan[0].keep_id == canonical.id
    apply_skill_merges(db_session, plan)
    assert len(first.skills) == 1 and first.skills[0].requirement_type == RequirementType.PREFERRED
    assert second.skills[0].skill_id == canonical.id
    db_session.commit()
    low = Skill(canonical_name="Tie.Tool", origin="extracted")
    high = Skill(canonical_name="TieTool", origin="extracted")
    db_session.add_all([low, high])
    db_session.commit()
    assert plan_skill_merges(db_session, {})[0].keep_id == low.id


def test_cv_match_sklearn_job_has_full_skill_score_after_seed(db_session):
    from types import SimpleNamespace

    from app.config import Settings
    from app.retrieval.types import SearchHit
    from app.services.cv_match import CVMatchService

    extracted = Skill(canonical_name="sklearn", origin="extracted")
    db_session.add(extracted)
    db_session.commit()
    job = add_job(db_session, [(extracted, RequirementType.REQUIRED)])
    seed_taxonomy(db_session, load_skill_aliases())
    db_session.commit()
    profile = CVExtraction(skills=[CVSkillClaim(name="Scikit-learn", evidence="Scikit-learn")])
    hit = SearchHit(
        job_id=job.id,
        title="AI",
        score=0.5,
        company=None,
        location=None,
        employment_type=None,
        experience_years_min=None,
        required_skills=["Scikit-learn"],
        preferred_skills=[],
    )
    service = CVMatchService(
        db_session,
        SimpleNamespace(load=lambda _c, **_kw: "Used Scikit-learn"),
        SimpleNamespace(extract=lambda _t: profile),
        SimpleNamespace(search=lambda _q, **_kw: [hit]),
        Settings(_env_file=None),
    )
    match = service.invoke(b"%PDF-test", limit=1, max_pages=5).jobs[0]
    assert match.matched_skills == ["Scikit-learn"]
    assert match.missing_required_skills == []
    assert match.skill_score == 1.0


def test_parenthesized_source_import_creates_individual_skills(db_session, fake_extractor):
    phrase = "Thư viện học máy (Scikit-learn, TensorFlow, Keras, PyTorch)"
    result = JobIngestionPipeline(db_session, fake_extractor).import_records(
        [{"title": "ML", "description": "Develop machine learning", "technical_skills": [phrase]}]
    )
    assert result.inserted == 1 and result.failed == 0
    names = set(db_session.scalars(select(Skill.canonical_name)))
    assert names == {"Scikit-learn", "TensorFlow", "Keras", "PyTorch"}
