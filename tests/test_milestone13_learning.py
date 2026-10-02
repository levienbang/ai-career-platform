import json
from decimal import Decimal
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.db.models import Job, JobSkill, RequirementType, Skill, SkillAlias, SkillDecision
from app.ingestion.normalizer import SkillNormalizer, skill_match_key
from app.ingestion.pipeline import JobIngestionPipeline
from app.services.skill_learning import SkillAliasLearner
from app.services.skill_taxonomy import (
    SkillMergeGroup,
    apply_skill_merges,
    load_skill_aliases,
    plan_skill_merges,
)
from scripts import learn_skill_aliases as learn_script
from scripts import review_skill_decisions as review_script


class FakeModel:
    def __init__(self, choose=None, error=None):
        self.choose = choose or (lambda name: {"decision": "new", "skill_id": None})
        self.error = error
        self.calls = []

    def invoke(self, messages):
        payload = json.loads(messages[1]["content"])
        self.calls.append(payload)
        if self.error:
            raise self.error
        return {
            "items": [
                dict(
                    name=entry["name"],
                    confidence=0.95,
                    category="tool",
                    reason="Alternative spelling",
                    **self.choose(entry["name"]),
                )
                for entry in payload
            ]
        }


def make_pipeline(session, extractor, model, **config):
    settings = Settings(_env_file=None, skill_alias_learning=True, **config)
    learner = SkillAliasLearner(session, settings, model=model)
    return JobIngestionPipeline(session, extractor, settings, alias_learner=learner), learner


def record(name, number=1):
    return {
        "title": f"Job {number}",
        "description": f"Use {name} for project {number}",
        "technical_skills": [name],
    }


def decision(session, name):
    return session.scalar(
        select(SkillDecision).where(SkillDecision.match_key == skill_match_key(name))
    )


def test_import_learns_alias_reuses_it_and_cv_match(db_session, fake_extractor, client):
    canonical = Skill(canonical_name="Scikit-learn", origin="curated")
    db_session.add(canonical)
    db_session.commit()
    model = FakeModel(lambda _: {"decision": "same_as", "skill_id": canonical.id})
    pipeline, learner = make_pipeline(db_session, fake_extractor, model)
    assert pipeline.import_records([record("Scikit learn lib")]).inserted == 1
    alias = db_session.scalar(select(SkillAlias))
    assert (alias.source, alias.confidence) == ("llm", Decimal("0.95"))
    assert decision(db_session, "Scikit learn lib").decision == "alias"
    job = db_session.scalar(select(Job))
    assert job.skills[0].skill_id == canonical.id
    db_session.commit()
    assert pipeline.import_records([record("Scikit learn lib", 2)]).inserted == 1
    assert learner.llm_calls == len(model.calls) == 1
    # Exercise the CV service used by /cv/match with deterministic search and extraction.
    from app.retrieval.types import SearchHit
    from app.schemas.cv import CVExtraction, CVSkillClaim
    from app.services.cv_match import CVMatchService

    profile = CVExtraction(skills=[CVSkillClaim(name="Scikit-learn", evidence="Scikit-learn")])
    hit = SearchHit(
        job_id=job.id,
        title=job.title,
        score=0.5,
        company=None,
        location=None,
        employment_type=None,
        experience_years_min=None,
        required_skills=["Scikit-learn"],
        preferred_skills=[],
    )
    match = (
        CVMatchService(
            db_session,
            SimpleNamespace(load=lambda _c, **_kw: "Used Scikit-learn"),
            SimpleNamespace(extract=lambda _t: profile),
            SimpleNamespace(search=lambda _q, **_kw: [hit]),
            Settings(_env_file=None),
        )
        .invoke(b"%PDF-test", limit=1, max_pages=5)
        .jobs[0]
    )
    assert match.matched_skills == ["Scikit-learn"] and match.skill_score == 1.0
    assert match.missing_required_skills == []
    from io import BytesIO

    from app.api.cv import get_cv_match_service
    from app.main import app

    service = CVMatchService(
        db_session,
        SimpleNamespace(load=lambda _c, **_kw: "Used Scikit-learn"),
        SimpleNamespace(extract=lambda _t: profile),
        SimpleNamespace(search=lambda _q, **_kw: [hit]),
        Settings(_env_file=None),
    )
    app.dependency_overrides[get_cv_match_service] = lambda: service
    response = client.post(
        "/cv/match",
        files={"file": ("cv.pdf", BytesIO(b"%PDF-test"), "application/pdf")},
        data={"limit": "1"},
    )
    assert response.status_code == 200
    assert response.json()["jobs"][0]["matched_skills"] == ["Scikit-learn"]
    assert response.json()["jobs"][0]["skill_score"] == 1.0


@pytest.mark.parametrize("outcome", ["new", "low", "invalid", "blocked"])
def test_new_and_pending_are_cached(db_session, fake_extractor, outcome):
    canonical = Skill(canonical_name="React", origin="curated")
    db_session.add(canonical)
    db_session.commit()
    name = "React Native" if outcome == "blocked" else "UnknownTool"

    class Model(FakeModel):
        def invoke(self, messages):
            result = super().invoke(messages)
            item = result["items"][0]
            if outcome != "new":
                item.update(decision="same_as", skill_id=canonical.id)
            if outcome == "low":
                item["confidence"] = 0.6
            if outcome == "invalid":
                item["skill_id"] = 999
            return result

    model = Model()
    pipeline, learner = make_pipeline(db_session, fake_extractor, model)
    assert pipeline.import_records([record(name)]).inserted == 1
    item = decision(db_session, name)
    assert item.decision == ("new" if outcome == "new" else "pending")
    skill = db_session.get(Skill, item.skill_id)
    assert skill.origin == "extracted" and skill.canonical_name == name
    if outcome == "new":
        assert skill.category == "tool"
    assert db_session.scalar(select(func.count()).select_from(SkillAlias)) == 0
    db_session.commit()
    assert pipeline.import_records([record(name, 2)]).inserted == 1
    assert learner.llm_calls == len(model.calls) == 1


@pytest.mark.parametrize("bad_output", ["exception", "category", "missing", "duplicate", "reason"])
def test_failed_llm_falls_back_without_caching_and_logs_safely(
    db_session, fake_extractor, caplog, bad_output
):
    class Model(FakeModel):
        def invoke(self, messages):
            result = super().invoke(messages)
            if bad_output == "exception":
                raise TimeoutError("private job text sk-secret123")
            if bad_output == "category":
                result["items"][0]["category"] = "invalid"
            elif bad_output == "missing":
                result["items"] = []
            elif bad_output == "duplicate":
                result["items"] *= 2
            elif bad_output == "reason":
                result["items"][0]["reason"] = "x" * 201
            return result

    model = Model()
    pipeline, learner = make_pipeline(db_session, fake_extractor, model)
    with caplog.at_level("WARNING"):
        assert pipeline.import_records([record("UnknownTool")]).inserted == 1
    assert "SkillAliasLearner" in caplog.text
    assert "private job text" not in caplog.text and "sk-secret123" not in caplog.text
    assert db_session.scalar(select(func.count()).select_from(SkillDecision)) == 0
    assert SkillNormalizer(db_session).resolve("UnknownTool").origin == "extracted"
    db_session.commit()
    pipeline.import_records([record("UnknownTool", 2)])
    assert learner.llm_calls == 2


def test_request_batches_and_evidence_filtering(db_session, fake_extractor):
    model = FakeModel()
    pipeline, learner = make_pipeline(db_session, fake_extractor, model, skill_alias_batch_size=2)
    values = ["Tool A", "Tool B", "Tool C", "Tool D", "Tool E"]
    records = [record(name, index) for index, name in enumerate(values)]
    records.extend(
        [record("Tool A", 100), record("one two three four five six", 101), record("x" * 51, 102)]
    )
    assert pipeline.import_records(records).inserted == 8
    assert [len(call) for call in model.calls] == [2, 2, 1]
    assert learner.llm_calls == 3
    db_session.commit()
    pipeline.import_records(
        [{"title": "Ungrounded", "description": "Python work", "required_skills": ["Hallucinated"]}]
    )
    assert learner.llm_calls == 3
    assert SkillNormalizer(db_session).resolve("Hallucinated") is None


def test_candidates_all_curated_plus_ten_nearest_and_disabled_learning(db_session, fake_extractor):
    curated = [Skill(canonical_name=f"Curated {n}", origin="curated") for n in range(15)]
    extracted = [Skill(canonical_name=f"Tool{n}", origin="extracted") for n in range(20)]
    db_session.add_all(curated + extracted)
    db_session.commit()
    model = FakeModel()
    pipeline, _ = make_pipeline(db_session, fake_extractor, model)
    pipeline.import_records([record("Tool22")])
    ids = {candidate["skill_id"] for candidate in model.calls[0][0]["candidates"]}
    assert {skill.id for skill in curated} <= ids
    assert len(ids - {skill.id for skill in curated}) == 10
    db_session.commit()
    disabled = JobIngestionPipeline(
        db_session, fake_extractor, Settings(_env_file=None, skill_alias_learning=False)
    )
    disabled.import_records([record("NewName", 2)])
    assert decision(db_session, "NewName") is None


def add_link(session, skill, evidence, number):
    job = Job(title=f"Job {number}", description=evidence, content_hash=f"{number:064x}")
    job.skills = [
        JobSkill(skill=skill, requirement_type=RequirementType.REQUIRED, evidence_text=evidence)
    ]
    session.add(job)
    session.commit()
    return job


def test_existing_learning_dry_run_apply_and_other_cwd(db_session, monkeypatch, tmp_path, capsys):
    canonical = Skill(canonical_name="Scikit-learn")
    old = Skill(canonical_name="Scikit learn lib", origin="extracted")
    db_session.add_all([canonical, old])
    db_session.commit()
    job = add_link(db_session, old, "Scikit learn lib", 1)
    canonical_id = canonical.id
    model = FakeModel(lambda _: {"decision": "same_as", "skill_id": canonical_id})
    monkeypatch.setattr(learn_script, "SessionLocal", sessionmaker(bind=db_session.get_bind()))
    monkeypatch.chdir(tmp_path)
    settings = Settings(_env_file=None)
    assert learn_script.learn_skill_aliases(model=model, settings=settings) == 1
    assert "Dry-run" in capsys.readouterr().out
    assert db_session.scalar(select(func.count()).select_from(Skill)) == 2
    assert db_session.scalar(select(func.count()).select_from(SkillDecision)) == 0
    db_session.commit()
    assert learn_script.learn_skill_aliases(apply=True, model=model, settings=settings) == 1
    db_session.expire_all()
    assert job.skills[0].skill_id == canonical_id
    assert db_session.scalar(select(SkillAlias)).source == "llm"
    assert decision(db_session, "Scikit learn lib").decision == "alias"
    db_session.commit()
    assert learn_script.learn_skill_aliases(apply=True, model=model, settings=settings) == 0


def test_review_approve_keep_new_revert_and_rejected_cache(db_session, monkeypatch, tmp_path):
    canonical = Skill(canonical_name="Scikit-learn")
    pending = Skill(canonical_name="Scikit learn lib", origin="extracted")
    distinct = Skill(canonical_name="Separate tool", origin="extracted")
    db_session.add_all([canonical, pending, distinct])
    db_session.commit()
    changed = add_link(db_session, pending, "Scikit learn lib", 1)
    unaffected = add_link(db_session, canonical, "Scikit-learn", 2)
    for skill in [pending, distinct]:
        db_session.add(
            SkillDecision(
                match_key=skill_match_key(skill.canonical_name),
                name=skill.canonical_name,
                decision="pending",
                skill_id=skill.id,
            )
        )
    db_session.commit()
    monkeypatch.setattr(review_script, "SessionLocal", sessionmaker(bind=db_session.get_bind()))
    monkeypatch.chdir(tmp_path)
    review_script.review_skill_decisions("approve", "Scikit learn lib", canonical="Scikit-learn")
    db_session.expire_all()
    assert changed.skills[0].skill_id == canonical.id
    assert decision(db_session, "Scikit learn lib").decision == "alias"
    db_session.commit()
    review_script.review_skill_decisions("keep-new", "Separate tool")
    db_session.expire_all()
    assert decision(db_session, "Separate tool").decision == "new"
    db_session.commit()
    review_script.review_skill_decisions("list", decision="alias")
    review_script.review_skill_decisions("revert", "Scikit learn lib")
    db_session.expire_all()
    assert changed.skills[0].skill.canonical_name == "Scikit learn lib"
    assert unaffected.skills[0].skill_id == canonical.id
    assert decision(db_session, "Scikit learn lib").decision == "rejected"
    assert db_session.scalar(select(SkillAlias)) is None
    db_session.commit()
    model = FakeModel()
    assert (
        SkillAliasLearner(
            db_session, Settings(_env_file=None), model=model, data_dir=learn_script.DATA_DIR
        ).propose(["Scikit learn lib"])
        == []
    )
    assert model.calls == []


@pytest.mark.parametrize("source", ["curated", "merge"])
def test_revert_refuses_non_llm_alias(db_session, source):
    from app.services.skill_review import revert_alias

    canonical = Skill(canonical_name="Python")
    db_session.add(canonical)
    db_session.flush()
    db_session.add(SkillAlias(skill_id=canonical.id, alias="Py", source=source))
    db_session.commit()
    with pytest.raises(ValueError, match="source='llm'"):
        revert_alias(db_session, "Py")


def test_merge_blocklist_and_canonical_display(db_session):
    java = Skill(canonical_name="Java")
    js = Skill(canonical_name="JavaScript", origin="extracted")
    elastic = Skill(canonical_name="Elastic Search", origin="curated")
    alias = Skill(canonical_name="ElasticSearch", origin="extracted")
    db_session.add_all([java, js, elastic, alias])
    db_session.commit()
    wrong = {"Java": {"category": "tool", "aliases": ["JavaScript"]}}
    assert all(
        group.keep_id not in (java.id, js.id) for group in plan_skill_merges(db_session, wrong)
    )
    with pytest.raises(ValueError, match="blocked"):
        apply_skill_merges(
            db_session, [SkillMergeGroup(java.id, "Java", (js.id,), ("JavaScript",), 0)]
        )
    plan = plan_skill_merges(db_session, load_skill_aliases())
    assert plan[0].keep_name == "Elasticsearch"
    survivor_id = plan[0].keep_id
    apply_skill_merges(db_session, plan)
    assert db_session.get(Skill, survivor_id).canonical_name == "Elasticsearch"
    aliases = {item.alias: item.source for item in db_session.scalars(select(SkillAlias))}
    assert aliases["Elastic Search"] == "merge" and aliases["ElasticSearch"] == "merge"
    assert SkillNormalizer(db_session).resolve("Elastic Search").id == survivor_id


@pytest.mark.parametrize(
    "first,second",
    [
        (first, second)
        for pair in json.loads((learn_script.DATA_DIR / "skill_alias_blocklist.json").read_text())
        for first, second in (pair, list(reversed(pair)))
    ],
)
def test_learning_blocks_all_pairs_in_both_directions(db_session, first, second):
    target = Skill(canonical_name=second)
    db_session.add(target)
    db_session.commit()
    model = FakeModel(lambda _: {"decision": "same_as", "skill_id": target.id})
    learner = SkillAliasLearner(db_session, Settings(_env_file=None), model=model)
    proposal = learner.propose([first])[0]
    assert proposal.decision == "pending"
    learner.apply([proposal])
    assert db_session.scalar(select(SkillAlias)) is None
    assert decision(db_session, first).decision == "pending"


def test_batch_aliases_can_target_a_skill_merged_earlier(db_session):
    target = Skill(canonical_name="Tool", origin="curated")
    intermediate = Skill(canonical_name="Tool old", origin="extracted")
    other = Skill(canonical_name="Tool older", origin="extracted")
    db_session.add_all([target, intermediate, other])
    db_session.commit()
    model = FakeModel(
        lambda name: {
            "decision": "same_as",
            "skill_id": (target.id if name == "Tool old" else intermediate.id),
        }
    )
    learner = SkillAliasLearner(db_session, Settings(_env_file=None), model=model)
    proposals = learner.propose(["Tool old", "Tool older"])
    # Applying either direction must not refer to a now-deleted target ID.
    learner.apply(sorted(proposals, key=lambda item: item.name != "Tool old"))
    assert SkillNormalizer(db_session).resolve("Tool older").id == target.id
    assert decision(db_session, "Tool older").skill_id == target.id


def test_revert_preserves_collision_evidence(db_session):
    from app.services.skill_review import revert_alias

    canonical = Skill(canonical_name="Tool")
    old = Skill(canonical_name="Tool lib", origin="extracted")
    db_session.add_all([canonical, old])
    db_session.commit()
    job = Job(title="Both claims", description="Tool; Tool lib", content_hash="a" * 64)
    job.skills = [
        JobSkill(skill=canonical, requirement_type=RequirementType.PREFERRED, evidence_text="Tool"),
        JobSkill(skill=old, requirement_type=RequirementType.REQUIRED, evidence_text="Tool lib"),
    ]
    db_session.add(job)
    db_session.commit()
    learner = SkillAliasLearner(
        db_session,
        Settings(_env_file=None),
        model=FakeModel(lambda _: {"decision": "same_as", "skill_id": canonical.id}),
    )
    learner.apply(learner.propose(["Tool lib"]))
    assert len(job.skills) == 1
    assert "Tool" in job.skills[0].evidence_text and "Tool lib" in job.skills[0].evidence_text
    revert_alias(db_session, "Tool lib")
    assert {link.skill.canonical_name for link in job.skills} == {"Tool", "Tool lib"}


def test_revert_alias_containing_canonical_does_not_keep_false_claim(db_session):
    from app.services.skill_review import revert_alias

    canonical = Skill(canonical_name="Tool")
    db_session.add(canonical)
    db_session.commit()
    learner = SkillAliasLearner(
        db_session,
        Settings(_env_file=None),
        model=FakeModel(lambda _: {"decision": "same_as", "skill_id": canonical.id}),
    )
    learner.apply(learner.propose(["Tool lib"]))
    db_session.commit()
    job = add_link(db_session, canonical, "Tool lib", 1)
    revert_alias(db_session, "Tool lib")
    assert [link.skill.canonical_name for link in job.skills] == ["Tool lib"]


def test_seed_alias_provenance_and_learning_canonical_display(db_session):
    from app.services.skill_taxonomy import seed_taxonomy

    old = Skill(canonical_name="Elastic Search", origin="curated")
    db_session.add(old)
    db_session.commit()
    learner = SkillAliasLearner(
        db_session,
        Settings(_env_file=None),
        model=FakeModel(lambda _: {"decision": "same_as", "skill_id": old.id}),
    )
    learner.apply(learner.propose(["ES legacy lib"]))
    aliases = {alias.alias: alias.source for alias in db_session.scalars(select(SkillAlias))}
    assert old.canonical_name == "Elasticsearch"
    assert aliases == {"Elastic Search": "merge", "ES legacy lib": "llm"}
    seed_taxonomy(db_session, load_skill_aliases())
    assert all(
        alias.source == "curated"
        for alias in db_session.scalars(select(SkillAlias))
        if alias.alias in {"sklearn", "elastic search"}
    )


def test_same_as_itself_is_kept_as_its_own_skill(db_session):
    java = Skill(canonical_name="Java", origin="extracted")
    db_session.add(java)
    db_session.commit()
    model = FakeModel(lambda name: {"decision": "same_as", "skill_id": java.id})
    learner = SkillAliasLearner(db_session, Settings(_env_file=None), model=model)
    (proposal,) = learner.propose(["Java"])
    assert (proposal.decision, proposal.skill_id) == ("new", None)


@pytest.mark.parametrize(("confidence", "expected"), [(0.75, "new"), (0.6, "pending")])
def test_new_skill_uses_its_own_lower_threshold(db_session, confidence, expected):
    class LowConfidence(FakeModel):
        def invoke(self, messages):
            output = super().invoke(messages)
            for item in output["items"]:
                item["confidence"] = confidence
            return output

    learner = SkillAliasLearner(db_session, Settings(_env_file=None), model=LowConfidence())
    (proposal,) = learner.propose(["Brand New Tool"])
    assert proposal.decision == expected


def test_opposite_alias_decisions_collapse_to_one_canonical(db_session):
    excel = Skill(canonical_name="Excel", origin="extracted")
    ms_excel = Skill(canonical_name="MS Excel", origin="extracted")
    job = Job(title="Analyst", description="Excel and MS Excel", content_hash="c" * 64)
    db_session.add_all([excel, ms_excel, job])
    db_session.flush()
    db_session.add_all(
        [
            JobSkill(job_id=job.id, skill_id=excel.id, requirement_type=RequirementType.REQUIRED),
            JobSkill(
                job_id=job.id, skill_id=ms_excel.id, requirement_type=RequirementType.PREFERRED
            ),
        ]
    )
    db_session.commit()
    ids = {"Excel": ms_excel.id, "MS Excel": excel.id}  # each points at the other
    model = FakeModel(lambda name: {"decision": "same_as", "skill_id": ids[name]})
    learner = SkillAliasLearner(db_session, Settings(_env_file=None), model=model)
    learner.apply(learner.propose(["Excel", "MS Excel"]))

    remaining = list(
        db_session.scalars(select(Skill).where(Skill.canonical_name.in_(["Excel", "MS Excel"])))
    )
    assert len(remaining) == 1
    canonical = remaining[0]
    assert SkillNormalizer(db_session).resolve("Excel").id == canonical.id
    assert SkillNormalizer(db_session).resolve("MS Excel").id == canonical.id
    links = list(db_session.scalars(select(JobSkill).where(JobSkill.job_id == job.id)))
    assert [(link.skill_id, link.requirement_type) for link in links] == [
        (canonical.id, RequirementType.REQUIRED)
    ]


def test_apply_saved_dry_run_without_calling_llm_again(db_session, monkeypatch, tmp_path):
    canonical = Skill(canonical_name="Scikit-learn")
    old = Skill(canonical_name="Scikit learn lib", origin="extracted")
    db_session.add_all([canonical, old])
    db_session.commit()
    job = add_link(db_session, old, "Scikit learn lib", 1)
    canonical_id = canonical.id
    monkeypatch.setattr(learn_script, "SessionLocal", sessionmaker(bind=db_session.get_bind()))
    settings = Settings(_env_file=None)
    saved = tmp_path / "proposals.json"

    model = FakeModel(lambda _: {"decision": "same_as", "skill_id": canonical_id})
    assert learn_script.learn_skill_aliases(model=model, settings=settings, save=saved) == 1
    assert json.loads(saved.read_text())["proposals"][0]["decision"] == "alias"
    db_session.commit()

    failing = FakeModel(error=AssertionError("LLM must not be called when applying a file"))
    calls = learn_script.learn_skill_aliases(
        apply=True, model=failing, settings=settings, source=saved
    )
    assert calls == 0 and failing.calls == []
    db_session.expire_all()
    assert job.skills[0].skill_id == canonical_id
    assert decision(db_session, "Scikit learn lib").decision == "alias"
