"""Tests del motor de match. Usan vacantes sintéticas de match alto/bajo conocido y
un embedder falso (bolsa de palabras) para no depender de descargar un modelo."""

import hashlib
import math
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from src.etl.match import (
    Matcher,
    Profile,
    assess_fit,
    load_profile,
    score_skills,
    split_sentences,
)
from src.etl.schema import Experience, Hours, Resolved, Vacante
from src.etl.transform import load_vocabulary

ROOT = Path(__file__).parent.parent
VOCAB = load_vocabulary(ROOT / "config" / "keyword_vocabulary.yaml")
PROFILE_PATH = ROOT / "config" / "skills_profile.yaml"
PROFILE = load_profile(PROFILE_PATH, VOCAB)
NOW = datetime(2026, 9, 22, tzinfo=UTC)


class BagOfWordsEmbedder:
    """Determinista: cada palabra suma 1 en una dimensión fija por hash."""

    dims = 256

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dims
            for word in re.findall(r"\w{3,}", text.casefold()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


def vac(**overrides: Any) -> Vacante:
    base: dict[str, Any] = {
        "id": "arbeitnow:x",
        "source": "arbeitnow",
        "source_id": "x",
        "title": "Data Intern",
        "company": "Acme",
        "url": "https://example.com/1",
        "description_text": "",
        "location": None,
        "is_quito": None,
        "modality": Resolved(value="remote", origin="structured", evidence="remote=true"),
        "seniority": Resolved(value="intern", origin="title", evidence="Data Intern"),
        "schedule": Resolved(unresolved="no_signal"),
        "hours": Hours(unresolved="no_signal"),
        "experience": Experience(unresolved="no_signal"),
        "language": "en",
        "keywords": [],
        "posted_at": None,
        "fetched_at": NOW,
    }
    return Vacante.model_validate({**base, **overrides})


HIGH = vac(
    title="Data Engineering Intern (Python, ETL)",
    description_text=(
        "You will build ETL pipelines in Python with pandas and pydantic validation. "
        "You write automated tests with pytest and version everything in Git. "
        "Automation of business processes with Power Automate is a plus."
    ),
    keywords=["etl", "git", "pandas", "power automate", "pydantic", "pytest", "python"],
    schedule=Resolved(value="part_time", origin="employment", evidence="Part time"),
    hours=Hours(min=20, max=25, period="week", evidence="20-25 hours per week"),
)
LOW = vac(
    title="Senior Platform Engineer (m/w/d)",
    description_text=(
        "Wir suchen eine erfahrene Person für Kubernetes, Terraform und Golang. "
        "Sehr gute Deutschkenntnisse sind erforderlich."
    ),
    keywords=["german", "golang", "kubernetes"],
    location="München",
    is_quito=False,
    modality=Resolved(value="onsite", origin="description", evidence="on-site"),
    seniority=Resolved(value="senior", origin="title", evidence="Senior"),
    schedule=Resolved(value="full_time", origin="employment", evidence="Full time"),
    language="de",
)


# --- Perfil ----------------------------------------------------------------------


def test_real_profile_loads_and_every_skill_is_in_vocabulary() -> None:
    assert set(PROFILE.skills) <= set(VOCAB)
    assert PROFILE.working_languages() == {"es", "en"}  # italiano A1 < B2


def _profile_data(**changes: Any) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))
    data.update(changes)
    return data


@pytest.mark.parametrize(
    "changes",
    [
        {"tier_weights": {"demostrado": 1.0}},  # falta peso para un tier
        {"languages": {"es": "fluido"}},  # nivel no CEFR
        {"semantic_calibration": {"low": 0.7, "high": 0.2}},
        {"skills": {"python": {"tier": "experto", "evidence": "x"}}},
        {"campo_nuevo": 1},
    ],
)
def test_profile_validation_fails_fast(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Profile.model_validate(_profile_data(**changes))


@pytest.mark.parametrize(
    ("skills", "match"),
    [
        ({"cobol": {"tier": "listado", "evidence": "x"}}, "no están en el vocabulario"),
        ({"german": {"tier": "listado", "evidence": "x"}}, "idiomas van en `languages`"),
    ],
)
def test_load_profile_rejects_unmatchable_skills(
    skills: dict[str, Any], match: str, tmp_path: Path
) -> None:
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(_profile_data(skills=skills)), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_profile(path, VOCAB)


# --- Skills ----------------------------------------------------------------------


def test_skills_breakdown_lists_matched_and_missing_with_tier_weights() -> None:
    b = score_skills(vac(keywords=["python", "power bi", "docker", "english"]), PROFILE)
    assert [(m.skill, m.tier) for m in b.matched] == [
        ("python", "demostrado"), ("power bi", "en_formacion"),
    ]  # fmt: skip
    assert b.missing == ["docker"]  # "english" no cuenta como skill
    assert b.value == pytest.approx((1.0 + 0.3) / 3)
    assert b.matched[0].evidence  # cada match cita su evidencia del CV


def test_few_keywords_do_not_inflate_coverage() -> None:
    b = score_skills(vac(keywords=["rest api"]), PROFILE)  # caso real: 1/1 daba 100
    assert b.value == pytest.approx(1 / 3)
    assert b.note is not None and "denominador mínimo 3" in b.note


def test_skills_without_keywords_is_unknown_not_zero() -> None:
    b = score_skills(vac(keywords=["english"]), PROFILE)
    assert b.value is None and b.note


# --- Fit -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "verdict", "fragment"),
    [
        ({}, "apta", None),
        ({"seniority": Resolved(value="senior")}, "no_apta", "seniority: senior"),
        ({"seniority": Resolved(value="student_job")}, "revisar", "seniority: student_job"),
        ({"modality": Resolved(value="hybrid"), "location": "Berlin", "is_quito": False},
         "no_apta", "fuera de Quito"),
        ({"modality": Resolved(value="hybrid"), "location": "Quito, Ecuador", "is_quito": True},
         "apta", None),
        ({"location": "Remote - Germany", "is_quito": False}, "revisar", "restringido"),
        # modalidad desconocida NUNCA descarta
        ({"modality": Resolved(unresolved="no_signal"), "location": "Berlin", "is_quito": False},
         "revisar", "podría exigir presencia"),
        ({"modality": Resolved(unresolved="no_signal")}, "apta", None),
        ({"language": "de"}, "no_apta", "alemán"),
        ({"schedule": Resolved(value="full_time")}, "revisar", "jornada completa"),
        ({"hours": Hours(min=40, max=40, period="week")}, "revisar", "fuera de tu rango"),
        ({"hours": Hours(min=4, max=4, period="day")}, "apta", None),
    ],
)  # fmt: skip
def test_fit_rules(overrides: dict[str, Any], verdict: str, fragment: str | None) -> None:
    fit = assess_fit(vac(**overrides), PROFILE)
    assert fit.verdict == verdict
    if fragment:
        assert any(fragment in m for m in fit.blockers + fit.reviews)


def test_unknown_modality_is_a_visible_warning() -> None:
    fit = assess_fit(vac(modality=Resolved(unresolved="conflict")), PROFILE)
    assert "modalidad desconocida (conflict)" in fit.warnings


# --- Extremos del score ------------------------------------------------------------


def test_high_and_low_match_extremes() -> None:
    """Extremos absolutos sobre el componente de skills (determinista). El semántico
    solo se prueba en orden relativo: la calibración 0.20–0.70 es para el modelo
    real, no para el embedder falso de estos tests."""
    high, low = Matcher(PROFILE).match(HIGH), Matcher(PROFILE).match(LOW)
    assert high.score == 100.0 and high.fit.verdict == "apta"
    assert low.score == 0.0 and low.fit.verdict == "no_apta"
    assert high.skills.missing == [] and set(low.skills.missing) == {"golang", "kubernetes"}

    matcher = Matcher(PROFILE, BagOfWordsEmbedder())
    high_sem, low_sem = matcher.match(HIGH), matcher.match(LOW)
    assert high_sem.semantic.value is not None and low_sem.semantic.value is not None
    assert high_sem.semantic.value > low_sem.semantic.value
    assert high_sem.score is not None and low_sem.score is not None
    assert high_sem.score > low_sem.score


def test_score_renormalizes_when_semantic_is_off() -> None:
    result = Matcher(PROFILE).match(HIGH)
    assert result.score == pytest.approx(100 * result.score_components["skills"], abs=0.1)
    assert result.score_note == "solo skills (pesos renormalizados)"


def test_no_components_means_no_score() -> None:
    result = Matcher(PROFILE).match(vac(keywords=[]))
    assert result.score is None and result.score_note == "sin componentes disponibles"


def test_ranking_blocked_last_then_score_then_verdict() -> None:
    apta_sin_skills = vac(id="arbeitnow:a", keywords=["salesforce"])
    revisar_con_skills = vac(
        id="arbeitnow:r", keywords=["python", "sql", "pandas"],
        schedule=Resolved(value="full_time"),
    )  # fmt: skip
    ranked = Matcher(PROFILE).match_all([LOW, apta_sin_skills, revisar_con_skills])
    assert [r.vacante_id for r in ranked] == ["arbeitnow:r", "arbeitnow:a", "arbeitnow:x"]
    assert [r.fit.verdict for r in ranked] == ["revisar", "apta", "no_apta"]


def test_explain_is_readable_and_cites_evidence() -> None:
    text = Matcher(PROFILE, BagOfWordsEmbedder()).match(LOW).explain()
    assert text.startswith("[NO_APTA]")
    assert "✗ faltan: golang, kubernetes" in text or "✗ faltan: kubernetes, golang" in text
    assert "⛔ seniority: senior" in text and "≈" in text


def test_split_sentences_filters_fragments() -> None:
    text = (
        "Short. This sentence is definitely long enough to count as one. "
        "• Another bullet that is long enough here"
    )
    assert split_sentences(text) == [
        "This sentence is definitely long enough to count as one.",
        "Another bullet that is long enough here",
    ]


@pytest.mark.parametrize(
    ("years", "verdict"),
    [(None, "apta"), (0.0, "apta"), (1.0, "apta"), (2.0, "no_apta"), (7.0, "no_apta")],
)
def test_experience_above_what_i_can_show_blocks(years: float | None, verdict: str) -> None:
    """Decisión estricta: pedir 2+ años bloquea (mi experiencia formal es de meses)."""
    exp = Experience(unresolved="no_signal") if years is None else Experience(years=years)
    fit = assess_fit(vac(experience=exp), PROFILE)
    assert fit.verdict == verdict
    if verdict == "no_apta":
        assert any("años de experiencia" in b for b in fit.blockers)
