"""Tests del matcher. Sin LLM: los requisitos del JD se arman a mano.

CV de prueba (tests/fixtures/cv_sample.tex):
  Pasante de Datos      · "Automaticé un reporte … Python y pandas"
                        · "Documenté el proceso en Google Apps Script."
  Pipeline de Calidad   · "Diseñé un pipeline ETL …"          stack: Python, pandas, pytest
  App de Eventos        · "API REST … base MySQL …"            stack: Java 17, Spring Boot, MySQL
  Habilidades           Python, SQL, Java · Power BI (en formación)
"""

from pathlib import Path
from typing import Any

import pytest

from src.cv_parser import parse_backing, parse_cv
from src.jd_extractor import JDRequirements
from src.matcher import (
    JDMatch,
    assess_fit,
    load_fit_policy,
    match,
    normalize_requirements,
)
from src.vocabulary import load_vocabulary

VOCAB = load_vocabulary()
POLICY = load_fit_policy()
SAMPLE = Path(__file__).parent / "fixtures" / "cv_sample.tex"
BULLETS = parse_cv(SAMPLE, VOCAB)
BACKING = parse_backing(SAMPLE, VOCAB)


def jd(hard: list[str], ats: list[str] | None = None, **overrides: Any) -> JDRequirements:
    base: dict[str, Any] = {
        "title": "Pasante de Datos",
        "hard_skills": hard,
        "soft_requirements": [],
        "seniority_signal": "internship",
        "seniority_evidence": "pasantía",
        "role_family": "data",
        "role_evidence": "análisis de datos",
        "modality": "undetermined",
        "modality_evidence": "",
        "workload": "undetermined",
        "hours_per_week": None,
        "workload_evidence": "",
        "ats_keywords": ats or [],
    }
    return JDRequirements.model_validate({**base, **overrides})


def run(hard: list[str], ats: list[str] | None = None, **overrides: Any) -> JDMatch:
    return match(BULLETS, BACKING, jd(hard, ats, **overrides), VOCAB, POLICY)


def bullet_result(m: JDMatch, text_start: str) -> Any:
    return next(r for r in m.bullets if r.bullet.text.startswith(text_start))


# --- Casos exactos -----------------------------------------------------------------


def test_exact_match_ranks_the_bullet_that_shows_everything_first() -> None:
    m = run(["Python", "pandas"])
    top = m.bullets[0]
    assert top.bullet.text.startswith("Automaticé un reporte")
    assert top.match_score == 1.0
    assert top.matched_keywords == ["pandas", "python"]
    assert m.covered == ["pandas", "python"] and m.coverage == 1.0


def test_score_is_the_fraction_of_the_jd_the_bullet_shows() -> None:
    m = run(["Python", "pandas", "ETL", "Kotlin"])
    assert bullet_result(m, "Automaticé").match_score == 0.5
    assert bullet_result(m, "Diseñé un pipeline").match_score == 0.25
    assert bullet_result(m, "Documenté").match_score == 0.0


# --- Sinónimos e implicaciones -----------------------------------------------------


@pytest.mark.parametrize(
    ("item", "canonical"),
    [
        ("Apps Script", "google apps script"),
        ("Low-Code/No-Code", "low-code"),
        ("Excel", "excel"),  # suelto en una lista es la herramienta, no el verbo
        ("MS Excel", "excel"),
        ("IA Agéntica", "ai agents"),
        ("Postgres", "postgresql"),
        ("multi-agent systems", "ai agents"),
        ("Back-end", "backend"),
        ("Full Stack", "full-stack"),
        ("ciencia de datos", "data science"),
        ("Prompting", "prompt engineering"),
    ],
)
def test_list_items_are_canonicalized(item: str, canonical: str) -> None:
    assert normalize_requirements(jd([item]), VOCAB)[0] == {canonical}


def test_synonym_in_the_jd_matches_the_cv_wording() -> None:
    m = run(["Apps Script"])
    assert bullet_result(m, "Documenté").matched_keywords == ["google apps script"]


def test_implication_lets_a_bullet_cover_a_skill_it_does_not_name() -> None:
    m = run(["SQL", "APIs"])  # el bullet dice "MySQL" y "API REST"
    r = bullet_result(m, "API REST")
    assert r.matched_keywords == ["apis", "sql"]
    assert m.covered == ["apis", "sql"] and m.gaps == []


# --- Las cuatro categorías ---------------------------------------------------------


def test_skill_in_project_stack_but_not_in_bullet_is_a_rewrite_opportunity() -> None:
    m = run(["pytest", "Spring Boot"])
    assert m.in_cv_not_in_bullets == ["pytest", "spring boot"]
    assert m.gaps == [] and m.covered == []
    assert bullet_result(m, "Diseñé un pipeline").missing_keywords == ["pytest"]
    assert bullet_result(m, "API REST").missing_keywords == ["spring boot"]
    assert m.backed == 1.0 and m.coverage == 0.0


def test_missing_keywords_only_come_from_that_bullets_own_project() -> None:
    m = run(["pytest"])
    # pytest está en el stack de Pipeline de Calidad, no en el de App de Eventos.
    assert bullet_result(m, "API REST").missing_keywords == []
    assert bullet_result(m, "Automaticé").missing_keywords == []


def test_skill_backed_outside_bullets_is_in_cv_not_in_bullets() -> None:
    m = run(["Java"])  # respaldado por el stack de App de Eventos y por la fila
    assert m.in_cv_not_in_bullets == ["java"]


def test_skill_in_training_is_not_presented_as_backed() -> None:
    m = run(["Power BI"])
    assert m.in_training == ["power bi"]
    assert m.in_cv_not_in_bullets == [] and m.gaps == []
    assert all("power bi" not in r.missing_keywords for r in m.bullets)


def test_gaps_are_listed_but_never_suggested() -> None:
    m = run(["Kotlin", "Python"])
    assert m.gaps == ["kotlin"]
    assert all("kotlin" not in r.missing_keywords for r in m.bullets)
    assert all("kotlin" not in r.matched_keywords for r in m.bullets)


def test_categories_partition_the_requested_skills() -> None:
    m = run(["Python", "pytest", "Power BI", "Kotlin", "SQL", "Spring Boot"])
    groups = [m.covered, m.in_cv_not_in_bullets, m.in_training, m.gaps]
    flat = [k for g in groups for k in g]
    assert sorted(flat) == m.requested and len(flat) == len(set(flat))


def test_unrecognized_items_are_reported_not_dropped() -> None:
    m = run(["Radford", "Python"], ats=["Mercer Comptryx"])
    assert m.unrecognized == ["Radford", "Mercer Comptryx"]
    assert m.requested == ["python"]


def test_languages_are_not_skills() -> None:
    m = run(["Python"], ats=["Inglés", "English"])
    assert m.requested == ["python"] and m.unrecognized == []


def test_hard_skills_and_ats_keywords_are_merged_without_duplicates() -> None:
    m = run(["Python"], ats=["Python", "python", "pandas"])
    assert m.requested == ["pandas", "python"]


def test_empty_requirements_give_zero_not_a_crash() -> None:
    m = run([])
    assert m.requested == [] and m.coverage == 0.0
    assert all(r.match_score == 0.0 for r in m.bullets)


# --- Veredicto de encaje -----------------------------------------------------------


@pytest.mark.parametrize(
    ("level", "role", "verdict"),
    [
        ("internship", "automation", "apta"),
        ("internship", "data", "apta"),
        ("junior", "data", "revisar"),
        ("undetermined", "software", "revisar"),
        ("mid", "software", "no_apta"),
        ("senior", "ml_ai", "no_apta"),
        ("internship", "non_technical", "no_apta"),  # el rol bloquea aunque sea pasantía
        ("junior", "non_technical", "no_apta"),
    ],
)
def test_fit_policy(level: str, role: str, verdict: str) -> None:
    req = jd([], seniority_signal=level, role_family=role)
    assert assess_fit(req, POLICY).verdict == verdict


def test_fit_reasons_cite_the_extractor_evidence() -> None:
    fit = assess_fit(jd([], role_family="non_technical", role_evidence="sales team"), POLICY)
    assert any("non_technical ⇒ no_apta" in r and "sales team" in r for r in fit.reasons)


def test_unrelated_jd_scores_zero_and_is_rejected() -> None:
    """Account Executive: el caso que el pipeline anterior marcaba apto."""
    m = run(["CRM software", "Salesforce"], seniority_signal="junior",
            role_family="non_technical", role_evidence="sales team")  # fmt: skip
    assert m.fit.verdict == "no_apta"
    assert m.coverage == 0.0 and m.gaps == ["crm", "salesforce"]


def test_full_skill_coverage_never_rescues_a_blocked_role() -> None:
    m = run(["Python", "pandas"], role_family="non_technical", role_evidence="x")
    assert m.coverage == 1.0 and m.fit.verdict == "no_apta"


# --- Política como datos -----------------------------------------------------------


_REST = (
    "modality: {remote: apta, hybrid: apta, onsite: revisar, undetermined: apta}\n"
    "workload: {max_hours_per_week: 30, over_max: revisar, full_time: revisar}\n"
)
_LEVELS = (
    "seniority: {internship: apta, junior: apta, mid: apta, senior: apta, undetermined: apta}\n"
)


@pytest.mark.parametrize(
    ("yaml_text", "match_text"),
    [
        ("blocking_roles: [non_technical]\nseniority: {internship: apta}\n" + _REST,
         "niveles sin veredicto"),
        ("blocking_roles: [sales]\n" + _LEVELS + _REST, "blocking_roles"),
        ("blocking_roles: []\n" + _LEVELS.replace("internship: apta", "internship: quizás")
         + _REST, "seniority"),
        ("blocking_roles: []\n" + _LEVELS + "modality: {remote: apta}\n"
         + _REST.split("\n")[1] + "\n", "modalidades sin veredicto"),
        ("blocking_roles: []\n" + _LEVELS + _REST.split("\n")[0] + "\n", "workload"),
    ],
)  # fmt: skip
def test_fit_policy_fails_fast_on_bad_config(
    tmp_path: Path, yaml_text: str, match_text: str
) -> None:
    bad = tmp_path / "fit.yaml"
    bad.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValueError, match=match_text):
        load_fit_policy(bad)


# --- Modalidad y jornada -----------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "verdict", "reason"),
    [
        ({"modality": "remote", "modality_evidence": "remoto"}, "apta", "modalidad remote"),
        ({"modality": "hybrid", "modality_evidence": "híbrido"}, "apta", "modalidad hybrid"),
        ({"modality": "onsite", "modality_evidence": "presencial"}, "revisar", "onsite ⇒ revisar"),
        ({}, "apta", "modalidad no indicada (se avisa)"),
        ({"workload": "part_time", "hours_per_week": 30.0, "workload_evidence": "6 horas"},
         "apta", "30 h/semana (máximo 30)"),
        ({"workload": "part_time", "hours_per_week": 32.0, "workload_evidence": "32 h"},
         "revisar", "32 h/semana"),
        ({"workload": "full_time", "workload_evidence": "full time"},
         "revisar", "tiempo completo, sin cifra"),
        ({"workload": "part_time", "workload_evidence": "part time"},
         "apta", "medio tiempo, sin cifra"),
        ({}, "apta", "jornada no indicada (se avisa)"),
    ],
)  # fmt: skip
def test_modality_and_workload_policy(overrides: dict[str, Any], verdict: str, reason: str) -> None:
    fit = assess_fit(jd([], **overrides), POLICY)
    assert fit.verdict == verdict
    assert any(reason in r for r in fit.reasons), fit.reasons


def test_most_restrictive_verdict_wins() -> None:
    req = jd([], seniority_signal="junior", modality="onsite", modality_evidence="presencial",
             role_family="non_technical", role_evidence="ventas")  # fmt: skip
    assert assess_fit(req, POLICY).verdict == "no_apta"


# --- Ruido en lo no reconocido -----------------------------------------------------


def test_title_and_education_are_ignored_not_hidden() -> None:
    m = run(
        ["Radford"],
        ats=["Pasante", "Data Engineer", "Ingeniería en Computación", "Bachelor's degree"],
        title="Pasante — AI Systems & Data Engineer",
    )
    assert m.unrecognized == ["Radford"]
    assert m.ignored == [
        "Pasante", "Data Engineer", "Ingeniería en Computación", "Bachelor's degree",
    ]  # fmt: skip


def test_known_skill_in_the_title_is_never_ignored() -> None:
    m = run(["Python"], title="Python Developer")
    assert m.requested == ["python"] and m.ignored == []


def test_context_engineering_is_not_prompt_engineering() -> None:
    assert normalize_requirements(jd(["context engineering"]), VOCAB)[0] == {"context engineering"}


def test_solid_is_left_out_of_the_vocabulary() -> None:
    """Principios SOLID: en inglés "solid" también es una palabra común."""
    assert normalize_requirements(jd(["SOLID"]), VOCAB) == (set(), ["SOLID"])


def test_unrecognized_are_deduplicated_ignoring_case() -> None:
    m = run(["Radford", "radford", "RADFORD "])
    assert m.unrecognized == ["Radford"]
