"""Tests de la reescritura. Offline: selección y validación son determinísticas, y
la llamada al LLM usa el cliente falso de test_jd_extractor.

CV de prueba: "Documenté el proceso en Google Apps Script." demuestra por
implicación process automation (RPA) y javascript sin nombrarlos; el bullet de
Pipeline de Calidad no nombra pytest aunque su stack lo usa; el de App de Eventos
no nombra Spring Boot. Python ya está escrito en otro bullet.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from src.cv_parser import Bullet, parse_backing, parse_cv
from src.jd_extractor import JDRequirements
from src.llm import CuotaAgotada
from src.matcher import JDMatch, load_fit_policy, match
from src.rewriter import (
    MAX_EXTRA_CHARS,
    RewriteTarget,
    rewrite,
    select_targets,
    validate_rewrite,
    with_suggestions,
)
from src.vocabulary import load_vocabulary
from tests.test_jd_extractor import FakeClient

VOCAB = load_vocabulary()
POLICY = load_fit_policy()
SAMPLE = Path(__file__).parent / "fixtures" / "cv_sample.tex"
HARD = ["RPA", "JavaScript", "pytest", "Spring Boot", "Python", "Kotlin"]


def jd(hard: list[str], **overrides: Any) -> JDRequirements:
    base: dict[str, Any] = {
        "title": "Pasante de Automatización", "hard_skills": hard, "soft_requirements": [],
        "seniority_signal": "internship", "seniority_evidence": "pasantía",
        "role_family": "automation", "role_evidence": "RPA", "ats_keywords": [],
        "modality": "undetermined", "modality_evidence": "", "workload": "undetermined",
        "hours_per_week": None, "workload_evidence": "",
    }  # fmt: skip
    return JDRequirements.model_validate({**base, **overrides})


def matched(req: JDRequirements, cv: Path = SAMPLE) -> JDMatch:
    return match(parse_cv(cv, VOCAB), parse_backing(cv, VOCAB), req, VOCAB, POLICY)


def targets_for(hard: list[str], cv: Path = SAMPLE, **overrides: Any) -> list[RewriteTarget]:
    req = jd(hard, **overrides)
    return select_targets(matched(req, cv), req, VOCAB)


def by_start(targets: list[RewriteTarget], start: str) -> RewriteTarget:
    return next(t for t in targets if t.bullet.text.startswith(start))


# --- Selección ---------------------------------------------------------------------


def test_selects_implicit_and_stack_terms_but_never_gaps_or_already_named() -> None:
    targets = targets_for(HARD)
    assert {t.bullet.text[:12]: t.add for t in targets} == {
        "Documenté el": ["javascript", "process automation"],  # implícitos
        "Diseñé un pi": ["pytest"],  # stack de su proyecto
        "API REST con": ["spring boot"],
    }
    added = {k for t in targets for k in t.add}
    assert "kotlin" not in added  # carencia
    assert "python" not in added  # ya escrito en otro bullet


def test_terms_are_passed_as_the_jd_writes_them() -> None:
    t = by_start(targets_for(HARD), "Documenté")
    assert t.add_as == ["JavaScript", "RPA"]


def test_rejected_jd_selects_nothing() -> None:
    assert targets_for(HARD, role_family="non_technical") == []


def test_each_term_goes_to_a_single_bullet(tmp_path: Path) -> None:
    cv = tmp_path / "cv.tex"
    cv.write_text(
        "\\begin{document}\n\\cvsection{Proyectos}\n\\cventry{QC}{Python, pytest}\n"
        "\\begin{itemize}\n\\item Validé datos con pandas.\n\\item Documenté el flujo.\n"
        "\\end{itemize}\n\\end{document}",
        encoding="utf-8",
    )
    targets = targets_for(["pytest", "pandas"], cv=cv)
    assert [(t.bullet.text, t.add) for t in targets] == [("Validé datos con pandas.", ["pytest"])]


def test_bullet_that_demonstrates_the_term_beats_one_that_only_shares_the_stack(
    tmp_path: Path,
) -> None:
    """SQL va al bullet de procedimientos almacenados, no al de mayor score del proyecto."""
    cv = tmp_path / "cv.tex"
    cv.write_text(
        "\\begin{document}\n\\cvsection{Proyectos}\n\\cventry{App}{Flask, MySQL}\n"
        "\\begin{itemize}\n\\item Implementé login con Flask y pandas.\n"
        "\\item Diseñé procedimientos almacenados.\n\\end{itemize}\n\\end{document}",
        encoding="utf-8",
    )
    targets = targets_for(["SQL", "pandas", "Flask"], cv=cv)
    assert [(t.bullet.text, t.add) for t in targets] == [
        ("Diseñé procedimientos almacenados.", ["sql"])
    ]


def test_skill_backed_only_by_a_skill_row_has_no_home(tmp_path: Path) -> None:
    cv = tmp_path / "cv.tex"
    cv.write_text(
        "\\begin{document}\n\\cvsection{Proyectos}\n\\cventry{App}{Python}\n"
        "\\begin{itemize}\n\\item Hice una app.\n\\end{itemize}\n"
        "\\skillrow{Lenguajes:}{SQL}\n\\end{document}",
        encoding="utf-8",
    )
    # SQL solo está en la fila de habilidades: ningún bullet lo demuestra ⇒ sin destino.
    # Python está en el stack del proyecto del bullet ⇒ sí.
    targets = targets_for(["SQL", "Python"], cv=cv)
    assert [t.add for t in targets] == [["python"]]


def test_at_most_max_bullets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.rewriter.MAX_BULLETS", 1)
    assert len(targets_for(HARD)) == 1


# --- Validación --------------------------------------------------------------------

TARGET = RewriteTarget(
    id=1,
    bullet=Bullet(
        project="Pasante",
        text="Documenté el proceso de +1,000 registros en Google Apps Script.",
        keywords=["google apps script"],
    ),
    add=["javascript", "process automation"],
    add_as=["JavaScript", "RPA"],
)
GOOD = "Documenté el proceso de RPA de +1,000 registros en Google Apps Script (JavaScript)."


def test_valid_rewrite_passes() -> None:
    assert validate_rewrite(TARGET, GOOD, VOCAB) == []


def test_thousands_separators_are_the_same_number() -> None:
    assert validate_rewrite(TARGET, GOOD.replace("+1,000", "1000"), VOCAB) == []


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (GOOD.replace(" (JavaScript)", ""), "no incluye ['javascript']"),
        (GOOD.replace("Script", "Script y Python"), "agrega skills no pedidas ['python']"),
        (GOOD.replace("en Google Apps Script", "con"), "pierde keywords"),
        (GOOD.replace("+1,000", "+1,500"), "introduce números"),
        (GOOD.replace("registros", "registros de 3 áreas"), "introduce números"),
        (GOOD.replace("RPA", "RPA con UiPath"), "nombres técnicos nuevos ['UiPath']"),
        (GOOD + " " + "x" * (MAX_EXTRA_CHARS + 1), "crece"),
        ("   ", "vacía"),
    ],
)
def test_each_rule_rejects(text: str, reason: str) -> None:
    problems = validate_rewrite(TARGET, text, VOCAB)
    assert any(reason in p for p in problems), problems


# --- LLM (cliente falso) -----------------------------------------------------------


@pytest.fixture
def no_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr("src.rewriter.CACHE_DIR", tmp_path)
    return tmp_path


REQ = jd(HARD)
M = matched(REQ)
TARGETS = select_targets(M, REQ, VOCAB)
VALID = {
    "Documenté": "Documenté el proceso de RPA en Google Apps Script (JavaScript).",
    "Diseñé un pipeline": "Diseñé un pipeline ETL con validación fail-fast y pytest.",
    "API REST": "API REST en Spring Boot con autenticación por roles, sobre una base MySQL "
    "con vistas de reportería.",
}


def batch(overrides: dict[str, str] | None = None, only: list[int] | None = None) -> str:
    texts = {**VALID, **(overrides or {})}
    items = [
        {"id": t.id, "text": next(v for k, v in texts.items() if t.bullet.text.startswith(k))}
        for t in TARGETS
        if only is None or t.id in only
    ]
    return json.dumps({"rewrites": items})


def test_all_valid_first_try(no_cache: Path) -> None:
    client = FakeClient(batch())
    outcomes = rewrite(M, REQ, VOCAB, client)
    assert client.calls == 1
    assert all(o.suggested and not o.rejected_reasons for o in outcomes)
    (record,) = [json.loads(p.read_text()) for p in no_cache.iterdir()]
    assert record["attempts"] == 1 and record["model"] == "fake-model"


def test_rejected_are_retried_once_alone_and_with_the_reason(no_cache: Path) -> None:
    bad = {"Diseñé un pipeline": "Diseñé 3 pipelines ETL con pytest."}
    pipeline_id = by_start(TARGETS, "Diseñé").id
    client = FakeClient(batch(bad), batch(only=[pipeline_id]))
    outcomes = rewrite(M, REQ, VOCAB, client)
    assert client.calls == 2
    assert all(o.suggested for o in outcomes)
    (record,) = [json.loads(p.read_text()) for p in no_cache.iterdir()]
    assert record["attempts"] == 2
    assert "introduce números" in record["first_attempt_rejections"][str(pipeline_id)][0]
    retry_prompt = client.prompts[1]
    assert "introduce números" in retry_prompt
    assert "Documenté" not in retry_prompt.split("<bullets>")[1]  # solo el rechazado


def test_second_rejection_leaves_no_suggestion_but_a_visible_reason(no_cache: Path) -> None:
    bad = {"Diseñé un pipeline": "Diseñé 3 pipelines ETL con pytest."}
    pipeline_id = by_start(TARGETS, "Diseñé").id
    client = FakeClient(batch(bad), batch(bad, only=[pipeline_id]))
    outcomes = rewrite(M, REQ, VOCAB, client)
    failed = next(o for o in outcomes if o.target.id == pipeline_id)
    assert failed.suggested is None and "introduce números" in failed.rejected_reasons[0]
    assert sum(o.suggested is not None for o in outcomes) == 2
    rewrite(M, REQ, VOCAB, client)  # desde el caché: no hay tercera llamada
    assert client.calls == 2


def test_invalid_json_retries_all(no_cache: Path) -> None:
    client = FakeClient("{no es json", batch())
    assert all(o.suggested for o in rewrite(M, REQ, VOCAB, client))
    assert client.calls == 2


def test_missing_id_counts_as_rejected(no_cache: Path) -> None:
    first = TARGETS[0].id
    others = [t.id for t in TARGETS if t.id != first]
    client = FakeClient(batch(only=others), batch(only=[first]))
    assert all(o.suggested for o in rewrite(M, REQ, VOCAB, client))
    assert client.calls == 2


def test_quota_error_propagates_without_retry(no_cache: Path) -> None:
    client = FakeClient(CuotaAgotada("cuota"), batch())
    with pytest.raises(CuotaAgotada):
        rewrite(M, REQ, VOCAB, client)
    assert client.calls == 1


def test_rejected_jd_makes_no_call(no_cache: Path) -> None:
    req = jd(HARD, role_family="non_technical")
    client = FakeClient()
    assert rewrite(matched(req), req, VOCAB, client) == []
    assert client.calls == 0


def test_cached_rewrites_are_revalidated(no_cache: Path) -> None:
    rewrite(M, REQ, VOCAB, FakeClient(batch()))
    (path,) = list(no_cache.iterdir())
    record = json.loads(path.read_text())
    record["rewrites"] = {k: v + " con UiPath" for k, v in record["rewrites"].items()}
    path.write_text(json.dumps(record))
    outcomes = rewrite(M, REQ, VOCAB, FakeClient())  # sin respuestas: no debe llamar
    assert all(o.suggested is None for o in outcomes)


def test_with_suggestions_fills_the_spec_field(no_cache: Path) -> None:
    outcomes = rewrite(M, REQ, VOCAB, FakeClient(batch()))
    filled = with_suggestions(M, outcomes)
    rewritten = [r for r in filled.bullets if r.suggested_rewrite]
    assert len(rewritten) == 3
    assert all(r.suggested_rewrite is None for r in M.bullets)  # el original no se toca


# --- Contra el modelo real (pytest -m llm) -----------------------------------------


@pytest.mark.llm
def test_strategia_rewrite_with_real_cv() -> None:
    """La validez se verifica sola; si la reescritura *suena bien* lo juzga Demian."""
    import os

    from dotenv import load_dotenv

    from src.cli import DEFAULT_CV
    from src.jd_extractor import load_cached
    from src.llm import MODELO_POR_DEFECTO, cliente_desde_env

    root = Path(__file__).parent.parent
    load_dotenv(root / ".env")
    jd_path = root / "data" / "jds" / "strategia.txt"
    model = os.environ.get("GEMINI_MODELO", "").strip() or MODELO_POR_DEFECTO
    req = load_cached(jd_path.read_text(encoding="utf-8"), model) if jd_path.exists() else None
    if req is None or not DEFAULT_CV.exists() or not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("falta el JD extraído, el CV real o GEMINI_API_KEY")
    m = match(parse_cv(DEFAULT_CV, VOCAB), parse_backing(DEFAULT_CV, VOCAB), req, VOCAB, POLICY)
    outcomes = rewrite(m, req, VOCAB, cliente_desde_env())
    assert outcomes, "StrategIA debería tener al menos un bullet que reescribir"
    for o in outcomes:
        assert o.suggested or o.rejected_reasons  # nunca en silencio
        if o.suggested:
            assert validate_rewrite(o.target, o.suggested, VOCAB) == []
