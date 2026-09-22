"""Tests de Transform. Los casos de modalidad/seniority salen de frases reales vistas
en el snapshot de Arbeitnow del 2026-09-21 (negaciones, falsos positivos, workation)."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from src.etl.extract import parse_arbeitnow
from src.etl.schema import RawVacante
from src.etl.transform import (
    classify_modality,
    classify_schedule,
    classify_seniority,
    clean_description,
    compile_vocabulary,
    dedup,
    detect_language,
    extract_hours,
    extract_keywords,
    is_quito,
    load_vocabulary,
    normalize,
    normalize_location,
    transform,
)

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
ROOT = Path(__file__).parent.parent
VOCAB = load_vocabulary(ROOT / "config" / "keyword_vocabulary.yaml")
COMPILED = compile_vocabulary(VOCAB)


def raw(**overrides: Any) -> RawVacante:
    base: dict[str, Any] = {
        "source": "arbeitnow",
        "source_id": "x-1",
        "title": "Data Analyst",
        "url": "https://example.com/jobs/1",
        "company": "Acme",
        "description_raw": "",
        "location_raw": "Berlin",
        "modality_raw": "remote=false",
        "fetched_at": NOW,
    }
    return RawVacante.model_validate({**base, **overrides})


# --- Texto y ubicación -----------------------------------------------------------


def test_clean_description_handles_double_escaped_html() -> None:
    raw_html = "&lt;p&gt;&amp;nbsp;Hola&lt;/p&gt;&lt;ul&gt;&lt;li&gt;SQL&lt;/li&gt;&lt;/ul&gt;"
    assert clean_description(raw_html) == "Hola SQL"


@pytest.mark.parametrize(
    ("location_raw", "expected"),
    [("", None), ("   ", None), (None, None), ("Remote job", None), ("Remote", None),
     ("Remote - Germany", "Remote - Germany"), ("  Quito,  Ecuador ", "Quito, Ecuador")],
)  # fmt: skip
def test_normalize_location(location_raw: str | None, expected: str | None) -> None:
    assert normalize_location(location_raw) == expected


def test_is_quito_is_unknown_without_location() -> None:
    assert is_quito(None) is None
    assert is_quito("Quito, Ecuador") is True
    assert is_quito("Berlin") is False


# --- Modalidad -------------------------------------------------------------------


def _modality(description: str = "", **kw: Any) -> tuple[str | None, str | None, str | None]:
    r = classify_modality(raw(**kw), description)
    return r.value, r.origin, r.unresolved


def test_structured_remote_flag_wins() -> None:
    assert _modality("hybrid work model", modality_raw="remote=true") == (
        "remote", "structured", None,
    )  # fmt: skip


def test_remote_false_does_not_mean_onsite() -> None:
    assert _modality("", modality_raw="remote=false") == (None, None, "no_signal")


@pytest.mark.parametrize(("value", "expected"), [("Híbrido", "hybrid"), ("presencial", "onsite"),
                                                  ("Remoto", "remote")])  # fmt: skip
def test_manual_csv_values_map_strictly(value: str, expected: str) -> None:
    assert _modality(modality_raw=value)[0] == expected


def test_manual_unknown_value_is_not_guessed() -> None:
    assert _modality("fully remote position", modality_raw="a veces") == (
        None, "structured", "unknown_value",
    )  # fmt: skip


def test_location_and_title_signals() -> None:
    assert _modality(location_raw="Remote - Germany")[:2] == ("remote", "location")
    assert _modality(title="Data Engineer (Hybrid)", location_raw="Berlin")[:2] == (
        "hybrid", "title",
    )  # fmt: skip


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("Ein hybrides Arbeitsmodell mit Büro und Remote.", ("hybrid", None)),
        ("Up to 50% remote work, depending on role.", ("hybrid", None)),
        ("Trust-based hours, hybrid work and home office.", ("hybrid", None)),
        ("This is a fully remote position.", ("remote", None)),
        ("The role is onsite, Monday to Friday.", ("onsite", None)),
        # negación real del snapshot: no debe contar como remoto
        ("We work from our office and do not offer remote work.", (None, "no_signal")),
        # falso positivo real: "Hybrid" no habla de modalidad
        ("HR se vuelve un Hybrid Resources Department.", (None, "no_signal")),
        # workation: beneficio, no modalidad
        ('Option to "work from anywhere" (6 weeks/year).', (None, "weak_signal")),
        ("Home office possible.", (None, "weak_signal")),
        ("Remote-first team. We also have a hybrid setup in Paris.", (None, "conflict")),
        ("You will be on-site with clients. Home office on Fridays.", (None, "conflict")),
    ],
)
def test_description_modality_rules(
    description: str, expected: tuple[str | None, str | None]
) -> None:
    value, _, unresolved = _modality(description)
    assert (value, unresolved) == expected


def test_conflict_keeps_evidence_of_both_sides() -> None:
    r = classify_modality(raw(), "Remote-first team. We also have a hybrid setup in Paris.")
    assert r.evidence is not None and "remote:" in r.evidence and "hybrid:" in r.evidence


# --- Seniority / jornada ---------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "employment", "expected"),
    [
        ("Praktikum Sales (m/f/d)", [], ("intern", None)),
        ("Stage de 6 mois UX/UI Designer", [], ("intern", None)),
        ("Werkstudent Marketing (m/w/d)", ["Entry", "Part time"], ("student_job", None)),
        ("Account Manager", ["Full time", "mid-senior"], ("mid", None)),
        ("Senior Data Engineer", [], ("senior", None)),
        ("International Sales Manager", [], (None, "no_signal")),  # "intern" ⊄ "international"
        ("Lead Generation Specialist", [], (None, "no_signal")),  # "lead" ≠ senior
        ("Director, Business Development", ["Entry"], (None, "conflict")),
        ("Working Student / Intern - Hardware", ["Intern"], (None, "conflict")),
    ],
)
def test_seniority_rules(
    title: str, employment: list[str], expected: tuple[str | None, str | None]
) -> None:
    r = classify_seniority(raw(title=title, employment_raw=employment))
    assert (r.value, r.unresolved) == expected


def test_schedule_both_signals_is_conflict() -> None:
    assert classify_schedule(raw(employment_raw=["Full or part time"])).unresolved == "conflict"
    assert classify_schedule(raw(employment_raw=["parttime fixed term"])).value == "part_time"


# --- Horas -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Unterstützung mit 15-20 h / Woche", (15.0, 20.0, "week")),
        ("Pasantía de 4 horas al día", (4.0, 4.0, "day")),
        ("full-time: 38.5 hrs/week", (38.5, 38.5, "week")),
    ],
)
def test_extract_hours(text: str, expected: tuple[float, float, str]) -> None:
    h = extract_hours(text)
    assert (h.min, h.max, h.period) == expected and h.unresolved is None


def test_extract_hours_rejects_implausible_and_flags_conflicts() -> None:
    assert extract_hours("Support 24 hours a day").unresolved == "no_signal"
    assert extract_hours("20 hours per week or 30 hours per week").unresolved == "conflict"


# --- Idioma y keywords -----------------------------------------------------------


def test_detect_language() -> None:
    assert detect_language("Wir suchen dich und die Zukunft ist mit uns für eine Zeit") == "de"
    assert detect_language("We are looking for you and the team with our tools to the") == "en"
    assert detect_language("Python SQL") is None


def test_keywords_respect_word_boundaries() -> None:
    kws = extract_keywords("JavaScript, C++ y Power BI; pandas.", COMPILED)
    assert kws == ["c++", "javascript", "pandas", "power bi"]
    assert "java" not in kws


@pytest.mark.parametrize("content", ["keywords: []", "otra: {}", "keywords: {python: []}"])
def test_load_vocabulary_fails_fast_on_bad_config(content: str, tmp_path: Path) -> None:
    path = tmp_path / "vocab.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_vocabulary(path)


# --- Dedup -----------------------------------------------------------------------


def test_dedup_keeps_newest_same_id() -> None:
    old = normalize(raw(fetched_at=NOW - timedelta(days=1)), COMPILED)
    new = normalize(raw(), COMPILED)
    kept, removed, _ = dedup([old, new])
    assert kept == [new] and removed[0].reason == "same_id"


def test_dedup_cross_source_needs_company_and_location() -> None:
    a = normalize(raw(), COMPILED)
    b = normalize(raw(source="manual", source_id="m-1", title="DATA analyst "), COMPILED)
    kept, removed, _ = dedup([a, b])
    assert len(kept) == 1 and removed[0].reason == "same_title_company_location"

    no_loc_a = normalize(raw(location_raw=""), COMPILED)
    no_loc_b = normalize(raw(source="manual", source_id="m-1", location_raw=""), COMPILED)
    assert len(dedup([no_loc_a, no_loc_b])[0]) == 2  # sin ubicación no hay confianza


def test_same_role_in_other_city_is_kept_but_reported() -> None:
    paris = normalize(raw(source_id="p", location_raw="Paris, France"), COMPILED)
    lyon = normalize(raw(source_id="l", location_raw="Lyon, France"), COMPILED)
    kept, removed, possible = dedup([paris, lyon])
    assert len(kept) == 2 and removed == []
    assert sorted(possible[0]) == ["arbeitnow:l", "arbeitnow:p"]


# --- Pipeline completo sobre el fixture de extract --------------------------------


def test_transform_on_fixture_counts_everything() -> None:
    payload = json.loads((ROOT / "tests" / "fixtures" / "arbeitnow_page1.json").read_text())
    records, rejected = parse_arbeitnow(payload, NOW)

    vacantes, report = transform(records, VOCAB, rejected)

    assert report.n_extract_ok + len(report.extract_rejected) == 6
    assert report.n_output == len(vacantes) == 4
    counts = report.counts()
    assert sum(counts["modality"].values()) == 4  # cada vacante contada una vez
    by_id = {v.source_id: v for v in vacantes}
    remote = by_id["remote-account-executive-deutschland-dusseldorf-204800"]
    assert remote.modality.value == "remote"
    assert by_id["senior-engineer-i-field-service-251145"].location is None
    summary = report.resumen()
    assert "Rechazados en extract (contrato):         2" in summary
    assert "Vacantes resultantes:                     4" in summary
