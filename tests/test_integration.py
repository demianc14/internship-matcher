"""Tests de integración: el pipeline completo con las tres fuentes.

Los unitarios prueban cada regla por separado; estos prueban que las piezas encajen
— contrato compartido, reporte que cuadra, ranking y sugerencias — y que la CLI
escriba lo que dice que escribe. Nada llama a la red: todo sale de fixtures.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import src.cli as cli
from src.cv.parser import parse_cv
from src.cv.suggest import Suggester
from src.etl.extract import parse_arbeitnow, parse_remoteok
from src.etl.manual import load_manual
from src.etl.match import Matcher, load_profile
from src.etl.schema import RawVacante, Vacante
from src.etl.transform import (
    compile_vocabulary,
    load_cities,
    load_implications,
    load_vocabulary,
    transform,
)
from tests.test_match import BagOfWordsEmbedder

ROOT = Path(__file__).parent.parent
FIXTURES = Path(__file__).parent / "fixtures"
VOCAB_PATH = ROOT / "config" / "keyword_vocabulary.yaml"
VOCAB = load_vocabulary(VOCAB_PATH)
CITIES = load_cities(ROOT / "config" / "locations.yaml")
PROFILE = load_profile(ROOT / "config" / "skills_profile.yaml", VOCAB)
CV = parse_cv(FIXTURES / "cv_sample.tex", compile_vocabulary(VOCAB))
NOW = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)

QUITO = """url: https://www.multitrabajos.com/empleos/pasante-datos-1
title: Pasante de Datos
company: Acme Ecuador
---
Buscamos pasante para el equipo de datos en Quito, modalidad híbrida.
Requisitos: Python, SQL y pandas. Deseable Power BI.
Pasantía de 4 horas al día.
"""


@pytest.fixture
def records(tmp_path: Path) -> tuple[list[RawVacante], list[Any]]:
    """Las tres fuentes juntas, como en una corrida real."""
    (tmp_path / "pasante-datos-acme.md").write_text(QUITO, encoding="utf-8")
    manual = load_manual(tmp_path, now=NOW)
    arbeitnow, rejected_a = parse_arbeitnow(
        json.loads((FIXTURES / "arbeitnow_page1.json").read_text()), NOW
    )
    remoteok, rejected_r = parse_remoteok(
        json.loads((FIXTURES / "remoteok.json").read_text()), NOW
    )
    return (
        [*arbeitnow, *remoteok, *manual.records],
        [*rejected_a, *rejected_r, *manual.rejected],
    )


@pytest.fixture
def pipeline(records: tuple[list[RawVacante], list[Any]]) -> tuple[list[Vacante], Any]:
    return transform(records[0], VOCAB, records[1], cities=CITIES)


def test_all_sources_share_the_same_contract(records: tuple[list[RawVacante], list[Any]]) -> None:
    raw, _ = records
    assert {r.source for r in raw} == {"arbeitnow", "remoteok", "manual"}
    assert all(isinstance(r, RawVacante) for r in raw)


def test_quality_report_adds_up(
    pipeline: tuple[list[Vacante], Any], records: tuple[list[RawVacante], list[Any]]
) -> None:
    vacantes, report = pipeline
    n_in = report.n_extract_ok + len(report.extract_rejected)
    assert n_in == len(records[0]) + len(records[1])
    # nada desaparece en silencio: entradas = salidas + rechazadas + duplicadas
    assert n_in == report.n_output + len(report.extract_rejected) + len(report.duplicates_removed)
    assert sum(report.counts()["source"].values()) == len(vacantes)


def test_every_classified_field_has_value_or_reason(pipeline: tuple[list[Vacante], Any]) -> None:
    """La invariante del proyecto: nunca un default silencioso."""
    for v in pipeline[0]:
        for field in (v.modality, v.seniority, v.schedule, v.hours):
            assert (field.unresolved is None) != (
                (field.value if hasattr(field, "value") else field.period) is None
            ), f"{v.id}: {field}"
        assert v.id == f"{v.source}:{v.source_id}" and v.id.count(":") >= 1
    assert len({v.id for v in pipeline[0]}) == len(pipeline[0])  # ids únicos


def test_manual_quito_vacancy_survives_to_the_top(pipeline: tuple[list[Vacante], Any]) -> None:
    vacantes, _ = pipeline
    quito = next(v for v in vacantes if v.source == "manual")
    assert (quito.is_quito, quito.modality.value, quito.seniority.value) == (
        True, "hybrid", "intern",
    )  # fmt: skip

    results = Matcher(PROFILE).match_all(vacantes)
    top = results[0]
    assert top.vacante_id == quito.id and top.fit.verdict == "apta"
    assert top.score is not None and top.score >= 60
    assert {"python", "sql", "pandas"} <= {m.skill for m in top.skills.matched}


def test_blocked_vacancies_never_outrank_viable_ones(pipeline: tuple[list[Vacante], Any]) -> None:
    results = Matcher(PROFILE, BagOfWordsEmbedder()).match_all(pipeline[0])
    verdicts = [r.fit.verdict for r in results]
    assert verdicts == sorted(verdicts, key=lambda v: v == "no_apta")
    assert all(r.score is None or 0 <= r.score <= 100 for r in results)


def test_suggestions_for_the_best_match_cite_real_bullets(
    pipeline: tuple[list[Vacante], Any],
) -> None:
    vacantes, _ = pipeline
    best = Matcher(PROFILE).match_all(vacantes)[0]
    vacante = next(v for v in vacantes if v.id == best.vacante_id)

    suggestion = Suggester(CV, PROFILE, implications=load_implications(VOCAB_PATH, VOCAB)).suggest(
        vacante
    )
    bullet_ids = {b.id for b in CV.bullets}
    assert suggestion.highlight and all(p.bullet_id in bullet_ids for p in suggestion.highlight)
    assert all(p.covers for p in suggestion.highlight)
    rendered = suggestion.render()
    assert suggestion.title in rendered


# --- CLI de punta a punta ----------------------------------------------------------


def test_cli_runs_the_whole_pipeline_and_writes_its_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_dir, processed, manual_dir = tmp_path / "raw", tmp_path / "processed", tmp_path / "manual"
    raw_dir.mkdir()
    manual_dir.mkdir()
    (manual_dir / "pasante-datos-acme.md").write_text(QUITO, encoding="utf-8")
    (raw_dir / "arbeitnow_20260922T120000Z.json").write_text(
        json.dumps(
            {
                "source": "arbeitnow",
                "fetched_at": NOW.isoformat(),
                "pages": [json.loads((FIXTURES / "arbeitnow_page1.json").read_text())],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "RAW_DIR", raw_dir)
    monkeypatch.setattr(cli, "PROCESSED_DIR", processed)
    monkeypatch.setattr(cli, "MANUAL_DIR", manual_dir)

    assert cli.main(["--top", "2"]) == 0

    out = capsys.readouterr().out
    assert "Reporte de calidad de datos" in out and "=== Match" in out
    lines = (processed / "vacantes.jsonl").read_text().splitlines()
    vacantes = [Vacante.model_validate_json(line) for line in lines]
    assert len(vacantes) == 5  # 4 del fixture + 1 manual
    report = json.loads((processed / "quality_report.json").read_text())
    assert report["n_output"] == 5 and len(report["extract_rejected"]) == 2
    matches = (processed / "matches.jsonl").read_text().splitlines()
    assert len(matches) == 5 and json.loads(matches[0])["fit"]["verdict"] == "apta"


def test_cli_without_snapshots_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "RAW_DIR", tmp_path / "vacio")
    monkeypatch.setattr(cli, "MANUAL_DIR", tmp_path / "manual")
    (tmp_path / "vacio").mkdir()

    assert cli.main([]) == 1
    assert "No hay vacantes" in capsys.readouterr().err


def test_cli_new_template_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "MANUAL_DIR", tmp_path)
    assert cli.main(["--nueva", "Pasante Datos Acme"]) == 0
    assert (tmp_path / "pasante-datos-acme.md").exists()
    assert cli.main(["--nueva", "pasante-datos-acme"]) == 1  # no sobrescribe


def test_cli_suggest_path_uses_the_cv_from_the_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Perfil apuntando al CV de prueba: cubre parse_cv + auditoría + sugerencias."""
    profile_yaml = (ROOT / "config" / "skills_profile.yaml").read_text(encoding="utf-8")
    profile_path = tmp_path / "perfil.yaml"
    profile_path.write_text(
        profile_yaml.replace(
            'cv_path: "~/Documents/Documentos Demi/Base CV.tex"',
            f'cv_path: "{FIXTURES / "cv_sample.tex"}"',
        ),
        encoding="utf-8",
    )
    manual_dir = tmp_path / "manual"
    manual_dir.mkdir()
    (manual_dir / "pasante-datos-acme.md").write_text(QUITO, encoding="utf-8")
    monkeypatch.setattr(cli, "PROFILE_PATH", profile_path)
    monkeypatch.setattr(cli, "RAW_DIR", tmp_path / "sin-snapshots")
    monkeypatch.setattr(cli, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(cli, "MANUAL_DIR", manual_dir)
    (tmp_path / "sin-snapshots").mkdir()

    # Sin snapshots de API, solo la fuente manual: debe correr igual
    assert cli.main(["--suggest", "1", "--top", "1"]) == 0

    out = capsys.readouterr().out
    assert "=== Sugerencias de CV (cv_sample.tex) ===" in out
    assert "Pasante de Datos" in out and "Destacar estos bullets" in out
