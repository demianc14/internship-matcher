"""Tests de match-all. Offline: cliente falso y caché en un directorio temporal."""

from pathlib import Path
from typing import Any

import pytest

from src.batch import Row, jd_files, match_all, rank, uncached
from src.cv_parser import parse_backing, parse_cv
from src.jd_extractor import JDRequirements
from src.llm import CuotaAgotada, ErrorDelLLM
from src.matcher import load_fit_policy, match
from src.vocabulary import load_vocabulary
from tests.test_jd_extractor import FakeClient

VOCAB = load_vocabulary()
POLICY = load_fit_policy()
SAMPLE = Path(__file__).parent / "fixtures" / "cv_sample.tex"
BULLETS, BACKING = parse_cv(SAMPLE, VOCAB), parse_backing(SAMPLE, VOCAB)
MODEL = "fake-model"


def jd_text(name: str) -> str:
    return f"{name}. Pasantía para estudiantes. Trabajo de análisis de datos con Python."


def req_json(name: str, **overrides: Any) -> str:
    base: dict[str, Any] = {
        "title": name, "hard_skills": ["Python"], "soft_requirements": [],
        "seniority_signal": "internship", "seniority_evidence": "Pasantía para estudiantes",
        "role_family": "data", "role_evidence": "análisis de datos con Python",
        "modality": "undetermined", "modality_evidence": "", "workload": "undetermined",
        "hours_per_week": None, "workload_evidence": "", "ats_keywords": [],
    }  # fmt: skip
    return JDRequirements.model_validate({**base, **overrides}).model_dump_json()


@pytest.fixture
def jds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("src.jd_extractor.CACHE_DIR", tmp_path / "cache")
    folder = tmp_path / "jds"
    folder.mkdir()
    for name in ["a", "b", "c"]:
        (folder / f"{name}.txt").write_text(jd_text(name), encoding="utf-8")
    return folder


def run(folder: Path, client: FakeClient | None = None) -> list[Row]:
    return match_all(jd_files(folder), BULLETS, BACKING, VOCAB, POLICY, MODEL, client)


def test_only_top_level_non_empty_files(jds: Path) -> None:
    (jds / "regresion").mkdir()
    (jds / "regresion" / "x.txt").write_text("otro aviso", encoding="utf-8")
    (jds / "vacio.txt").write_text("  \n", encoding="utf-8")
    assert [p.name for p in jd_files(jds)] == ["a.txt", "b.txt", "c.txt"]


def test_without_client_makes_no_calls(jds: Path) -> None:
    rows = run(jds)
    assert [r.status for r in rows] == ["sin_extraer"] * 3
    assert all("--extract" in r.detail for r in rows)


def test_extracts_only_what_is_missing(jds: Path) -> None:
    run(jds, FakeClient(req_json("a"), req_json("b"), req_json("c"), modelo=MODEL))
    assert uncached(jd_files(jds), MODEL) == []
    (jds / "d.txt").write_text(jd_text("d"), encoding="utf-8")
    client = FakeClient(req_json("d"), modelo=MODEL)
    rows = run(jds, client)
    assert client.calls == 1  # a, b, c salen del caché
    assert all(r.status == "ok" for r in rows)


def test_quota_stops_further_calls(jds: Path) -> None:
    client = FakeClient(req_json("a"), CuotaAgotada("cuota"), req_json("c"), modelo=MODEL)
    rows = run(jds, client)
    assert client.calls == 2  # a bien, b sin cuota, c ni se intenta
    by_file = {r.file: r for r in rows}
    assert by_file["a.txt"].status == "ok"
    assert by_file["b.txt"].status == "sin_extraer" and "sin cuota" in by_file["b.txt"].detail
    assert by_file["c.txt"].status == "sin_extraer" and by_file["c.txt"].detail == "sin cuota"


def test_one_failure_does_not_stop_the_rest(jds: Path) -> None:
    client = FakeClient(req_json("a"), ErrorDelLLM("timeout"), req_json("c"), modelo=MODEL)
    rows = run(jds, client)
    assert {r.file: r.status for r in rows} == {"a.txt": "ok", "b.txt": "error", "c.txt": "ok"}
    assert client.calls == 3


def _row(file: str, **overrides: Any) -> Row:
    req = JDRequirements.model_validate_json(req_json(file, **overrides))
    return Row(file=file, status="ok", result=match(BULLETS, BACKING, req, VOCAB, POLICY))


def test_rank_by_verdict_then_coverage_then_how_many_skills() -> None:
    rows = [
        Row(file="sin.txt", status="sin_extraer"),
        _row("no_apta.txt", role_family="non_technical"),
        _row("revisar.txt", seniority_signal="junior", seniority_evidence="x"),
        _row("apta_poca.txt", hard_skills=["Python", "Kotlin"]),  # 50 % sobre 2
        _row("apta_mucha.txt", hard_skills=["Python", "pandas"]),  # 100 %
        _row("apta_poca_n4.txt", hard_skills=["Python", "pandas", "Kotlin", "Rust", "Scala"]),
    ]
    assert [r.file for r in rank(rows)] == [
        "apta_mucha.txt", "apta_poca_n4.txt", "apta_poca.txt", "revisar.txt", "no_apta.txt",
        "sin.txt",
    ]  # fmt: skip
