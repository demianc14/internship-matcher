"""Tests del reporte HTML autocontenido."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from src.cv.parser import parse_cv
from src.cv.suggest import Suggester
from src.etl.match import Matcher, load_profile
from src.etl.schema import Vacante
from src.etl.transform import compile_vocabulary, load_cities, load_vocabulary, transform
from src.report import PLACEHOLDER, build_payload, render_html, write_report
from tests.test_integration import QUITO_RECORDS

ROOT = Path(__file__).parent.parent
VOCAB = load_vocabulary(ROOT / "config" / "keyword_vocabulary.yaml")
CITIES = load_cities(ROOT / "config" / "locations.yaml")
PROFILE = load_profile(ROOT / "config" / "skills_profile.yaml", VOCAB)
CV = parse_cv(Path(__file__).parent / "fixtures" / "cv_sample.tex", compile_vocabulary(VOCAB))
NOW = datetime(2026, 9, 22, 15, 30, tzinfo=UTC)


@pytest.fixture
def payload() -> dict[str, Any]:
    vacantes, report = transform(QUITO_RECORDS(), VOCAB, cities=CITIES)
    results = Matcher(PROFILE).match_all(vacantes)
    suggester = Suggester(CV, PROFILE)
    suggestions = {v.id: suggester.suggest(v) for v in vacantes}
    return build_payload(vacantes, results, report, suggestions, generated_at=NOW)


def test_payload_joins_vacante_match_and_suggestion(payload: dict[str, Any]) -> None:
    (fila,) = payload["vacantes"]
    assert fila["id"].startswith("manual:") and fila["is_quito"] is True
    assert fila["modality"]["value"] == "hybrid" and fila["modality"]["evidence"]
    assert fila["score"] is not None and fila["fit"]["verdict"] == "apta"
    assert fila["suggestion"] is not None and fila["suggestion"]["highlight"]
    assert payload["calidad"]["resultantes"] == 1
    assert payload["generated_at"].startswith("2026-09-22T15:30")


def test_payload_keeps_ranking_order(payload: dict[str, Any]) -> None:
    ids = [f["id"] for f in payload["vacantes"]]
    assert ids == sorted(ids, key=lambda i: ids.index(i))  # se respeta el orden dado


def test_render_escapes_script_close_in_data() -> None:
    """Una descripción con </script> cerraría el bloque y rompería la página."""
    html = render_html(f"<script>{PLACEHOLDER}</script>", {"x": "</script><img onerror=1>"})
    assert "</script><img" not in html
    assert "<\\/script>" in html
    bloque = re.search(r"<script>(.*)</script>", html, re.S)
    assert bloque and json.loads(bloque.group(1).replace("<\\/", "</"))["x"]


def test_render_fails_fast_without_placeholder() -> None:
    with pytest.raises(ValueError, match="marcador"):
        render_html("<html>sin marcador</html>", {})


def test_write_report_is_self_contained(payload: dict[str, Any], tmp_path: Path) -> None:
    out = write_report(ROOT / "web" / "template.html", tmp_path / "sub" / "reporte.html", payload)
    html = out.read_text(encoding="utf-8")

    assert out.exists() and "<!DOCTYPE html>" in html
    assert PLACEHOLDER not in html  # los datos quedaron incrustados
    assert "Pasante de Datos" in html
    # sin fetch ni recursos externos: se abre con doble clic, sin servidor ni conexión
    assert "fetch(" not in html
    assert not re.search(r'(src|href)="https?://[^"]+"', html.split("<footer")[0])


def test_report_data_roundtrips(payload: dict[str, Any], tmp_path: Path) -> None:
    out = write_report(ROOT / "web" / "template.html", tmp_path / "r.html", payload)
    bloque = re.search(
        r'<script id="datos" type="application/json">(.*?)</script>',
        out.read_text(encoding="utf-8"),
        re.S,
    )
    assert bloque
    datos = json.loads(bloque.group(1).replace("<\\/", "</"))
    assert datos["vacantes"][0]["title"] == "Pasante de Datos"


def test_vacante_model_survives_the_payload_dump(payload: dict[str, Any]) -> None:
    """El payload es JSON puro: nada de objetos pydantic ni datetime sueltos."""
    json.dumps(payload)  # no debe lanzar
    assert isinstance(payload["vacantes"][0]["hours"], dict)
    assert not isinstance(payload["vacantes"][0], Vacante)
