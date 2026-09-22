"""Tests de la fuente manual (vacantes de Quito pegadas a mano).

El punto de esta fuente es escribir lo mínimo: los tests verifican que con url,
title y el aviso pegado, transform deduce ciudad, modalidad, seniority y horas."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.etl.manual import TEMPLATE, load_manual, parse_manual_file, write_template
from src.etl.schema import RawVacante, RecordError
from src.etl.transform import infer_location, load_cities, load_vocabulary, transform

ROOT = Path(__file__).parent.parent
VOCAB = load_vocabulary(ROOT / "config" / "keyword_vocabulary.yaml")
CITIES = load_cities(ROOT / "config" / "locations.yaml")
NOW = datetime(2026, 9, 22, tzinfo=UTC)

AVISO = """url: https://www.multitrabajos.com/empleos/pasante-de-datos-1234567
title: Pasante de Análisis de Datos
company: Acme Ecuador
---
Empresa busca pasante para su equipo de datos en Quito (sector Cumbayá).
Modalidad híbrida: 3 días en oficina y 2 desde casa.
Requisitos: manejo de SQL y Python (pandas). Deseable Power BI.
Ofrecemos: pasantía de 4 horas al día, horario flexible.
"""


def write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_minimal_file_derives_everything_else(tmp_path: Path) -> None:
    write(tmp_path, "pasante-datos-acme.md", AVISO)
    result = load_manual(tmp_path, now=NOW)
    assert result.ok and len(result.records) == 1 and result.rejected == []

    (vacante,), _ = transform(result.records, VOCAB, cities=CITIES)
    assert vacante.id == "manual:pasante-datos-acme"  # id del nombre del archivo
    assert (vacante.location, vacante.is_quito) == ("Quito", True)
    assert vacante.location_source == "description"  # deducida, no escrita
    assert vacante.modality.value == "hybrid"
    assert vacante.seniority.value == "intern"
    assert (vacante.hours.min, vacante.hours.period) == (4.0, "day")
    assert vacante.language == "es"
    assert {"sql", "python", "pandas", "power bi"} <= set(vacante.keywords)


def test_optional_field_overrides_the_guess(tmp_path: Path) -> None:
    write(tmp_path, "v.md", AVISO.replace("---", "modality: remoto\n---", 1))
    (record,) = load_manual(tmp_path, now=NOW).records
    (vacante,), _ = transform([record], VOCAB, cities=CITIES)
    assert vacante.modality.value == "remote" and vacante.modality.origin == "structured"


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("url: https://x.io\ntitle: A\nsin separador", "falta la línea '---'"),
        ("title: A\n---\ncuerpo", "faltan campos obligatorios: url"),
        ("url: https://x.io\n---\ncuerpo", "faltan campos obligatorios: title"),
        ("url: https://x.io\ntitle: A\n---\n   ", "el texto del aviso está vacío"),
        ("url: https://x.io\ntitle: A\ncompani: X\n---\ncuerpo", "campo desconocido 'compani'"),
        ("url: no-es-url\ntitle: A\n---\ncuerpo", "url:"),
        ("url: https://x.io\ntitle: A\nlinea suelta\n---\ncuerpo", "se esperaba 'campo: valor'"),
    ],
)
def test_malformed_file_is_reported_not_raised(content: str, reason: str, tmp_path: Path) -> None:
    parsed = parse_manual_file(write(tmp_path, "mala.md", content), NOW)
    assert isinstance(parsed, RecordError) and reason in parsed.reason


def test_one_bad_file_does_not_break_the_rest(tmp_path: Path) -> None:
    write(tmp_path, "buena.md", AVISO)
    write(tmp_path, "mala.md", "sin nada")
    result = load_manual(tmp_path, now=NOW)
    assert len(result.records) == 1 and len(result.rejected) == 1
    assert result.ok  # la fuente no falló: falló un archivo


def test_missing_directory_is_not_an_error(tmp_path: Path) -> None:
    result = load_manual(tmp_path / "no-existe", now=NOW)
    assert result.ok and result.records == [] and result.rejected == []


def test_template_is_valid_once_filled(tmp_path: Path) -> None:
    path = write_template(tmp_path, "Pasante Datos Acme")
    assert path.name == "pasante-datos-acme.md"
    assert isinstance(parse_manual_file(path, NOW), RecordError)  # vacía todavía

    filled = TEMPLATE.replace("url:", "url: https://x.io/1").replace("title:", "title: Pasante")
    filled = filled.replace("(pega aquí el aviso completo, tal cual)", "Pasantía en Quito.")
    path.write_text(filled, encoding="utf-8")
    assert isinstance(parse_manual_file(path, NOW), RawVacante)

    with pytest.raises(FileExistsError):
        write_template(tmp_path, "pasante-datos-acme")


# --- Inferencia de ciudad ---------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Oficina en Cumbayá, jornada matutina", "Quito"),  # valle → Quito
        ("Puesto en Guayaquil", "Guayaquil"),
        ("Trabajo en Sangolquí", "Quito"),
        ("Vacante en Quito y Guayaquil", None),  # dos ciudades: no se adivina
        ("Puesto remoto para LatAm", None),
        ("Se requiere equito de trabajo", None),  # límites de palabra
    ],
)
def test_infer_location(text: str, expected: str | None) -> None:
    city, evidence = infer_location(text, CITIES)
    assert city == expected
    if expected:
        assert evidence and expected.split()[0].casefold() in evidence.casefold() or evidence
