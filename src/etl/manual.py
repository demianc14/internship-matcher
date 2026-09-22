"""Fuente manual: vacantes locales (Quito) que pego a mano.

Pensada para escribir lo MÍNIMO. Un archivo por vacante en `data/manual/*.md`:

    url: https://www.multitrabajos.com/empleos/pasante-datos-123
    title: Pasante de Análisis de Datos
    ---
    (aquí se pega el texto del aviso, tal cual, tan largo como venga)

Solo `url`, `title` y el cuerpo son obligatorios. Modalidad, ubicación, seniority,
jornada, horas, idioma y keywords NO se escriben: los deduce `transform` del texto
pegado, con exactamente las mismas reglas que las vacantes de API.

Campos opcionales, solo cuando el texto no alcanza o quieres corregir la deducción:
`company`, `location`, `modality`, `employment` (separado por comas), `posted_at`.

Por qué archivos y no un CSV: el aviso pegado trae saltos de línea, comas y comillas
que hay que escapar a mano en un CSV — es la parte más fácil de arruinar. Aquí se
pega y ya. La ingesta sigue siendo la misma que la de las APIs (`FetchResult` con
`RawVacante`), así que el resto del pipeline no distingue la fuente.

Un archivo mal formado NO rompe el run: se reporta en `rejected` con su motivo.
"""

import re
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from src.etl.extract import validation_reason
from src.etl.schema import FetchResult, RawVacante, RecordError

SEPARATOR = re.compile(r"^-{3,}\s*$", re.MULTILINE)
REQUIRED = ("url", "title")
OPTIONAL = ("company", "location", "modality", "employment", "posted_at", "tags")

TEMPLATE = """url:
title:
# Opcionales (bórralos si no los usas): company, location, modality, employment,
# posted_at. Todo lo demás (modalidad, ciudad, horas, seniority, idioma, keywords)
# se deduce del texto de abajo.
company:
---
(pega aquí el aviso completo, tal cual)
"""


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")


def parse_manual_file(path: Path, fetched_at: datetime) -> RawVacante | RecordError:
    """Un archivo → RawVacante, o RecordError con el motivo (nunca lanza)."""
    source_id = _slug(path.stem)
    raw_text = path.read_text(encoding="utf-8")
    parts = SEPARATOR.split(raw_text, maxsplit=1)
    if len(parts) != 2:
        return RecordError(
            source="manual", source_id=source_id,
            reason="falta la línea '---' que separa la cabecera del texto del aviso",
        )  # fmt: skip

    header, body = parts
    fields: dict[str, str] = {}
    for lineno, line in enumerate(header.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition(":")
        key = key.strip().casefold()
        if not sep:
            return RecordError(source="manual", source_id=source_id,
                               reason=f"línea {lineno}: se esperaba 'campo: valor'")  # fmt: skip
        if key not in REQUIRED + OPTIONAL:
            return RecordError(source="manual", source_id=source_id,
                               reason=f"campo desconocido {key!r} (¿typo?)")  # fmt: skip
        if value.strip():
            fields[key] = value.strip()

    missing = [k for k in REQUIRED if k not in fields]
    if missing:
        return RecordError(source="manual", source_id=source_id,
                           reason=f"faltan campos obligatorios: {', '.join(missing)}")  # fmt: skip
    if not body.strip():
        return RecordError(source="manual", source_id=source_id,
                           reason="el texto del aviso está vacío")  # fmt: skip

    def split_list(value: str | None) -> list[str]:
        return [x.strip() for x in value.split(",") if x.strip()] if value else []

    try:
        return RawVacante.model_validate(
            {
                "source": "manual",
                "source_id": source_id,
                "title": fields["title"],
                "url": fields["url"],
                "company": fields.get("company"),
                "description_raw": body.strip(),
                "location_raw": fields.get("location"),
                "modality_raw": fields.get("modality"),
                "employment_raw": split_list(fields.get("employment")),
                "tags_raw": split_list(fields.get("tags")),
                "posted_at": fields.get("posted_at"),
                "fetched_at": fetched_at,
            }
        )
    except ValidationError as exc:
        return RecordError(source="manual", source_id=source_id, reason=validation_reason(exc))


def load_manual(directory: Path, now: datetime | None = None) -> FetchResult:
    """Carga data/manual/*.md. Directorio vacío o inexistente = 0 vacantes, sin error:
    la fuente manual es opcional y el pipeline corre igual."""
    fetched_at = now or datetime.now(UTC)
    result = FetchResult(source="manual", fetched_at=fetched_at)
    if not directory.is_dir():
        return result
    for path in sorted(directory.glob("*.md")):
        parsed = parse_manual_file(path, fetched_at)
        if isinstance(parsed, RawVacante):
            result.records.append(parsed)
        else:
            result.rejected.append(parsed)
    result.pages_fetched = 1
    return result


def write_template(directory: Path, slug: str) -> Path:
    """Crea data/manual/<slug>.md para llenar. No sobrescribe."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_slug(slug)}.md"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(TEMPLATE, encoding="utf-8")
    return path
