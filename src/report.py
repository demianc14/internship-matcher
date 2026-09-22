"""Reporte HTML autocontenido: un archivo que se abre con doble clic.

Los datos van **incrustados** en el HTML, no se leen con `fetch`: una página abierta
desde `file://` no puede leer archivos vecinos (el navegador lo bloquea). Así el
reporte funciona sin servidor y sin conexión, que es el punto de tenerlo.

La plantilla (`web/template.html`) es HTML normal con un marcador `/*__DATOS__*/`;
aquí solo se arma el payload y se sustituye. Cero lógica de negocio: lo que se
muestra ya viene decidido por transform/match/suggest.
"""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.cv.suggest import Suggestion
from src.etl.match import MatchResult
from src.etl.schema import Vacante
from src.etl.transform import TransformReport

PLACEHOLDER = "/*__DATOS__*/"


def build_payload(
    vacantes: Sequence[Vacante],
    results: Sequence[MatchResult],
    report: TransformReport,
    suggestions: dict[str, Suggestion] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Une vacante + match + sugerencia por id, en el orden del ranking."""
    by_id = {v.id: v for v in vacantes}
    suggestions = suggestions or {}
    filas = []
    for r in results:
        v = by_id[r.vacante_id]
        filas.append(
            {
                "id": v.id,
                "source": v.source,
                "title": v.title,
                "company": v.company,
                "url": str(v.url),
                "location": v.location,
                "location_source": v.location_source,
                "is_quito": v.is_quito,
                "language": v.language,
                "posted_at": v.posted_at.isoformat() if v.posted_at else None,
                "modality": v.modality.model_dump(),
                "seniority": v.seniority.model_dump(),
                "schedule": v.schedule.model_dump(),
                "hours": v.hours.model_dump(),
                "experience": v.experience.model_dump(),
                "keywords": v.keywords,
                "score": r.score,
                "score_components": r.score_components,
                "score_note": r.score_note,
                "skills": r.skills.model_dump(),
                "semantic": r.semantic.model_dump(),
                "fit": r.fit.model_dump(),
                "suggestion": (
                    suggestions[v.id].model_dump() if v.id in suggestions else None
                ),
            }
        )
    return {
        "generated_at": (generated_at or datetime.now(UTC)).isoformat(timespec="minutes"),
        "calidad": {
            "recibidas": report.n_extract_ok + len(report.extract_rejected),
            "rechazadas": len(report.extract_rejected),
            "duplicados": len(report.duplicates_removed),
            "posibles_duplicados": len(report.possible_duplicates),
            "resultantes": report.n_output,
            "conteos": {k: dict(c) for k, c in report.counts().items()},
        },
        "vacantes": filas,
    }


def render_html(template: str, payload: dict[str, Any]) -> str:
    """Sustituye el marcador por el JSON del payload.

    `</script>` dentro de los datos cerraría el bloque y rompería la página (una
    descripción de vacante puede traer cualquier cosa), así que se escapa.
    """
    if PLACEHOLDER not in template:
        raise ValueError(f"la plantilla no tiene el marcador {PLACEHOLDER}")
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return template.replace(PLACEHOLDER, data)


def write_report(template_path: Path, out_path: Path, payload: dict[str, Any]) -> Path:
    html = render_html(template_path.read_text(encoding="utf-8"), payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
