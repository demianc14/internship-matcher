"""Extract: fetch crudo por fuente, sin lógica de negocio.

Cada fuente expone:
  - `parse_<fuente>(payload, fetched_at)`: función pura, payload JSON → RawVacante[]
  - `fetch_<fuente>(...)`: HTTP + snapshot crudo en data/raw/, nunca lanza; los
    fallos de la fuente van en `FetchResult.source_error`.

Aquí solo se mapean nombres de campo al contrato. No se limpia HTML, no se
clasifica modalidad/ubicación y no se deduplica: todo eso es trabajo de transform.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from pydantic import ValidationError

from src.etl.schema import FetchResult, RawVacante, RecordError

ARBEITNOW_URL = "https://www.arbeitnow.com/api/job-board-api"
USER_AGENT = "internship-matcher/0.1 (proyecto personal; uso no comercial)"
DEFAULT_TIMEOUT_S = 20.0


def _validation_reason(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
        for err in exc.errors()
    )


# --- Arbeitnow -----------------------------------------------------------------


def _arbeitnow_modality(remote: Any) -> str | None:
    if remote is None:
        return None
    if isinstance(remote, bool):
        return f"remote={str(remote).lower()}"
    return f"remote={remote}"


def parse_arbeitnow(
    payload: Any, fetched_at: datetime
) -> tuple[list[RawVacante], list[RecordError]]:
    """Mapea una página de Arbeitnow al contrato.

    Lanza ValueError si la forma de la página cambió (sin lista `data`): eso es un
    cambio de API, no un registro malo, y `fetch_arbeitnow` lo reporta como fallo
    de la fuente.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("respuesta de Arbeitnow sin lista 'data': ¿cambió la API?")

    records: list[RawVacante] = []
    rejected: list[RecordError] = []
    for item in payload["data"]:
        if not isinstance(item, dict):
            rejected.append(
                RecordError(source="arbeitnow", source_id=None, reason="registro no es objeto")
            )
            continue
        slug = item.get("slug")
        try:
            records.append(
                RawVacante.model_validate(
                    {
                        "source": "arbeitnow",
                        "source_id": slug,
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "company": item.get("company_name"),
                        "description_raw": item.get("description"),
                        "location_raw": item.get("location"),
                        "modality_raw": _arbeitnow_modality(item.get("remote")),
                        "employment_raw": item.get("job_types") or [],
                        "tags_raw": item.get("tags") or [],
                        "posted_at": item.get("created_at"),
                        "fetched_at": fetched_at,
                    }
                )
            )
        except ValidationError as exc:
            rejected.append(
                RecordError(
                    source="arbeitnow",
                    source_id=slug if isinstance(slug, str) else None,
                    reason=_validation_reason(exc),
                )
            )
    return records, rejected


def fetch_arbeitnow(
    session: requests.Session | None = None,
    max_pages: int = 1,
    snapshot_dir: Path | None = None,
    now: datetime | None = None,
) -> FetchResult:
    """Trae hasta `max_pages` páginas (250 vacantes c/u). Nunca lanza.

    `max_pages=1` por defecto: los términos de la API piden "please do not abuse".
    Si una página intermedia falla, se conservan las anteriores y el fallo queda
    en `source_error` (resultado parcial, explícito).
    """
    fetched_at = now or datetime.now(UTC)
    http = session or requests.Session()
    result = FetchResult(source="arbeitnow", fetched_at=fetched_at)
    pages: list[Any] = []

    url: str | None = ARBEITNOW_URL
    while url and result.pages_fetched < max_pages:
        page_no = result.pages_fetched + 1
        try:
            resp = http.get(
                url, headers={"User-Agent": USER_AGENT}, timeout=DEFAULT_TIMEOUT_S
            )
            resp.raise_for_status()
            payload = resp.json()
            records, rejected = parse_arbeitnow(payload, fetched_at)
        except (requests.RequestException, ValueError) as exc:
            result.source_error = f"página {page_no}: {type(exc).__name__}: {exc}"
            break
        pages.append(payload)
        result.records.extend(records)
        result.rejected.extend(rejected)
        result.pages_fetched = page_no
        links = payload.get("links")
        url = links.get("next") if isinstance(links, dict) else None

    if snapshot_dir is not None and pages:
        save_snapshot("arbeitnow", pages, snapshot_dir, fetched_at)
    return result


# --- Snapshots -----------------------------------------------------------------


def save_snapshot(source: str, pages: list[Any], directory: Path, fetched_at: datetime) -> Path:
    """Guarda las páginas tal cual llegaron, para poder re-procesar sin re-fetch."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{source}_{fetched_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(
        json.dumps(
            {"source": source, "fetched_at": fetched_at.isoformat(), "pages": pages},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


_PARSERS = {"arbeitnow": parse_arbeitnow}


def records_from_snapshot(path: Path) -> tuple[list[RawVacante], list[RecordError]]:
    """Re-parsea un snapshot de data/raw/ sin volver a llamar a la API."""
    snap = json.loads(path.read_text(encoding="utf-8"))
    parser = _PARSERS.get(snap.get("source"))
    if parser is None:
        raise ValueError(f"{path}: fuente desconocida {snap.get('source')!r}")
    fetched_at = datetime.fromisoformat(snap["fetched_at"])
    records: list[RawVacante] = []
    rejected: list[RecordError] = []
    for page in snap["pages"]:
        page_records, page_rejected = parser(page, fetched_at)
        records += page_records
        rejected += page_rejected
    return records, rejected
