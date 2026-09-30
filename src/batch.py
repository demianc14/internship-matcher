"""Match de todos los avisos de una carpeta, ordenados para decidir a cuál postular.

Cuidado con la cuota: sin cliente, solo usa lo que ya está en el caché y marca el
resto como "sin extraer" (cero llamadas). Con cliente, extrae lo que falte; si la
cuota se agota, deja de llamar y marca el resto, sin reintentar. Un error en un
aviso (timeout, cita inválida) no detiene los demás.

Orden: apta, revisar, no_apta; dentro de cada grupo, por cobertura en bullets,
luego por respaldo del CV y luego por cuántas skills se reconocieron (mayor primero:
un 50 % sobre 12 skills pesa más que sobre 2). Los que no tienen resultado van al final.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.cv_parser import Bullet, CVBacking
from src.jd_extractor import (
    ExtractionError,
    JDRequirements,
    extract_requirements,
    load_cached,
)
from src.llm import ClienteLLM, CuotaAgotada, ErrorDelLLM
from src.matcher import FitPolicy, JDMatch, match
from src.vocabulary import Vocabulary

Status = Literal["ok", "sin_extraer", "error"]
_VERDICT_ORDER = {"apta": 0, "revisar": 1, "no_apta": 2}


class Row(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str
    status: Status
    result: JDMatch | None = None
    detail: str = ""


def jd_files(jd_dir: Path) -> list[Path]:
    """Solo el primer nivel: data/jds/regresion/ son casos de prueba, no postulaciones."""
    return sorted(p for p in jd_dir.glob("*.txt") if p.read_text(encoding="utf-8").strip())


def uncached(files: list[Path], model: str) -> list[Path]:
    return [f for f in files if _cached_or_error(f, model) is None]


def _cached_or_error(path: Path, model: str) -> JDRequirements | ExtractionError | None:
    try:
        return load_cached(path.read_text(encoding="utf-8"), model)
    except ExtractionError as e:
        return e  # caché presente pero inválido: se reporta como error, no se re-extrae


def match_all(
    files: list[Path],
    bullets: list[Bullet],
    backing: CVBacking,
    vocab: Vocabulary,
    policy: FitPolicy,
    model: str,
    client: ClienteLLM | None = None,
) -> list[Row]:
    rows: list[Row] = []
    quota_gone = False
    for path in files:
        text = path.read_text(encoding="utf-8")
        cached = _cached_or_error(path, model)
        if isinstance(cached, ExtractionError):
            rows.append(Row(file=path.name, status="error", detail=f"caché inválido: {cached}"))
            continue
        req: JDRequirements | None = cached
        if req is None:
            if client is None or quota_gone:
                why = "sin cuota" if quota_gone else "usa --extract"
                rows.append(Row(file=path.name, status="sin_extraer", detail=why))
                continue
            try:
                req = extract_requirements(text, client)
            except CuotaAgotada as e:
                quota_gone = True
                rows.append(Row(file=path.name, status="sin_extraer", detail=f"sin cuota: {e}"))
                continue
            except (ErrorDelLLM, ExtractionError) as e:
                rows.append(Row(file=path.name, status="error", detail=str(e)))
                continue
        rows.append(
            Row(file=path.name, status="ok", result=match(bullets, backing, req, vocab, policy))
        )
    return rank(rows)


def rank(rows: list[Row]) -> list[Row]:
    def key(row: Row) -> tuple[int, float, float, int, str]:
        if row.result is None:
            return (3, 0.0, 0.0, 0, row.file)
        m = row.result
        return (_VERDICT_ORDER[m.fit.verdict], -m.coverage, -m.backed, -len(m.requested), row.file)

    return sorted(rows, key=key)
