"""Contrato compartido de vacantes crudas.

`RawVacante` es la forma que emiten TODAS las fuentes (Arbeitnow, RemoteOK, Adzuna,
CSV manual). Solo valida estructura: campos presentes, tipos y forma de URL.

Lo que requiere criterio de negocio (modalidad, ubicación, seniority, carga horaria)
se guarda como texto crudo (`*_raw`) sin clasificar. Clasificarlo es trabajo de
`transform` (Fase 2). Ejemplo: Arbeitnow manda `remote=false`, lo que NO significa
"presencial", solo que la vacante no está marcada como remota.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

Source = Literal["arbeitnow", "remoteok", "adzuna", "manual"]


class RawVacante(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=False)

    source: Source
    source_id: str = Field(min_length=1, description="ID estable dentro de la fuente")
    title: str = Field(min_length=1)
    url: HttpUrl
    company: str | None = None
    description_raw: str = Field(description="Tal cual llega (puede traer HTML escapado)")
    location_raw: str | None = Field(
        default=None, description="Texto de ubicación sin interpretar; '' se conserva"
    )
    modality_raw: str | None = Field(
        default=None,
        description="Señal de modalidad sin clasificar, p. ej. 'remote=true' o 'híbrido'",
    )
    employment_raw: list[str] = Field(
        default_factory=list, description="Tipos de empleo/seniority/jornada tal cual"
    )
    tags_raw: list[str] = Field(default_factory=list)
    posted_at: datetime | None = None
    fetched_at: datetime


class RecordError(BaseModel):
    """Registro rechazado por no cumplir el contrato. Se reporta, no se corrige."""

    model_config = ConfigDict(frozen=True)

    source: Source
    source_id: str | None
    reason: str


class FetchResult(BaseModel):
    """Resultado de una fuente. `source_error` != None ⇒ la fuente falló entera,
    y el pipeline sigue con las demás (degradación elegante)."""

    source: Source
    fetched_at: datetime
    records: list[RawVacante] = Field(default_factory=list)
    rejected: list[RecordError] = Field(default_factory=list)
    source_error: str | None = None
    pages_fetched: int = 0

    @property
    def ok(self) -> bool:
        return self.source_error is None


# --- Transform (Fase 2) ----------------------------------------------------------

Modality = Literal["remote", "hybrid", "onsite"]
Seniority = Literal["intern", "student_job", "entry", "mid", "senior"]
Schedule = Literal["full_time", "part_time"]
SignalOrigin = Literal["structured", "location", "title", "employment", "description"]
Unresolved = Literal["no_signal", "conflict", "weak_signal", "unknown_value"]


class Resolved(BaseModel):
    """Campo clasificado con su evidencia. `value=None` ⇒ `unresolved` dice por qué.

    Nunca hay default silencioso: o hay un valor con evidencia citada, o queda
    explícitamente pendiente.
    """

    model_config = ConfigDict(frozen=True)

    value: str | None = None
    origin: SignalOrigin | None = None
    evidence: str | None = None
    unresolved: Unresolved | None = None


class Hours(BaseModel):
    """Carga horaria explícita. No se convierte semana↔día (sería asumir 5 días)."""

    model_config = ConfigDict(frozen=True)

    min: float | None = None
    max: float | None = None
    period: Literal["day", "week"] | None = None
    evidence: str | None = None
    unresolved: Unresolved | None = None


class Vacante(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="'<source>:<source_id>'")
    source: Source
    source_id: str
    title: str
    company: str | None
    url: HttpUrl
    description_text: str
    location: str | None
    location_source: Literal["field", "description"] | None = Field(
        default=None, description="'description' ⇒ deducida del texto, no vino como campo"
    )
    location_evidence: str | None = None
    is_quito: bool | None
    modality: Resolved
    seniority: Resolved
    schedule: Resolved
    hours: Hours
    language: str | None = Field(description="Idioma dominante de la descripción o None")
    keywords: list[str]
    posted_at: datetime | None
    fetched_at: datetime
