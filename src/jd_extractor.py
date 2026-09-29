"""Texto de un JD pegado a mano → JDRequirements, vía un LLM con salida estructurada.

El LLM hace lo que las reglas no lograron en el pipeline anterior: leer el aviso
entero y decidir qué nivel pide y qué tipo de trabajo es. Pero no se le cree a
ciegas, y con un modelo chico (el gratuito) esta capa pesa más, no menos:

- `seniority` y `role_family` son categorías cerradas (Literal), no texto libre.
- Cada una trae `*_evidence`: una cita LITERAL del aviso. `check_evidence` verifica
  sin LLM que la cita exista en el texto.
- JSON que no valida o cita que no aparece ⇒ UN reintento (con el motivo del
  rechazo en el prompt). Si vuelve a fallar ⇒ ExtractionError con la salida cruda.
- Respuesta cortada, bloqueada o cuota agotada ⇒ ErrorDelLLM / CuotaAgotada del
  cliente, sin reintento. No hay defaults silenciosos.

El extractor no conoce ningún SDK: recibe un `ClienteLLM` por inyección.

Las respuestas válidas se guardan en data/cache/ por hash de (versión de prompt,
prompt, modelo, JD), junto con el modelo que las produjo: el mismo JD no vuelve a
gastar cuota y resultados de modelos distintos nunca se mezclan.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from src.llm import ClienteLLM

PROMPT_VERSION = "3"  # 2: junior = menos de 2 años; 3: modalidad y jornada
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"


Seniority = Literal["internship", "junior", "mid", "senior", "undetermined"]
RoleFamily = Literal[
    "data", "software", "ml_ai", "automation", "it_ops", "non_technical",
]  # fmt: skip
Modality = Literal["remote", "hybrid", "onsite", "undetermined"]
Workload = Literal["part_time", "full_time", "undetermined"]
FULL_TIME_HOURS = 35  # desde aquí, horas por semana ⇒ full_time


class JDRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    hard_skills: list[str]
    soft_requirements: list[str]
    seniority_signal: Seniority
    seniority_evidence: str
    role_family: RoleFamily
    role_evidence: str
    modality: Modality
    modality_evidence: str
    workload: Workload
    hours_per_week: float | None
    workload_evidence: str
    ats_keywords: list[str]


class ExtractionError(RuntimeError):
    """El LLM respondió, pero su salida no cumple el contrato (JSON o cita)."""

    def __init__(self, message: str, raw_outputs: list[str] | None = None) -> None:
        super().__init__(message)
        self.raw_outputs = raw_outputs or []  # para diagnosticar qué devolvió el modelo


INSTRUCTIONS = """\
Extraes requisitos de avisos de empleo para que un estudiante de computación decida
si postular. Devuelve solo lo que el aviso dice; si algo no está, no lo supongas.

title: el título del cargo tal como lo nombra el aviso.

hard_skills: tecnologías, lenguajes, herramientas, plataformas y métodos técnicos
concretos que el aviso exige o valora (p. ej. "Python", "RPA", "Power BI", "ETL").
Una entrada por skill, nombre corto, sin frases. Vacío si el cargo no pide ninguna.

soft_requirements: todo lo demás que se le pide al candidato: formación, idiomas
(con nivel), años de experiencia, habilidades blandas, disponibilidad o jornada.
Frases cortas.

seniority_signal: el nivel que el aviso EXIGE al candidato.
- internship: pasantía, prácticas, internship, trainee o estudiante explícitos.
- junior: entry level, recién graduado, junior, o menos de 2 años de experiencia
  ("18 months", "1.5+ years" y "1 año" son junior).
- mid: pide de 2 a 4 años de experiencia, o dice mid/intermediate.
- senior: senior, lead, staff, principal, manager, head, director, general manager;
  o 5+ años; o el cargo dirige un equipo, es dueño de un P&L o reporta al CEO/C-level.
- undetermined: el aviso no da ninguna señal de nivel. Un título sin calificador
  ("Software Engineer") NO es señal por sí solo: busca años, responsabilidades o
  a quién reporta antes de decidir.

role_family: el TRABAJO DIARIO del cargo, no el sector de la empresa. Una empresa
de IA o cripto que contrata ventas, marketing, comunicación o RR. HH. es
non_technical.
- data: análisis de datos, BI, ingeniería de datos, estadística.
- software: desarrollo de software (backend, frontend, móvil, plataforma).
- ml_ai: construir modelos o sistemas de ML/IA (incluye LLMs y agentes).
- automation: automatización de procesos, RPA, low-code/no-code.
- it_ops: soporte, infraestructura, redes, seguridad operativa, DevOps.
- non_technical: ventas, cuentas, marketing, comunicación, RR. HH., finanzas,
  legal, operaciones de negocio o dirección general.
Si el cargo mezcla dos familias técnicas, elige la que ocupa más responsabilidades.

modality: dónde se trabaja.
- remote: remoto, 100% remoto, remote, virtual, teletrabajo.
- hybrid: híbrido, hybrid, algunos días en oficina.
- onsite: presencial, en oficina, onsite.
- undetermined: el aviso no lo dice. Una ciudad o país sola NO indica presencialidad.

workload y hours_per_week: la dedicación que pide el cargo.
- part_time: medio tiempo, part-time, o menos de 35 horas por semana.
- full_time: tiempo completo, full-time, o 35 horas por semana o más.
- undetermined: el aviso no lo dice.
hours_per_week: horas por semana si el aviso da una cifra ("6 horas diarias, de
lunes a viernes" = 30; si da un rango, el máximo). null si no da ninguna cifra.

seniority_evidence, role_evidence, modality_evidence y workload_evidence: copia
LITERAL de un fragmento del aviso (de 3 a 30 palabras) que justifique la categoría,
carácter por carácter, sin corregir erratas, sin traducir y sin puntos suspensivos.
Si la categoría es undetermined, su evidencia es "".

ats_keywords: de 5 a 15 términos que un sistema ATS buscaría en un CV para este
aviso, copiados tal como aparecen en el aviso (título del cargo, herramientas,
áreas). Sin sinónimos inventados.
"""


# --- Validación sin LLM -------------------------------------------------------------


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def check_evidence(req: JDRequirements, jd_text: str) -> None:
    """Cada evidencia es una cita literal del JD (salvo espacios y mayúsculas), y las
    horas son coherentes con su cita y con la categoría de jornada."""
    haystack = _normalize(jd_text)
    problems: list[str] = []
    pairs = [
        ("seniority_evidence", req.seniority_signal),
        ("role_evidence", req.role_family),
        ("modality_evidence", req.modality),
        ("workload_evidence", req.workload),
    ]
    for field, value in pairs:
        quote = _normalize(getattr(req, field))
        if not quote:
            if value != "undetermined":
                problems.append(f"{field} vacía")
        elif quote not in haystack:  # también si la categoría es undetermined
            problems.append(f"{field} no aparece en el JD: {getattr(req, field)!r}")

    hours = req.hours_per_week
    if hours is not None:
        if req.workload == "undetermined":
            problems.append("hours_per_week sin categoría de jornada")
        elif not re.search(r"\d", req.workload_evidence):
            problems.append("hours_per_week sin una cifra en workload_evidence")
        elif hours <= 0 or hours > 80:
            problems.append(f"hours_per_week fuera de rango: {hours}")
        elif (req.workload == "full_time") != (hours >= FULL_TIME_HOURS):
            problems.append(f"workload {req.workload} contradice {hours} h/semana")
    if problems:
        raise ExtractionError("; ".join(problems))


# --- Llamada al LLM -----------------------------------------------------------------


def build_prompt(jd_text: str, rejection: str | None = None) -> str:
    prompt = f"{INSTRUCTIONS}\n<aviso>\n{jd_text}\n</aviso>\n"
    if rejection:
        prompt += (
            f"\nTu respuesta anterior fue rechazada: {rejection}\n"
            "Corrígela. Las citas deben copiarse literalmente del aviso.\n"
        )
    return prompt


def _cache_path(jd_text: str, model: str) -> Path:
    key = hashlib.sha256(f"{PROMPT_VERSION}\0{INSTRUCTIONS}\0{model}\0{jd_text}".encode())
    return CACHE_DIR / f"{key.hexdigest()[:24]}.json"


def _parse(raw: str, jd_text: str) -> JDRequirements:
    try:
        req = JDRequirements.model_validate_json(raw)
    except ValidationError as e:
        raise ExtractionError(f"la respuesta no valida contra JDRequirements: {e}") from e
    check_evidence(req, jd_text)
    return req


def _ask(client: ClienteLLM, jd_text: str) -> tuple[JDRequirements, int]:
    """Una llamada y, si el JSON o la cita no pasan, un único reintento.

    Devuelve también cuántas llamadas hizo (1 o 2): es lo que gastó de cuota."""
    schema = JDRequirements.model_json_schema()
    raw_outputs: list[str] = []
    rejection: str | None = None
    for _ in range(2):
        raw = client.completar_json(build_prompt(jd_text, rejection), schema)
        raw_outputs.append(raw)
        try:
            return _parse(raw, jd_text), len(raw_outputs)
        except ExtractionError as e:
            rejection = str(e)
    raise ExtractionError(f"2 intentos rechazados; último motivo: {rejection}", raw_outputs)


def load_cached(jd_text: str, model: str) -> JDRequirements | None:
    cache = _cache_path(jd_text, model)
    if not cache.exists():
        return None
    record = json.loads(cache.read_text(encoding="utf-8"))
    if record.get("model") != model:  # defensa extra: el hash ya incluye el modelo
        raise ExtractionError(f"{cache}: generado por {record.get('model')!r}, no por {model!r}")
    req = JDRequirements.model_validate(record["requirements"])
    check_evidence(req, jd_text)
    return req


def extract_requirements(
    jd_text: str, client: ClienteLLM, use_cache: bool = True
) -> JDRequirements:
    if not jd_text.strip():
        raise ExtractionError("el JD está vacío")
    if use_cache and (cached := load_cached(jd_text, client.modelo)) is not None:
        return cached
    req, attempts = _ask(client, jd_text)
    if use_cache:
        cache = _cache_path(jd_text, client.modelo)
        cache.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "model": client.modelo,
            "prompt_version": PROMPT_VERSION,
            "attempts": attempts,
            "requirements": req.model_dump(),
        }
        cache.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return req
