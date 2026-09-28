"""Tests del extractor de JDs.

Dos grupos:
- Offline (siempre corren): validación de evidencia y manejo de respuestas malas,
  con un cliente falso. No prueban al LLM, prueban lo que hacemos con su salida.
- `llm` (`pytest -m llm`): los JDs reales de regresión contra el modelo de
  GEMINI_MODELO. Usan el caché de data/cache/, así que tras la primera corrida no
  vuelven a gastar cuota. Se saltan si no hay key ni caché.
"""

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

from src.jd_extractor import (
    ExtractionError,
    JDRequirements,
    _cache_path,
    check_evidence,
    extract_requirements,
)
from src.llm import MODELO_POR_DEFECTO, CuotaAgotada, ErrorDelLLM, cliente_desde_env

ROOT = Path(__file__).parent.parent
JD = (
    "Pasante de Automatización e IA. Buscamos estudiante de últimos semestres.\n"
    "Asistir en el desarrollo de procesos automatizados con herramientas de RPA."
)


def req(**overrides: Any) -> JDRequirements:
    base: dict[str, Any] = {
        "title": "Pasante de Automatización e IA",
        "hard_skills": ["RPA"],
        "soft_requirements": ["estudiante de últimos semestres"],
        "seniority_signal": "internship",
        "seniority_evidence": "Buscamos estudiante de últimos semestres",
        "role_family": "automation",
        "role_evidence": "procesos automatizados con herramientas de RPA",
        "ats_keywords": ["RPA", "automatización"],
    }
    return JDRequirements.model_validate({**base, **overrides})


Reply = str | Exception | Callable[[], str]


class FakeClient:
    """Implementa ClienteLLM: devuelve (o lanza) las respuestas en orden."""

    def __init__(self, *replies: Reply, modelo: str = "fake-model") -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []
        self._modelo = modelo

    @property
    def modelo(self) -> str:
        return self._modelo

    def completar_json(self, prompt: str, schema: dict[str, Any]) -> str:
        assert schema["additionalProperties"] is False  # el contrato viaja al proveedor
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else reply()

    @property
    def calls(self) -> int:
        return len(self.prompts)


def as_json(r: JDRequirements) -> str:
    return r.model_dump_json()


# --- Evidencia ---------------------------------------------------------------------


def test_evidence_quoted_literally_passes() -> None:
    check_evidence(req(), JD)


def test_evidence_ignores_whitespace_and_case() -> None:
    check_evidence(req(role_evidence="PROCESOS   automatizados con\nherramientas de rpa"), JD)


def test_invented_evidence_fails_fast() -> None:
    with pytest.raises(ExtractionError, match="role_evidence no aparece"):
        check_evidence(req(role_evidence="experiencia con UiPath"), JD)


def test_paraphrased_evidence_fails_fast() -> None:
    with pytest.raises(ExtractionError, match="seniority_evidence"):
        paraphrase = "buscamos un estudiante de los últimos semestres"
        check_evidence(req(seniority_evidence=paraphrase), JD)


def test_undetermined_seniority_needs_no_evidence() -> None:
    check_evidence(req(seniority_signal="undetermined", seniority_evidence=""), JD)


def test_empty_role_evidence_fails() -> None:
    with pytest.raises(ExtractionError, match="role_evidence vacía"):
        check_evidence(req(role_evidence="  "), JD)


def test_categories_are_closed() -> None:
    with pytest.raises(ValueError):
        req(seniority_signal="entry")
    with pytest.raises(ValueError):
        req(role_family="sales")


# --- Respuesta del LLM (cliente falso) ---------------------------------------------


@pytest.fixture
def no_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr("src.jd_extractor.CACHE_DIR", tmp_path)
    return tmp_path


def test_valid_response_is_returned_and_cached(no_cache: Path) -> None:
    client = FakeClient(as_json(req()))
    assert extract_requirements(JD, client) == req()
    assert extract_requirements(JD, client) == req()
    assert client.calls == 1  # la segunda vez sale del caché


def test_cache_records_the_model_that_produced_it(no_cache: Path) -> None:
    extract_requirements(JD, FakeClient(as_json(req()), modelo="modelo-a"))
    (record,) = [json.loads(p.read_text()) for p in no_cache.iterdir()]
    assert record["model"] == "modelo-a"
    assert record["attempts"] == 1


def test_cache_records_that_the_retry_was_spent(no_cache: Path) -> None:
    extract_requirements(JD, FakeClient("no es json", as_json(req())))
    (record,) = [json.loads(p.read_text()) for p in no_cache.iterdir()]
    assert record["attempts"] == 2


def test_cache_never_mixes_models(no_cache: Path) -> None:
    a = FakeClient(as_json(req()), modelo="modelo-a")
    b = FakeClient(as_json(req(role_family="ml_ai")), modelo="modelo-b")
    assert extract_requirements(JD, a).role_family == "automation"
    assert extract_requirements(JD, b).role_family == "ml_ai"  # no reutiliza lo de modelo-a
    assert (a.calls, b.calls) == (1, 1)
    assert _cache_path(JD, "modelo-a") != _cache_path(JD, "modelo-b")


@pytest.mark.parametrize(
    "error", [CuotaAgotada("cuota agotada"), ErrorDelLLM("respuesta no completada")]
)
def test_provider_errors_propagate_without_retry(no_cache: Path, error: Exception) -> None:
    """Cuota agotada, respuesta cortada o bloqueada: error visible, sin reintentar."""
    client = FakeClient(error, as_json(req()))
    with pytest.raises(type(error)):
        extract_requirements(JD, client)
    assert client.calls == 1


def test_invalid_json_is_retried_once(no_cache: Path) -> None:
    client = FakeClient('{"title": "Pasante"', as_json(req()))
    assert extract_requirements(JD, client) == req()
    assert client.calls == 2
    assert "rechazada" in client.prompts[1] and "JDRequirements" in client.prompts[1]


def test_hallucinated_evidence_is_retried_once_with_the_reason(no_cache: Path) -> None:
    client = FakeClient(as_json(req(role_evidence="experiencia con UiPath")), as_json(req()))
    assert extract_requirements(JD, client) == req()
    assert "experiencia con UiPath" in client.prompts[1]


def test_second_failure_is_explicit_and_keeps_raw_outputs(no_cache: Path) -> None:
    bad = as_json(req(role_evidence="cita inventada"))
    client = FakeClient("no es json", bad, as_json(req()))
    with pytest.raises(ExtractionError, match="2 intentos") as exc:
        extract_requirements(JD, client)
    assert exc.value.raw_outputs == ["no es json", bad]
    assert client.calls == 2  # nunca un tercer intento
    assert list(no_cache.iterdir()) == []  # lo rechazado no se cachea


def test_categories_outside_the_enum_are_rejected(no_cache: Path) -> None:
    out_of_enum = as_json(req()).replace('"automation"', '"sales"')
    with pytest.raises(ExtractionError):
        extract_requirements(JD, FakeClient(out_of_enum, out_of_enum))


def test_empty_jd_fails_before_calling() -> None:
    client = FakeClient(as_json(req()))
    with pytest.raises(ExtractionError, match="vacío"):
        extract_requirements("  \n", client)
    assert client.calls == 0


# --- Regresión contra Claude (pytest -m llm) ---------------------------------------
# Los cargos que el pipeline anterior marcó mal. Donde el aviso admite dos lecturas
# razonables se acepta cualquiera de las dos; lo que no se acepta es el error viejo.

load_dotenv(ROOT / ".env")
JDS = ROOT / "data" / "jds"
REGRESSION: list[tuple[str, set[str], set[str]]] = [
    # archivo, niveles aceptables, familias aceptables
    # "At least 18 months": menos de 2 años ⇒ junior (prompt v2 cerró el hueco 1–2 años).
    ("regresion/account_executive_smb.txt", {"junior"}, {"non_technical"}),
    ("regresion/social_comms.txt", {"junior", "mid", "senior", "undetermined"},
     {"non_technical"}),
    ("regresion/total_rewards_specialist.txt", {"mid", "senior"}, {"non_technical"}),
    ("regresion/gm_ai_services.txt", {"senior"}, {"non_technical"}),
    ("regresion/software_engineer_jvm.txt", {"mid", "senior"}, {"software", "data"}),
    # "1.5+ years": junior. Con prompt v1 este salió junior y el de arriba mid.
    ("regresion/ai_engineer_full_time.txt", {"junior"}, {"ml_ai", "software"}),
    ("strategia.txt", {"internship"}, {"automation", "ml_ai"}),
]  # fmt: skip


@pytest.mark.llm
@pytest.mark.parametrize(("name", "levels", "families"), REGRESSION)
def test_regression_jds(name: str, levels: set[str], families: set[str]) -> None:
    path = JDS / name
    if not path.exists():
        pytest.skip(f"{name} no está en esta máquina")
    text = path.read_text(encoding="utf-8")
    model = os.environ.get("GEMINI_MODELO", "").strip() or MODELO_POR_DEFECTO
    if not os.environ.get("GEMINI_API_KEY") and not _cache_path(text, model).exists():
        pytest.skip("sin GEMINI_API_KEY ni respuesta en caché")
    # Sin key, un cliente que solo sirve para leer el caché (falla si intenta llamar).
    client = cliente_desde_env() if os.environ.get("GEMINI_API_KEY") else FakeClient(modelo=model)
    r = extract_requirements(text, client)
    where = f"[{client.modelo}]"
    assert r.seniority_signal in levels, f"{where} {r.seniority_signal} ← {r.seniority_evidence!r}"
    assert r.role_family in families, f"{where} {r.role_family} ← {r.role_evidence!r}"
