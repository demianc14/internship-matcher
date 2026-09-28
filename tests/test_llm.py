"""Tests del cliente de Gemini. No sale a la red: el SDK está falseado."""

from types import SimpleNamespace
from typing import Any

import pytest

from src.llm import (
    INTENTOS_HTTP,
    MODELO_POR_DEFECTO,
    TIMEOUT_MS,
    CuotaAgotada,
    ErrorDelLLM,
    GeminiCliente,
    cliente_desde_env,
)


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Reemplaza google.genai.Client; `registro["crear"]` decide qué responde."""
    registro: dict[str, Any] = {}

    def client_falso(**kwargs: Any) -> Any:
        registro["kwargs"] = kwargs

        def crear(**k: Any) -> Any:
            registro["llamada"] = k
            return registro["crear"]()

        return SimpleNamespace(interactions=SimpleNamespace(create=crear))

    monkeypatch.setattr("google.genai.Client", client_falso)
    return registro


def respuesta(texto: str | None, status: str = "completed") -> Any:
    return lambda: SimpleNamespace(status=status, output_text=texto)


def lanza(error: Exception) -> Any:
    def f() -> Any:
        raise error

    return f


def test_configura_timeout_y_sin_reintentos_del_sdk(sdk: dict[str, Any]) -> None:
    GeminiCliente("fake-key")
    http = sdk["kwargs"]["http_options"]
    assert http.timeout == TIMEOUT_MS
    assert http.retry_options.attempts == INTENTOS_HTTP == 1


def test_pide_salida_estructurada_con_el_esquema(sdk: dict[str, Any]) -> None:
    sdk["crear"] = respuesta('{"a": 1}')
    schema = {"type": "object"}
    assert GeminiCliente("k", modelo="m").completar_json("prompt", schema) == '{"a": 1}'
    assert sdk["llamada"]["model"] == "m"
    assert sdk["llamada"]["response_format"]["mime_type"] == "application/json"
    assert sdk["llamada"]["response_format"]["schema"] is schema


@pytest.mark.parametrize("status", ["incomplete", "failed", "cancelled"])
def test_respuesta_no_completada_falla_explicito(sdk: dict[str, Any], status: str) -> None:
    sdk["crear"] = respuesta('{"a"', status)  # cortada: no se le pasa al parser
    with pytest.raises(ErrorDelLLM, match=status):
        GeminiCliente("k").completar_json("p", {})


@pytest.mark.parametrize("vacia", ["", "   ", None])
def test_respuesta_vacia_falla_explicito(sdk: dict[str, Any], vacia: str | None) -> None:
    sdk["crear"] = respuesta(vacia)
    with pytest.raises(ErrorDelLLM, match="vacía"):
        GeminiCliente("k").completar_json("p", {})


@pytest.mark.parametrize(
    "mensaje",
    ["Error code: 429 - {'error': {'message': 'Rate limit exceeded'}}",
     "RESOURCE_EXHAUSTED: quota exceeded"],
)  # fmt: skip
def test_la_cuota_agotada_se_distingue_y_dice_que_hacer(sdk: dict[str, Any], mensaje: str) -> None:
    sdk["crear"] = lanza(RuntimeError(mensaje))
    with pytest.raises(CuotaAgotada, match="vuelve mañana"):
        GeminiCliente("k").completar_json("p", {})


def test_un_error_cualquiera_no_se_confunde_con_cuota(sdk: dict[str, Any]) -> None:
    sdk["crear"] = lanza(TimeoutError("deadline exceeded"))
    with pytest.raises(ErrorDelLLM, match="no respondió") as exc:
        GeminiCliente("k").completar_json("p", {})
    assert not isinstance(exc.value, CuotaAgotada)


def test_cliente_desde_env(sdk: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ErrorDelLLM, match="GEMINI_API_KEY"):
        cliente_desde_env()
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.delenv("GEMINI_MODELO", raising=False)
    assert cliente_desde_env().modelo == MODELO_POR_DEFECTO
    monkeypatch.setenv("GEMINI_MODELO", "gemini-otro")
    assert cliente_desde_env().modelo == "gemini-otro"
