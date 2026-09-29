"""Cliente de LLM. Única parte del paquete que habla con un proveedor real.

Mismo patrón que family-expense-bot: el extractor depende del protocolo
`ClienteLLM`, no de un SDK. Los tests usan un cliente falso y cambiar de proveedor
no obliga a tocar la validación.

Implementación por defecto: Gemini con salida estructurada nativa (esquema JSON),
plan gratuito. Configuración en .env: GEMINI_API_KEY y GEMINI_MODELO.
"""

import os
from typing import Any, Protocol

MODELO_POR_DEFECTO = "gemini-3.5-flash-lite"
"""El que menos cuota gratuita consume. Los modelos grandes agotan el límite diario."""

TIMEOUT_MS = 120_000
"""Con 60 s hubo 2 timeouts en ~14 llamadas (las normales tardan ~8-10 s; una corrida
de extracción + reescritura tardó 80 s en total). Cada timeout gasta cuota sin
resultado. 120 s sigue acotando la espera."""

INTENTOS_HTTP = 1
"""Sin reintentos del SDK. Su backoff ante un 429 solo gasta tiempo (y, si el
límite es por minuto, más cuota) contra un límite que ya se alcanzó. El único
reintento del sistema es el del extractor, y solo ante JSON o cita inválidos."""


class ErrorDelLLM(Exception):
    """El proveedor falló: timeout, red, respuesta vacía, bloqueada o cortada."""


class CuotaAgotada(ErrorDelLLM):
    """El proveedor respondió 429 / RESOURCE_EXHAUSTED."""


class ClienteLLM(Protocol):
    @property
    def modelo(self) -> str:
        """Entra en la clave del caché: resultados de modelos distintos no se mezclan."""
        ...

    def completar_json(self, prompt: str, schema: dict[str, Any]) -> str: ...


class GeminiCliente:
    def __init__(
        self, api_key: str, modelo: str = MODELO_POR_DEFECTO, timeout_ms: int = TIMEOUT_MS
    ) -> None:
        from google import genai  # import perezoso: solo hace falta al llamar de verdad
        from google.genai import types

        self._cliente = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=timeout_ms,
                retry_options=types.HttpRetryOptions(attempts=INTENTOS_HTTP),
            ),
        )
        self._modelo = modelo

    @property
    def modelo(self) -> str:
        return self._modelo

    def completar_json(self, prompt: str, schema: dict[str, Any]) -> str:
        try:
            interaccion: Any = self._cliente.interactions.create(
                model=self._modelo,
                input=prompt,
                response_format={"type": "text", "mime_type": "application/json", "schema": schema},
            )
        except Exception as e:
            if _es_cuota_agotada(e):
                raise CuotaAgotada(
                    f"cuota de Gemini agotada ({self._modelo}). Si el límite es por minuto, "
                    f"espera un minuto; si persiste, es el diario: vuelve mañana. Detalle: {e}"
                ) from e
            raise ErrorDelLLM(f"Gemini no respondió: {e}") from e

        if interaccion.status != "completed":
            # "incomplete" = respuesta cortada; "failed"/"cancelled" = rechazada o caída.
            raise ErrorDelLLM(f"respuesta no completada (status={interaccion.status})")
        salida = interaccion.output_text
        if not isinstance(salida, str) or not salida.strip():
            raise ErrorDelLLM("Gemini devolvió una respuesta vacía (¿bloqueada?)")
        return salida


def _es_cuota_agotada(error: Exception) -> bool:
    codigo = getattr(error, "code", None) or getattr(error, "status_code", None)
    return codigo == 429 or "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error)


def cliente_desde_env() -> GeminiCliente:
    """Lee GEMINI_API_KEY y GEMINI_MODELO (cargados antes desde .env). Sin key ⇒ error claro."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ErrorDelLLM("falta GEMINI_API_KEY en .env (https://aistudio.google.com/apikey)")
    return GeminiCliente(api_key, os.environ.get("GEMINI_MODELO", "").strip() or MODELO_POR_DEFECTO)
