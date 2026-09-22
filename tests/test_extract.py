"""Tests de Extract. Nunca llaman a la API real: usan un fixture recortado de una
respuesta real de Arbeitnow (2026-09-21) más dos registros inválidos a propósito."""

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import requests
from pydantic import ValidationError

from src.etl.extract import ARBEITNOW_URL, fetch_arbeitnow, parse_arbeitnow
from src.etl.schema import FetchResult, RawVacante

FIXTURE = Path(__file__).parent / "fixtures" / "arbeitnow_page1.json"
NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)


@pytest.fixture
def payload() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data


class FakeResponse:
    def __init__(self, body: Any = None, status: int = 200, bad_json: bool = False) -> None:
        self.body, self.status, self.bad_json = body, status, bad_json

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status} Server Error")

    def json(self) -> Any:
        if self.bad_json:
            raise requests.JSONDecodeError("Expecting value", "<html>", 0)
        return self.body


class FakeSession:
    """Devuelve respuestas en orden; una Exception en la cola se lanza en get()."""

    def __init__(self, *responses: FakeResponse | Exception) -> None:
        self.queue = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, **_: Any) -> FakeResponse:
        self.calls.append(url)
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _fetch(session: FakeSession, **kw: Any) -> FetchResult:
    return fetch_arbeitnow(session=session, now=NOW, **kw)  # type: ignore[arg-type]


# --- parse_arbeitnow (función pura) ----------------------------------------------


def test_parse_emits_valid_records_and_reports_invalid(payload: dict[str, Any]) -> None:
    records, rejected = parse_arbeitnow(payload, NOW)

    assert len(records) == 4
    assert all(isinstance(r, RawVacante) and r.source == "arbeitnow" for r in records)
    assert {e.source_id for e in rejected} == {
        "registro-invalido-url",
        "registro-invalido-sin-titulo",
    }
    reasons = {e.source_id: e.reason for e in rejected}
    assert reasons["registro-invalido-url"].startswith("url:")
    assert reasons["registro-invalido-sin-titulo"].startswith("title:")


def test_parse_keeps_raw_values_without_interpreting(payload: dict[str, Any]) -> None:
    by_id = {r.source_id: r for r in parse_arbeitnow(payload, NOW)[0]}

    no_location = by_id["senior-engineer-i-field-service-251145"]
    assert no_location.location_raw == ""  # vacío se conserva; transform decide
    assert no_location.modality_raw == "remote=false"  # no se traduce a "presencial"
    assert "&lt;" in no_location.description_raw  # HTML escapado intacto

    remote = by_id["remote-account-executive-deutschland-dusseldorf-204800"]
    assert remote.modality_raw == "remote=true"
    assert remote.location_raw == "Remote job"
    assert remote.employment_raw == ["Mid", "fulltime fixed term"]


def test_parse_converts_epoch_to_utc(payload: dict[str, Any]) -> None:
    record = parse_arbeitnow(payload, NOW)[0][0]
    epoch = payload["data"][0]["created_at"]
    assert record.posted_at == datetime.fromtimestamp(epoch, tz=UTC)
    assert record.fetched_at == NOW


@pytest.mark.parametrize("bad", [None, [], {"jobs": []}, {"data": "x"}])
def test_parse_raises_on_api_shape_change(bad: Any) -> None:
    with pytest.raises(ValueError, match="cambió la API"):
        parse_arbeitnow(bad, NOW)


def test_parse_rejects_non_object_item() -> None:
    records, rejected = parse_arbeitnow({"data": ["texto"]}, NOW)
    assert records == [] and rejected[0].reason == "registro no es objeto"


# --- fetch_arbeitnow (HTTP simulado) ---------------------------------------------


def test_fetch_happy_path_writes_snapshot(payload: dict[str, Any], tmp_path: Path) -> None:
    result = _fetch(FakeSession(FakeResponse(payload)), snapshot_dir=tmp_path)

    assert result.ok and result.pages_fetched == 1
    assert len(result.records) == 4 and len(result.rejected) == 2
    (snapshot,) = tmp_path.glob("arbeitnow_*.json")
    saved = json.loads(snapshot.read_text(encoding="utf-8"))
    assert saved["pages"] == [payload]  # snapshot = crudo exacto


def test_fetch_respects_max_pages(payload: dict[str, Any]) -> None:
    payload["links"]["next"] = ARBEITNOW_URL + "?page=2"
    session = FakeSession(FakeResponse(payload))
    result = _fetch(session, max_pages=1)
    assert session.calls == [ARBEITNOW_URL] and result.pages_fetched == 1


def test_fetch_keeps_partial_results_when_later_page_fails(payload: dict[str, Any]) -> None:
    page1 = copy.deepcopy(payload)
    page1["links"]["next"] = ARBEITNOW_URL + "?page=2"
    session = FakeSession(FakeResponse(page1), FakeResponse(status=503))

    result = _fetch(session, max_pages=2)

    assert not result.ok and result.source_error is not None
    assert result.source_error.startswith("página 2:")
    assert len(result.records) == 4 and result.pages_fetched == 1


@pytest.mark.parametrize(
    "failure",
    [
        requests.ConnectionError("DNS failure"),
        requests.Timeout("read timeout"),
        FakeResponse(status=500),
        FakeResponse(bad_json=True),
        FakeResponse({"unexpected": True}),
    ],
    ids=["connection", "timeout", "http500", "bad-json", "shape-change"],
)
def test_fetch_never_raises_on_source_failure(
    failure: FakeResponse | Exception, tmp_path: Path
) -> None:
    result = _fetch(FakeSession(failure), snapshot_dir=tmp_path)

    assert result.source_error is not None and result.source_error.startswith("página 1:")
    assert result.records == [] and result.pages_fetched == 0
    assert list(tmp_path.iterdir()) == []  # sin datos, sin snapshot


# --- Contrato ------------------------------------------------------------------


def test_contract_accepts_manual_source_for_quito_csv() -> None:
    v = RawVacante(
        source="manual",
        source_id="quito-001",
        title="Pasante de Datos",
        url="https://ejemplo.ec/vacante/1",  # type: ignore[arg-type]
        description_raw="Pasantía 4h diarias",
        location_raw="Quito, Ecuador",
        modality_raw="híbrido",
        fetched_at=NOW,
    )
    assert v.company is None and v.posted_at is None and v.employment_raw == []


def test_contract_rejects_unknown_fields_and_sources() -> None:
    base: dict[str, Any] = dict(
        source_id="x", title="t", url="https://a.io", description_raw="", fetched_at=NOW
    )
    with pytest.raises(ValidationError):
        RawVacante.model_validate({**base, "source": "linkedin"})
    with pytest.raises(ValidationError):
        RawVacante.model_validate({**base, "source": "manual", "modality": "remoto"})
