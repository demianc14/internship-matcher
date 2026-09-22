"""Smoke test: los paquetes importan y el tooling corre (viene de la Fase 0)."""

import src
import src.cv
import src.etl


def test_packages_import() -> None:
    for module in (src, src.etl, src.cv):
        assert module.__name__.startswith("src")
