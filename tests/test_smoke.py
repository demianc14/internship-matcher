"""Test dummy de la Fase 0: verifica que el paquete importa y el tooling corre."""

import src
import src.api
import src.cv
import src.etl


def test_packages_import() -> None:
    for module in (src, src.etl, src.cv, src.api):
        assert module.__name__.startswith("src")
