"""Vocabulario de keywords: config/keyword_vocabulary.yaml → patrones compilados.

Es la capa determinística que comparten el parser del CV y el matcher: reconoce
términos (con alias en es/en/de) y sus implicaciones (MySQL ⇒ SQL). No usa LLM,
así que todo lo que dependa de esto se testea sin mocks.

Fail fast: un YAML malformado o una implicación con términos fuera del vocabulario
lanzan ValueError al cargar, no producen un vocabulario a medias.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from re import Pattern

import yaml

DEFAULT_PATH = Path(__file__).parent.parent / "config" / "keyword_vocabulary.yaml"


@dataclass(frozen=True)
class Vocabulary:
    aliases: dict[str, list[str]]  # canónico → alias
    implies: dict[str, list[str]]  # canónico → canónicos que demuestra
    patterns: dict[str, Pattern[str]]

    def extract(self, text: str) -> list[str]:
        """Keywords canónicas presentes en `text`, ordenadas."""
        return sorted(k for k, p in self.patterns.items() if p.search(text))

    def expand(self, keywords: Iterable[str]) -> set[str]:
        """Keywords + las que implican. Un solo nivel a propósito: encadenar
        implicaciones abriría deducciones que nadie revisó a mano."""
        found = set(keywords)
        for k in list(found):
            found.update(self.implies.get(k, []))
        return found

    def canonical(self, term: str) -> str | None:
        """'Postgres' → 'postgresql'. None si el término no está en el vocabulario."""
        hits = [k for k, p in self.patterns.items() if p.fullmatch(term.strip())]
        return hits[0] if len(hits) == 1 else None


def _compile(aliases: list[str]) -> Pattern[str]:
    # Límites propios en vez de \b: "c++" y "c#" terminan en no-palabra, y "java"
    # no debe matchear dentro de "javascript".
    return re.compile(
        r"(?<![\w+#])(?:" + "|".join(re.escape(a) for a in aliases) + r")(?![\w+#])",
        re.IGNORECASE,
    )


def load_vocabulary(path: Path = DEFAULT_PATH) -> Vocabulary:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    keywords = data.get("keywords") if isinstance(data, dict) else None
    if not isinstance(keywords, dict) or not keywords:
        raise ValueError(f"{path}: se esperaba un mapa 'keywords' no vacío")
    aliases: dict[str, list[str]] = {}
    for canonical, alias_list in keywords.items():
        if (
            not isinstance(canonical, str)
            or not isinstance(alias_list, list)
            or not alias_list
            or not all(isinstance(a, str) and a.strip() for a in alias_list)
        ):
            raise ValueError(f"{path}: entrada inválida para {canonical!r}: {alias_list!r}")
        aliases[canonical] = alias_list

    implies_raw = data.get("implies") or {}
    if not isinstance(implies_raw, dict):
        raise ValueError(f"{path}: 'implies' debe ser un mapa")
    implies: dict[str, list[str]] = {}
    for key, targets in implies_raw.items():
        if not isinstance(targets, list) or not targets:
            raise ValueError(f"{path}: implies.{key} debe ser una lista no vacía")
        unknown = [k for k in [key, *targets] if k not in aliases]
        if unknown:
            raise ValueError(f"{path}: implies.{key} usa términos fuera del vocabulario: {unknown}")
        implies[str(key)] = [str(t) for t in targets]

    return Vocabulary(
        aliases=aliases,
        implies=implies,
        patterns={k: _compile(v) for k, v in aliases.items()},
    )
