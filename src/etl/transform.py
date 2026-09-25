"""Transform: RawVacante[] → Vacante[] + reporte de calidad cuantificado.

Reglas de clasificación (modalidad, seniority, jornada, horas) explícitas y con
evidencia citada. Si no hay señal confiable, el campo queda en None con el motivo
(`no_signal`, `conflict`, `weak_signal`, `unknown_value`), nunca con un default.

Prioridad de señales para modalidad: structured > location > title > description.
Se usa el primer nivel que dé alguna señal; un conflicto dentro de ese nivel deja
el campo sin resolver en vez de "bajar" a un nivel menos confiable.
"""

import html
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from re import Pattern

import yaml

from src.etl.schema import (
    Experience,
    Hours,
    RawVacante,
    RecordError,
    Resolved,
    SignalOrigin,
    Vacante,
)

# --- Limpieza de texto -----------------------------------------------------------

_BLOCK_TAGS = {"p", "br", "li", "div", "ul", "ol", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK_TAGS:
            self.chunks.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _BLOCK_TAGS:
            self.chunks.append(" ")

    def handle_data(self, data: str) -> None:
        self.chunks.append(data)


_MOJIBAKE = re.compile(r"[âÂ][\x80-\x9f]")


def fix_mojibake(text: str) -> str:
    """Repara UTF-8 leído como cp1252, que es como llega el texto de RemoteOK
    (“Iâ\x80\x99m” en vez de “I’m”).

    Solo actúa si el texto tiene ese patrón y la reinterpretación no falla; si no,
    lo devuelve tal cual (nunca corrompe un texto que ya estaba bien)."""
    if not _MOJIBAKE.search(text):
        return text
    for encoding in ("latin-1", "cp1252"):  # RemoteOK manda latin-1; cp1252 por si acaso
        try:
            repaired = text.encode(encoding, errors="strict").decode("utf-8", errors="strict")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if not _MOJIBAKE.search(repaired):
            return repaired
    return text


def clean_description(raw: str) -> str:
    """HTML (posiblemente escapado más de una vez, como en Arbeitnow) → texto plano.
    De paso repara el mojibake con el que llega RemoteOK."""
    text = fix_mojibake(raw)
    for _ in range(3):
        unescaped = html.unescape(text)
        if unescaped == text:
            break
        text = unescaped
    parser = _TextExtractor()
    parser.feed(text)
    parser.close()
    return re.sub(r"\s+", " ", "".join(parser.chunks).replace("\xa0", " ")).strip()


_LINKEDIN_JOB = re.compile(r"(?:currentJobId=|/jobs/view/(?:[^/?#]*-)?)(\d{6,})")


def canonical_url(url: str) -> str:
    """Un enlace de LinkedIn copiado desde una búsqueda trae la vacante en
    `currentJobId` más parámetros de rastreo; se reduce a /jobs/view/<id>/, que es
    el enlace permanente. Cualquier otro URL queda tal cual."""
    if "linkedin.com" not in url:
        return url
    m = _LINKEDIN_JOB.search(url)
    return f"https://www.linkedin.com/jobs/view/{m.group(1)}/" if m else url


def normalize_key(text: str | None) -> str | None:
    """Para comparar (dedup): sin acentos, minúsculas, solo alfanumérico."""
    if text is None:
        return None
    ascii_ = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    key = re.sub(r"[^a-z0-9]+", " ", ascii_.casefold()).strip()
    return key or None


def _snippet(text: str, match: re.Match[str], pad: int = 40) -> str:
    start, end = max(0, match.start() - pad), min(len(text), match.end() + pad)
    return ("…" if start else "") + text[start:end].strip() + ("…" if end < len(text) else "")


_NEGATION = re.compile(
    r"\b(?:no|not|don't|do not|does not|doesn't|without|non|cannot|can't|"
    r"kein\w*|nicht|ohne|sin|ningún)\b[^.;:!?]{0,40}$",
    re.IGNORECASE,
)


def _is_negated(text: str, match: re.Match[str]) -> bool:
    return bool(_NEGATION.search(text[max(0, match.start() - 50) : match.start()]))


def _first_hit(text: str, patterns: Sequence[Pattern[str]]) -> re.Match[str] | None:
    for pattern in patterns:
        for m in pattern.finditer(text):
            if not _is_negated(text, m):
                return m
    return None


def _compile(*patterns: str) -> list[Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


# --- Ubicación -------------------------------------------------------------------

_REMOTE_ONLY_LOCATION = re.compile(r"^\s*(?:remote|remoto)(?:\s+job)?\s*$", re.IGNORECASE)


def normalize_location(location_raw: str | None) -> str | None:
    """'' / solo espacios / 'Remote' (no es un lugar) → None. El resto, tal cual."""
    if location_raw is None:
        return None
    loc = re.sub(r"\s+", " ", location_raw).strip()
    if not loc or _REMOTE_ONLY_LOCATION.match(loc):
        return None
    return loc


Cities = dict[str, list[str]]


def load_cities(path: Path) -> Cities:
    """Lee config/locations.yaml. Config malformada ⇒ ValueError (fail fast)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    cities = data.get("cities") if isinstance(data, dict) else None
    if not isinstance(cities, dict) or not cities:
        raise ValueError(f"{path}: se esperaba un mapa 'cities' no vacío")
    for canonical, aliases in cities.items():
        ok = isinstance(aliases, list) and aliases and all(isinstance(a, str) for a in aliases)
        if not ok:
            raise ValueError(f"{path}: alias inválidos para {canonical!r}")
    return {str(c): [str(a) for a in aliases] for c, aliases in cities.items()}


def infer_location(text: str, cities: Cities) -> tuple[str | None, str | None]:
    """Ciudad mencionada en texto libre → (ciudad, evidencia). Dos ciudades distintas
    ⇒ (None, motivo): no se adivina cuál es la sede."""
    found: dict[str, str] = {}
    for canonical, aliases in cities.items():
        pattern = re.compile(
            r"(?<![\w])(?:" + "|".join(re.escape(a) for a in aliases) + r")(?![\w])",
            re.IGNORECASE,
        )
        m = pattern.search(text)
        if m:
            found[canonical] = _snippet(text, m, 30)
    if len(found) == 1:
        ((city, evidence),) = found.items()
        return city, evidence
    if len(found) > 1:
        return None, f"conflicto: {', '.join(sorted(found))}"
    return None, None


def is_quito(location: str | None) -> bool | None:
    if location is None:
        return None
    return "quito" in (normalize_key(location) or "").split()


# --- Modalidad -------------------------------------------------------------------

_MODALITY_VOCAB = {
    "remote": "remote", "remoto": "remote", "teletrabajo": "remote",
    "hybrid": "hybrid", "hibrido": "hybrid", "hibrida": "hybrid", "mixto": "hybrid",
    "onsite": "onsite", "on site": "onsite", "presencial": "onsite",
}  # fmt: skip

_REMOTE_STRONG = _compile(
    r"\b(?:fully|100\s?%|full)[\s-]remote\b",
    r"\bremote[\s-](?:first|only|position|role|job|based|opportunity|contract)\b",
    r"\b(?:position|role|job) is (?:fully )?remote\b",
    r"\bteletrabajo\b",
    r"\b(?:trabajo|modalidad) remot[oa]\b",
)
_HYBRID = _compile(
    r"\bhybrid(?:e|es|en)?[\s-](?:work\w*|role|position|model|setup|set-up|office|"
    r"schedule|arbeit\w*|job)\b",
    r"\b(?:work|working|role|position|model)\s+(?:is\s+|in\s+a\s+)?hybrid\b",
    r"\bup to \d{1,3}\s?%\s(?:remote|home[\s-]?office|mobile)",
    r"\b\d(?:\s?-\s?\d)?\s?days? (?:per|a|each) week (?:in|at|from) (?:the |our )?"
    r"(?:office|home)\b",
    r"\b(?:modalidad|trabajo|esquema) h[ií]brid[oa]\b",
)
_ONSITE = _compile(r"\bon[\s-]?site\b", r"\boffice[\s-]based\b", r"\bpresencial\b")
_REMOTE_WEAK = _compile(
    r"\bhome[\s-]?office\b",
    r"\bmobiles? arbeiten\b",
    r"\bwork(?:ing)? from home\b",
    r"\bremote work(?:ing)?\b",
    r"\bremote (?:option|possible|friendly)\b",
    r"\bwork from anywhere\b",  # suele ser beneficio de "workation" (x semanas/año)
)
_TITLE_MODALITY = {
    "remote": _compile(r"\bremote\b", r"\bremoto\b"),
    "hybrid": _compile(r"\bhybrid\b", r"\bh[ií]brido\b"),
    "onsite": _compile(r"\bon[\s-]?site\b", r"\bpresencial\b"),
}


def _labels_in(text: str, rules: dict[str, list[Pattern[str]]]) -> dict[str, re.Match[str]]:
    found: dict[str, re.Match[str]] = {}
    for label, patterns in rules.items():
        m = _first_hit(text, patterns)
        if m:
            found[label] = m
    return found


def _from_labels(
    found: dict[str, re.Match[str]], text: str, origin: SignalOrigin
) -> Resolved | None:
    if not found:
        return None
    if len(found) > 1:
        evidence = " | ".join(f"{k}: {_snippet(text, m, 20)}" for k, m in sorted(found.items()))
        return Resolved(origin=origin, evidence=evidence, unresolved="conflict")
    ((label, m),) = found.items()
    return Resolved(value=label, origin=origin, evidence=_snippet(text, m))


def classify_modality(raw: RawVacante, description: str) -> Resolved:
    # 1. Señal estructurada de la fuente
    if raw.modality_raw is not None:
        value = raw.modality_raw.strip()
        if value == "remote=true":
            return Resolved(value="remote", origin="structured", evidence=value)
        if value != "remote=false":  # remote=false = "no marcada remota": no informa
            mapped = _MODALITY_VOCAB.get(normalize_key(value) or "")
            if mapped is None:
                return Resolved(origin="structured", evidence=value, unresolved="unknown_value")
            return Resolved(value=mapped, origin="structured", evidence=value)

    # 2. Ubicación y 3. título
    for origin, text in (("location", raw.location_raw or ""), ("title", raw.title)):
        resolved = _from_labels(_labels_in(text, _TITLE_MODALITY), text, origin)  # type: ignore[arg-type]
        if resolved:
            return resolved

    # 4. Descripción
    found = _labels_in(
        description, {"remote": _REMOTE_STRONG, "hybrid": _HYBRID, "onsite": _ONSITE}
    )
    weak = _first_hit(description, _REMOTE_WEAK)
    if not found:
        if weak:
            return Resolved(
                origin="description", evidence=_snippet(description, weak), unresolved="weak_signal"
            )
        return Resolved(unresolved="no_signal")
    if set(found) == {"onsite"} and weak:
        # "on-site" + "home office": puede ser híbrido, pero no hay regla explícita
        found["remote (weak)"] = weak
    return _from_labels(found, description, "description") or Resolved(unresolved="no_signal")


# --- Seniority y jornada (solo título + employment_raw) ---------------------------

_SENIORITY = {
    "intern": _compile(
        r"\b(?:intern|internship|praktikum|praktikant(?:in)?|pasant[ea]s?|pasant[ií]a|"
        r"stagiaire|becari[oa])\b",
        r"\bstage (?:de|d')",  # francés: "Stage de 6 mois"
    ),
    "student_job": _compile(
        r"\b(?:working student|werkstudent(?:in)?|studentische \w+|student(?: college)?)\b"
    ),
    "entry": _compile(
        r"\b(?:entry(?:[\s-]level)?|junior|jr\.?|graduate|trainee|berufseinsteiger\w*)\b"
    ),
    "mid": _compile(r"\b(?:mid(?:[\s-](?:level|senior))?|intermediate)\b"),
    "senior": _compile(
        r"(?<!mid-)(?<!mid )\b(?:senior|sr\.?|principal|staff|director|head of|vp)\b",
        # "Lead" en el título es un puesto de liderazgo ("Technical Lead", "AI Solutions
        # Lead", "Lead Product Designer"). Única excepción: "Lead Generation" (ventas).
        # La versión anterior solo aceptaba "team/tech lead" y dejaba pasar 11 de 15.
        r"\blead\b(?!\s*gen)",
    ),
}  # fmt: skip
_SCHEDULE = {
    "full_time": _compile(
        r"\bfull[\s-]?time\b",
        r"\bfull(?=\s*(?:or|oder|/)\s*part)",  # "Full or part time" ⇒ ambas señales
        r"\bvollzeit\b",
        r"\btiempo completo\b",
    ),
    "part_time": _compile(
        r"\bpart[\s-]?time\b", r"\bteilzeit\b", r"\bmedio tiempo\b", r"\btiempo parcial\b"
    ),
}


# "Entry" es un nivel; intern/student_job son un tipo de puesto: no se contradicen,
# y gana la etiqueta más específica. Cualquier otra combinación es conflicto.
_COMPATIBLE = {frozenset({"entry", "intern"}): "intern",
               frozenset({"entry", "student_job"}): "student_job"}  # fmt: skip


def _classify_title_employment(
    raw: RawVacante, rules: dict[str, list[Pattern[str]]]
) -> Resolved:
    """Une etiquetas de título y employment_raw. Dos etiquetas distintas ⇒ conflicto,
    salvo las combinaciones de `_COMPATIBLE`."""
    employment = " ; ".join(raw.employment_raw)
    in_title = _labels_in(raw.title, rules)
    in_employment = _labels_in(employment, rules)
    labels = set(in_title) | set(in_employment)
    if frozenset(labels) in _COMPATIBLE:
        labels = {_COMPATIBLE[frozenset(labels)]}
    if not labels:
        return Resolved(unresolved="no_signal")
    if len(labels) > 1:
        parts = [f"title: {raw.title}"] if in_title else []
        parts += [f"employment: {employment}"] if in_employment else []
        return Resolved(evidence=" | ".join(parts), unresolved="conflict")
    (label,) = labels
    if label in in_title:
        return Resolved(value=label, origin="title", evidence=raw.title)
    return Resolved(value=label, origin="employment", evidence=employment)


def classify_seniority(raw: RawVacante) -> Resolved:
    return _classify_title_employment(raw, _SENIORITY)


def classify_schedule(raw: RawVacante) -> Resolved:
    return _classify_title_employment(raw, _SCHEDULE)


# --- Horas -----------------------------------------------------------------------

_HOURS = re.compile(
    r"\b(\d{1,2}(?:[.,]5)?)\s*(?:(?:-|–|to|bis|a)\s*(\d{1,2}(?:[.,]5)?)\s*)?"
    r"(?:hours?|hrs?|h|stunden|std\.?|horas)\s*(?:per|a|/|pro|in der|por|al|a la|each)?\s*"
    r"(week|woche|wk|semana|day|tag|d[ií]a)\b",
    re.IGNORECASE,
)
_PERIOD = {"week": "week", "woche": "week", "wk": "week", "semana": "week",
           "day": "day", "tag": "day", "dia": "day", "día": "day"}  # fmt: skip
_MAX_HOURS = {"day": 12.0, "week": 60.0}


def extract_hours(text: str) -> Hours:
    """Solo horas explícitas y plausibles ('20 hours per week', '4-6 horas al día')."""
    found: dict[tuple[float, float, str], str] = {}
    for m in _HOURS.finditer(text):
        period = _PERIOD[m.group(3).lower()]
        lo = float(m.group(1).replace(",", "."))
        hi = float(m.group(2).replace(",", ".")) if m.group(2) else lo
        if 0 < lo <= hi <= _MAX_HOURS[period]:
            found.setdefault((lo, hi, period), _snippet(text, m, 25))
    if not found:
        return Hours(unresolved="no_signal")
    if len(found) > 1:
        return Hours(evidence=" | ".join(list(found.values())[:3]), unresolved="conflict")
    ((lo, hi, period), evidence), = found.items()
    return Hours(min=lo, max=hi, period=period, evidence=evidence)  # type: ignore[arg-type]


# --- Años de experiencia ---------------------------------------------------------

_EXPERIENCE = re.compile(
    r"\b(\d{1,2})\s*(?:\+|plus)?\s*(?:-|–|to|a|bis|à)?\s*(?:\d{1,2})?\s*\+?\s*"
    r"(?:years?|yrs?|jahre|años|anos|ans)\b[^.;!?]{0,60}?"
    r"(?:experience|experien[cz]ia|erfahrung|expérience|berufserfahrung)",
    re.IGNORECASE,
)
# "With more than 40 years of experience" habla de la trayectoria de la empresa, no
# de quien postula. La guardia es ESTRECHA a propósito: una versión amplia (cualquier
# "we/our/company" cerca) descartaba requisitos reales como
# "What We're Looking For: 2+ years of experience".
_EMPRESA = re.compile(
    r"(?:with (?:more than|over)|con más de|mit über|seit über)\s*$", re.IGNORECASE
)
_MAX_YEARS = 15


def extract_experience(text: str) -> Experience:
    """Años de experiencia pedidos.

    Dos guardias, ambas medidas sobre datos reales: se ignora lo precedido por
    "with more than …" (la empresa hablando de sí misma) y todo lo que pase de 15
    años, que en el snapshot real era siempre eso mismo ("28+ years of fintech
    experience", "With 30 years of experience")."""
    found: list[tuple[float, str]] = []
    for m in _EXPERIENCE.finditer(text):
        antes = text[max(0, m.start() - 25) : m.start()]
        if _EMPRESA.search(antes):
            continue
        years = float(m.group(1))
        if 0 <= years <= _MAX_YEARS:
            found.append((years, _snippet(text, m, 25)))
    if not found:
        return Experience(unresolved="no_signal")
    years, evidence = min(found, key=lambda t: t[0])
    return Experience(years=years, evidence=evidence)


# --- Idioma ----------------------------------------------------------------------

_STOPWORDS = {
    "en": {"the", "and", "you", "with", "for", "our", "are", "will", "of", "to", "your"},
    "de": {"und", "der", "die", "das", "wir", "sie", "mit", "für", "ist", "ein", "eine", "zu"},
    "es": {"el", "los", "las", "y", "que", "con", "para", "una", "por", "del", "nuestro"},
    "fr": {"le", "les", "et", "des", "une", "pour", "avec", "vous", "nous", "est", "du"},
}


def detect_language(text: str, min_hits: int = 5, min_hits_unopposed: int = 3) -> str | None:
    """Idioma dominante por stopwords. Mezcla sin ganador claro (<2x) ⇒ None.

    Con evidencia nula de los otros idiomas basta `min_hits_unopposed`: los avisos
    locales pegados a mano suelen ser cortos (3–4 stopwords) y quedaban sin idioma.
    """
    tokens = Counter(re.findall(r"[a-zäöüßéèêàáíóúñç]+", text.casefold()))
    scores = sorted(
        ((sum(tokens[w] for w in words), lang) for lang, words in _STOPWORDS.items()),
        reverse=True,
    )
    (best, lang), (second, _) = scores[0], scores[1]
    if best >= min_hits and best >= 2 * second:
        return lang
    if second == 0 and best >= min_hits_unopposed:
        return lang
    return None


# --- Keywords --------------------------------------------------------------------

Vocabulary = dict[str, list[str]]
Implications = dict[str, list[str]]  # skill → skills que demuestra (vocabulario `implies`)


def load_vocabulary(path: Path) -> Vocabulary:
    """Lee config/keyword_vocabulary.yaml. Config malformada ⇒ ValueError (fail fast)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    keywords = data.get("keywords") if isinstance(data, dict) else None
    if not isinstance(keywords, dict) or not keywords:
        raise ValueError(f"{path}: se esperaba un mapa 'keywords' no vacío")
    vocab: Vocabulary = {}
    for canonical, aliases in keywords.items():
        if (
            not isinstance(canonical, str)
            or not isinstance(aliases, list)
            or not aliases
            or not all(isinstance(a, str) and a.strip() for a in aliases)
        ):
            raise ValueError(f"{path}: entrada inválida para {canonical!r}: {aliases!r}")
        vocab[canonical] = aliases
    return vocab


def load_implications(path: Path, vocab: Vocabulary) -> Implications:
    """Lee la sección opcional `implies`. Toda clave/valor debe existir en el vocabulario."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    implies = data.get("implies") or {}
    if not isinstance(implies, dict):
        raise ValueError(f"{path}: 'implies' debe ser un mapa")
    for key, targets in implies.items():
        if not isinstance(targets, list) or not targets:
            raise ValueError(f"{path}: implies.{key} debe ser una lista no vacía")
        unknown = [k for k in [key, *targets] if k not in vocab]
        if unknown:
            raise ValueError(f"{path}: implies.{key} usa términos fuera del vocabulario: {unknown}")
    return {str(k): [str(x) for x in v] for k, v in implies.items()}


def expand_keywords(keywords: Iterable[str], implications: Implications) -> set[str]:
    """Keywords + las que implican (Power Automate ⇒ process automation, low-code).

    Un solo nivel a propósito: las implicaciones del vocabulario son directas y
    encadenarlas abriría la puerta a deducciones que ya nadie revisó a mano."""
    found = set(keywords)
    for k in list(found):
        found.update(implications.get(k, []))
    return found


def compile_vocabulary(vocab: Vocabulary) -> dict[str, Pattern[str]]:
    return {
        canonical: re.compile(
            r"(?<![\w+#])(?:" + "|".join(re.escape(a) for a in aliases) + r")(?![\w+#])",
            re.IGNORECASE,
        )
        for canonical, aliases in vocab.items()
    }


def extract_keywords(text: str, compiled: dict[str, Pattern[str]]) -> list[str]:
    return sorted(k for k, pattern in compiled.items() if pattern.search(text))


# --- Normalización de un registro ------------------------------------------------


def normalize(
    raw: RawVacante, compiled_vocab: dict[str, Pattern[str]], cities: Cities | None = None
) -> Vacante:
    description = clean_description(raw.description_raw)
    location = normalize_location(raw.location_raw)
    location_source: str | None = "field" if location else None
    location_evidence: str | None = None
    if location is None and cities:  # sin campo de ubicación: intentar deducirla del texto
        location, location_evidence = infer_location(f"{raw.title}. {description}", cities)
        location_source = "description" if location else None
    # El título y la empresa pasan por la misma limpieza que la descripción: llegan
    # con entidades HTML ("Machine Operator &amp; Labourers") y con el mojibake de
    # RemoteOK ("Oracle Fusion Cloud Lead â€”"). Si la limpieza los vaciara, se
    # conserva el original en vez de quedarse sin título.
    title = clean_description(raw.title) or raw.title.strip()
    company = clean_description(raw.company or "") or None
    return Vacante(
        id=f"{raw.source}:{raw.source_id}",
        source=raw.source,
        source_id=raw.source_id,
        title=title,
        company=company,
        url=canonical_url(str(raw.url)),  # type: ignore[arg-type]
        description_text=description,
        location=location,
        location_source=location_source,  # type: ignore[arg-type]
        location_evidence=location_evidence,
        is_quito=is_quito(location),
        modality=classify_modality(raw, description),
        seniority=classify_seniority(raw),
        schedule=classify_schedule(raw),
        hours=extract_hours(f"{title}. {description}"),
        experience=extract_experience(description),
        language=detect_language(description),
        keywords=extract_keywords(f"{title}. {description}", compiled_vocab),
        posted_at=raw.posted_at,
        fetched_at=raw.fetched_at,
    )


# --- Dedup -----------------------------------------------------------------------


@dataclass(frozen=True)
class DuplicateRecord:
    removed_id: str
    kept_id: str
    reason: str  # "same_id" | "same_title_company_location"


def dedup(
    vacantes: Sequence[Vacante],
) -> tuple[list[Vacante], list[DuplicateRecord], list[list[str]]]:
    """Quita duplicados seguros y reporta los posibles sin tocarlos.

    - mismo id (misma vacante en dos snapshots) → se queda la más reciente
    - mismo título+empresa+ubicación normalizados (misma vacante en dos fuentes)
      → solo si empresa y ubicación existen; sin ellas no hay confianza suficiente
    - mismo título+empresa en ubicaciones distintas → se CONSERVAN (suele ser el
      mismo puesto abierto en varias ciudades) y se listan como posibles duplicados
    """
    ordered = sorted(vacantes, key=lambda v: v.fetched_at, reverse=True)
    kept: list[Vacante] = []
    removed: list[DuplicateRecord] = []
    by_id: dict[str, str] = {}
    by_content: dict[tuple[str, str, str], str] = {}

    for v in ordered:
        if v.id in by_id:
            removed.append(DuplicateRecord(v.id, by_id[v.id], "same_id"))
            continue
        title, company, loc = (normalize_key(x) for x in (v.title, v.company, v.location))
        content_key = (title, company, loc) if title and company and loc else None
        if content_key and content_key in by_content:
            removed.append(
                DuplicateRecord(v.id, by_content[content_key], "same_title_company_location")
            )
            continue
        by_id[v.id] = v.id
        if content_key:
            by_content[content_key] = v.id
        kept.append(v)

    groups: dict[tuple[str | None, str | None], list[str]] = defaultdict(list)
    for v in kept:
        if v.company:
            groups[(normalize_key(v.title), normalize_key(v.company))].append(v.id)
    possible = [ids for ids in groups.values() if len(ids) > 1]
    return kept, removed, possible


# --- Reporte de calidad ----------------------------------------------------------


def _field_counts(values: Iterable[Resolved | Hours]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for r in values:
        if r.unresolved:
            counts[f"sin resolver: {r.unresolved}"] += 1
        elif isinstance(r, Hours):
            counts[f"{r.min:g}-{r.max:g} h/{r.period}"] += 1
        else:
            counts[str(r.value)] += 1
    return counts


@dataclass
class TransformReport:
    """Cuantifica cada decisión de limpieza. Mismo espíritu que DataQualityReport
    de MetalurgicaAndina_QC: nada se descarta ni se asume sin quedar contado aquí."""

    n_extract_ok: int
    extract_rejected: list[RecordError]
    duplicates_removed: list[DuplicateRecord]
    possible_duplicates: list[list[str]]
    vacantes: list[Vacante] = field(repr=False)

    @property
    def n_output(self) -> int:
        return len(self.vacantes)

    def counts(self) -> dict[str, Counter[str]]:
        v = self.vacantes
        return {
            "modality": _field_counts(x.modality for x in v),
            "seniority": _field_counts(x.seniority for x in v),
            "schedule": _field_counts(x.schedule for x in v),
            "hours": _field_counts(x.hours for x in v),
            "experience": Counter(
                "sin resolver" if x.experience.years is None else f"{x.experience.years:g}+ años"
                for x in v
            ),
            "source": Counter(x.source for x in v),
            "language": Counter(x.language or "sin resolver" for x in v),
            "is_quito": Counter(
                "ubicación desconocida" if x.is_quito is None else str(x.is_quito) for x in v
            ),
        }

    def resumen(self) -> str:
        n_in = self.n_extract_ok + len(self.extract_rejected)
        dup_reasons = Counter(d.reason for d in self.duplicates_removed)
        lines = [
            "=== Reporte de calidad de datos: vacantes ===",
            f"Registros recibidos de las fuentes:       {n_in}",
            f"Rechazados en extract (contrato):         {len(self.extract_rejected)}",
            f"Duplicados removidos:                     {len(self.duplicates_removed)}"
            + (f"  {dict(dup_reasons)}" if dup_reasons else ""),
            f"Posibles duplicados conservados (grupos): {len(self.possible_duplicates)}",
            f"Vacantes resultantes:                     {self.n_output}",
            f"Sin descripción:                          "
            f"{sum(1 for x in self.vacantes if not x.description_text)}",
            f"Sin keywords reconocidas:                 "
            f"{sum(1 for x in self.vacantes if not x.keywords)}",
        ]
        for name, counter in self.counts().items():
            lines.append(f"\n{name}:")
            lines += [f"  {k:<32} {n}" for k, n in counter.most_common()]
        return "\n".join(lines)


def transform(
    records: Sequence[RawVacante],
    vocab: Vocabulary,
    extract_rejected: Sequence[RecordError] = (),
    cities: Cities | None = None,
) -> tuple[list[Vacante], TransformReport]:
    compiled = compile_vocabulary(vocab)
    kept, removed, possible = dedup([normalize(r, compiled, cities) for r in records])
    report = TransformReport(
        n_extract_ok=len(records),
        extract_rejected=list(extract_rejected),
        duplicates_removed=removed,
        possible_duplicates=possible,
        vacantes=kept,
    )
    return kept, report
