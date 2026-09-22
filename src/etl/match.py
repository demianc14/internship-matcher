"""Match: Vacante × perfil → score explicable + veredicto de encaje.

Dos cosas separadas a propósito:
  - `score` (0–100): qué tanto encajan mis skills con la vacante. Sale de
    componentes con desglose (skills que matchean/faltan, frases más similares).
  - `fit` (apta / revisar / no_apta): restricciones prácticas (seniority,
    modalidad+ubicación, idioma, jornada). No se mezclan en el score: una vacante
    senior en alemán puede tener buen score de skills y aun así no ser para mí.

El perfil y todas las reglas numéricas viven en config/skills_profile.yaml.
"""

import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.etl.schema import Seniority, Vacante
from src.etl.transform import Vocabulary

Tier = Literal["demostrado", "listado", "en_formacion"]
Verdict = Literal["apta", "revisar", "no_apta"]
FitAction = Literal["ok", "review", "blocker"]

# Keywords de idioma del vocabulario → código ISO. No cuentan como skills.
LANGUAGE_KEYWORDS = {"english": "en", "german": "de", "spanish": "es", "french": "fr",
                     "italian": "it"}  # fmt: skip
_CEFR = ["A1", "A2", "B1", "B2", "C1", "C2"]
_LANG_NAMES = {"en": "inglés", "de": "alemán", "es": "español", "fr": "francés",
               "it": "italiano"}  # fmt: skip


# --- Perfil ----------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SkillEntry(_Strict):
    tier: Tier
    evidence: str = Field(min_length=1)


class FitRules(_Strict):
    seniority: dict[Literal["ok", "review", "blocker"], list[Seniority]]
    home_city: str = Field(min_length=1)
    hours_per_day: tuple[float, float]
    hours_per_week: tuple[float, float]
    full_time: FitAction
    max_years_experience: float = Field(ge=0)


class Calibration(_Strict):
    low: float
    high: float

    @model_validator(mode="after")
    def _ordered(self) -> "Calibration":
        if not self.low < self.high:
            raise ValueError("semantic_calibration: low debe ser < high")
        return self


class Profile(_Strict):
    cv_path: str
    tier_weights: dict[Tier, float]
    skills: dict[str, SkillEntry]
    languages: dict[str, str]
    min_working_level: str
    summary: str = Field(min_length=1)
    fit: FitRules
    score_weights: dict[Literal["skills", "semantic"], float]
    min_required_skills: int = Field(ge=1)
    semantic_calibration: Calibration

    @model_validator(mode="after")
    def _consistent(self) -> "Profile":
        missing_tiers = set(Tier.__args__) - set(self.tier_weights)  # type: ignore[attr-defined]
        if missing_tiers:
            raise ValueError(f"tier_weights sin peso para: {sorted(missing_tiers)}")
        for lang, level in {**self.languages, "min_working_level": self.min_working_level}.items():
            if level != "nativo" and level not in _CEFR:
                raise ValueError(f"nivel de idioma inválido para {lang}: {level!r}")
        return self

    def working_languages(self) -> set[str]:
        """Idiomas en los que puedo trabajar (nivel >= min_working_level)."""

        def rank(level: str) -> int:
            return len(_CEFR) if level == "nativo" else _CEFR.index(level)

        return {
            lang for lang, level in self.languages.items()
            if rank(level) >= rank(self.min_working_level)
        }  # fmt: skip


def load_profile(path: Path, vocab: Vocabulary) -> Profile:
    """Lee y valida el perfil. Fail fast si una skill no existe en el vocabulario:
    una skill que transform nunca va a extraer no podría matchear jamás."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    profile = Profile.model_validate(data)
    unknown = sorted(set(profile.skills) - set(vocab))
    if unknown:
        raise ValueError(f"{path}: skills que no están en el vocabulario: {unknown}")
    as_language = sorted(set(profile.skills) & set(LANGUAGE_KEYWORDS))
    if as_language:
        raise ValueError(f"{path}: los idiomas van en `languages`, no en `skills`: {as_language}")
    return profile


# --- Componente: skills ----------------------------------------------------------


class SkillMatch(_Strict):
    skill: str
    tier: Tier
    weight: float
    evidence: str


class SkillsBreakdown(_Strict):
    value: float | None
    matched: list[SkillMatch]
    missing: list[str]
    note: str | None = None


def score_skills(v: Vacante, profile: Profile) -> SkillsBreakdown:
    """Cobertura ponderada: Σ peso(tier) de las skills pedidas que tengo, dividido por
    max(# pedidas, min_required_skills) para no inflar vacantes con 1–2 keywords."""
    required = [k for k in v.keywords if k not in LANGUAGE_KEYWORDS]
    if not required:
        return SkillsBreakdown(
            value=None, matched=[], missing=[],
            note="la vacante no menciona ninguna skill del vocabulario",
        )  # fmt: skip
    matched = [
        SkillMatch(
            skill=k,
            tier=profile.skills[k].tier,
            weight=profile.tier_weights[profile.skills[k].tier],
            evidence=profile.skills[k].evidence,
        )
        for k in required
        if k in profile.skills
    ]
    missing = [k for k in required if k not in profile.skills]
    denominator = max(len(required), profile.min_required_skills)
    note = None
    if denominator > len(required):
        note = (f"pide solo {len(required)} skill(s) reconocida(s); "
                f"denominador mínimo {profile.min_required_skills}")  # fmt: skip
    return SkillsBreakdown(
        value=sum(m.weight for m in matched) / denominator,
        matched=matched,
        missing=missing,
        note=note,
    )


# --- Componente: semántico -------------------------------------------------------


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class SemanticPair(_Strict):
    vacancy_sentence: str
    profile_passage: str
    similarity: float


class SemanticBreakdown(_Strict):
    value: float | None
    raw_similarity: float | None
    top_pairs: list[SemanticPair]
    note: str | None = None


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\s+[•·▪]\s+|\s+-\s+(?=[A-ZÄÖÜ])")


def split_sentences(text: str, min_len: int = 30, max_len: int = 400, limit: int = 60) -> list[str]:
    out = [s.strip().lstrip("•·▪- ").strip() for s in _SENTENCE_SPLIT.split(text)]
    return [s for s in out if min_len <= len(s) <= max_len][:limit]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def score_semantic(
    v: Vacante,
    passages: Sequence[str],
    passage_vecs: Sequence[Sequence[float]],
    embedder: Embedder,
    calibration: Calibration,
    top_k: int = 3,
) -> SemanticBreakdown:
    """Similitud del TÍTULO de la vacante con el passage más parecido del perfil.

    Medido sobre el snapshot real (2026-09-21, 249 vacantes): comparar el cuerpo
    completo NO discrimina — la media de las mejores frases daba 0.464 a una
    pasantía de research y 0.467 a atención al cliente en alemán, porque todos los
    avisos comparten relleno genérico ("buscamos a alguien que…"). El título sí:
    0.522 vs 0.28 en ese mismo par. Por eso el componente usa el título, y el
    cuerpo queda para las keywords del componente de skills.
    """
    (title_vec,) = embedder.encode([v.title])
    pairs = sorted(
        (
            SemanticPair(
                vacancy_sentence=v.title,
                profile_passage=passage,
                similarity=round(_cosine(title_vec, pv), 4),
            )
            for passage, pv in zip(passages, passage_vecs, strict=True)
        ),
        key=lambda p: p.similarity,
        reverse=True,
    )
    raw = pairs[0].similarity
    span = calibration.high - calibration.low
    value = min(1.0, max(0.0, (raw - calibration.low) / span))
    return SemanticBreakdown(value=value, raw_similarity=raw, top_pairs=pairs[:top_k])


# --- Encaje (fit) ----------------------------------------------------------------


class Fit(_Strict):
    verdict: Verdict
    blockers: list[str]
    reviews: list[str]
    warnings: list[str]


def _overlaps(lo: float, hi: float, rng: tuple[float, float]) -> bool:
    return lo <= rng[1] and hi >= rng[0]


def assess_fit(v: Vacante, profile: Profile) -> Fit:
    rules = profile.fit
    blockers: list[str] = []
    reviews: list[str] = []
    warnings: list[str] = []

    def apply(action: FitAction, msg: str) -> None:
        {"blocker": blockers, "review": reviews, "ok": []}[action].append(msg)

    # Seniority
    s = v.seniority.value
    if s is None:
        warnings.append(f"seniority desconocida ({v.seniority.unresolved})")
    else:
        for action in ("blocker", "review"):
            if s in rules.seniority.get(action, []):
                apply(action, f"seniority: {s}")

    # Modalidad + ubicación. Modalidad desconocida NUNCA descarta (decisión 2026-09-22).
    m, loc, quito = v.modality.value, v.location, v.is_quito
    if m == "remote":
        if loc and not quito:
            reviews.append(f"remoto, pero con ubicación '{loc}': puede estar restringido ahí")
    elif m in ("hybrid", "onsite"):
        if quito is False:
            blockers.append(f"{m} en '{loc}', fuera de {rules.home_city.title()}")
        elif quito is None:
            reviews.append(f"{m} sin ubicación conocida")
    else:
        warnings.append(f"modalidad desconocida ({v.modality.unresolved})")
        if quito is False:
            reviews.append(f"modalidad desconocida con ubicación '{loc}': podría exigir presencia")

    # Idioma
    working = profile.working_languages()
    if v.language is None:
        warnings.append("idioma de la descripción no determinado")
    elif v.language not in working:
        name = _LANG_NAMES.get(v.language, v.language)
        blockers.append(f"descripción en {name}: no es uno de tus idiomas de trabajo")
    for kw, code in LANGUAGE_KEYWORDS.items():
        if kw in v.keywords and code not in working and code != v.language:
            warnings.append(f"menciona {_LANG_NAMES[code]} (puede ser requisito)")

    # Jornada y horas
    if v.schedule.value == "full_time":
        lo_d, hi_d = rules.hours_per_day
        apply(rules.full_time, f"jornada completa; buscas {lo_d:g}–{hi_d:g} h/día")
    h = v.hours
    if h.period is not None and h.min is not None and h.max is not None:
        rng = rules.hours_per_day if h.period == "day" else rules.hours_per_week
        if not _overlaps(h.min, h.max, rng):
            unit = "día" if h.period == "day" else "semana"
            reviews.append(
                f"{h.min:g}–{h.max:g} h/{unit} fuera de tu rango {rng[0]:g}–{rng[1]:g}"
            )

    # Años de experiencia: el texto pide más de lo que puedo acreditar
    exp = v.experience
    if exp.years is not None and exp.years > rules.max_years_experience:
        blockers.append(
            f"pide {exp.years:g}+ años de experiencia (acreditas {rules.max_years_experience:g})"
        )

    verdict: Verdict = "no_apta" if blockers else "revisar" if reviews else "apta"
    return Fit(verdict=verdict, blockers=blockers, reviews=reviews, warnings=warnings)


# --- Resultado -------------------------------------------------------------------


class MatchResult(_Strict):
    vacante_id: str
    title: str
    company: str | None
    url: str
    score: float | None
    score_components: dict[str, float]
    score_note: str | None
    skills: SkillsBreakdown
    semantic: SemanticBreakdown
    fit: Fit

    def explain(self) -> str:
        score = "n/d" if self.score is None else f"{self.score:.0f}/100"
        head = f"[{self.fit.verdict.upper()}] {score}  {self.title}"
        lines = [head + (f" — {self.company}" if self.company else ""), f"  {self.url}"]
        if self.score_components:
            parts = ", ".join(f"{k} {v * 100:.0f}" for k, v in self.score_components.items())
            note = f"  ({self.score_note})" if self.score_note else ""
            lines.append(f"  componentes: {parts}{note}")
        if self.skills.matched:
            skills = ", ".join(f"{m.skill} ({m.tier})" for m in self.skills.matched)
            lines.append(f"  ✓ skills: {skills}")
        if self.skills.missing:
            lines.append("  ✗ faltan: " + ", ".join(self.skills.missing))
        if self.skills.note:
            lines.append(f"  · {self.skills.note}")
        for p in self.semantic.top_pairs[:2]:
            lines.append(
                f'  ≈ {p.similarity:.2f} "{p.vacancy_sentence[:70]}" ↔ "{p.profile_passage[:60]}"'
            )
        lines += [f"  ⛔ {b}" for b in self.fit.blockers]
        lines += [f"  ⚠ {r}" for r in self.fit.reviews]
        lines += [f"  · {w}" for w in self.fit.warnings]
        return "\n".join(lines)


_VERDICT_ORDER = {"apta": 0, "revisar": 1, "no_apta": 2}


class Matcher:
    def __init__(
        self,
        profile: Profile,
        embedder: Embedder | None = None,
        passages: Sequence[str] | None = None,
    ) -> None:
        """`passages`: textos del perfil contra los que se compara la vacante. Por
        defecto, el resumen + la evidencia de cada skill; la CLI pasa los bullets
        del CV, que son más específicos."""
        self.profile = profile
        self.embedder = embedder
        self.passages = list(passages) if passages else [
            profile.summary.strip(), *(s.evidence for s in profile.skills.values())
        ]
        self.passage_vecs = embedder.encode(self.passages) if embedder else []

    def match(self, v: Vacante) -> MatchResult:
        skills = score_skills(v, self.profile)
        if self.embedder is None:
            semantic = SemanticBreakdown(
                value=None, raw_similarity=None, top_pairs=[], note="embeddings desactivados"
            )
        else:
            semantic = score_semantic(
                v, self.passages, self.passage_vecs, self.embedder,
                self.profile.semantic_calibration,
            )  # fmt: skip

        available = {
            name: value
            for name, value in (("skills", skills.value), ("semantic", semantic.value))
            if value is not None
        }
        weights: dict[str, float] = {str(k): w for k, w in self.profile.score_weights.items()}
        score: float | None = None
        note: str | None = None
        if available:
            total_w = sum(weights[k] for k in available)
            weighted = sum(weights[k] * value for k, value in available.items())
            score = round(100 * weighted / total_w, 1)
            if len(available) < len(weights):
                note = "solo " + ", ".join(available) + " (pesos renormalizados)"
        else:
            note = "sin componentes disponibles"

        return MatchResult(
            vacante_id=v.id,
            title=v.title,
            company=v.company,
            url=str(v.url),
            score=score,
            score_components={k: round(x, 4) for k, x in available.items()},
            score_note=note,
            skills=skills,
            semantic=semantic,
            fit=assess_fit(v, self.profile),
        )

    def match_all(self, vacantes: Sequence[Vacante]) -> list[MatchResult]:
        """Orden: no_apta siempre al final; entre apta/revisar manda el score (n/d al
        final) y el veredicto desempata. Una "apta" sin skills en común no debe quedar
        por encima de una "revisar" con buen encaje (caso real del snapshot)."""
        results = [self.match(v) for v in vacantes]
        return sorted(
            results,
            key=lambda r: (
                r.fit.verdict == "no_apta",
                r.score is None,
                -(r.score or 0),
                _VERDICT_ORDER[r.fit.verdict],
            ),
        )
