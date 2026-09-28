"""Bullets del CV × JDRequirements → JDMatch. Capa determinística: sin LLM.

Dos preguntas separadas, a propósito:

1. ¿Vale la pena mirar el aviso? (`assess_fit`) Solo con role_family y
   seniority_signal, que el extractor respalda con citas literales, contra la
   política de config/fit.yaml. El score de skills nunca cambia este veredicto:
   un Account Executive que pide Excel no se vuelve apto por pedir Excel.

2. ¿Qué parte del aviso cubre el CV? Las skills del JD (hard_skills + ats_keywords)
   se pasan a canónicos del vocabulario, y cada una cae en exactamente una de:
   - covered: algún bullet la demuestra (con implicaciones: MySQL ⇒ SQL);
   - in_cv_not_in_bullets: el CV la respalda (stack de un proyecto o fila de
     habilidades) pero ningún bullet la nombra → oportunidad de reescritura honesta;
   - in_training: solo aparece marcada "(en formación)" → no se sugiere presentarla;
   - gaps: no está en el CV → se lista, nunca se sugiere agregarla.
   Lo que el vocabulario no reconoce va a `unrecognized`: se reporta, no se descarta.

Por bullet (`MatchResult`, modelo del spec):
- match_score = |matched_keywords| / |skills del JD|;
- matched_keywords = skills del JD que el bullet demuestra, con implicaciones
  (Power Automate ⇒ process automation). Campo extra respecto del spec: sin él, el
  score no dice de dónde sale;
- missing_keywords = skills del JD que el PROYECTO de ese bullet respalda (su línea
  de stack) y el bullet no nombra. Es lo que se puede agregar a ese bullet sin
  inventar nada; "te falta todo el aviso" repetido en cada bullet no sirve.
- suggested_rewrite = None en esta capa (la reescritura con LLM llega después).
"""

from pathlib import Path
from typing import Literal, get_args

import yaml
from pydantic import BaseModel, ConfigDict

from src.cv_parser import Bullet, CVBacking
from src.jd_extractor import JDRequirements, RoleFamily, Seniority
from src.vocabulary import Vocabulary

DEFAULT_FIT_PATH = Path(__file__).parent.parent / "config" / "fit.yaml"

Verdict = Literal["apta", "revisar", "no_apta"]
_SEVERITY: dict[str, int] = {"apta": 0, "revisar": 1, "no_apta": 2}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FitPolicy(_Model):
    blocking_roles: list[RoleFamily]
    seniority: dict[Seniority, Verdict]


class Fit(_Model):
    verdict: Verdict
    reasons: list[str]


class MatchResult(_Model):
    bullet: Bullet
    match_score: float
    matched_keywords: list[str]
    missing_keywords: list[str]
    suggested_rewrite: str | None = None


class JDMatch(_Model):
    title: str
    fit: Fit
    requested: list[str]  # skills canónicas del JD (sin idiomas)
    unrecognized: list[str]  # ítems del JD que el vocabulario no reconoce
    covered: list[str]
    in_cv_not_in_bullets: list[str]
    in_training: list[str]
    gaps: list[str]
    coverage: float  # covered / requested
    backed: float  # (covered + in_cv_not_in_bullets) / requested
    bullets: list[MatchResult]  # ordenados por match_score, mayor primero


def load_fit_policy(path: Path = DEFAULT_FIT_PATH) -> FitPolicy:
    """Fail fast: YAML malformado, valores fuera de los enums o un nivel sin mapear."""
    policy = FitPolicy.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if missing := set(get_args(Seniority)) - policy.seniority.keys():
        raise ValueError(f"{path}: niveles sin veredicto: {sorted(missing)}")
    return policy


def assess_fit(req: JDRequirements, policy: FitPolicy) -> Fit:
    by_level = policy.seniority[req.seniority_signal]
    by_role: Verdict = "no_apta" if req.role_family in policy.blocking_roles else "apta"
    verdict = max(by_level, by_role, key=_SEVERITY.__getitem__)
    level_quote = f" ← {req.seniority_evidence!r}" if req.seniority_evidence else ""
    return Fit(
        verdict=verdict,
        reasons=[
            f"nivel {req.seniority_signal} ⇒ {by_level}{level_quote}",
            f"rol {req.role_family} ⇒ {by_role} ← {req.role_evidence!r}",
        ],
    )


def normalize_requirements(req: JDRequirements, vocab: Vocabulary) -> tuple[set[str], list[str]]:
    """hard_skills + ats_keywords → (canónicos sin idiomas, ítems no reconocidos)."""
    requested: set[str] = set()
    unrecognized: list[str] = []
    for item in dict.fromkeys([*req.hard_skills, *req.ats_keywords]):  # dedup, orden estable
        found = vocab.canonicalize(item)
        if found:
            requested.update(found)
        elif item.strip():
            unrecognized.append(item.strip())
    return requested - vocab.non_skills, unrecognized


def _ratio(part: set[str], whole: set[str]) -> float:
    return round(len(part) / len(whole), 3) if whole else 0.0


def match(
    bullets: list[Bullet],
    backing: CVBacking,
    req: JDRequirements,
    vocab: Vocabulary,
    policy: FitPolicy,
) -> JDMatch:
    requested, unrecognized = normalize_requirements(req, vocab)

    shown = {id(b): vocab.expand(b.keywords) for b in bullets}
    covered = requested & set().union(*shown.values())
    backed_anywhere = vocab.expand(
        [*backing.skills, *(k for ks in backing.projects.values() for k in ks)]
    )
    in_cv = (requested & backed_anywhere) - covered
    in_training = (requested & set(backing.in_training)) - covered - in_cv
    gaps = requested - covered - in_cv - in_training

    results = []
    for b in bullets:
        project_backing = vocab.expand(backing.projects.get(b.project, []))
        matched = requested & shown[id(b)]
        results.append(
            MatchResult(
                bullet=b,
                match_score=_ratio(matched, requested),
                matched_keywords=sorted(matched),
                missing_keywords=sorted((requested & project_backing) - shown[id(b)]),
            )
        )
    results.sort(key=lambda r: r.match_score, reverse=True)  # estable: empate ⇒ orden del CV

    return JDMatch(
        title=req.title,
        fit=assess_fit(req, policy),
        requested=sorted(requested),
        unrecognized=unrecognized,
        covered=sorted(covered),
        in_cv_not_in_bullets=sorted(in_cv),
        in_training=sorted(in_training),
        gaps=sorted(gaps),
        coverage=_ratio(covered, requested),
        backed=_ratio(covered | in_cv, requested),
        bullets=results,
    )
