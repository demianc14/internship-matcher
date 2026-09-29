"""Reescritura de bullets con LLM, acotada y validada sin LLM.

Objetivo: que un ATS encuentre términos que el CV YA respalda pero ningún bullet
nombra literalmente. Dos casos:
- implícitos: el bullet los demuestra por implicación (Power Automate ⇒ RPA,
  low-code) pero no los escribe;
- de stack: el proyecto del bullet los usa (Python en World Cup Predictor) y el
  bullet no los nombra.
Nunca entran las carencias (no están en el CV) ni lo que solo respalda una fila de
habilidades: sin un bullet que lo demuestre no hay dónde ponerlo con honestidad.

1. `select_targets` (sin LLM): solo avisos apta/revisar. Cada término va a UN solo
   bullet en todo el CV (a un ATS le basta encontrarlo una vez): primero los que lo
   demuestran por implicación, y si no hay, los cuyo proyecto lo usa; dentro de
   cada grupo, el de mayor match_score (empate ⇒ orden del CV). Máximo MAX_BULLETS.
2. Una llamada al LLM por aviso con todos los candidatos (salida estructurada).
3. `validate_rewrite` (sin LLM), por reescritura: contiene los términos pedidos; no
   agrega otras skills del vocabulario; no pierde keywords del original; todo
   número está en el original; no crece más de MAX_EXTRA_CHARS; no trae nombres
   técnicos nuevos (MAYÚSCULAS, CamelCase o con dígitos) ajenos a lo pedido.
4. Las rechazadas tienen UN reintento, solo ellas y con el motivo. Si vuelven a
   fallar: sin sugerencia y con el motivo visible. Nunca se muestra una
   reescritura que no pasó la validación.

Caché en data/cache/ por hash de (versión de prompt, prompt, modelo, payload), con
el modelo y los intentos. Las rechazadas también se cachean (como None + motivo):
volver a pedirlas gastaría cuota por el mismo resultado; `use_cache=False` fuerza.
"""

import hashlib
import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from src.cv_parser import Bullet
from src.jd_extractor import CACHE_DIR, JDRequirements
from src.llm import ClienteLLM
from src.matcher import JDMatch
from src.vocabulary import Vocabulary

PROMPT_VERSION = "1"
MAX_BULLETS = 5
MAX_EXTRA_CHARS = 60


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RewriteTarget(_Model):
    id: int
    bullet: Bullet
    add: list[str]  # canónicos del vocabulario
    add_as: list[str]  # cómo los escribe el aviso ("RPA", "Low-Code/No-Code")


class RewriteOutcome(_Model):
    target: RewriteTarget
    suggested: str | None
    rejected_reasons: list[str]  # vacío si hay sugerencia


class _Rewrite(_Model):
    id: int
    text: str


class _RewriteBatch(_Model):
    rewrites: list[_Rewrite]


# --- 1. Selección (sin LLM) --------------------------------------------------------


def surface_forms(req: JDRequirements, vocab: Vocabulary) -> dict[str, str]:
    """Canónico → primera forma en que lo escribe el aviso."""
    forms: dict[str, str] = {}
    for item in [*req.hard_skills, *req.ats_keywords]:
        for canonical in vocab.canonicalize(item):
            forms.setdefault(canonical, item.strip())
    return forms


def select_targets(m: JDMatch, req: JDRequirements, vocab: Vocabulary) -> list[RewriteTarget]:
    if m.fit.verdict == "no_apta":
        return []
    named_somewhere = {k for r in m.bullets for k in r.bullet.keywords}
    home: dict[str, int] = {}  # término → índice (en m.bullets) del bullet que lo recibe
    # Primero los bullets que DEMUESTRAN el término (por implicación) y después los que
    # solo lo tienen en el stack de su proyecto: "procedimientos almacenados" es mejor
    # lugar para SQL que un bullet de autenticación del mismo proyecto. Dentro de cada
    # pasada, m.bullets ya viene por score (empate ⇒ orden del CV).
    for i, r in enumerate(m.bullets):
        for k in sorted(set(r.matched_keywords) - set(r.bullet.keywords) - named_somewhere):
            home.setdefault(k, i)
    for i, r in enumerate(m.bullets):
        for k in sorted(set(r.missing_keywords) - named_somewhere):
            home.setdefault(k, i)

    by_bullet: dict[int, list[str]] = {}
    for k, i in home.items():
        by_bullet.setdefault(i, []).append(k)
    forms = surface_forms(req, vocab)
    targets = []
    for n, i in enumerate(sorted(by_bullet)[:MAX_BULLETS], start=1):  # mejor score primero
        add = sorted(by_bullet[i])
        targets.append(
            RewriteTarget(
                id=n, bullet=m.bullets[i].bullet, add=add, add_as=[forms.get(k, k) for k in add]
            )
        )
    return targets


# --- 3. Validación (sin LLM) -------------------------------------------------------

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_THOUSANDS = re.compile(r"^\d{1,3}(?:[.,]\d{3})+$")
_TOKEN = re.compile(r"[A-Za-zÀ-ÿ][\w+#./-]*")


def _numbers(text: str) -> set[str]:
    """'+1,000', '~400', '10,000' → {'1000', '400', '10000'}; '0.5' se conserva."""
    out = set()
    for n in _NUMBER.findall(text):
        out.add(re.sub(r"[.,]", "", n) if _THOUSANDS.match(n) else n)
    return out


def _looks_technical(token: str) -> bool:
    letters = [c for c in token if c.isalpha()]
    return (
        any(c.isdigit() for c in token)
        or (len(letters) >= 2 and all(c.isupper() for c in letters))
        or any(c.isupper() for c in token[1:])
    )


def validate_rewrite(target: RewriteTarget, text: str, vocab: Vocabulary) -> list[str]:
    """Motivos de rechazo; lista vacía = la reescritura es aceptable."""
    original = target.bullet.text
    if not text.strip():
        return ["reescritura vacía"]
    problems: list[str] = []
    before, after, wanted = set(vocab.extract(original)), set(vocab.extract(text)), set(target.add)
    if missing := wanted - after:
        problems.append(f"no incluye {sorted(missing)}")
    if extra := after - before - wanted:
        problems.append(f"agrega skills no pedidas {sorted(extra)}")
    if lost := before - after:
        problems.append(f"pierde keywords del original {sorted(lost)}")
    if invented := _numbers(text) - _numbers(original):
        problems.append(f"introduce números que no están en el original {sorted(invented)}")
    if (grown := len(text) - len(original)) > MAX_EXTRA_CHARS:
        problems.append(f"crece {grown} caracteres (máximo {MAX_EXTRA_CHARS})")
    allowed = {t.casefold() for t in _TOKEN.findall(original + " " + " ".join(target.add_as))}
    for k in target.add:
        allowed |= {t.casefold() for a in vocab.aliases[k] for t in _TOKEN.findall(a)}
    novel = sorted({t for t in _TOKEN.findall(text)
                    if _looks_technical(t) and t.casefold() not in allowed})  # fmt: skip
    if novel:
        problems.append(f"introduce nombres técnicos nuevos {novel}")
    return problems


# --- 2 y 4. Llamada al LLM con un reintento ----------------------------------------

INSTRUCTIONS = """\
Reescribes bullets de un CV en español para que un sistema ATS encuentre términos
que el candidato YA demuestra en ese bullet o en ese proyecto.

Para cada bullet agrega, de forma natural, TODOS los términos de "agregar", escritos
exactamente como aparecen ahí. Reglas:
- No agregues hechos, cifras, herramientas ni tecnologías distintas de "agregar".
- Conserva todas las herramientas, tecnologías y cifras del original.
- Mantén el español y el estilo: empieza con el mismo verbo en primera persona.
- Como máximo {max_extra} caracteres más que el original.
Devuelve una reescritura por cada id recibido.
""".format(max_extra=MAX_EXTRA_CHARS)


def _payload(targets: list[RewriteTarget]) -> list[dict[str, object]]:
    return [{"id": t.id, "bullet": t.bullet.text, "agregar": t.add_as} for t in targets]


def build_prompt(
    title: str, targets: list[RewriteTarget], rejections: dict[int, list[str]] | None = None
) -> str:
    prompt = (
        f"{INSTRUCTIONS}\nAviso: {title}\n<bullets>\n"
        f"{json.dumps(_payload(targets), ensure_ascii=False, indent=2)}\n</bullets>\n"
    )
    if rejections:
        prompt += "\nTus reescrituras anteriores de estos ids fueron rechazadas:\n"
        prompt += "".join(f"- id {i}: {'; '.join(r)}\n" for i, r in rejections.items())
        prompt += "Corrígelas respetando las reglas.\n"
    return prompt


def _cache_path(title: str, targets: list[RewriteTarget], model: str) -> Path:
    blob = json.dumps(_payload(targets), ensure_ascii=False, sort_keys=True)
    key = hashlib.sha256(
        f"rewrite\0{PROMPT_VERSION}\0{INSTRUCTIONS}\0{model}\0{title}\0{blob}".encode()
    )
    return CACHE_DIR / f"rw-{key.hexdigest()[:24]}.json"


def _ask(
    client: ClienteLLM,
    title: str,
    targets: list[RewriteTarget],
    rejections: dict[int, list[str]] | None = None,
) -> dict[int, str | None]:
    """{id: texto o None}. JSON que no valida ⇒ None para todos (cuenta como rechazo).

    Errores del proveedor (cuota, timeout, respuesta cortada) se propagan sin reintento."""
    raw = client.completar_json(
        build_prompt(title, targets, rejections), _RewriteBatch.model_json_schema()
    )
    try:
        batch = _RewriteBatch.model_validate_json(raw)
    except ValidationError:
        return {t.id: None for t in targets}
    got = {r.id: r.text for r in batch.rewrites}
    return {t.id: got.get(t.id) for t in targets}


def _reasons(target: RewriteTarget, text: str | None, vocab: Vocabulary) -> list[str]:
    if text is None:
        return ["el modelo no devolvió una reescritura válida para este bullet"]
    return validate_rewrite(target, text, vocab)


def rewrite(
    m: JDMatch,
    req: JDRequirements,
    vocab: Vocabulary,
    client: ClienteLLM,
    use_cache: bool = True,
) -> list[RewriteOutcome]:
    targets = select_targets(m, req, vocab)
    if not targets:
        return []  # sin llamada: aviso no apto o nada que reescribir con honestidad
    cache = _cache_path(req.title, targets, client.modelo)
    if use_cache and cache.exists():
        record = json.loads(cache.read_text(encoding="utf-8"))
        texts: dict[int, str | None] = {int(i): t for i, t in record["rewrites"].items()}
    else:
        texts = _ask(client, req.title, targets)
        attempts = 1
        rejected = {t.id: r for t in targets if (r := _reasons(t, texts[t.id], vocab))}
        if rejected:  # un único reintento, solo de las rechazadas y con el motivo
            retry = [t for t in targets if t.id in rejected]
            texts.update(_ask(client, req.title, retry, rejected))
            attempts = 2
        if use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            record = {"model": client.modelo, "prompt_version": PROMPT_VERSION,
                      "attempts": attempts, "first_attempt_rejections": rejected,
                      "rewrites": texts}  # fmt: skip
            cache.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    # Se valida siempre, también lo que sale del caché: las reglas pudieron cambiar.
    outcomes = []
    for t in targets:
        text = texts.get(t.id)
        reasons = _reasons(t, text, vocab)
        outcomes.append(
            RewriteOutcome(target=t, suggested=None if reasons else text, rejected_reasons=reasons)
        )
    return outcomes


def with_suggestions(m: JDMatch, outcomes: list[RewriteOutcome]) -> JDMatch:
    """Copia de `m` con `suggested_rewrite` lleno en los bullets reescritos y validados."""
    by_text = {o.target.bullet.text: o.suggested for o in outcomes if o.suggested}
    return m.model_copy(
        update={
            "bullets": [
                r.model_copy(update={"suggested_rewrite": by_text.get(r.bullet.text)})
                for r in m.bullets
            ]
        }
    )
