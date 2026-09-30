"""CLI de cv-matcher.

    python -m src.cli bullets [cv.tex]      bullets del CV con sus keywords
    python -m src.cli extract <jd.txt>      requisitos del JD (llama a Gemini, con caché)
    python -m src.cli match <jd.txt> [cv] [--rewrite]
                                            veredicto + cobertura del CV frente al JD;
                                            --rewrite sugiere reescrituras (llama a Gemini)
    python -m src.cli match-all [cv] [--extract]
                                            todos los avisos de data/jds/, ordenados; solo
                                            caché, salvo --extract (extrae los que falten)

GEMINI_API_KEY y GEMINI_MODELO se leen del entorno o de .env en la raíz del repo.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.batch import Row, jd_files, match_all, uncached
from src.cv_parser import parse_backing, parse_cv
from src.jd_extractor import ExtractionError, JDRequirements, extract_requirements
from src.llm import MODELO_POR_DEFECTO, CuotaAgotada, ErrorDelLLM, cliente_desde_env
from src.matcher import JDMatch, load_fit_policy, match
from src.rewriter import RewriteOutcome, rewrite
from src.vocabulary import load_vocabulary

ROOT = Path(__file__).parent.parent
JD_DIR = ROOT / "data" / "jds"
DEFAULT_CV = Path.home() / "Documents" / "Documentos Demi" / "Base CV.tex"


def cmd_bullets(args: list[str]) -> int:
    cv = Path(args[0]) if args else DEFAULT_CV
    for b in parse_cv(cv, load_vocabulary()):
        print(f"[{b.project}] {b.text}\n    keywords: {', '.join(b.keywords) or '—'}")
    return 0


def _extract(jd_path: str) -> JDRequirements | int:
    """Requisitos del JD, o el código de salida si falló (con el error ya impreso)."""
    try:
        client = cliente_desde_env()
        req = extract_requirements(Path(jd_path).read_text(encoding="utf-8"), client)
    except CuotaAgotada as e:
        print(f"sin cuota: {e}", file=sys.stderr)
        return 3
    except ErrorDelLLM as e:
        print(f"error del proveedor: {e}", file=sys.stderr)
        return 1
    except ExtractionError as e:
        print(f"error de extracción: {e}", file=sys.stderr)
        for i, raw in enumerate(e.raw_outputs, 1):
            print(f"--- salida cruda, intento {i} ---\n{raw}", file=sys.stderr)
        return 1
    print(f"{req.title}   [modelo: {client.modelo}]")
    return req


def cmd_extract(args: list[str]) -> int:
    if not args:
        print("falta la ruta del JD", file=sys.stderr)
        return 2
    req = _extract(args[0])
    if isinstance(req, int):
        return req
    print(f"  nivel: {req.seniority_signal}  ← {req.seniority_evidence!r}")
    print(f"  rol:   {req.role_family}  ← {req.role_evidence!r}")
    print(f"  modalidad: {req.modality}  ← {req.modality_evidence!r}")
    hours = f", {req.hours_per_week:g} h/semana" if req.hours_per_week is not None else ""
    print(f"  jornada:   {req.workload}{hours}  ← {req.workload_evidence!r}")
    print(f"  hard skills: {', '.join(req.hard_skills) or '—'}")
    print(f"  otros requisitos: {'; '.join(req.soft_requirements) or '—'}")
    print(f"  ATS: {', '.join(req.ats_keywords)}")
    return 0


def _items(values: list[str]) -> str:
    return ", ".join(values) or "—"


def print_match(m: JDMatch, top: int = 5) -> None:
    print(f"\nVEREDICTO: {m.fit.verdict.upper()}")
    for reason in m.fit.reasons:
        print(f"  · {reason}")
    n = len(m.requested)
    print(f"\nSKILLS DEL AVISO ({n} reconocidas; {len(m.unrecognized)} fuera del vocabulario)")
    print(f"  en tus bullets ({len(m.covered)}/{n}):        {_items(m.covered)}")
    print(f"  en tu CV, sin bullet ({len(m.in_cv_not_in_bullets)}/{n}): "
          f"{_items(m.in_cv_not_in_bullets)}")  # fmt: skip
    print(f"  en formación ({len(m.in_training)}/{n}):          {_items(m.in_training)}")
    print(f"  no están en tu CV ({len(m.gaps)}/{n}):     {_items(m.gaps)}")
    print(f"  no reconocidas: {_items(m.unrecognized)}")
    if m.ignored:
        print(f"  ignoradas (título/formación, no son skills): {_items(m.ignored)}")
    print(f"  cobertura en bullets {m.coverage:.0%} · respaldadas por el CV {m.backed:.0%}")

    relevant = [r for r in m.bullets if r.match_score > 0 or r.missing_keywords][:top]
    print(f"\nBULLETS MÁS RELEVANTES ({len(relevant)})")
    for r in relevant:
        print(f"  [{r.match_score:.0%}] {r.bullet.project}: {r.bullet.text[:110]}")
        print(f"        demuestra: {_items(r.matched_keywords)}")
        if r.missing_keywords:
            print(f"        tu proyecto usa y el bullet no nombra: {_items(r.missing_keywords)}")
    if m.gaps:
        print("\nLo de 'no están en tu CV' no se sugiere agregar: solo si de verdad lo tienes.")


def print_rewrites(outcomes: list[RewriteOutcome]) -> None:
    if not outcomes:
        print("\nREESCRITURAS: ninguna. El aviso no es apto, o todo lo que tu CV respalda ya")
        print("  aparece literalmente en algún bullet.")
        return
    print(f"\nREESCRITURAS ({len(outcomes)} bullet(s); cada término va a un solo bullet)")
    for o in outcomes:
        print(f"\n  [{o.target.bullet.project}] agregar: {_items(o.target.add_as)}")
        print(f"    original:  {o.target.bullet.text}")
        if o.suggested:
            print(f"    sugerido:  {o.suggested}")
        else:
            print(f"    sin sugerencia válida: {'; '.join(o.rejected_reasons)}")


def cmd_match(args: list[str]) -> int:
    do_rewrite = "--rewrite" in args
    args = [a for a in args if a != "--rewrite"]
    if not args:
        print("falta la ruta del JD", file=sys.stderr)
        return 2
    cv = Path(args[1]) if len(args) > 1 else DEFAULT_CV
    vocab = load_vocabulary()
    bullets, backing = parse_cv(cv, vocab), parse_backing(cv, vocab)
    req = _extract(args[0])
    if isinstance(req, int):
        return req
    result = match(bullets, backing, req, vocab, load_fit_policy())
    print_match(result)
    if not do_rewrite:
        return 0
    try:
        outcomes = rewrite(result, req, vocab, cliente_desde_env())
    except CuotaAgotada as e:
        print(f"\nsin cuota para reescribir: {e}", file=sys.stderr)
        return 3
    except ErrorDelLLM as e:
        print(f"\nerror del proveedor al reescribir: {e}", file=sys.stderr)
        return 1
    print_rewrites(outcomes)
    return 0


def _short(values: list[str], limit: int = 3) -> str:
    if not values:
        return "—"
    extra = f" +{len(values) - limit}" if len(values) > limit else ""
    return ", ".join(values[:limit]) + extra


def print_table(rows: list[Row], model: str) -> None:
    print(f"{len(rows)} avisos en {JD_DIR.relative_to(ROOT)}/  [modelo: {model}]\n")
    print(f"  {'veredicto':11} {'bullets':>7} {'CV':>5} {'n':>3}  {'archivo':18} {'aviso':36} "
          "carencias")  # fmt: skip
    for row in rows:
        name = Path(row.file).stem[:18]
        m = row.result
        if m is None:
            print(f"  {row.status:11} {'':>7} {'':>5} {'':>3}  {name:18} {row.detail[:70]}")
            continue
        title = m.title if len(m.title) <= 36 else m.title[:35] + "…"
        print(f"  {m.fit.verdict:11} {m.coverage:>7.0%} {m.backed:>5.0%} {len(m.requested):>3}  "
              f"{name:18} {title:36} {_short(m.gaps)}")  # fmt: skip
    print(
        "\nbullets: skills del aviso en tus bullets · CV: respaldadas en cualquier parte del CV ·"
    )
    print("n: skills reconocidas (con n chico el porcentaje dice poco). Detalle de uno:")
    print("  python -m src.cli match data/jds/<archivo>.txt")


def cmd_match_all(args: list[str]) -> int:
    do_extract = "--extract" in args
    args = [a for a in args if a != "--extract"]
    cv = Path(args[0]) if args else DEFAULT_CV
    model = os.environ.get("GEMINI_MODELO", "").strip() or MODELO_POR_DEFECTO
    files = jd_files(JD_DIR)
    pending = uncached(files, model)
    client = None
    if do_extract and pending:
        print(f"{len(pending)} aviso(s) sin extraer: hasta {2 * len(pending)} llamadas a {model}.")
        try:
            client = cliente_desde_env()
        except ErrorDelLLM as e:
            print(f"error del proveedor: {e}", file=sys.stderr)
            return 1
    vocab = load_vocabulary()
    rows = match_all(
        files, parse_cv(cv, vocab), parse_backing(cv, vocab), vocab, load_fit_policy(), model,
        client,
    )  # fmt: skip
    print_table(rows, model)
    return 3 if any("sin cuota" in r.detail for r in rows) else 0


COMMANDS = {
    "bullets": cmd_bullets,
    "extract": cmd_extract,
    "match": cmd_match,
    "match-all": cmd_match_all,
}


def main(argv: list[str]) -> int:
    load_dotenv(ROOT / ".env")
    if not argv or argv[0] not in COMMANDS:
        print(__doc__, file=sys.stderr)
        return 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
