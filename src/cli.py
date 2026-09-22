"""CLI del pipeline (Extract → Transform). Se amplía en la Fase 5.

Uso:
  python -m src.cli              # reprocesa el snapshot más reciente de data/raw/
  python -m src.cli --fetch      # trae datos frescos de Arbeitnow + RemoteOK
  python -m src.cli --semantic   # activa embeddings (requiere el extra [semantic])
  python -m src.cli --top 15     # cuántas vacantes explicar en consola
  python -m src.cli --suggest 3  # + sugerencias de CV para las 3 mejores
  python -m src.cli --nueva pasante-datos-acme   # plantilla de vacante manual

Las vacantes de data/manual/*.md se cargan siempre: son la fuente para Quito.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from src.cv.parser import parse_cv
from src.cv.suggest import Suggester, profile_consistency
from src.etl.extract import fetch_arbeitnow, fetch_remoteok, records_from_snapshot
from src.etl.manual import load_manual, write_template
from src.etl.match import Embedder, Matcher, load_profile
from src.etl.transform import (
    compile_vocabulary,
    load_cities,
    load_implications,
    load_vocabulary,
    transform,
)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
VOCAB_PATH = ROOT / "config" / "keyword_vocabulary.yaml"
PROFILE_PATH = ROOT / "config" / "skills_profile.yaml"
CITIES_PATH = ROOT / "config" / "locations.yaml"
MANUAL_DIR = ROOT / "data" / "manual"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="traer datos frescos de la API")
    parser.add_argument("--semantic", action="store_true", help="activar embeddings")
    parser.add_argument("--top", type=int, default=10, help="vacantes a explicar")
    parser.add_argument(
        "--suggest", type=int, default=0, metavar="N",
        help="sugerencias de adaptación del CV para las N mejores vacantes",
    )  # fmt: skip
    parser.add_argument(
        "--nueva", metavar="SLUG", help="crea una plantilla de vacante manual y termina"
    )
    args = parser.parse_args(argv)

    if args.nueva:
        try:
            path = write_template(MANUAL_DIR, args.nueva)
        except FileExistsError as exc:
            print(f"Ya existe: {exc}", file=sys.stderr)
            return 1
        print(f"Creado {path.relative_to(ROOT)} — llena url/title y pega el aviso.")
        return 0

    records, rejected = [], []
    if args.fetch:
        for fetch in (fetch_arbeitnow, fetch_remoteok):
            result = fetch(snapshot_dir=RAW_DIR)
            if not result.ok:  # una fuente caída no tumba el run
                print(f"[{result.source}] fuente con error: {result.source_error}", file=sys.stderr)
            records += result.records
            rejected += result.rejected
            print(f"Extract ({result.source}): {len(result.records)} válidas, "
                  f"{len(result.rejected)} rechazadas")  # fmt: skip
    else:
        latest = {}  # el snapshot más reciente de cada fuente
        for path in sorted(RAW_DIR.glob("*.json")):
            latest[path.name.rsplit("_", 1)[0]] = path
        if not latest:
            print("No hay snapshots en data/raw/. Corre con --fetch.", file=sys.stderr)
            return 1
        for source, path in latest.items():
            snap_records, snap_rejected = records_from_snapshot(path)
            records += snap_records
            rejected += snap_rejected
            print(f"Extract (snapshot {path.name}): {len(snap_records)} válidas")

    manual = load_manual(MANUAL_DIR)
    records = [*records, *manual.records]
    rejected = [*rejected, *manual.rejected]
    print(f"Extract (manual): {len(manual.records)} válidas, {len(manual.rejected)} rechazadas")

    vocab = load_vocabulary(VOCAB_PATH)
    profile = load_profile(PROFILE_PATH, vocab)  # antes de trabajar: config inválida falla ya
    vacantes, report = transform(records, vocab, rejected, cities=load_cities(CITIES_PATH))
    print()
    print(report.resumen())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "vacantes.jsonl"
    out.write_text("".join(v.model_dump_json() + "\n" for v in vacantes), encoding="utf-8")
    report_path = PROCESSED_DIR / "quality_report.json"
    report_path.write_text(
        json.dumps(
            {
                "n_input": report.n_extract_ok + len(report.extract_rejected),
                "extract_rejected": [e.model_dump() for e in report.extract_rejected],
                "duplicates_removed": [vars(d) for d in report.duplicates_removed],
                "possible_duplicates": report.possible_duplicates,
                "n_output": report.n_output,
                "counts": {k: dict(c) for k, c in report.counts().items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    embedder: Embedder | None = None
    if args.semantic:
        from src.etl.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()

    # Los bullets del CV son passages más específicos que las evidencias del perfil.
    cv_path = Path(profile.cv_path).expanduser()
    cv = parse_cv(cv_path, compile_vocabulary(vocab)) if cv_path.exists() else None
    passages = [profile.summary.strip(), *(b.text for b in cv.bullets)] if cv else None
    results = Matcher(profile, embedder, passages).match_all(vacantes)
    matches_path = PROCESSED_DIR / "matches.jsonl"
    matches_path.write_text(
        "".join(r.model_dump_json() + "\n" for r in results), encoding="utf-8"
    )

    verdicts = Counter(r.fit.verdict for r in results)
    print(f"\n=== Match ({'skills + semántico' if embedder else 'solo skills'}) ===")
    print("Veredictos: " + ", ".join(f"{k} {verdicts[k]}" for k in ("apta", "revisar", "no_apta")))
    for r in results[: args.top]:
        print()
        print(r.explain())

    if args.suggest:
        if cv is None:
            print(f"\nNo encuentro el CV en {cv_path} (cv_path en skills_profile.yaml)",
                  file=sys.stderr)  # fmt: skip
            return 1
        audit = profile_consistency(cv, profile)
        if any(audit.values()):
            print(f"\n⚠ perfil y CV desalineados: {audit}", file=sys.stderr)
        suggester = Suggester(cv, profile, embedder, load_implications(VOCAB_PATH, vocab))
        by_id = {v.id: v for v in vacantes}
        print(f"\n=== Sugerencias de CV ({cv_path.name}) ===")
        for r in results[: args.suggest]:
            print()
            print(suggester.suggest(by_id[r.vacante_id]).render())

    written = (out, report_path, matches_path)
    print("\nEscrito: " + ", ".join(str(p.relative_to(ROOT)) for p in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
