"""CLI del pipeline (Extract → Transform). Se amplía en la Fase 5.

Uso:
  python -m src.cli              # reprocesa el snapshot más reciente de data/raw/
  python -m src.cli --fetch      # trae 1 página fresca de Arbeitnow y la procesa
  python -m src.cli --semantic   # activa embeddings (requiere el extra [semantic])
  python -m src.cli --top 15     # cuántas vacantes explicar en consola
  python -m src.cli --suggest 3  # + sugerencias de CV para las 3 mejores
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from src.cv.parser import parse_cv
from src.cv.suggest import Suggester, profile_consistency
from src.etl.extract import fetch_arbeitnow, records_from_snapshot
from src.etl.match import Embedder, Matcher, load_profile
from src.etl.transform import compile_vocabulary, load_implications, load_vocabulary, transform

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
VOCAB_PATH = ROOT / "config" / "keyword_vocabulary.yaml"
PROFILE_PATH = ROOT / "config" / "skills_profile.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="traer datos frescos de la API")
    parser.add_argument("--semantic", action="store_true", help="activar embeddings")
    parser.add_argument("--top", type=int, default=10, help="vacantes a explicar")
    parser.add_argument(
        "--suggest", type=int, default=0, metavar="N",
        help="sugerencias de adaptación del CV para las N mejores vacantes",
    )  # fmt: skip
    args = parser.parse_args(argv)

    if args.fetch:
        result = fetch_arbeitnow(snapshot_dir=RAW_DIR)
        if not result.ok:
            print(f"[arbeitnow] fuente con error: {result.source_error}", file=sys.stderr)
        records, rejected = result.records, result.rejected
        print(f"Extract (API): {len(records)} válidas, {len(rejected)} rechazadas")
    else:
        snapshots = sorted(RAW_DIR.glob("*.json"))
        if not snapshots:
            print("No hay snapshots en data/raw/. Corre con --fetch.", file=sys.stderr)
            return 1
        records, rejected = records_from_snapshot(snapshots[-1])
        print(f"Extract (snapshot {snapshots[-1].name}): {len(records)} válidas")

    vocab = load_vocabulary(VOCAB_PATH)
    profile = load_profile(PROFILE_PATH, vocab)  # antes de trabajar: config inválida falla ya
    vacantes, report = transform(records, vocab, rejected)
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
    results = Matcher(profile, embedder).match_all(vacantes)
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
        cv_path = Path(profile.cv_path).expanduser()
        if not cv_path.exists():
            print(f"\nNo encuentro el CV en {cv_path} (cv_path en skills_profile.yaml)",
                  file=sys.stderr)  # fmt: skip
            return 1
        cv = parse_cv(cv_path, compile_vocabulary(vocab))
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
