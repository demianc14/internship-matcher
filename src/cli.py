"""CLI de cv-matcher.

    python -m src.cli bullets [cv.tex]      bullets del CV con sus keywords
    python -m src.cli extract <jd.txt>      requisitos del JD (llama a Gemini, con caché)

GEMINI_API_KEY y GEMINI_MODELO se leen del entorno o de .env en la raíz del repo.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

from src.cv_parser import parse_cv
from src.jd_extractor import ExtractionError, extract_requirements
from src.llm import CuotaAgotada, ErrorDelLLM, cliente_desde_env
from src.vocabulary import load_vocabulary

ROOT = Path(__file__).parent.parent
DEFAULT_CV = Path.home() / "Documents" / "Documentos Demi" / "Base CV.tex"


def cmd_bullets(args: list[str]) -> int:
    cv = Path(args[0]) if args else DEFAULT_CV
    for b in parse_cv(cv, load_vocabulary()):
        print(f"[{b.project}] {b.text}\n    keywords: {', '.join(b.keywords) or '—'}")
    return 0


def cmd_extract(args: list[str]) -> int:
    if not args:
        print("falta la ruta del JD", file=sys.stderr)
        return 2
    try:
        client = cliente_desde_env()
        req = extract_requirements(Path(args[0]).read_text(encoding="utf-8"), client)
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
    print(f"  nivel: {req.seniority_signal}  ← {req.seniority_evidence!r}")
    print(f"  rol:   {req.role_family}  ← {req.role_evidence!r}")
    print(f"  hard skills: {', '.join(req.hard_skills) or '—'}")
    print(f"  otros requisitos: {'; '.join(req.soft_requirements) or '—'}")
    print(f"  ATS: {', '.join(req.ats_keywords)}")
    return 0


COMMANDS = {"bullets": cmd_bullets, "extract": cmd_extract}


def main(argv: list[str]) -> int:
    load_dotenv(ROOT / ".env")
    if not argv or argv[0] not in COMMANDS:
        print(__doc__, file=sys.stderr)
        return 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
