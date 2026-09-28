"""Tests del parser de CV y del vocabulario que usa.

El CV real (con datos personales) no está en el repo: los tests usan
tests/fixtures/cv_sample.tex, con las mismas macros y marcas. Hay un test extra
que corre contra el CV real solo si existe en la máquina."""

from pathlib import Path

import pytest

from src.cv_parser import Bullet, parse_backing, parse_cv, strip_comment, to_plain
from src.vocabulary import load_vocabulary

VOCAB = load_vocabulary()
SAMPLE = Path(__file__).parent / "fixtures" / "cv_sample.tex"
BULLETS = parse_cv(SAMPLE, VOCAB)
REAL_CV = Path.home() / "Documents" / "Documentos Demi" / "Base CV.tex"


# --- LaTeX → texto ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("tex", "expected"),
    [
        (r"\textbf{Reduje a cero} la revisión", "Reduje a cero la revisión"),
        (r"de \textasciitilde400 estudiantes", "de ~400 estudiantes"),
        (r"margen de error $\pm0.5\,\%$", "margen de error ±0.5%"),
        (r"corrección $\rho=-0.13$", "corrección ρ=-0.13"),
        (r"2023 -- presente", "2023 – presente"),
        (r"QC --- Producción", "QC — Producción"),
        (r"{\small Python, pytest \ \href{https://x.io}{\raisebox{-0.05\height}\faGithub}}",
         "Python, pytest"),
        (r"\textbf{Diseñé \emph{un} pipeline}", "Diseñé un pipeline"),
    ],
)  # fmt: skip
def test_to_plain(tex: str, expected: str) -> None:
    assert to_plain(tex) == expected


def test_strip_comment_respects_escaped_percent() -> None:
    assert strip_comment(r"con 3\% de error % CONFIRMAR") == r"con 3\% de error "
    assert strip_comment("sin comentario") == "sin comentario"


# --- Parser ------------------------------------------------------------------------


def test_one_bullet_per_item_with_its_project() -> None:
    assert [(b.project, b.text) for b in BULLETS] == [
        # "~" en LaTeX es espacio no separable, no una tilde literal.
        ("Pasante de Datos",
         "Automaticé un reporte de 500 filas con Python y pandas, con 3% de error."),
        ("Pasante de Datos", "Documenté el proceso en Google Apps Script."),
        ("Pipeline de Calidad", "Diseñé un pipeline ETL con validación fail-fast."),
        ("App de Eventos",
         "API REST con autenticación por roles, sobre una base MySQL con vistas de reportería."),
    ]  # fmt: skip


def test_author_comments_never_reach_the_bullet() -> None:
    assert not any("PENDIENTE" in b.text or "CONFIRMAR" in b.text for b in BULLETS)


def test_bullet_keywords_come_only_from_its_own_text() -> None:
    by_text = {b.text[:20]: b.keywords for b in BULLETS}
    assert by_text["Automaticé un report"] == ["pandas", "python"]
    assert by_text["Documenté el proceso"] == ["google apps script"]
    assert by_text["Diseñé un pipeline E"] == ["etl"]
    # El stack de la entrada ("Python, pandas, pytest") no se cuela en el bullet.
    assert "pytest" not in by_text["Diseñé un pipeline E"]
    assert by_text["API REST con autenti"] == ["apis", "mysql", "rest api"]


def test_bullet_is_the_spec_model() -> None:
    assert set(Bullet.model_fields) == {"project", "text", "keywords"}
    with pytest.raises(ValueError):
        Bullet(project="x", text="y", keywords=[], extra="z")  # type: ignore[call-arg]


def test_item_outside_an_entry_takes_the_section_as_project(tmp_path: Path) -> None:
    tex = tmp_path / "cv.tex"
    tex.write_text(
        "\\begin{document}\n\\cvsection{Logros}\n\\begin{itemize}\n"
        "  \\item Primer lugar en hackathon de Python.\n\\end{itemize}\n\\end{document}",
        encoding="utf-8",
    )
    assert parse_cv(tex, VOCAB) == [
        Bullet(project="Logros", text="Primer lugar en hackathon de Python.", keywords=["python"])
    ]


def test_new_section_resets_the_entry(tmp_path: Path) -> None:
    tex = tmp_path / "cv.tex"
    tex.write_text(
        "\\begin{document}\n\\cvsection{Experiencia}\n\\cventry{Pasante}{2026}\n"
        "\\begin{itemize}\n\\item a\n\\end{itemize}\n"
        "\\cvsection{Voluntariado}\n\\begin{itemize}\n\\item b\n\\end{itemize}\n\\end{document}",
        encoding="utf-8",
    )
    assert [b.project for b in parse_cv(tex, VOCAB)] == ["Pasante", "Voluntariado"]


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (r"\documentclass{article}", "begin"),
        ("\\begin{document}\n\\cvsection{Perfil}\ntexto\n\\end{document}", "item"),
        ("\\begin{document}\n\\begin{itemize}\n\\item suelto\n\\end{itemize}", "fuera"),
        ("\\begin{document}\n\\cventry{Sin cerrar}{2026\n", "llaves"),
    ],
)
def test_fails_fast_on_non_cv(tmp_path: Path, body: str, match: str) -> None:
    tex = tmp_path / "x.tex"
    tex.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        parse_cv(tex, VOCAB)


@pytest.mark.skipif(not REAL_CV.exists(), reason="el CV real no está en esta máquina")
def test_real_cv_parses() -> None:
    bullets = parse_cv(REAL_CV, VOCAB)
    assert len(bullets) >= 10
    assert all(b.text and b.project for b in bullets)
    assert {"etl", "python"} <= {k for b in bullets for k in b.keywords}


# --- Vocabulario -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("JavaScript y Java", ["java", "javascript"]),  # java no matchea dentro de javascript
        ("C++ y C#", ["c#", "c++"]),
        ("análisis de datos con Postgres", ["data analysis", "postgresql"]),
        ("herramientas de RPA", ["process automation"]),
        ("Excel at communication", []),  # verbo inglés, no la herramienta
    ],
)
def test_vocabulary_extract(text: str, expected: list[str]) -> None:
    assert VOCAB.extract(text) == expected


def test_vocabulary_expand_is_one_level() -> None:
    assert VOCAB.expand(["mysql"]) == {"mysql", "sql", "databases"}
    assert VOCAB.expand(["power automate"]) == {"power automate", "process automation", "low-code"}


def test_vocabulary_canonical() -> None:
    assert VOCAB.canonical("Postgres") == "postgresql"
    assert VOCAB.canonical("  PySpark ") == "spark"
    assert VOCAB.canonical("Radford") is None


def test_vocabulary_fails_fast_on_bad_config(tmp_path: Path) -> None:
    bad = tmp_path / "v.yaml"
    bad.write_text("keywords:\n  sql: [sql]\nimplies:\n  mysql: [sql]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fuera del vocabulario"):
        load_vocabulary(bad)
    bad.write_text("keywords: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no vacío"):
        load_vocabulary(bad)


# --- Respaldo fuera de los bullets -------------------------------------------------


def test_backing_reads_project_stacks_and_skill_rows() -> None:
    backing = parse_backing(SAMPLE, VOCAB)
    assert backing.projects == {
        "Pasante de Datos": [],
        "Pipeline de Calidad": ["pandas", "pytest", "python"],
        "App de Eventos": ["java", "mysql", "spring boot"],
    }
    assert backing.skills == ["java", "python", "sql"]


def test_backing_separates_skills_in_training() -> None:
    assert parse_backing(SAMPLE, VOCAB).in_training == ["power bi"]


def test_skill_row_items_split_outside_parentheses(tmp_path: Path) -> None:
    tex = tmp_path / "cv.tex"
    tex.write_text(
        "\\begin{document}\n"
        "\\skillrow{BD:}{MySQL (vistas, triggers), Power BI (en formación), Excel avanzado}\n"
        "\\end{document}",
        encoding="utf-8",
    )
    backing = parse_backing(tex, VOCAB)
    assert backing.skills == ["excel", "mysql"]
    assert backing.in_training == ["power bi"]
