"""Tests del parser de CV y del motor de sugerencias.

El CV real (con datos personales) no está en el repo: los tests usan
tests/fixtures/cv_sample.tex, con las mismas macros y marcas. Hay un test extra
que corre contra el CV real solo si existe en la máquina."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from src.cv.parser import parse_cv, split_comment, to_plain
from src.cv.suggest import Suggester, profile_consistency
from src.etl.match import load_profile
from src.etl.schema import Hours, Resolved, Vacante
from src.etl.transform import compile_vocabulary, load_implications, load_vocabulary

ROOT = Path(__file__).parent.parent
VOCAB_PATH = ROOT / "config" / "keyword_vocabulary.yaml"
VOCAB = load_vocabulary(VOCAB_PATH)
COMPILED = compile_vocabulary(VOCAB)
IMPLIES = load_implications(VOCAB_PATH, VOCAB)
PROFILE = load_profile(ROOT / "config" / "skills_profile.yaml", VOCAB)
CV = parse_cv(Path(__file__).parent / "fixtures" / "cv_sample.tex", COMPILED)


def vac(**overrides: Any) -> Vacante:
    base: dict[str, Any] = {
        "id": "arbeitnow:x", "source": "arbeitnow", "source_id": "x",
        "title": "Data Intern", "company": "Acme", "url": "https://example.com/1",
        "description_text": "", "location": None, "is_quito": None,
        "modality": Resolved(unresolved="no_signal"),
        "seniority": Resolved(value="intern"), "schedule": Resolved(unresolved="no_signal"),
        "hours": Hours(unresolved="no_signal"), "language": "en", "keywords": [],
        "posted_at": None, "fetched_at": datetime(2026, 9, 22, tzinfo=UTC),
    }  # fmt: skip
    return Vacante.model_validate({**base, **overrides})


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
        (r"7.\,° semestre \enspace|\enspace Minor", "7.° semestre | Minor"),
    ],
)  # fmt: skip
def test_to_plain(tex: str, expected: str) -> None:
    assert to_plain(tex) == expected


def test_split_comment_respects_escaped_percent() -> None:
    assert split_comment(r"con 3\% de error % CONFIRMAR") == (r"con 3\% de error ", "CONFIRMAR")
    assert split_comment("sin comentario") == ("sin comentario", None)


# --- Parser ------------------------------------------------------------------------


def test_parses_structure() -> None:
    assert [s.name for s in CV.sections] == [
        "Perfil", "Experiencia Profesional", "Proyectos Destacados", "Habilidades Técnicas",
    ]  # fmt: skip
    assert [e.title for e in CV.entries] == [
        "Pasante de Datos", "Pipeline de Calidad", "App de Eventos",
    ]  # fmt: skip
    assert len(CV.bullets) == 4
    assert CV.entry("Pasante de Datos").bullet_ids == ["b01", "b02"]
    assert CV.entry("Pipeline de Calidad").keywords == ["pandas", "pytest", "python"]
    assert [r.label for r in CV.skill_rows] == ["Lenguajes", "BI / Analítica"]


def test_entry_details_and_section_text_feed_keywords() -> None:
    entry = CV.entry("Pasante de Datos")
    assert "Empresa Ejemplo – Quito" in entry.details
    assert "data analysis" in entry.keywords  # viene de "análisis de datos" en el detalle
    summary = CV.summary_section
    assert summary is not None and "pasantía híbrida en Quito" in summary.text


def test_annotations_attach_to_the_right_element() -> None:
    legend = [a for s in CV.sections for a in s.annotations if "leyenda" in a.note]
    assert legend == []  # los comentarios antes de \begin{document} se ignoran

    summary = CV.summary_section
    assert summary is not None and [a.tag for a in summary.annotations] == ["ADAPTAR"]

    b01 = CV.bullets[0]
    assert [a.tag for a in b01.annotations] == ["CONFIRMAR"]
    assert "y no de otro reporte" in b01.annotations[0].note  # continuación multilínea

    assert [a.tag for a in CV.bullets[2].annotations] == ["PENDIENTE"]  # inline
    assert [a.tag for a in CV.entry("App de Eventos").annotations] == ["OPCIONAL"]
    proyectos = next(s for s in CV.sections if s.name == "Proyectos Destacados")
    assert [a.tag for a in proyectos.annotations] == ["ADAPTAR"]


def test_parser_fails_fast_on_non_cv(tmp_path: Path) -> None:
    no_doc = tmp_path / "a.tex"
    no_doc.write_text(r"\documentclass{article}", encoding="utf-8")
    with pytest.raises(ValueError, match="begin"):
        parse_cv(no_doc, COMPILED)

    no_items = tmp_path / "b.tex"
    no_items.write_text(
        "\\begin{document}\n\\cvsection{Perfil}\ntexto\n\\end{document}", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="item"):
        parse_cv(no_items, COMPILED)


# --- Sugerencias -------------------------------------------------------------------


def suggester(**kw: Any) -> Suggester:
    return Suggester(CV, PROFILE, implications=IMPLIES, **kw)


def test_highlights_bullets_that_cover_requested_skills() -> None:
    s = suggester().suggest(vac(keywords=["python", "pandas"]))
    assert s.highlight[0].bullet_id == "b01" and s.highlight[0].covers == ["pandas", "python"]
    assert s.highlight[0].entry == "Pasante de Datos"


def test_bullet_with_pending_tag_carries_a_warning() -> None:
    s = suggester().suggest(vac(keywords=["etl"]))
    assert s.highlight[0].bullet_id == "b03"
    assert s.highlight[0].warnings and "PENDIENTE" in s.highlight[0].warnings[0]


def test_mentions_skill_backed_only_by_the_stack_line() -> None:
    hint = suggester().suggest(vac(keywords=["pytest"])).mention[0]
    assert hint.skill == "pytest" and "Pipeline de Calidad" in hint.where


def test_mentions_skill_implied_by_another() -> None:
    hint = suggester().suggest(vac(keywords=["sql"])).mention[0]
    assert hint.skill == "sql" and "App de Eventos" in hint.where
    assert "mysql" in hint.reason


def test_skill_in_training_is_not_presented_as_mastered() -> None:
    hint = suggester().suggest(vac(keywords=["power bi"])).mention[0]
    assert "en formación" in hint.reason


def test_gaps_are_listed_without_suggesting_to_invent_them() -> None:
    s = suggester().suggest(vac(keywords=["kubernetes", "tableau"]))
    assert s.gaps == ["kubernetes", "tableau"] and s.mention == []
    assert "no agregar sin respaldo" in s.render()


def test_reorders_only_sections_marked_adaptar() -> None:
    s = suggester().suggest(vac(keywords=["java", "spring boot"]))
    assert s.entry_order == {"Proyectos Destacados": ["App de Eventos", "Pipeline de Calidad"]}
    # Experiencia no está marcada ADAPTAR: no se reordena
    assert "Experiencia Profesional" not in s.entry_order


def test_optional_entry_is_droppable_only_when_it_adds_nothing() -> None:
    assert suggester().suggest(vac(keywords=["python"])).drop_optional == ["App de Eventos"]
    assert suggester().suggest(vac(keywords=["java"])).drop_optional == []


def test_profile_note_flags_modality_contradiction() -> None:
    remote = vac(keywords=["python"], modality=Resolved(value="remote"))
    notes = suggester().suggest(remote).profile_notes
    assert any("híbrida en Quito" in n for n in notes)
    sin_modalidad = suggester().suggest(vac(keywords=["python"])).profile_notes
    assert not any("híbrida en Quito" in n for n in sin_modalidad)


def test_suggestion_without_recognized_skills_says_so() -> None:
    s = suggester().suggest(vac(keywords=["english"]))
    assert s.highlight == [] and s.gaps == []
    assert "Sin sugerencias" in s.render()


def test_profile_consistency_audit() -> None:
    audit = profile_consistency(CV, PROFILE)
    # El CV de prueba es un subconjunto: el perfil real tiene más skills que él
    assert audit["en_cv_no_en_perfil"] == []
    assert "kubernetes" not in audit["en_perfil_no_en_cv"]


@pytest.mark.skipif(
    not Path(PROFILE.cv_path).expanduser().exists(), reason="el CV real no está en esta máquina"
)
def test_real_cv_parses_and_matches_the_profile() -> None:
    cv = parse_cv(Path(PROFILE.cv_path).expanduser(), COMPILED)
    assert len(cv.bullets) >= 15 and cv.summary_section is not None
    audit = profile_consistency(cv, PROFILE)
    assert audit == {"en_perfil_no_en_cv": [], "en_cv_no_en_perfil": []}


def test_implication_makes_a_bullet_cover_the_skill() -> None:
    """Un bullet con Power Automate cubre 'process automation' aunque no lo diga."""
    cv = parse_cv(Path(__file__).parent / "fixtures" / "cv_sample.tex", COMPILED)
    implies = {"google apps script": ["process automation"]}
    s = Suggester(cv, PROFILE, implications=implies).suggest(vac(keywords=["process automation"]))
    assert [p.bullet_id for p in s.highlight] == ["b02"]  # "…en Google Apps Script"
    assert s.gaps == [] and s.mention == []


def test_consistency_accepts_skills_backed_by_implication() -> None:
    cv = parse_cv(Path(__file__).parent / "fixtures" / "cv_sample.tex", COMPILED)
    sin = profile_consistency(cv, PROFILE)
    con = profile_consistency(cv, PROFILE, {"google apps script": ["process automation"]})
    assert "process automation" in sin["en_perfil_no_en_cv"]
    assert "process automation" not in con["en_perfil_no_en_cv"]
