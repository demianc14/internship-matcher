"""Parser de Base CV.tex → estructura con bullets, entradas, skills y anotaciones.

Entiende las macros del CV (`\\cvsection`, `\\cventry`, `\\skillrow`, `\\item`) y
las anotaciones de trabajo del autor en comentarios:
  % ADAPTAR / CONFIRMAR / AGREGAR / PENDIENTE / OPCIONAL
Un comentario con tag en su propia línea se asocia al SIGUIENTE elemento
(sección, entrada o bullet); las líneas de comentario sin tag que lo siguen son
continuación de su nota. Un comentario con tag al final de una línea de código se
asocia a ESE elemento. Los comentarios antes de \\begin{document} son la leyenda del
archivo y se ignoran.

Fail fast: un .tex sin \\begin{document}, sin secciones o sin bullets no es un CV
que este parser entienda, y lanza ValueError en vez de devolver algo vacío.
"""

import re
from pathlib import Path
from re import Pattern
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.etl.transform import extract_keywords

Tag = Literal["ADAPTAR", "CONFIRMAR", "AGREGAR", "PENDIENTE", "OPCIONAL"]
_TAG = re.compile(r"^\s*(ADAPTAR|CONFIRMAR|AGREGAR|PENDIENTE|OPCIONAL)\b[:\s]*(.*)$")
_SUMMARY_SECTIONS = {"perfil", "profile", "summary", "resumen"}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Annotation(_Model):
    tag: Tag
    note: str
    line: int


class Bullet(_Model):
    id: str
    section: str
    entry: str | None
    text: str
    line: int
    annotations: list[Annotation] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class Entry(_Model):
    title: str
    meta: str
    section: str
    line: int
    details: list[str] = Field(default_factory=list)
    annotations: list[Annotation] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    bullet_ids: list[str] = Field(default_factory=list)


class Section(_Model):
    name: str
    line: int
    text: str = ""
    annotations: list[Annotation] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class SkillRow(_Model):
    label: str
    items: str
    line: int
    keywords: list[str] = Field(default_factory=list)


class CVDocument(_Model):
    path: str
    sections: list[Section]
    entries: list[Entry]
    bullets: list[Bullet]
    skill_rows: list[SkillRow]

    @property
    def summary_section(self) -> Section | None:
        return next((s for s in self.sections if s.name.casefold() in _SUMMARY_SECTIONS), None)

    def entry(self, title: str) -> Entry:
        return next(e for e in self.entries if e.title == title)

    def keywords_in_bullets(self) -> set[str]:
        return {k for b in self.bullets for k in b.keywords}

    def keywords_anywhere(self) -> set[str]:
        return (
            self.keywords_in_bullets()
            | {k for e in self.entries for k in e.keywords}
            | {k for s in self.sections for k in s.keywords}
            | {k for r in self.skill_rows for k in r.keywords}
        )


# --- LaTeX → texto plano -----------------------------------------------------------

_SIMPLE = [
    (r"\\raisebox\{[^{}]*\}", ""), (r"\\fa[A-Za-z]+", ""),
    (r"\\href\{[^{}]*\}\{([^{}]*)\}", r"\1"),
    (r"\\\\(\[[^\]]*\])?", " "), (r"\\(?:enspace|quad|qquad|\s)", " "),
    (r"(?<!\\)~", " "), (r"\\,", ""), (r"\\textasciitilde\s?", "~"),
    (r"\\rho", "ρ"), (r"\\pm", "±"), (r"\\times", "×"),
    (r"\\([%&#_$])", r"\1"), (r"---", "—"), (r"--", "–"),
    (r"\$([^$]*)\$", r"\1"),
    (r"\\(?:small|large|Large|Huge|footnotesize|scriptsize|normalsize)\b", ""),
    (r"\\(?:vspace|hspace)\*?\{[^{}]*\}", ""),
]  # fmt: skip
_SIMPLE_RE = [(re.compile(p), r) for p, r in _SIMPLE]
_WRAP = re.compile(r"\\(?:textbf|textit|emph|underline|texttt|textsc)\{([^{}]*)\}")


def to_plain(tex: str) -> str:
    text = tex
    for pattern, repl in _SIMPLE_RE:
        text = pattern.sub(repl, text)
    while _WRAP.search(text):
        text = _WRAP.sub(r"\1", text)
    text = re.sub(r"\\[A-Za-z]+\*?", "", text)  # comandos restantes sin argumento útil
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


def split_comment(line: str) -> tuple[str, str | None]:
    """'código % comentario' → (código, comentario). Respeta el \\% escapado."""
    m = re.search(r"(?<!\\)%", line)
    if m is None:
        return line, None
    return line[: m.start()], line[m.end() :].strip()


def brace_args(text: str, start: int, n: int) -> tuple[list[str], int]:
    """Lee `n` argumentos {…} (con llaves anidadas) desde `start`."""
    args: list[str] = []
    i = start
    for _ in range(n):
        while i < len(text) and text[i] in " \t":
            i += 1
        if i >= len(text) or text[i] != "{":
            raise ValueError(f"se esperaba '{{' en: {text.strip()[:80]!r}")
        depth, j = 0, i
        while j < len(text):
            if text[j] == "{" and text[j - 1] != "\\":
                depth += 1
            elif text[j] == "}" and text[j - 1] != "\\":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            raise ValueError(f"llaves sin cerrar en: {text.strip()[:80]!r}")
        args.append(text[i + 1 : j])
        i = j + 1
    return args, i


# --- Parser ------------------------------------------------------------------------

_CMD = re.compile(r"^\s*\\(cvsection|cventry|skillrow|langrow|item)\b")
_STRUCTURAL_SKIP = re.compile(
    r"^\s*\\(begin|end|setlength|noindent|vspace|par|needspace|newpage)\b"
)


def parse_cv(path: Path, compiled_vocab: dict[str, Pattern[str]]) -> CVDocument:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        body_start = next(i for i, ln in enumerate(lines) if r"\begin{document}" in ln)
    except StopIteration:
        raise ValueError(f"{path}: no tiene \\begin{{document}}") from None

    sections: list[Section] = []
    entries: list[Entry] = []
    bullets: list[Bullet] = []
    skill_rows: list[SkillRow] = []
    pending: list[Annotation] = []  # comentarios con tag que esperan su elemento
    last_was_tag_comment = False
    current_bullet: Bullet | None = None
    in_itemize = False

    def kws(text: str) -> list[str]:
        return extract_keywords(text, compiled_vocab)

    def take_pending() -> list[Annotation]:
        out = list(pending)
        pending.clear()
        return out

    def inline(comment: str | None, lineno: int) -> list[Annotation]:
        m = _TAG.match(comment or "")
        return [Annotation(tag=m.group(1), note=m.group(2).strip(), line=lineno)] if m else []  # type: ignore[arg-type]

    def close_bullet() -> None:
        nonlocal current_bullet
        if current_bullet is not None:
            current_bullet.keywords = kws(current_bullet.text)
            current_bullet = None

    for idx in range(body_start + 1, len(lines)):
        lineno = idx + 1
        code, comment = split_comment(lines[idx])
        stripped = code.strip()

        # Línea de solo comentario
        if not stripped:
            if comment is None:
                last_was_tag_comment = False
                continue
            m = _TAG.match(comment)
            if m:
                pending.append(Annotation(tag=m.group(1), note=m.group(2).strip(), line=lineno))  # type: ignore[arg-type]
                last_was_tag_comment = True
            elif last_was_tag_comment and pending:
                prev = pending[-1]
                pending[-1] = prev.model_copy(update={"note": f"{prev.note} {comment}".strip()})
            continue
        last_was_tag_comment = False

        cmd = _CMD.match(code)
        section = sections[-1].name if sections else ""
        entry = entries[-1] if entries and entries[-1].section == section else None

        if cmd and cmd.group(1) == "cvsection":
            close_bullet()
            (name,), _ = brace_args(code, cmd.end(), 1)
            sections.append(
                Section(name=to_plain(name), line=lineno,
                        annotations=take_pending() + inline(comment, lineno))
            )  # fmt: skip
        elif cmd and cmd.group(1) == "cventry":
            close_bullet()
            (title, meta), _ = brace_args(code, cmd.end(), 2)
            entries.append(
                Entry(
                    title=to_plain(title), meta=to_plain(meta), section=section, line=lineno,
                    annotations=take_pending() + inline(comment, lineno),
                )
            )  # fmt: skip
        elif cmd and cmd.group(1) in ("skillrow", "langrow"):
            (label, items), _ = brace_args(code, cmd.end(), 2)
            row = SkillRow(label=to_plain(label).rstrip(":"), items=to_plain(items), line=lineno)
            row.keywords = kws(row.items)
            skill_rows.append(row)
        elif cmd and cmd.group(1) == "item":
            close_bullet()
            text = to_plain(code[cmd.end() :])
            current_bullet = Bullet(
                id=f"b{len(bullets) + 1:02d}", section=section,
                entry=entry.title if entry else None, text=text, line=lineno,
                annotations=take_pending() + inline(comment, lineno),
            )  # fmt: skip
            bullets.append(current_bullet)
            if entry:
                entry.bullet_ids.append(current_bullet.id)
        elif re.match(r"^\s*\\begin\{itemize\}", code):
            in_itemize = True
        elif re.match(r"^\s*\\end\{itemize\}", code):
            close_bullet()
            in_itemize = False
        elif _STRUCTURAL_SKIP.match(code) or not to_plain(code):
            continue
        elif in_itemize and current_bullet is not None:
            current_bullet.text = f"{current_bullet.text} {to_plain(code)}".strip()
        elif sections:  # texto de párrafo: detalle de la entrada o texto de la sección
            text = to_plain(code)
            if entry is not None:
                entry.details.append(text)
            else:
                sec = sections[-1]
                sec.text = f"{sec.text} {text}".strip()
                sec.annotations += inline(comment, lineno)
    close_bullet()
    for e in entries:  # título + meta + párrafos de detalle
        e.keywords = kws(" . ".join([e.title, e.meta, *e.details]))
    for sec in sections:
        sec.keywords = kws(sec.text)

    if not sections:
        raise ValueError(f"{path}: no se encontró ninguna \\cvsection")
    if not bullets:
        raise ValueError(f"{path}: no se encontró ningún \\item")
    return CVDocument(
        path=str(path), sections=sections, entries=entries, bullets=bullets,
        skill_rows=skill_rows,
    )  # fmt: skip
