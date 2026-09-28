"""Base CV.tex → list[Bullet]. Determinístico, sin LLM y sin lógica de negocio.

Entiende las macros del CV: `\\cvsection{nombre}`, `\\cventry{título}{meta}` e
`\\item` dentro de `itemize`. Cada `\\item` es un Bullet cuyo `project` es la
entrada (`\\cventry`) bajo la que aparece, o la sección si no hay entrada. Los
comentarios LaTeX (`% ADAPTAR`, `% PENDIENTE`…) son notas de trabajo del autor:
se descartan, nunca llegan al texto del bullet.

`parse_backing` lee lo que el CV respalda FUERA de los bullets: la línea de stack
de cada `\\cventry{título}{stack}` y las filas `\\skillrow`. El matcher lo usa para
distinguir "lo tienes pero no lo nombras en un bullet" de "no está en tu CV". Los
ítems marcados "(en formación)" van aparte: no se sugiere presentarlos como dominados.

Fail fast: un .tex sin \\begin{document} o sin ningún \\item no es un CV que este
parser entienda, y lanza ValueError en vez de devolver una lista vacía.
"""

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.vocabulary import Vocabulary


class Bullet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project: str
    text: str
    keywords: list[str]


class CVBacking(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    projects: dict[str, list[str]]  # título de la entrada → keywords de su línea de stack
    skills: list[str]  # keywords de las filas de habilidades (sin "en formación")
    in_training: list[str]  # keywords de ítems marcados "(en formación)"


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


def strip_comment(line: str) -> str:
    """'código % comentario' → 'código'. Respeta el \\% escapado."""
    m = re.search(r"(?<!\\)%", line)
    return line if m is None else line[: m.start()]


def brace_args(text: str, start: int, n: int) -> list[str]:
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
    return args


# --- Parser ------------------------------------------------------------------------

_CMD = re.compile(r"^\s*\\(cvsection|cventry|item)\b")
_SKILLROW = re.compile(r"^\s*\\skillrow\b")
_IN_TRAINING = re.compile(r"en formaci[oó]n", re.IGNORECASE)
_BEGIN_ITEMIZE = re.compile(r"^\s*\\begin\{itemize\}")
_END_ITEMIZE = re.compile(r"^\s*\\end\{itemize\}")


def _body(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        body_start = next(i for i, ln in enumerate(lines) if r"\begin{document}" in ln)
    except StopIteration:
        raise ValueError(f"{path}: no tiene \\begin{{document}}") from None
    return lines[body_start + 1 :]


def _split_items(text: str) -> list[str]:
    """'Python, SQL (vistas, triggers), Java' → ítems, sin cortar dentro de paréntesis."""
    items, depth, current = [], 0, ""
    for ch in text:
        depth += (ch == "(") - (ch == ")")
        if ch == "," and depth == 0:
            items.append(current)
            current = ""
        else:
            current += ch
    return [i.strip() for i in [*items, current] if i.strip()]


def parse_backing(path: Path, vocab: Vocabulary) -> CVBacking:
    projects: dict[str, list[str]] = {}
    skills: set[str] = set()
    in_training: set[str] = set()
    for line in _body(path):
        code = strip_comment(line)
        if (cmd := _CMD.match(code)) and cmd.group(1) == "cventry":
            title, stack = brace_args(code, cmd.end(), 2)
            projects[to_plain(title)] = vocab.extract(to_plain(stack))
        elif m := _SKILLROW.match(code):
            _, items = brace_args(code, m.end(), 2)
            for item in _split_items(to_plain(items)):
                target = in_training if _IN_TRAINING.search(item) else skills
                target.update(vocab.extract(item))
    return CVBacking(
        projects=projects, skills=sorted(skills), in_training=sorted(in_training - skills)
    )


def parse_cv(path: Path, vocab: Vocabulary) -> list[Bullet]:
    body = _body(path)

    raw: list[tuple[str, str]] = []  # (project, texto) antes de extraer keywords
    section: str | None = None
    entry: str | None = None
    item_open = False  # hay un \item abierto que puede seguir en la próxima línea

    for line in body:
        code = strip_comment(line)
        if not code.strip():
            continue
        cmd = _CMD.match(code)
        if cmd and cmd.group(1) != "item":
            item_open = False
        if cmd and cmd.group(1) == "cvsection":
            section, entry = to_plain(brace_args(code, cmd.end(), 1)[0]), None
        elif cmd and cmd.group(1) == "cventry":
            entry = to_plain(brace_args(code, cmd.end(), 2)[0])
        elif cmd and cmd.group(1) == "item":
            project = entry or section
            if project is None:
                raise ValueError(f"{path}: \\item fuera de toda sección: {line.strip()[:80]!r}")
            raw.append((project, to_plain(code[cmd.end() :])))
            item_open = True
        elif _BEGIN_ITEMIZE.match(code) or _END_ITEMIZE.match(code):
            item_open = False
        elif item_open:  # \item partido en varias líneas
            project, text = raw[-1]
            raw[-1] = (project, f"{text} {to_plain(code)}".strip())

    if not raw:
        raise ValueError(f"{path}: no se encontró ningún \\item")
    return [Bullet(project=p, text=t, keywords=vocab.extract(t)) for p, t in raw]
