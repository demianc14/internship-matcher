"""Sugerencias de adaptación del CV por vacante.

Regla central (la misma que el CV se pone a sí mismo: "solo lo respaldado por un
proyecto o experiencia"): el motor NUNCA sugiere agregar una skill que no está
respaldada en el CV. Sugiere solo cuatro cosas:

  1. Bullets a destacar: los que ya nombran skills que la vacante pide.
  2. Skills a nombrar: las que tienes (en el stack de un proyecto o en Habilidades)
     pero ningún bullet menciona, y dónde nombrarlas.
  3. Orden de entradas en secciones marcadas `% ADAPTAR` (p. ej. Proyectos) y
     entradas `% OPCIONAL` que se pueden quitar para esta vacante.
  4. Ajustes al Perfil (`% ADAPTAR`) cuando la vacante contradice lo que dice.

Lo que la vacante pide y no está en el CV se lista como brecha, sin sugerir
inventarlo. Los bullets sugeridos que tienen `% CONFIRMAR`/`% PENDIENTE` se
advierten: no se envía un dato que el propio CV marca como no verificado.
"""

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.cv.parser import Bullet, CVDocument
from src.etl.match import LANGUAGE_KEYWORDS, Embedder, Profile, _cosine, split_sentences
from src.etl.schema import Vacante
from src.etl.transform import expand_keywords


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BulletPick(_Model):
    bullet_id: str
    entry: str | None
    text: str
    covers: list[str]
    similarity: float | None = None
    warnings: list[str] = Field(default_factory=list)


class MentionHint(_Model):
    skill: str
    where: str
    reason: str


class Suggestion(_Model):
    vacante_id: str
    title: str
    highlight: list[BulletPick]
    mention: list[MentionHint]
    gaps: list[str]
    entry_order: dict[str, list[str]]  # sección → orden sugerido (solo si cambia)
    drop_optional: list[str]
    profile_notes: list[str]

    def render(self) -> str:
        out = [f"## {self.title}  ({self.vacante_id})"]
        if self.highlight:
            out.append("\nDestacar estos bullets (ya cubren lo que piden):")
            for p in self.highlight:
                sim = f", similitud {p.similarity:.2f}" if p.similarity is not None else ""
                covers = ", ".join(p.covers) if p.covers else "—"
                out.append(f"  • [{p.bullet_id}] {p.text[:110]}…")
                out.append(f"      cubre: {covers}{sim}  ·  {p.entry or 'sin entrada'}")
                out += [f"      ⚠ {w}" for w in p.warnings]
        if self.mention:
            out.append("\nNombrar explícitamente (la tienes, pero ningún bullet la dice):")
            out += [f"  • {m.skill} → {m.where}: {m.reason}" for m in self.mention]
        if self.entry_order:
            for section, order in self.entry_order.items():
                out.append(f"\nReordenar «{section}» (el CV lo marca ADAPTAR):")
                out += [f"  {i}. {t}" for i, t in enumerate(order, 1)]
        if self.drop_optional:
            out.append("\nSe puede quitar para esta vacante (marcado OPCIONAL, no aporta aquí):")
            out += [f"  • {t}" for t in self.drop_optional]
        if self.profile_notes:
            out.append("\nPerfil (marcado ADAPTAR):")
            out += [f"  • {n}" for n in self.profile_notes]
        if self.gaps:
            out.append("\nBrechas reales (no están en tu CV; no agregar sin respaldo):")
            out.append("  " + ", ".join(self.gaps))
        if len(out) == 1:
            out.append("\nSin sugerencias: la vacante no menciona skills reconocidas.")
        return "\n".join(out)


def _bullet_warnings(b: Bullet) -> list[str]:
    return [
        f"{a.tag} (línea {a.line}): {a.note}"
        for a in b.annotations
        if a.tag in ("CONFIRMAR", "PENDIENTE")
    ]


class Suggester:
    def __init__(
        self,
        cv: CVDocument,
        profile: Profile,
        embedder: Embedder | None = None,
        implications: dict[str, list[str]] | None = None,
        max_highlight: int = 5,
    ) -> None:
        self.cv, self.profile, self.embedder = cv, profile, embedder
        self.implications = implications or {}
        self.max_highlight = max_highlight
        self.bullet_vecs = embedder.encode([b.text for b in cv.bullets]) if embedder else []

    def _similarities(self, v: Vacante) -> list[float] | None:
        if self.embedder is None:
            return None
        vecs = self.embedder.encode([v.title, *split_sentences(v.description_text)])
        return [max(_cosine(bv, sv) for sv in vecs) for bv in self.bullet_vecs]

    def suggest(self, v: Vacante) -> Suggestion:
        cv = self.cv
        required = [k for k in v.keywords if k not in LANGUAGE_KEYWORDS]
        req = set(required)
        sims = self._similarities(v)

        # 1. Bullets a destacar: por # de skills cubiertas, luego similitud semántica
        scored = []
        for i, b in enumerate(cv.bullets):
            # Un bullet que nombra Power Automate también cubre "process automation"
            covers = sorted(req & expand_keywords(b.keywords, self.implications))
            sim = sims[i] if sims is not None else None
            if covers or (sim is not None and sim >= self.profile.semantic_calibration.high):
                scored.append((len(covers), sim or 0.0, b, covers, sim))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        highlight = [
            BulletPick(
                bullet_id=b.id, entry=b.entry, text=b.text, covers=covers,
                similarity=None if sim is None else round(sim, 3), warnings=_bullet_warnings(b),
            )  # fmt: skip
            for _, _, b, covers, sim in scored[: self.max_highlight]
        ]

        # 2. Skills a nombrar y 5. brechas
        in_bullets = expand_keywords(cv.keywords_in_bullets(), self.implications)
        mention: list[MentionHint] = []
        gaps: list[str] = []
        for skill in required:
            if skill in in_bullets:
                continue
            # Entradas con bullets primero: sugerir un bullet en Educación no sirve.
            direct = [e for e in cv.entries if skill in e.keywords]
            implied = [
                (e, k) for e in cv.entries for k in e.keywords
                if skill in self.implications.get(k, []) and e not in direct
            ]  # fmt: skip
            direct.sort(key=lambda e: not e.bullet_ids)
            implied.sort(key=lambda t: not t[0].bullet_ids)
            rows = [r.label for r in cv.skill_rows if skill in r.keywords]
            tier = self.profile.skills[skill].tier if skill in self.profile.skills else None
            if direct:
                mention.append(MentionHint(
                    skill=skill, where=f"un bullet de «{direct[0].title}»",
                    reason="está en el stack/descripción de esa entrada pero no en sus bullets",
                ))  # fmt: skip
            elif implied:
                entry, via = implied[0]
                mention.append(MentionHint(
                    skill=skill, where=f"un bullet de «{entry.title}»",
                    reason=f"esa entrada usa {via}, que implica {skill}; nómbrala explícitamente",
                ))  # fmt: skip
            elif rows:
                reason = "solo aparece en Habilidades; respáldala en un bullet si es real"
                if tier == "en_formacion":
                    reason = "el CV la marca en formación; no la presentes como dominada"
                mention.append(MentionHint(skill=skill, where=f"Habilidades ({rows[0]})",
                                           reason=reason))  # fmt: skip
            else:
                gaps.append(skill)

        # 3. Orden de entradas en secciones ADAPTAR y entradas OPCIONAL
        def relevance(title: str) -> int:
            e = cv.entry(title)
            bullet_kws = {k for b in cv.bullets if b.id in e.bullet_ids for k in b.keywords}
            return len(req & expand_keywords({*e.keywords, *bullet_kws}, self.implications))

        entry_order: dict[str, list[str]] = {}
        for section in cv.sections:
            if not any(a.tag == "ADAPTAR" for a in section.annotations):
                continue
            titles = [e.title for e in cv.entries if e.section == section.name]
            ranked = sorted(titles, key=lambda t: -relevance(t))  # sort estable
            if len(titles) > 1 and ranked != titles and req:
                entry_order[section.name] = ranked
        drop_optional = [
            e.title for e in cv.entries
            if any(a.tag == "OPCIONAL" for a in e.annotations) and req and relevance(e.title) == 0
        ]  # fmt: skip

        return Suggestion(
            vacante_id=v.id, title=v.title, highlight=highlight, mention=mention, gaps=gaps,
            entry_order=entry_order, drop_optional=drop_optional,
            profile_notes=self._profile_notes(v, required),
        )  # fmt: skip

    def _profile_notes(self, v: Vacante, required: Sequence[str]) -> list[str]:
        section = self.cv.summary_section
        if section is None or not any(a.tag == "ADAPTAR" for a in section.annotations):
            return []
        notes: list[str] = []
        text = section.text
        hybrid_quito = re.search(r"h[ií]brid\w*[^.]*quito", text, re.IGNORECASE)
        if hybrid_quito and v.modality.value == "remote":
            notes.append(
                f"La vacante es remota y el Perfil dice «{hybrid_quito.group(0)}»: "
                "ajústalo (p. ej. «remota o híbrida en Quito»)."
            )
        in_summary = set(section.keywords)
        missing = [
            s for s in required
            if s in self.profile.skills and self.profile.skills[s].tier == "demostrado"
            and s not in in_summary
        ][:3]  # fmt: skip
        if missing:
            notes.append(
                "Pide skills que ya demostraste y el Perfil no nombra: " + ", ".join(missing)
            )
        return notes


def profile_consistency(
    cv: CVDocument, profile: Profile, implications: dict[str, list[str]] | None = None
) -> dict[str, list[str]]:
    """Audita que skills_profile.yaml siga derivado del CV (el perfil se escribió a
    mano a partir del CV; esto detecta cuando uno cambia y el otro no).

    Una skill del perfil cuenta como respaldada si el CV la nombra o si la implica
    algo que el CV nombra. Al revés no: que el CV implique "low-code" no obliga a
    declararlo en el perfil, porque `effective_skills` ya lo deduce."""
    in_cv = cv.keywords_anywhere() - set(LANGUAGE_KEYWORDS)
    backed = expand_keywords(in_cv, implications or {})
    return {
        "en_perfil_no_en_cv": sorted(set(profile.skills) - backed),
        "en_cv_no_en_perfil": sorted(in_cv - set(profile.skills)),
    }
