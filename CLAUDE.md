# internship-matcher — Brief de Proyecto para Claude Code

> Guarda este archivo como `CLAUDE.md` en la raíz del repo. Claude Code lo lee
> automáticamente al iniciar sesión ahí y lo trata como contexto persistente
> del proyecto — no hace falta repetir esto en cada prompt.

## Rol de Claude Code en este proyecto

Actúa como ingeniero de datos/backend senior construyendo esto junto a Demian,
no como generador de snippets sueltos. Antes de escribir código en cada fase,
propone el diseño (esquema de datos, contrato de funciones, qué falla y cómo)
y espera confirmación si algo es ambiguo. Prioriza corrección y honestidad
sobre volumen de código: si un dato no se puede resolver sin más contexto,
se documenta como pendiente — no se inventa.

## Objetivo

Sistema que automatiza la búsqueda de pasantías/empleos y sugiere qué partes
de mi CV (`Base_CV.tex`) activar o reescribir según cada vacante, con un
score de compatibilidad explicable (no una caja negra).

**No es un proyecto de scraping agresivo ni un chatbot conversacional.** Es
un pipeline de datos con un componente de matching semántico encima —mismo
espíritu que `world-cup-predictor` y `MetalurgicaAndina_QC`: ETL con
principios fail-fast, funciones puras testeables, reporte de calidad
cuantificado, y un `README.md` que documenta decisiones y limitaciones, no
solo "cómo correrlo".

## Principios no negociables (mismos que en mis otros proyectos)

1. **Fail-fast + escalación documentada.** Si un campo de una vacante no se
   puede parsear con confianza (modalidad, ubicación, seniority), no se
   asume un valor por defecto silencioso — se marca explícitamente como
   `null`/pendiente en el reporte, igual que las fechas ambiguas en
   MetalurgicaAndina_QC.
2. **Extract sin lógica de negocio.** La capa que llama a las APIs de
   empleo solo trae datos crudos tal cual. Toda normalización, dedup y
   scoring va en `transform`/`match`, nunca mezclado con el fetch.
3. **El perfil de habilidades es un archivo de datos, no código.** Vive en
   `config/skills_profile.yaml`, no hardcodeado en el motor de matching.
   Así se actualiza sin tocar lógica, y es auditable.
4. **El score de matching debe ser explicable.** No "la IA dice 87% match"
   sin más — el output debe listar qué keywords/skills sí matchean, cuáles
   faltan, y qué bullets del CV cubren cuáles requisitos.
5. **Tests desde la fase 1**, no al final. `pytest` + `ruff check` + `mypy`
   corriendo desde el primer commit funcional, mismo stack que ya usas.
6. **Nada de scraping de sitios que lo prohíban en su ToS** (LinkedIn
   incluido). Solo APIs oficiales/públicas o feeds que lo permitan
   explícitamente. Si una fuente que quiero no tiene API gratuita, se
   documenta como limitación, no se scrapea igual.

## Contexto real que debe moldear el diseño

- Busco pasantía **remota o híbrida, 4–6 horas diarias**; ideal 4h remoto,
  pero mi prioridad real es híbrido en Quito. Esto significa que el
  filtro de vacantes necesita al menos: modalidad (remoto/híbrido/
  presencial), carga horaria si el dato existe, y ubicación.
- **Limitación honesta que hay que documentar desde ya, no descubrir a
  mitad de proyecto:** las APIs de empleo gratuitas y confiables
  (Arbeitnow, RemoteOK) cubren sobre todo remoto global, no vacantes
  locales/híbridas en Quito. No hay API pública gratuita robusta para
  bolsas ecuatorianas (Computrabajo, Multitrabajos no exponen API
  pública). Dos caminos, no elegir en silencio cuál:
  - (a) el MVP cubre bien el segmento remoto y para Quito/híbrido queda
    como límite documentado en el README, o
  - (b) se agrega una fuente manual: un CSV/formulario donde yo pego
    vacantes locales que encuentro a mano y el pipeline las procesa igual
    que las de API (mismo esquema, mismo scoring).
  Claude Code: preguntar cuál prefiero antes de construir la capa de
  ingesta, no asumir.

## Fuentes de datos (fase de Extract)

- **Arbeitnow API** — `https://arbeitnow.com/api/job-board-api` — gratis,
  sin key, remoto/tech.
- **RemoteOK API** — `https://remoteok.com/api` — gratis, sin key.
- **Adzuna API** — requiere `app_id`+`app_key` gratuitos; verificar
  cobertura geográfica real antes de asumir que incluye Ecuador o LatAm
  (Adzuna no cubre todos los países).
- Fuente manual (CSV) para vacantes locales, si se decide el camino (b)
  de arriba.

Cada fuente es un módulo independiente en `extract.py` con su propio
manejo de errores — si una fuente falla o cambia su API, el pipeline
degrada elegante (mismo patrón que los scrapers de `world-cup-predictor`),
no rompe todo el run.

## Arquitectura propuesta

```
src/
  etl/
    extract.py          # fetch crudo de cada fuente, sin lógica de negocio
    transform.py        # normaliza (modalidad, ubicación, seniority),
                         # dedup, extrae requisitos/keywords de cada vacante
    match.py            # embeddings (sentence-transformers, local, gratis)
                         # + scoring explicable contra skills_profile.yaml
  cv/
    parser.py           # extrae bullets, tags % ADAPTAR y keywords ya
                         # presentes desde Base_CV.tex
    suggest.py           # cruza gaps de la vacante contra bullets
                          # disponibles y sugiere qué activar/reescribir
  api/
    main.py              # FastAPI: endpoints para listar vacantes,
                          # ver score, pedir sugerencia de CV
  cli.py                 # corre el pipeline completo, resumen en consola
config/
  skills_profile.yaml     # fuente de verdad de mi perfil de habilidades
data/
  raw/                    # snapshots crudos por fuente
  processed/              # vacantes normalizadas + scores
tests/
README.md                 # arquitectura, reporte de calidad, limitaciones
```

## Fases de desarrollo (ir una por una, no todo de golpe)

**Fase 0 — Setup.** Estructura de carpetas, `pyproject.toml`, entorno
virtual, `pytest`/`ruff`/`mypy` configurados y corriendo (aunque sea sobre
un test dummy). Cero lógica de negocio todavía.

**Fase 1 — Extract.** Antes del primer fetcher, definir `src/etl/schema.py`:
un modelo Pydantic `RawVacante` que sea el **contrato compartido** por
todas las fuentes futuras (Arbeitnow, RemoteOK, Adzuna, CSV manual). Solo
valida estructura (campos presentes, tipos, forma de URL) — los campos
que requieren criterio de negocio (modalidad, ubicación) se guardan como
texto crudo (`modality_raw`, `location_raw`), sin clasificar todavía; eso
es trabajo de Transform en la Fase 2, no de aquí. Con el contrato ya
definido, un solo fetcher (Arbeitnow) trayendo datos crudos a
`data/raw/`, con manejo de error si la API cae. Test que verifica que el
fetcher emite `RawVacante` válidos (con un fixture, no llamando a la API
real en cada test run).

**Fase 2 — Transform.** Normalización + dedup + extracción de
requisitos/keywords de cada vacante. Reporte de calidad cuantificado
(cuántas vacantes llegaron, cuántas se descartaron y por qué, cuántas
quedan con modalidad/ubicación sin poder determinar) — mismo formato que
el reporte de MetalurgicaAndina_QC.

**Fase 3 — `skills_profile.yaml` + Match engine.** Definir mi perfil de
habilidades como datos. Motor de embeddings + scoring explicable. Tests
con vacantes sintéticas de match alto/bajo conocido, para verificar que
el score se comporta como se espera en los extremos.

**Fase 4 — CV parser + suggestion engine.** Parsear `Base_CV.tex` para
extraer bullets y keywords ya cubiertas. Cruzar contra los gaps de cada
vacante y sugerir qué bullets activar o qué keyword falta mencionar.
Output legible, no solo JSON crudo.

**Fase 5 — API/CLI + segunda fuente de datos.** Exponer todo vía FastAPI
y/o CLI. Agregar RemoteOK (y Adzuna si la cobertura lo justifica, o la
fuente manual CSV si se eligió ese camino).

**Fase 6 — Tests de integración + README final.** Cobertura de pytest en
las capas críticas (transform, match, cv/suggest), `mypy`/`ruff` limpios,
y `README.md` con arquitectura, reporte de calidad y limitaciones
honestas — igual que mis otros dos proyectos de portafolio.

Al cerrar cada fase, Claude Code resume qué quedó funcionando, qué se
descartó y por qué, y qué decisión quedó pendiente para la fase siguiente
— no avanza a la fase N+1 sin ese cierre explícito.

## Qué NO hacer

- No scrapear LinkedIn ni sitios que lo prohíban en su ToS.
- No hardcodear mi perfil de habilidades dentro de la lógica de matching.
- No inventar un score sin poder explicar de qué keywords/requisitos sale.
- No mezclar fetch de datos con normalización/lógica de negocio.
- No avanzar de fase sin que la fase anterior tenga tests pasando.

## Decisiones tomadas

- **2026-09-21 — Fuente Quito: camino (b), CSV manual** en `data/manual/`. Mismo
  esquema y scoring que las fuentes API. Loader en Fase 5; el esquema común de
  vacante (Fase 1) debe contemplar esta fuente desde el inicio.
- **Entorno:** `venv` + `pip`, `pip install -e ".[dev]"`.
- **2026-09-21 — Esquema común en Fase 1** (`src/etl/schema.py`, `RawVacante`),
  antes del primer fetcher. Sin clasificar modalidad/ubicación.
- **2026-09-21 — Hallazgo:** Arbeitnow página 1 = 3/250 remotas, mayoría Alemania.
  Su aporte para "pasantía remota" es bajo; RemoteOK/CSV manual pesan más.
- **2026-09-21 — Fase 2:** modalidad por reglas explícitas con evidencia y `None` +
  motivo si no hay confianza; dedup por (título, empresa, ubicación), misma vacante en
  otra ciudad se conserva y se reporta. Vocabulario genérico de keywords en
  `config/keyword_vocabulary.yaml` (distinto de `skills_profile.yaml`).
- **2026-09-22 — CV:** la fuente es `~/Documents/Documentos Demi/Base CV.tex` (con
  espacio). `cv.tex` en la misma carpeta está desactualizado; no usarlo.
- **2026-09-22 — Fase 3:** modalidad desconocida no descarta (solo avisa). Score
  (skills + semántico opcional) separado del veredicto de encaje. Tiers del perfil
  derivados de dónde aparece la skill en el CV, no autoevaluados.
- **2026-09-22 — Ranking:** no_apta al final; entre apta/revisar ordena el score y el
  veredicto desempata. C++ se mantiene como `listado`.
- **2026-09-22 — Fase 4:** sugerencias nunca inventan skills; `implies` en el
  vocabulario (MySQL → SQL) ubica skills respaldadas indirectamente. El CV real no
  entra al repo: los tests usan `tests/fixtures/cv_sample.tex`.
