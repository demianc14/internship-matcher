# internship-matcher

Pipeline de datos que busca pasantías/empleos, las normaliza y calcula un score de
compatibilidad **explicable** contra mi perfil de habilidades y `Base_CV.tex`.

> Estado: **Fase 4** cerrada (parser de `Base CV.tex` + motor de sugerencias por vacante).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest && ruff check . && mypy src tests

python -m src.cli            # reprocesa el snapshot más reciente de data/raw/
python -m src.cli --fetch    # trae 1 página fresca de Arbeitnow
python -m src.cli --semantic # + embeddings (requiere pip install -e ".[semantic]")
python -m src.cli --suggest 3 # + sugerencias de CV para las 3 mejores
```

## Decisiones

- **Vacantes locales (Quito, híbrido/presencial): fuente manual CSV** — camino (b).
  Computrabajo y Multitrabajos no exponen API pública, y no se scrapean. Las vacantes
  que encuentre a mano van en `data/manual/*.csv` y se procesan con el mismo esquema,
  transform y scoring que las de API. El esquema común se diseña desde la Fase 1
  pensando en esta fuente; el loader se implementa en la Fase 5.

## Extract (Fase 1)

- `src/etl/schema.py`: `RawVacante`, contrato compartido por todas las fuentes. Solo
  valida estructura; modalidad/ubicación/jornada quedan como texto crudo (`*_raw`).
- `src/etl/extract.py`: `parse_arbeitnow` (pura) + `fetch_arbeitnow` (HTTP, nunca
  lanza). Registros que no cumplen el contrato se **rechazan con motivo**, no se
  corrigen. Si la API cae o cambia de forma, queda en `source_error` y el run sigue.
- Snapshot crudo de cada fetch en `data/raw/<fuente>_<timestamp>.json`.
- Por defecto se trae 1 página (250 vacantes): los términos de Arbeitnow piden no abusar.

## Transform (Fase 2)

`src/etl/transform.py` convierte `RawVacante` → `Vacante`. Cada campo clasificado es un
`Resolved` con **valor + origen + evidencia citada**, o `None` + motivo:
`no_signal` (no hay dato), `conflict` (señales contradictorias, se citan ambas),
`weak_signal` (solo "home office"/"work from anywhere") o `unknown_value` (valor del
CSV manual que no está en el vocabulario). Nunca hay un default silencioso.

| Campo | Fuentes de señal | Regla clave |
|---|---|---|
| Modalidad | structured > location > title > description | Se usa el primer nivel con señal. `remote=false` no implica presencial. Menciones negadas ("do not offer remote work") se ignoran. |
| Seniority | título + `employment_raw` | Etiquetas distintas = conflicto, salvo `entry` + `intern`/`student_job` (nivel vs. tipo de puesto: gana el más específico). |
| Jornada | título + `employment_raw` | "Full or part time" = conflicto. No se infiere de la descripción (los beneficios mencionan "part-time options"). |
| Horas | título + descripción | Solo explícitas y plausibles (≤12 h/día, ≤60 h/semana). No se convierte semana↔día. |
| Idioma | descripción | Stopwords en/de/es/fr; sin ganador claro (≥2x) = `None`. |
| Keywords | título + descripción | Vocabulario genérico en `config/keyword_vocabulary.yaml` (no es mi perfil). |

**Dedup:** mismo id → se queda el más reciente. Mismo título+empresa+ubicación
normalizados → duplicado entre fuentes (solo si hay empresa y ubicación). Mismo
título+empresa en **otra ciudad se conserva**: en los datos reales casi siempre es el
mismo puesto abierto en varias ciudades; se lista como "posible duplicado".

### Reporte de calidad (snapshot real Arbeitnow, 2026-09-21)

| Decisión | Cantidad |
|---|---|
| Registros recibidos / rechazados por contrato | 250 / 0 |
| Duplicados removidos (título+empresa+ubicación) | 1 |
| Posibles duplicados conservados (grupos, otra ciudad) | 6 |
| Vacantes resultantes | 249 |
| Modalidad: híbrido / presencial / remoto | 28 / 20 / 19 |
| Modalidad sin resolver: sin señal / conflicto / señal débil | 153 / 16 / 13 |
| Seniority: senior / student_job / mid / intern / entry | 84 / 10 / 9 / 5 / 1 |
| Seniority sin resolver: sin señal / conflicto | 133 / 7 |
| Jornada: full / part / sin señal / conflicto | 128 / 11 / 109 / 1 |
| Con horas explícitas | 4 |
| Idioma: en / de / fr / sin resolver | 187 / 34 / 26 / 2 |
| En Quito | 0 (ubicación desconocida: 9) |
| Sin keywords reconocidas | 36 |

La salida queda en `data/processed/vacantes.jsonl` y `data/processed/quality_report.json`.

## Match (Fase 3)

`config/skills_profile.yaml` es la fuente de verdad de mi perfil, derivada de
`Base CV.tex`. Sin niveles autoevaluados: el **tier** sale de dónde aparece la skill
en el CV (`demostrado` = en un bullet de experiencia/proyecto, `listado` = solo en
Habilidades o por certificación, `en_formacion` = el CV lo dice). Cada skill cita su
evidencia. El loader falla si una skill no existe en el vocabulario (nunca podría
matchear) o si un idioma se cuela como skill.

`src/etl/match.py` separa dos cosas a propósito:

- **Score 0–100** = qué tanto encajan mis skills. Componentes con desglose:
  - *skills*: Σ peso(tier) de las skills pedidas que tengo ÷ max(# pedidas, 3). El
    mínimo 3 evita el caso real de "100/100" por una vacante que solo menciona
    `rest api`. Lista ✓ qué matchea (con tier y evidencia del CV) y ✗ qué falta.
  - *semántico* (opcional): por cada frase de la vacante, el passage del perfil más
    similar; se reportan los pares. Si no está activo, los pesos se renormalizan y
    la explicación lo dice.
- **Veredicto** `apta / revisar / no_apta` = restricciones prácticas, no afecta el
  score: seniority, modalidad+ubicación (híbrido/presencial fuera de Quito bloquea;
  **modalidad desconocida nunca descarta**, solo avisa), idioma de la descripción vs.
  mis idiomas de trabajo (≥B2), jornada completa y horas fuera de 4–6 h/día.

Resultado sobre el snapshot real (solo skills, sin embeddings): **3 aptas, 83 a
revisar, 163 no aptas**. Bloqueos: senior 84, descripción en alemán 34 / francés 26,
híbrido fuera de Quito 27, presencial fuera de Quito 19. 142 de 249 vacantes no
mencionan ninguna skill del vocabulario (score n/d, no 0).

## CV: parser y sugerencias (Fase 4)

`src/cv/parser.py` lee `Base CV.tex` (ruta en `cv_path` del perfil; el CV no está en
el repo) y devuelve secciones, entradas, bullets y filas de habilidades. Entiende las
marcas de trabajo del propio CV y las asocia a su elemento: `% ADAPTAR`, `% CONFIRMAR`,
`% AGREGAR`, `% PENDIENTE`, `% OPCIONAL` (en línea propia van al elemento siguiente; al
final de una línea, a esa; la leyenda del encabezado se ignora). Un `.tex` sin
`\begin{document}`, sin secciones o sin bullets lanza ValueError en vez de devolver algo
vacío.

`src/cv/suggest.py` cruza los gaps de la vacante contra el CV. **Nunca sugiere agregar
una skill que el CV no respalde** — la misma regla que el CV se pone a sí mismo:

| Situación | Sugerencia |
|---|---|
| Skill pedida, ya nombrada en un bullet | destacar ese bullet (ordenado por # de skills que cubre) |
| Skill pedida, en el stack de una entrada pero no en sus bullets | nombrarla en un bullet de esa entrada |
| Skill pedida, implícita en otra (`implies` del vocabulario: MySQL → SQL) | nombrarla, diciendo de dónde sale |
| Skill pedida, solo en Habilidades | respaldarla en un bullet si es real; si el CV la marca "en formación", no presentarla como dominada |
| Skill pedida, ausente del CV | brecha: se lista, no se inventa |

Además reordena las secciones marcadas `% ADAPTAR` según la vacante, señala entradas
`% OPCIONAL` que no aportan, y avisa cuando el Perfil contradice la vacante (dice
"pasantía híbrida en Quito" y la vacante es remota). Los bullets sugeridos que tengan
`% CONFIRMAR`/`% PENDIENTE` salen con esa advertencia: no se manda un dato que el CV
marca como no verificado.

`profile_consistency()` audita que `skills_profile.yaml` siga derivado del CV. Sobre el
CV real: 0 skills de más y 0 de menos.

## Limitaciones conocidas

- **Arbeitnow no es "remoto/tech" como se asumía.** Corrida real del 2026-09-21:
  250 vacantes, solo 3 con `remote=true`; el resto son mayormente puestos en Alemania.
  Además `remote=false` no significa "presencial" (solo "no marcada remota"), así que
  transform no debe inferir presencialidad de ese flag.

- Las APIs gratuitas (Arbeitnow, RemoteOK) cubren sobre todo remoto global. La
  cobertura de Quito depende de lo que se cargue manualmente en el CSV.
- Adzuna: la cobertura de Ecuador/LatAm está **pendiente de verificar**.
- **61% de las vacantes quedan con modalidad sin resolver (182/249)**, casi todas por
  falta de señal. Es a propósito: bajar el umbral significaría adivinar. El match
  (Fase 3) tiene que tratar `modality=None` como "desconocida", no como "descartar".
- **Las reglas de modalidad son regex con contexto, no NLP.** Se calibraron sobre un
  solo snapshot (en/de/fr). Un "Office-based role… four days a week" queda como
  presencial aunque probablemente sea híbrido (el número va en palabras). La evidencia
  queda citada en cada vacante para poder auditarlo.
- **Keywords de idioma son ruidosas:** "German" matchea tanto "fluent German" como
  "a German company". Sirven como pista, no como requisito confirmado.
- **No se separa la sección de requisitos** de la descripción: las keywords salen del
  texto completo (beneficios incluidos). Queda pendiente para la Fase 3 si el match lo
  necesita.
- **Arbeitnow casi no tiene vacantes para mi perfil.** Las 3 "aptas" son de ventas
  (Account Executive) con 0 skills en común: pasan las restricciones pero no encajan.
  Sin el componente semántico, el score no distingue dominio (ventas vs. datos) cuando
  la vacante no nombra herramientas. El pipeline funciona; la fuente no sirve para
  este perfil. RemoteOK y el CSV manual de Quito (Fase 5) son los que importan.
- **La calibración semántica (0.20–0.70) es provisional:** no se ha validado con el
  modelo real sobre vacantes reales.
