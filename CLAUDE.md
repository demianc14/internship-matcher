# cv-matcher — Brief de Proyecto para Claude Code

> Pivot del antiguo `internship-matcher` (el pipeline de descubrimiento de
> vacantes quedó en el tag `pre-pivot`). Spec de origen: `PROJECT_SPEC.md` de Demian,
> 2026-09-25.

## Rol de Claude Code en este proyecto

Actúa como ingeniero de datos/backend senior construyendo esto junto a Demian,
no como generador de snippets sueltos. Antes de escribir código en cada paso,
propone el diseño (modelos, contrato de funciones, qué falla y cómo) y espera
confirmación si algo es ambiguo. Prioriza corrección y honestidad sobre volumen
de código: si un dato no se puede resolver sin más contexto, se documenta como
pendiente, no se inventa.

## Objetivo

Dado un JD **ya encontrado y pegado a mano**, analizar el fit contra mi CV
(`Base CV.tex`) y sugerir ediciones cuantificadas, con lógica de matching
testeable. No es "preguntarle al LLM y pegar lo que diga".

Mismo patrón que Metalúrgica Andina y World Cup Predictor: Extract → Match →
Output, fail-fast con Pydantic, decisiones de diseño documentadas en el README.

## Alcance

**Sí construye:**
- `src/cv_parser.py`: `Base CV.tex` → `list[Bullet]`. Determinístico, regex sobre LaTeX.
- `src/jd_extractor.py`: texto del JD → `JDRequirements` vía LLM (Pydantic, fail-fast).
- `src/matcher.py`: score de cada bullet contra las keywords del JD, keywords
  faltantes y sugerencia de reescritura.
- `src/cli.py`: `python -m src.cli match data/jds/<archivo>.txt`
- Tests reales sobre `cv_parser` y `matcher`: casos exactos, sinónimos y avisos sin relación.

**NO construye (decisión de diseño, no pendiente):**
- Scraping o descubrimiento de vacantes: el input siempre es un JD pegado a mano.
- UI: se queda en CLI/terminal.
- Integraciones con portales de empleo.

## Modelos (Pydantic)

```python
class Bullet(BaseModel):
    project: str
    text: str
    keywords: list[str]

class JDRequirements(BaseModel):          # ampliado 2026-09-28, ver Decisiones
    title: str
    hard_skills: list[str]
    soft_requirements: list[str]
    seniority_signal: Literal["internship", "junior", "mid", "senior", "undetermined"]
    seniority_evidence: str                # cita literal del JD
    role_family: Literal["data", "software", "ml_ai", "automation", "it_ops", "non_technical"]
    role_evidence: str                     # cita literal del JD
    ats_keywords: list[str]

class MatchResult(BaseModel):
    bullet: Bullet
    match_score: float
    missing_keywords: list[str]
    suggested_rewrite: str | None
```

## Orden de construcción

1. `cv_parser.py` + `test_cv_parser.py`: 100 % determinístico. **Hecho.**
2. `jd_extractor.py`: probar contra JDs reales como fixtures (`data/jds/`). **Código y tests
   offline hechos; falta la corrida real (`pytest -m llm`).**
3. `matcher.py`: depende de que los dos anteriores den datos limpios. Empezar con
   overlap de keywords (sets) antes de meter un segundo pase de LLM para reescritura.
   **Capa determinística hecha (2026-09-28).** Pendiente: `suggested_rewrite` con LLM,
   validado sin LLM (keywords ⊆ bullet ∪ respaldo del proyecto; números solo del bullet
   original).

## Reglas del pivot (para no reintroducir lo descartado)

- No agregar dependencias de scraping (requests a portales externos, BeautifulSoup
  contra sitios de empleo, etc.) sin que Demian lo pida explícitamente.
- Toda decisión de diseño no obvia va en el README bajo "Limitaciones", con el porqué.
- El matching tiene al menos una capa no-LLM (overlap de sets sobre
  `config/keyword_vocabulary.yaml`) antes de cualquier llamada al LLM,
  para que el matcher sea testeable sin mockear el LLM.
- Fail-fast: si el LLM devuelve algo que no valida contra el modelo, error explícito.
  No se completa con defaults silenciosos.
- `pytest` + `ruff check` + `mypy` limpios antes de pasar al paso siguiente.
- Las sugerencias nunca inventan skills: solo reordenan o reformulan lo que el CV respalda.

## Decisiones tomadas

- **Entorno:** `venv` + `pip`, `pip install -e ".[dev]"`.
- **CV:** la fuente es `~/Documents/Documentos Demi/Base CV.tex` (con espacio).
  `cv.tex` en la misma carpeta está desactualizado; no usarlo. El CV real no entra
  al repo: los tests usan `tests/fixtures/cv_sample.tex`.
- **2026-09-25 — Pivot en este repo** (no en un repo nuevo). Motivo medido: la
  clasificación por reglas marcaba mal los cargos. Hubo senior/full-time que
  pasaban (Software Engineer JVM #1 con 86.7, GM AI Services) y roles no técnicos
  aptos (Account Executive, Social Comms). Esos JDs están en `data/jds/regresion/`
  (gitignored, texto de terceros) como regresión para `jd_extractor`.
- **2026-09-25 — Se conservan** `config/keyword_vocabulary.yaml` (con `implies`:
  MySQL ⇒ SQL, Power Automate ⇒ process automation/low-code) como capa de
  sinónimos no-LLM, cargado por `src/vocabulary.py`.
- **2026-09-25 — Keywords de un bullet = solo su texto**, no el stack de la
  `\cventry`. Los comentarios LaTeX (`% ADAPTAR`…) se descartan en el parser.
- **2026-09-28 — `JDRequirements` ampliado** respecto al spec: seniority cerrada y
  `role_family`, cada una con cita literal del JD que `check_evidence` verifica sin LLM
  (cita inventada ⇒ ExtractionError). `role_family` clasifica el trabajo diario, no el
  sector de la empresa: ese era el error de fondo con ventas/RR. HH. en empresas de IA.
- **2026-09-28 — LLM gratuito, no la API de Anthropic** (Demian no quiere gastar).
  Mismo patrón que family-expense-bot: protocolo `ClienteLLM` (`modelo` +
  `completar_json(prompt, schema)`) en `src/llm.py`; el extractor lo recibe por
  inyección y no importa ningún SDK. Implementación: `GeminiCliente`,
  `gemini-3.5-flash-lite` con salida estructurada nativa. `.env`: `GEMINI_API_KEY`,
  `GEMINI_MODELO`. `anthropic` fuera de las dependencias.
- **2026-09-28 — Contrato y validación intactos** tras el cambio de modelo: enums
  cerrados, cita literal verificada, error explícito. Un único reintento (con el motivo
  del rechazo en el prompt) solo si el JSON no valida o la cita no aparece. Respuesta
  cortada (`status != completed`), vacía o bloqueada ⇒ `ErrorDelLLM` sin reintento.
- **2026-09-28 — Cuota:** 429/RESOURCE_EXHAUSTED ⇒ `CuotaAgotada` con el motivo, sin
  reintentos (`INTENTOS_HTTP = 1`, sin backoff del SDK). Límites del plan gratuito: solo
  en AI Studio (Google no los publica); el diario se reinicia a medianoche del Pacífico.
- **2026-09-28 — Caché** en `data/cache/` (gitignored) por hash de versión de prompt +
  prompt + modelo + JD, y el archivo guarda `model`. Cambiar el prompt ⇒ subir
  `PROMPT_VERSION`. **Al leer resultados de regresión, revisar de qué modelo salieron.**
- **2026-09-28 — Prompt v2: junior = menos de 2 años.** Con v1 (junior ≤ 1 año, mid 2–4)
  el hueco de 1–2 años hizo que "18 months" saliera mid y "1.5+ years" junior en la misma
  corrida. Los tests de esos dos avisos ahora exigen `junior` (se endurecieron, no se
  relajaron). Regresión v2 con `gemini-3.5-flash-lite`: 7/7, todos en 1 intento. El caché
  registra `attempts` (1 o 2) para medir la cuota gastada; los timeouts no quedan registrados.
- **2026-09-28 — Regresión que falla ⇒ no se relaja el test.** Se muestra la salida real,
  la cita y la regla incumplida, y se ajusta el prompt con esa evidencia. Si con
  flash-lite no se puede, se documenta como limitación con datos.
- **2026-09-28 — Keys** en `.env` (gitignored, chmod 600), cargado con python-dotenv por
  la CLI y los tests `llm`. Tests `llm` excluidos por defecto: `pytest -m llm`.
- **2026-09-28 — Matcher (capa sin LLM).** Veredicto separado del score, política en
  `config/fit.yaml` (non_technical, mid y senior ⇒ no_apta; junior e indeterminado ⇒
  revisar; pasantía ⇒ apta). Cada skill del JD cae en una de cuatro categorías: en un
  bullet / en el CV sin bullet / en formación / carencia. Las carencias nunca se
  sugieren. `missing_keywords` = lo que el stack del proyecto respalda y el bullet no
  nombra. Modelos extra respecto del spec: `JDMatch` y `MatchResult.matched_keywords`.
  `parse_backing` lee el stack de cada `\cventry` y las filas `\skillrow`; los ítems
  "(en formación)" van aparte.
- **2026-09-28 — Vocabulario:** `canonicalize` para ítems de una lista de skills ("Excel"
  suelto vale ahí); `non_skills` (idiomas); términos nuevos solo inequívocos. La
  cobertura del vocabulario sobre las skills extraídas es una métrica a vigilar (era 15/31).
- **Pendiente:** fixtures `agents_booster.txt` y `movmo.txt` (Demian los pega después);
  jornada/carga horaria no está en `JDRequirements` (relevante para 4–6 h/día).
