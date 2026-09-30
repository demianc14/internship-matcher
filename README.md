# cv-matcher

Dado un aviso de empleo (JD) que encontré y pegué a mano, analiza el fit contra mi
CV (`Base CV.tex`) y sugiere ediciones concretas: qué bullets ya cubren los
requisitos, qué keywords faltan y cómo reescribir un bullet sin inventar nada.

```
Base CV.tex ──cv_parser──► list[Bullet] ─┐
                                         ├─matcher──► list[MatchResult] ──► CLI
JD (.txt) ──jd_extractor (LLM)──► JDRequirements ─┘
```

> **Estado:** hecho: `cv_parser`, vocabulario, `jd_extractor` (nivel, rol, modalidad
> y jornada con cita literal; regresión 10/10 con `gemini-3.5-flash-lite`, prompt v3),
> el `matcher` determinístico con `cli match` y la reescritura de bullets validada sin
> LLM (`match --rewrite`).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m src.cli bullets                 # bullets del CV real con sus keywords
python -m src.cli bullets tests/fixtures/cv_sample.tex
cp .env.example .env                      # y pega tu GEMINI_API_KEY
python -m src.cli extract data/jds/strategia.txt
python -m src.cli match data/jds/strategia.txt   # veredicto + cobertura contra tu CV
python -m src.cli match data/jds/strategia.txt --rewrite   # + reescrituras sugeridas
python -m src.cli match-all               # todos los avisos, ordenados (solo caché)
python -m src.cli match-all --extract     # extrae antes los que falten (gasta cuota)
pytest && ruff check . && mypy src tests  # offline
pytest -m llm                             # regresión contra Gemini (usa caché)
```

## Arquitectura

| Módulo | Entrada → salida | LLM |
|---|---|---|
| `src/vocabulary.py` | `config/keyword_vocabulary.yaml` → patrones con alias es/en/de + implicaciones (MySQL ⇒ SQL) | no |
| `src/cv_parser.py` | `.tex` → `list[Bullet(project, text, keywords)]` | no |
| `src/llm.py` | protocolo `ClienteLLM` + `GeminiCliente` (única parte que habla con un proveedor) | sí |
| `src/jd_extractor.py` | texto del JD → `JDRequirements` validado con Pydantic; recibe el cliente por inyección | vía `ClienteLLM` |
| `src/matcher.py` | bullets + respaldo del CV × requisitos → `JDMatch` (veredicto, cobertura, `MatchResult` por bullet) con política en `config/fit.yaml` | no |
| `src/batch.py` | todos los avisos de `data/jds/` → tabla ordenada por veredicto y cobertura; solo caché salvo `--extract` | solo con `--extract` |
| `src/rewriter.py` | `JDMatch` → reescrituras de bullets: selección y validación sin LLM, redacción vía `ClienteLLM` | solo la redacción |

El matcher tiene una capa determinística (overlap de sets sobre el vocabulario)
**antes** de cualquier llamada al LLM, así que se testea sin mockear la API.

## Por qué el pivot (2026-09-25)

Este repo empezó como `internship-matcher`: un pipeline que *descubría* vacantes
(Arbeitnow, RemoteOK, fuente manual), las clasificaba por reglas y las rankeaba.
El código quedó en el tag `pre-pivot`. Se descartó por dos razones medidas:

1. **No hay fuentes para el mercado que importa.** Arbeitnow es mayoría Alemania (3 de 250 remotas en la página 1);
   RemoteOK es remoto global; Adzuna no cubre Ecuador; las bolsas ecuatorianas no
   tienen API y LinkedIn prohíbe el scraping. Las vacantes reales de Quito
   llegaban igual a mano.
2. **La clasificación por reglas marcaba mal los cargos.** Con 348 vacantes, el
   primer lugar del ranking fue *Software Engineer JVM* (un puesto full-time, no
   una pasantía), con 86.7. *GM AI Services* (General Manager) pasó como no-senior.
   *Account Executive - SMB* y *Social Comms* salieron **aptas**, y *Total Rewards
   Specialist* quedó en "revisar" con 66.7: el score solo miraba keywords técnicas
   y nada decía que el rol no era técnico. Cada regex nueva ("Lead" ⇒ senior salvo "Lead
   Generation", años de experiencia) arreglaba un caso y dejaba pasar otros.

Esos seis avisos quedaron en `data/jds/regresion/` (solo local, no se versionan
porque son texto de terceros) como casos de regresión para `jd_extractor`.

## Limitaciones y decisiones

- **El input siempre es un JD pegado a mano.** Sin scraping ni integración con
  portales, por decisión de diseño, no porque esté pendiente.
- **Sin UI.** Se usa desde la CLI.
- **Las keywords de un bullet salen solo de su texto.** El stack de la entrada
  (`\cventry{…}{Python, pandas, pytest}`) no se cuela en cada bullet: un
  reclutador lee el bullet, y "falta pytest en este bullet" es justo la señal útil.
- **El vocabulario es cerrado.** Lo que no está en `keyword_vocabulary.yaml` no se
  reconoce (por ejemplo, en el CV real "esquema relacional, triggers" no marca
  `sql`). Es lo que hace testeable a la capa determinística; el LLM del extractor
  ve el texto completo, pero el matcher compara contra términos canónicos.
- **El extractor amplía el modelo del spec.** `seniority_signal` es una categoría
  cerrada y hay un campo `role_family`, ambos con una cita literal del JD. La cita
  se verifica sin LLM: si el modelo inventa o parafrasea, la extracción falla en vez
  de devolver una clasificación sin respaldo. `role_family` es el trabajo diario del
  cargo, no el sector de la empresa; confundir las dos cosas producía los falsos
  "aptos" del pipeline anterior.
- **Modelo gratuito: Gemini `gemini-3.5-flash-lite`** (plan gratuito de AI Studio),
  para no pagar por uso. Es el mismo patrón que family-expense-bot: el extractor
  depende del protocolo `ClienteLLM`, no de un SDK, así que cambiar de proveedor no
  toca la validación. Con un modelo chico la validación pesa más: categorías
  cerradas, cita literal verificada y **un único reintento** (con el motivo del
  rechazo) solo si el JSON no valida o la cita no aparece en el JD.
- **Cuota limitada.** Google no publica los números del plan gratuito; se ven en
  [AI Studio](https://aistudio.google.com/rate-limit). La cuota diaria se reinicia a
  medianoche del Pacífico (02:00 en Ecuador con horario de verano de EE. UU., 03:00
  sin él). Ante un 429 el sistema falla con un mensaje visible, sin reintentos: ni
  del SDK (`INTENTOS_HTTP = 1`) ni del extractor. Si el límite es por minuto, basta
  esperar; si es el diario, hay que volver al día siguiente. En el peor caso, un JD
  consume 2 llamadas (la original y el reintento).
- **Caché por modelo.** Cada respuesta válida se guarda en `data/cache/` con clave
  hash(versión de prompt + prompt + modelo + JD), y el archivo registra qué modelo la
  produjo. El mismo aviso no gasta cuota dos veces y los resultados de modelos
  distintos nunca se mezclan. **Al leer resultados de regresión, revisa de qué
  modelo salieron** (campo `model` del caché, o `[modelo]` en la CLI y en los
  mensajes de fallo). Que un test pase con un modelo no dice nada del otro.
- **Veredicto y score son independientes.** El veredicto (`apta`/`revisar`/`no_apta`)
  sale de rol, nivel, modalidad y jornada, contra `config/fit.yaml`, y gana el más
  restrictivo: rol no técnico, mid o senior ⇒ no apta; junior o nivel indeterminado,
  presencial, más de 30 h/semana o tiempo completo ⇒ revisar. Una modalidad o
  jornada no indicada no descarta, pero se avisa. Ninguna cobertura
  de skills rescata un rol bloqueado. Ejemplo: Social Comms tiene 100 % de cobertura
  porque pide una sola skill que tengo (LLM), y sigue siendo no apta.
- **Cuatro categorías por skill, sin inventar.** Cada skill del aviso cae en una sola:
  en un bullet · en el CV pero sin bullet (stack de un proyecto o fila de
  habilidades: se puede nombrar con honestidad) · en formación (no se presenta como
  dominada) · no está en el CV (se lista, nunca se sugiere). Motivo medido: Software
  Engineer JVM pide SQL; está en mis habilidades y en el stack del proyecto con MySQL,
  pero en ningún bullet, y el cruce solo por bullets lo daba como carencia.
- **`missing_keywords` por bullet = lo que su proyecto respalda y el bullet no nombra.**
  Se agregó `matched_keywords` a `MatchResult` (no está en el spec): sin ese campo, un
  bullet con Power Automate marcaba 22 % sin decir por qué (lo explica la implicación
  a "process automation" y "low-code").
- **Una herramienta que el vocabulario no conoce no cuenta como carencia.** Se
  reporta como "no reconocida", y la cobertura la ignora. CE20261298 (Movmo) daba
  "100 % respaldado" mientras pedía n8n, Twilio, VAPI, GoHighLevel y WhatsApp
  Business API: se agregaron al vocabulario. Revisar siempre la línea de "no
  reconocidas" antes de creerle al porcentaje.
- **"No reconocidas" separa el ruido, sin esconderlo.** El extractor mete en las
  keywords ATS el título del cargo y la carrera pedida ("Pasante", "Ingeniería en
  Computación"). Esos términos no reconocidos que son parte del título o nombran una
  carrera se muestran aparte como "ignoradas", no se borran. Un término del
  vocabulario nunca se ignora aunque esté en el título ("Python Developer"). Los
  nombres de empresa ("Agents Booster", "Checkatrade") siguen como ruido: el
  extractor no captura la empresa. Con los 9 avisos: 50 de 105 términos reconocidos
  antes; 52 de 87 después de separar el ruido y ampliar el vocabulario.
- **Prompt engineering cuenta como cubierto por una certificación.** "Prompting para
  tareas de trabajo – Google" respalda prompt engineering ("prompting" es alias), igual
  que "Introducción a la IA – IBM" respalda "artificial intelligence". Es respaldo
  real pero débil: el matcher no pondera la fuerza de la evidencia (ver arriba).
- **`match-all` cuida la cuota.** Sin `--extract` no hace ninguna llamada: lo que no
  está en el caché aparece como "sin extraer". Con `--extract` avisa antes cuántas
  llamadas hará como máximo; si la cuota se agota deja de llamar, y un error en un
  aviso no detiene los demás. Orden: veredicto, cobertura en bullets, respaldo del
  CV y, en empate, el aviso con más skills reconocidas (50 % sobre 12 pesa más que
  sobre 2). Solo lee `data/jds/*.txt`, no la carpeta de regresión.
- **Cobertura con pocas skills engaña.** El porcentaje se muestra siempre junto con
  cuántas skills reconoció el aviso ("1/1" no es "7/9"). Los avisos no técnicos
  reconocen pocas: la mayoría de sus términos (ventas, compensaciones) están fuera
  del vocabulario, y se reportan como "no reconocidas".
- **Las certificaciones cuentan como bullets.** "Introducción a la IA – IBM" cubre
  "artificial intelligence". Es respaldo real pero débil; el matcher no pondera la
  fuerza de la evidencia.
- **Vocabulario ampliado con datos:** de las 31 skills que devolvió el extractor en los
  7 avisos, el vocabulario reconocía 15. Se sumaron solo términos inequívocos (RAG,
  APIs, Kotlin, CRM, bases de datos, agentes de IA, visión por computadora, IA). En
  una lista de skills, "Excel" suelto ahora cuenta (en texto libre sigue excluido
  por ser un verbo en inglés). Los idiomas se reconocen pero no cuentan como skill.
- **La reescritura solo hace visible lo que el CV ya respalda.** Agrega a un bullet
  términos que ese bullet demuestra por implicación (Power Automate ⇒ "RPA",
  "Low-Code/No-Code") o que usa el stack de su proyecto, escritos como los pone el
  aviso, para que un ATS los encuentre literalmente. Nunca agrega carencias, y cada
  término va a un solo bullet de todo el CV: a un ATS le basta encontrarlo una vez, y
  sugerir "agrega Python" en 9 bullets era ruido (medido con el CV real).
- **Cada reescritura se valida sin LLM** antes de mostrarse: contiene lo pedido, no
  agrega otras skills, no pierde keywords, todo número está en el original, crece
  como mucho 60 caracteres y no trae nombres técnicos nuevos. Esta última regla es
  heurística: detecta MAYÚSCULAS, CamelCase y dígitos ("UiPath"); una herramienta
  escrita en minúsculas y fuera del vocabulario podría pasar. Rechazada ⇒ un
  reintento con el motivo; si vuelve a fallar se muestra el motivo y no se sugiere nada.
- **Si una reescritura "suena bien" no es testeable.** Se verifica que sea válida; la
  calidad de la redacción la juzga quien lee. Primera corrida real (StrategIA):
  válida, pero en el segundo intento, y agrega los términos como un paréntesis
  ("Power Automate (Low-Code/No-Code y RPA)") más que como prosa natural.
- **Caso real que la validación no atrapó.** Con "automatizaciones" como alias de
  automatización de procesos, la reescritura de Agents Booster produjo "…diseñando e
  implementando flujos con Automatizaciones y un flujo de validación…": cumplía todas
  las reglas y era mala (torpe, y agregaba unos "flujos" que el original no tenía).
  Se quitó el alias. Un alias debe ser un término que un ATS buscaría, no una palabra
  común; y toda reescritura se lee antes de pegarla en el CV.
- **Alcance real con mi CV:** de los 9 avisos, 2 producen una reescritura de un
  bullet cada uno. StrategIA agrega "Low-Code" y "RPA" al bullet de Power Automate, y
  Movmo agrega "SQL" al bullet de procedimientos almacenados. Agents Booster y AI
  Engineer no producen ninguna: lo que el CV respalda ya aparece escrito en algún
  bullet. Los otros 5 no son aptos y no gastan llamadas.
- **Los comentarios LaTeX del CV** (`% ADAPTAR`, `% PENDIENTE`) son notas de
  trabajo y el parser los descarta.
- **El CV real no entra al repo.** Los tests usan `tests/fixtures/cv_sample.tex`.
  Hay un test extra contra el CV real que solo corre si el archivo existe.
