# internship-matcher

Pipeline de datos que reúne vacantes de varias fuentes, las normaliza y calcula un
score de compatibilidad **explicable** contra mi perfil, más sugerencias de qué partes
de mi CV activar para cada vacante.

No es un scraper ni un chatbot: es un ETL con principios fail-fast, funciones puras
testeables y un reporte de calidad cuantificado — mismo espíritu que
[world-cup-predictor](https://github.com/demianc14/WC_predictor) y
[MetalurgicaAndina_QC](https://github.com/demianc14/MetalurgicaAndina_QC).

> Estado: fases 0–6 cerradas, más un reporte HTML para revisar los resultados.
> 200 tests, 96 % de cobertura, `ruff` y `mypy` limpios.

## Qué produce

```
[APTA] 82/100  Pasante de Datos — Acme Ecuador
  https://www.multitrabajos.com/empleos/pasante-datos-1
  componentes: skills 82
  ✓ skills: pandas (demostrado), power bi (en_formacion), python (demostrado), sql (demostrado)

## Pasante de Datos  (manual:pasante-datos-acme)

Destacar estos bullets (ya cubren lo que piden):
  • [b01] Automaticé un reporte de ~500 filas con Python y pandas…
      cubre: pandas, python  ·  Pasante de Datos
      ⚠ CONFIRMAR (línea 24): que las 500 filas sean del proceso correcto y no de otro reporte.

Nombrar explícitamente (la tienes, pero ningún bullet la dice):
  • sql → un bullet de «App de Eventos»: esa entrada usa mysql, que implica sql
  • power bi → Habilidades: el CV la marca en formación; no la presentes como dominada

Brechas reales (no están en tu CV; no agregar sin respaldo):
  looker, tableau
```

## Reporte HTML

Cada corrida escribe `data/processed/reporte.html`: **un archivo autocontenido** que se
abre con doble clic. Filtra por veredicto, fuente, "solo Quito", score mínimo y texto;
cada vacante despliega su desglose (skills con tier y evidencia del CV, frases
similares, bloqueos y avisos) junto a las sugerencias de CV.

Los datos van **incrustados en el HTML**, no se leen con `fetch`: una página abierta
desde `file://` no puede leer archivos vecinos porque el navegador lo bloquea. Así
funciona sin servidor y sin conexión. La plantilla vive aparte en `web/template.html`
(HTML, CSS y JS planos, sin dependencias ni CDN) y `src/report.py` solo inyecta el
JSON, escapando `</` para que una descripción con `</script>` no rompa la página.

Se descartó un servidor local: el CLI ya corre el pipeline, y lo que faltaba era poder
**leer** 348 resultados sin scrollear la terminal.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # + ".[semantic]" para embeddings (~1,5 GB)

python -m src.cli                       # reprocesa los snapshots de data/raw/
python -m src.cli --fetch               # trae datos frescos de Arbeitnow + RemoteOK
python -m src.cli --suggest 3           # + sugerencias de CV para las 3 mejores
python -m src.cli --semantic            # + componente semántico (embeddings locales)
python -m src.cli --nueva pasante-acme  # plantilla para pegar una vacante de Quito

pytest && ruff check . && mypy src tests
```

## Arquitectura

```
src/etl/schema.py      contrato compartido: RawVacante (crudo) y Vacante (normalizada)
src/etl/extract.py     fetch crudo por fuente (Arbeitnow, RemoteOK) + snapshots
src/etl/manual.py      fuente manual: un .md por vacante pegada a mano
src/etl/transform.py   normaliza, deduplica, extrae keywords, arma el reporte de calidad
src/etl/match.py       score explicable + veredicto de encaje contra el perfil
src/etl/embeddings.py  adaptador opcional a sentence-transformers
src/cv/parser.py       Base CV.tex → secciones, entradas, bullets y sus marcas
src/cv/suggest.py      qué bullets activar / qué nombrar / qué falta, por vacante
src/report.py          arma el payload e inyecta los datos en la plantilla HTML
src/cli.py             corre todo y escribe data/processed/
web/template.html      plantilla del reporte (sin dependencias externas)
config/                perfil, vocabulario y ciudades: datos, no código
```

**Principios que sostienen el diseño**

1. **Extract sin lógica de negocio.** Las fuentes solo mapean campos al contrato. Toda
   interpretación vive en `transform`/`match`.
2. **Un solo contrato.** API y fuente manual emiten `RawVacante`; de ahí en adelante el
   pipeline no sabe de dónde vino cada vacante.
3. **Fail-fast con escalación documentada.** Ningún default silencioso: cada campo
   clasificado tiene valor + evidencia citada, o `None` + motivo.
4. **Degradación elegante.** Si una fuente cae o cambia de forma, queda en
   `source_error` y el run sigue con las demás.
5. **La configuración es data.** Perfil, vocabulario y ciudades son YAML auditables,
   no constantes escondidas en el motor.

## Fuentes

| Fuente | Cobertura | Notas |
|---|---|---|
| [Arbeitnow](https://www.arbeitnow.com/) | Europa, sobre todo Alemania (250/página) | Gratis, sin key. 1 página por corrida: sus términos piden no abusar. |
| [Remote OK](https://remoteok.com/) | Remoto global (~99 por llamada) | Gratis, sin key. Todo su catálogo es remoto → señal estructurada. Sus términos exigen mencionar y enlazar la fuente; las vacantes enlazan a remoteok.com. |
| Manual (`data/manual/*.md`) | Quito, híbrido y presencial | La única que cubre mi prioridad real. |
| ~~Adzuna~~ | — | **Descartada:** su API cubre 19 países (`gb us at au be br ca ch de es fr in it mx nl nz pl sg za`). Ecuador no está; en LatAm solo Brasil y México. |
| ~~LinkedIn, Computrabajo, Multitrabajos~~ | — | **No se scrapean:** sus ToS lo prohíben y no exponen API pública. Por eso existe la fuente manual. |

### Fuente manual: escribir lo mínimo

`python -m src.cli --nueva <slug>` crea la plantilla. Solo se llenan **tres cosas**:

```
url: https://www.multitrabajos.com/empleos/pasante-de-datos-1234567
title: Pasante de Análisis de Datos
---
(se pega el aviso completo, tal cual)
```

Ciudad, modalidad, seniority, jornada, horas, idioma y keywords **no se escriben**: los
deduce `transform` del texto pegado, con las mismas reglas que las vacantes de API. De
esas tres líneas, un aviso real produjo: `Quito` (deducido de "Cumbayá"), `hybrid`,
`intern`, `4 h/día`, idioma `es` y 6 keywords. Campos opcionales (`company`,
`location`, `modality`, `employment`, `posted_at`) solo si el texto no alcanza o hay
que corregir la deducción.

Se eligió un archivo por vacante en vez de un CSV: pegar un aviso con saltos de línea,
comas y comillas dentro de una celda es la parte más fácil de arruinar a mano. Un
archivo mal formado no rompe el run — se reporta con su motivo, como cualquier registro
rechazado.

## Transform: reglas de clasificación

Cada campo clasificado es un `Resolved` con **valor + origen + evidencia citada**, o
`None` + motivo: `no_signal`, `conflict` (se citan ambas señales), `weak_signal`
(solo "home office") o `unknown_value`.

| Campo | Señales | Regla clave |
|---|---|---|
| Modalidad | structured > location > title > description | Se usa el primer nivel con señal. `remote=false` no implica presencial. Las menciones negadas ("do not offer remote work") se ignoran. |
| Ubicación | campo de la fuente; si falta, el texto | Ciudades de `config/locations.yaml`; los valles de Quito mapean a Quito. Dos ciudades distintas ⇒ sin ubicación. |
| Seniority | título + `employment_raw` | Etiquetas distintas = conflicto, salvo `entry` + `intern`/`student_job` (nivel vs. tipo de puesto: gana el específico). |
| Jornada | título + `employment_raw` | "Full or part time" = conflicto. No se infiere de la descripción: los beneficios mencionan "part-time options". |
| Horas | título + descripción | Solo explícitas y plausibles (≤12 h/día, ≤60 h/semana). No se convierte semana↔día. |
| Experiencia | descripción | Años pedidos ("2+ years of experience"). Se ignora la trayectoria de la empresa ("With more than 40 years…") y todo lo que pase de 15 años. Con varios requisitos se guarda el menor. |
| Idioma | descripción | Stopwords en/de/es/fr; sin ganador claro ⇒ `None`. Umbral más bajo si no hay evidencia de otro idioma (los avisos locales son cortos). |
| Keywords | título + descripción | `config/keyword_vocabulary.yaml`, con alias en es/en/de ("RPA" y "automatización de procesos" son la misma skill). |

**Implicaciones** (`implies` en el vocabulario): tener una skill demuestra otras —
Power Automate ⇒ automatización de procesos y low-code; MySQL ⇒ SQL; Apps Script ⇒
JavaScript. Valen en todo el sistema: en el score (la skill implícita hereda el tier de
la que la implica y cita de dónde sale), en qué bullet cubre qué requisito y en la
auditoría perfil↔CV. Un solo nivel, sin encadenar. Se descubrió con la primera vacante
real cargada a mano: pedía RPA y low-code, y el sistema sugería destacar las
simulaciones Monte Carlo en vez del flujo de Power Automate.

**URLs:** un enlace de LinkedIn copiado desde una búsqueda (`?currentJobId=…&geoId=…`)
se reduce al enlace permanente `/jobs/view/<id>/`.

**Dedup:** mismo id ⇒ se queda el más reciente. Mismo título+empresa+ubicación
normalizados ⇒ duplicado entre fuentes. Mismo título+empresa en **otra ciudad se
conserva**: en los datos reales casi siempre es el mismo puesto abierto en varias
sedes; se lista como "posible duplicado".

### Reporte de calidad (corrida real, 2026-09-22)

| Decisión | Cantidad |
|---|---|
| Recibidas: Arbeitnow / RemoteOK / manual | 250 / 99 / 0 |
| Rechazadas por contrato | 0 |
| Duplicados removidos / posibles duplicados conservados | 1 / 8 grupos |
| Vacantes resultantes | 348 |
| Modalidad: remoto / híbrido / presencial | 125 / 36 / 11 |
| Modalidad sin resolver: sin señal / débil / conflicto | 150 / 14 / 12 |
| Seniority: senior / intern / entry / student_job / mid | 101 / 18 / 9 / 6 / 3 |
| Seniority sin resolver | 211 |
| Jornada: completa / parcial / sin señal | 177 / 4 / 166 |
| Con horas explícitas | 11 |
| Idioma: en / de / fr / es / sin resolver | 274 / 59 / 9 / 2 / 4 |
| En Quito | 0 (56 sin ubicación) |
| Sin keywords reconocidas | 87 |

## Match: score y veredicto, separados

- **Score 0–100** = cuánto encajan mis skills.
  - *skills*: Σ peso(tier) de las skills pedidas que tengo ÷ max(nº pedidas, 3). El
    mínimo de 3 evita el caso real de "100/100" por una vacante que solo menciona
    `rest api`. Lista ✓ lo que matchea (con tier y evidencia del CV) y ✗ lo que falta.
  - *semántico* (opcional): similitud del **título** de la vacante con los bullets del
    CV. Si está apagado, los pesos se renormalizan y la explicación lo dice.
- **Veredicto** `apta / revisar / no_apta` = restricciones prácticas, fuera del score:
  seniority, **años de experiencia pedidos** (más de 1 bloquea: mi experiencia formal
  son 3 meses de pasantía), modalidad+ubicación (híbrido/presencial fuera de Quito bloquea;
  **modalidad desconocida nunca descarta**, solo avisa), idioma de la descripción vs.
  mis idiomas de trabajo (≥B2), jornada completa y horas fuera de 4–6 h/día.

Una vacante senior en alemán puede tener buen score de skills y aun así no ser para mí:
por eso son dos números distintos y no uno solo.

Sobre la corrida real: **35 vacantes con score ≥60**; bloqueos por seniority senior
(101), descripción en alemán (59) o francés (9), e híbrido/presencial fuera de Quito
(47).

## Perfil y CV

`config/skills_profile.yaml` es la fuente de verdad, derivada de `Base CV.tex`. No hay
niveles autoevaluados: el **tier** sale de dónde aparece la skill en el CV
(`demostrado` = en un bullet de experiencia/proyecto, `listado` = solo en Habilidades o
por certificación, `en_formacion` = el CV lo dice) y cada una cita su evidencia. El
loader falla si una skill no está en el vocabulario (nunca podría matchear) o si un
idioma se cuela como skill.

`src/cv/parser.py` lee el `.tex` y entiende las marcas de trabajo del propio CV
(`% ADAPTAR`, `% CONFIRMAR`, `% AGREGAR`, `% PENDIENTE`, `% OPCIONAL`), asociándolas a
su elemento. `src/cv/suggest.py` cruza los gaps de cada vacante contra el CV y **nunca
sugiere agregar una skill que el CV no respalde** — la misma regla que el CV se pone a
sí mismo:

| Situación | Sugerencia |
|---|---|
| Skill pedida, ya en un bullet | destacar ese bullet |
| Skill pedida, en el stack de una entrada pero no en sus bullets | nombrarla ahí |
| Skill pedida, implícita en otra (MySQL ⇒ SQL) | nombrarla, diciendo de dónde sale |
| Skill pedida, solo en Habilidades | respaldarla en un bullet si es real; si el CV la marca "en formación", no presentarla como dominada |
| Skill pedida, ausente del CV | brecha: se lista, no se inventa |

Además reordena las secciones marcadas `% ADAPTAR`, señala entradas `% OPCIONAL` que no
aportan y avisa cuando el Perfil contradice la vacante (dice "pasantía híbrida en
Quito" y la vacante es remota). Los bullets sugeridos que tengan `% CONFIRMAR` o
`% PENDIENTE` salen con esa advertencia: no se manda un dato que el propio CV marca
como no verificado. `profile_consistency()` audita que el perfil siga derivado del CV
(hoy: 0 skills de más, 0 de menos).

## Calidad

```
200 tests · 96 % de cobertura · ruff y mypy limpios
transform 97 % · match 98 % · cv/parser 97 % · cv/suggest 95 % · manual 100 % · report 100 %
```

Ningún test llama a la red: las fuentes se prueban con fixtures recortados de
respuestas reales, incluidos registros inválidos a propósito. Los tests de integración
recorren las tres fuentes → transform → match → sugerencias, y verifican la invariante
del proyecto: **nada desaparece en silencio** (entradas = salidas + rechazadas +
duplicadas) y todo campo clasificado tiene valor o motivo.

El CV real no está en el repo (tiene datos personales): los tests usan
`tests/fixtures/cv_sample.tex`, y hay un test contra el CV real que se salta si no está
en la máquina.

## Limitaciones honestas

- **Las APIs gratuitas casi no sirven para mi caso.** De 348 vacantes reales, **0 en
  Quito**. Arbeitnow es sobre todo Alemania y RemoteOK es remoto global con mucho
  puesto senior. El segmento que me importa depende de la fuente manual. El pipeline
  funciona; el problema es la oferta disponible por API.
- **43 % de las vacantes quedan con modalidad sin resolver** (150/348, casi todas por
  falta de señal). Es deliberado: bajar el umbral sería adivinar. El match las trata
  como "desconocida", no las descarta.
- **Las reglas son regex con contexto, no NLP.** Se calibraron sobre snapshots reales
  en en/de/fr. Un "Office-based role… four days a week" queda como presencial aunque
  probablemente sea híbrido (el número va en palabras). Cada clasificación cita su
  evidencia para poder auditarla.
- **El componente semántico usa solo el título.** Medido sobre el snapshot real,
  comparar el cuerpo completo **no discrimina**: 0.464 para una pasantía de research vs.
  0.467 para atención al cliente en alemán, porque todos los avisos comparten relleno
  genérico. El título sí separa (0.522 vs. 0.28). La calibración 0.30–0.60 sale de la
  distribución real (p50 0.33, p90 0.49) con `paraphrase-multilingual-MiniLM-L12-v2`;
  hay que rehacerla si cambia el modelo.
- **Un título llega truncado desde RemoteOK** (`…Attributeâ`, un "™" cortado a medias
  en el propio feed). No se repara: completar bytes que la fuente no mandó sería
  inventar. Es 1 de 348.
- **Las keywords de idioma son ruidosas:** "German" matchea tanto "fluent German" como
  "a German company". Sirven como aviso, no como requisito confirmado.
- **No se separa la sección de requisitos** de la descripción: las keywords salen del
  texto completo, beneficios incluidos.
- **`src/etl/embeddings.py` no tiene cobertura de tests** (0 %): probarlo exigiría
  descargar el modelo en cada corrida de tests. Es un adaptador de ~10 líneas y se
  valida a mano; el resto del match se prueba con un embedder falso determinista.

## Decisiones

- **Fuente para Quito: archivos manuales, no CSV** ni scraping de bolsas locales.
- **Adzuna descartada** tras verificar que su API no cubre Ecuador.
- **Sin FastAPI:** el CLI cubre el flujo completo y no hay otro consumidor; agregar
  endpoints habría sido superficie sin uso. La interfaz es un HTML autocontenido que
  se abre con doble clic, no una app servida.
- **Score y encaje separados**, en vez de un único número que mezcle "sé hacer esto"
  con "puedo tomar este trabajo".
- **Modalidad desconocida no descarta:** con 43 % de vacantes sin señal, descartarlas
  habría escondido justo las que hay que revisar a mano.
