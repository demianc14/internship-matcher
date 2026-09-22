# internship-matcher

Pipeline de datos que busca pasantías/empleos, las normaliza y calcula un score de
compatibilidad **explicable** contra mi perfil de habilidades y `Base_CV.tex`.

> Estado: **Fase 1 — Extract** cerrada (contrato `RawVacante` + fetcher de Arbeitnow).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest && ruff check . && mypy src tests
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

## Limitaciones conocidas

- **Arbeitnow no es "remoto/tech" como se asumía.** Corrida real del 2026-09-21:
  250 vacantes, solo 3 con `remote=true`; el resto son mayormente puestos en Alemania.
  Además `remote=false` no significa "presencial" (solo "no marcada remota"), así que
  transform no debe inferir presencialidad de ese flag.

- Las APIs gratuitas (Arbeitnow, RemoteOK) cubren sobre todo remoto global. La
  cobertura de Quito depende de lo que se cargue manualmente en el CSV.
- Adzuna: la cobertura de Ecuador/LatAm está **pendiente de verificar**.
