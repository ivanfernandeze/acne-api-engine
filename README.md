# acne-ai-engine

Motor de inferencia CNN, interpretabilidad Grad-CAM y análisis estadístico del prototipo de
diagnóstico asistido de acné — **Tesis UNT 2026**.

Python 3.11 · FastAPI · PyTorch + timm + Captum · SciPy

---

## Contexto

Soporta la tesis **«Agilización del proceso de diagnóstico de acné mediante inteligencia
artificial en un centro especializado en dermatología»** (Universidad Nacional de Trujillo, 2026).

Este servicio **es la variable independiente** del experimento: el modelo CNN cuya presencia o
ausencia distingue el pretest (IIA = 0) del postest (IIA = 1). Además aloja el análisis estadístico
con SciPy, porque es donde el stack obligatorio lo sitúa.

> **Aviso:** Herramienta de apoyo diagnóstico con fines de investigación académica. No sustituye el
> juicio clínico del profesional médico. Prototipo académico UNT — no certificado como dispositivo
> médico.

---

## Estado actual del modelo

**No hay una CNN entrenada todavía.** El motor arranca con un **clasificador stub determinista**
que deriva una predicción estable del SHA-256 de la imagen. Sirve para que toda la tubería clínica
—cola, Grad-CAM, TPD, matriz de confusión, contraste de HE₂— sea construible y demostrable antes
de que exista un modelo, y **no disimula lo que es**:

```bash
curl http://localhost:8000/audit/iia-check
# → { "operational": false, "model_backend": "stub", "warnings": [...] }
```

Ese `operational: false` es deliberado y consecuente: `acne-api` lo lee y su matriz de cobertura
degrada automáticamente **OE2, OG e IIA a `partial`**. La tesis no puede declarar IIA = 1 mientras
las predicciones no vengan de un modelo entrenado, y el sistema lo dice por sí mismo en lugar de
depender de que alguien lo recuerde.

---

## Levantamiento en menos de 10 minutos

Requiere [`uv`](https://docs.astral.sh/uv/). `uv` instala y fija **Python 3.11** por su cuenta, lo
que además resuelve la restricción de la máquina del estudio, que solo tiene 3.14 —donde PyTorch
aún no publica ruedas estables.

```bash
uv python install 3.11
uv venv --python 3.11
uv pip install -e ".[dev]"

cp .env.example .env
uv run uvicorn app.main:app --reload --port 8000
```

Verificación:

```bash
curl http://localhost:8000/health
```

- Swagger UI: <http://localhost:8000/docs>
- OpenAPI: <http://localhost:8000/openapi.json>

---

## Contrato del servicio (Sección 9.1)

| Método | Ruta | Objetivo |
|---|---|---|
| `POST` | `/infer/single` | OE2 — clase IGA + confianza + Grad-CAM |
| `POST` | `/model/eval` | OE2 — métricas sobre el split congelado |
| `GET` | `/audit/iia-check` | RF-TR-04 — ficha de operatividad |
| `POST` | `/stats/paired-test` | HE₁–HE₄ |
| `GET` | `/health` | Estado del servicio |

### Regla de decisión estadística

Shapiro-Wilk sobre las diferencias pareadas → **t de Student para muestras relacionadas** si no se
rechaza la normalidad, o **Wilcoxon de rangos con signo** en caso contrario. α = 0,05.
Tamaño de efecto: Cohen's *dz* o *r* rank-biserial, según la prueba aplicada. Para HE₃ el diseño es
de una muestra contra la línea base declarada (RF-OE3-05).

Dos casos límite se tratan explícitamente porque afectan a la validez del resultado:

- **Todas las diferencias exactamente cero** → se rechaza el contraste: no hubo cambio alguno.
- **Diferencia constante no nula** → la varianza es cero y la t queda indefinida, así que se aplica
  Wilcoxon. Se usa una tolerancia *relativa*: en punto flotante `8,2 − 5,2` no es idéntico a
  `8,0 − 5,0`, y esa dispersión de ~1e-16 produciría una t de ~1e16 y un p-valor de cero, un
  resultado «abrumadoramente significativo» fabricado por el redondeo.

---

## Entrenamiento (fases 2–5 de CRISP-DM)

```bash
# Extra con PyTorch. En la RTX 5060 (Blackwell, sm_120) hace falta CUDA 12.8 o superior:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[torch]"

# Fase 3 — preparación: detección facial, recorte y manifiesto
uv run python -m training.preprocess --source ./data/raw --output ./data/prepared

# Fase 4 — entrenamiento con semilla fija
uv run python -m training.train --config training/config.yaml

# Fase 5 — evaluación sobre el test congelado
uv run python -m training.evaluate
```

`evaluate.py` escribe `runs/<dataset_version>/metrics-test.json`, que es exactamente lo que sirve
`POST /model/eval` y lo que alimenta `REPORTE-EVALUACION-MODELO.pdf`.

Para activar el modelo real:

```bash
MODEL_BACKEND=torch
MODEL_CHECKPOINT_PATH=./checkpoints/efficientnet_b2_iga.pt
```

Si el checkpoint no existe o falta el extra, el motor **no finge**: registra el error, vuelve al
stub y mantiene `operational: false`.

### Decisiones de entrenamiento que importan

- **Splits estables por hash, no por barajado.** La partición se deriva del hash de la ruta más la
  semilla, de modo que añadir imágenes después no mueve silenciosamente una imagen del test al
  train, lo que invalidaría toda métrica ya reportada en la tesis.
- **Nunca se altera el `hue`.** La aumentación toca rotación, volteo, brillo y contraste, pero deja
  saturación y tono intactos: desplazarlos distorsionaría el tono de piel y el eritema, que son
  justamente la señal que la gradación evalúa.
- **Pesos de clase inversos a la frecuencia.** Los datasets públicos de acné están muy
  desbalanceados hacia los grados intermedios; sin compensar, el modelo aprende a no predecir
  nunca `severo`, que es precisamente la clase que un triaje no puede perderse.
- **Criterio de éxito declarado:** F1 macro ≥ 0,752 sobre el test congelado, el 85 % del 0,885 que
  reporta Watanabe (2025).

---

## Calidad

```bash
uv run ruff check app tests
uv run mypy app
uv run pytest
```

Las pruebas del stub son **deterministas con imagen sintética y semilla fija** (Sección 12.5): la
misma imagen produce siempre la misma predicción, lo que también es lo que exige RNF-03.

---

## Arquitectura

Tres repositorios independientes que se comunican solo por contratos OpenAPI:

| Proyecto | Rol | Puerto |
|---|---|---|
| [`acne-web`](../acne-web) | Portal del dermatólogo y PWA del paciente | 3000 |
| [`acne-api`](../acne-api) | Orquestación clínica, indicadores y estadística | 3001 |
| **`acne-ai-engine`** (este) | Inferencia CNN, Grad-CAM y SciPy | 8000 |

`app/domain/iga.py` es un **espejo** de `acne-api/src/common/domain/iga.ts`: el motor no puede
depender del orquestador en tiempo de ejecución. Cualquier cambio en la lista de clases debe
hacerse en ambos sitios y anotarse en el `CHANGELOG.md` de cada repositorio.

---

## Trazabilidad

Toda tanda de cambios actualiza [`../acne-infra/TRACEABILITY.md`](../acne-infra/TRACEABILITY.md) y
este `CHANGELOG.md`.
