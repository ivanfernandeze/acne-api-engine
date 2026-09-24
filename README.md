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

**Desde el 17/09/2026 el motor sirve una CNN entrenada** (`MODEL_BACKEND=torch`) sobre ACNE04
(1 406 imágenes, escala Hayashi mapeada a IGA; ver `acne-infra/docs/02_data_understanding.md`).
Se sirve un **ensemble** (`MODEL_ARCH=ensemble`) de los tres clasificadores comparados en
`training/compare.py` —YOLOv8s-cls, EfficientNet-B2 (Sección 8.1) y ResNet-50— que se ejecutan a la
par sobre cada imagen; la sugerencia es la lectura íntegra del modelo más seguro
(`ENSEMBLE_RULE=mas_confiable`; empate → mayor F1 en el test congelado; `voto_ponderado` conserva el
promedio ponderado por F1), y la respuesta lleva además la lectura y el Grad-CAM de cada modelo y el
nivel de acuerdo (`members[]`, `agreement`). Las cifras están en `acne-infra/docs/05_evaluation.md`;
ninguna variante alcanza todavía el F1 macro ≥ 0,752 de la Fase 1.

```bash
curl http://localhost:8000/audit/iia-check
# → { "operational": true, "model_backend": "torch", "model_kind": "cnn", "weights_checksum": "...", ... }
```

Sin checkpoint (o sin el extra `[torch]`) el motor arranca con un **clasificador stub
determinista** que deriva una predicción estable del SHA-256 de la imagen, y lo dice:
`operational: false`. `acne-api` lo lee y su matriz de cobertura degrada **OE2, OG e IIA a
`partial`**: la tesis no puede declarar IIA = 1 con un stub, y el sistema lo recuerda por sí mismo.

### Backend `gemini` — un LLM externo como alternativa

Para poder medir el TPD antes de que exista un modelo entrenado, el motor admite
`MODEL_BACKEND=gemini`: un modelo multimodal de propósito general al que se le describe la escala
IGA y se le pide clase, confianza, distribución sobre las cinco clases, GAGS estimado y las regiones
más afectadas.

```bash
# acne-ai-engine/.env  — el archivo está en .gitignore; la clave NUNCA va a .env.example
MODEL_BACKEND=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TIMEOUT_SECONDS=45
```

**`operational` sigue siendo `false`, y no es un descuido.** El diseño exige «una única CNN
validada» (Sección 3) y nombra EfficientNet-B2 (Sección 8.1); lo que responde aquí es otra cosa. Lo
que sí cambia es `model_kind`, que pasa de `stub` a `llm_externo`: los dos fallan la verificación de
IIA, pero no por lo mismo, y `GET /audit/iia-check` lo dice con todas sus letras.

Tres consecuencias que el código asume de forma explícita:

- **No hay reproducibilidad (RNF-03).** No hay pesos que hashear, `temperature: 0` no la garantiza y
  la versión remota puede cambiar sin aviso. `model_version` guarda
  `gemini/<modelo>/<revisión del prompt>` —**el prompt es parte del modelo**: cambiarlo cambia las
  graduaciones, así que un resultado solo es interpretable junto a la revisión que lo produjo.
- **No hay Grad-CAM.** Un mapa de atribución necesita los gradientes, que una API no expone. En su
  lugar se dibujan las regiones que el propio modelo declara haber mirado, y el portal las rotula
  **«regiones señaladas por el modelo»**. Llamarlas Grad-CAM sería falso.

#### Lectura clínica y anamnesis

Desde la revisión de prompt `p2` el backend no devuelve solo una clase: emite la lectura del caso en
prosa, las lesiones que distingue, las zonas comprometidas, **por qué esa clase y no la contigua**,
diferenciales, aspectos de manejo a valorar y las banderas que aporta la historia. Una clase sin
fundamento solo se puede creer o ignorar, y ninguna de las dos cosas es juicio clínico.

`POST /infer/single` acepta además `clinical_context` —edad, sexo, fototipo y las respuestas del
cuestionario, **sin identidad del paciente**— y el prompt fija la regla que lo hace admisible:

> La anamnesis contextualiza y puede señalar banderas, pero **NO decide la graduación**: la
> severidad la fija la imagen.

Sin esa regla, un paciente que declara «acné severo» podría inducir la clase que OE2 mide contra el
patrón de consenso. El backend `torch` recibe el mismo contexto por uniformidad de protocolo y lo
descarta: una CNN ve píxeles, y su `assessment` queda en `null`.

La prosa cuesta latencia. Una medición aislada con `p2` dio **23 598 ms** frente a los 8 712 ms de
mediana con `p1`, porque el modelo genera muchos más tokens de salida — otra razón por la que un
backend remoto no cabe en el presupuesto de RNF-01.

#### Cribado de admisión — `POST /image/screen`

Decide si una imagen puede entrar al registro clínico. `acne-api` lo llama **antes** de cifrar y
guardar: rechazar después dejaría la basura en disco y en la ficha, que es justo lo que se quiere
evitar.

```bash
curl -s localhost:8000/image/screen -H 'Content-Type: application/json'   -d '{"image_base64":"...","mime_type":"image/png"}'
# → {"acceptable":false,"checked":true,"subject":"otro","issues":["no_es_fotografia"],
#    "reason":"Por favor, envíe una fotografía de un rostro humano.", ...}
```

Tres decisiones que lo hacen admisible dentro del experimento:

- **No dice nada clínico.** La respuesta no tiene campo para gravedad, lesiones ni clase IGA, y el
  prompt lo prohíbe. El cribado corre en la admisión, que ocurre en las **dos** condiciones; si
  revelara algo de la enfermedad metería la intervención en el pretest y rompería RF-TR-01. Una
  prueba comprueba la forma del tipo, no solo su comportamiento.
- **Es idéntico en pretest y postest**, y un factor constante en ambos lados de un contraste
  pareado no puede explicar una diferencia entre ellos.
- **No busca acné.** `sin_lesiones` es una de las cinco clases IGA: una cara limpia en una visita de
  control es exactamente el aspecto de un caso resuelto, y cribar por «muestra acné» rechazaría la
  evidencia de que el tratamiento funcionó.

**Desde el 17/09/2026 el cribado es local por defecto** (`SCREENING_BACKEND=local`,
`app/models/local_screener.py`) y ya no depende del clasificador que sirva: OpenCV **YuNet**
(detector de rostros, ONNX de 230 KB vendido en `app/models/assets/`, Apache-2.0) a dos tamaños más
tres reglas — hay contenido, hay un rostro humano de tamaño plausible, y dentro del rostro hay color
de piel real (tono YCrCb y saturación). Rechaza objetos, paisajes, documentos, dibujos a lápiz,
imágenes planas y cartoons sin color de piel; admite una fotografía de rostro, de frente o de perfil.
Calibrado sobre las 1 406 fotografías de ACNE04: 6 rechazadas (0,43 %). **Límite conocido**: una
ilustración coloreada con tono de piel realista puede pasar; para eso está `SCREENING_BACKEND=gemini`
(a costa de enviar la imagen a un tercero). Ninguna imagen sale del equipo con el cribado local.

Los problemas de calidad (rostro pequeño o más de uno en cuadro, desenfoque, luz insuficiente) son
avisos que no bloquean. `SCREENING_BACKEND=none` desactiva el cribado y el endpoint responde
`checked: false` en lugar de inventarse un veredicto.
- **La respuesta se valida sin piedad y no se corrige.** Clase fuera de la escala, confianza fuera
  de [0, 1], probabilidades que no suman ~1 o GAGS fuera de [0, 44] hacen fallar la inferencia. Un
  valor recortado en silencio sería un juicio clínico inventado, y además reventaría los `CHECK` de
  la base lejos de donde se originó.

Las pruebas de este backend (`tests/test_gemini_backend.py`) **no llaman a la API**, y
`tests/conftest.py` fija `MODEL_BACKEND=stub` para toda la suite: sin eso, un `.env` con la clave
haría que cada prueba de contrato gastara cuota y dependiera de la red.

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

### Modelos entrenados (Git LFS)

Los pesos que sirve el motor van en el repositorio con **Git LFS**:
`checkpoints/*.pt` (EfficientNet-B2 v1.1 y v1.0, ResNet-50 y YOLOv8s-cls, ~167 MB) y el detector
facial del cribado `app/models/assets/face_detection_yunet_2023mar.onnx`. Instale Git LFS **antes**
de clonar; si no, esos archivos llegan como punteros de texto de 130 bytes y el motor cae al stub
(`operational: false`) y el cribado se desactiva.

```bash
git lfs install
git clone <url-del-repositorio>
# si ya se clonó sin LFS:
git lfs pull
```

El dataset **ACNE04** (`archive/`) y las corridas (`runs/`, `data/`) **no** se versionan: ACNE04 es
de terceros (Wu et al., 2019) y solo hace falta para reentrenar. Descárguelo del repositorio del
paper (`github.com/xpwu95/ldl`) o de una copia redimensionada a 1024 px (Kaggle), y deje las
imágenes `levle<N>_<id>.jpg` en `archive/acne_1024/all_1024/`.

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
uv pip install timm captum scikit-learn
# facenet-pytorch (MTCNN) fija torch<2.3 y no es compatible con el wheel cu128; sin él el
# preprocesado y la inferencia usan un recorte cuadrado centrado.

# Fase 2 — ingesta de ACNE04 (archive/acne_1024) al layout data/raw/<clase_iga>/.
# Mapeo Hayashi→IGA: levle0→casi_sin_lesiones, levle1→leve, levle2→moderado, levle3→severo.
uv run python -m training.ingest_acne04 --source ./archive/acne_1024/all_1024 --output ./data/raw

# Fase 3 — preparación: recorte y manifiesto (ACNE04 ya viene recortado al rostro: --detector none).
# --size debe coincidir con model.input_size de config.yaml y con MODEL_INPUT_SIZE de .env.
uv run python -m training.preprocess --source ./data/raw --output ./data/prepared --detector none --size 320

# Fase 4 — entrenamiento con semilla fija
uv run python -m training.train --config training/config.yaml

# Fase 5 — evaluación sobre el test congelado
uv run python -m training.evaluate

# Fase 5 — validación cruzada 5-fold sobre train+val (el test congelado no participa).
# Escribe runs/<dataset_version>/cv-<run_id>.json con F1 macro media ± desviación.
uv run python -m training.cross_validate

# Selección de modelo — los rivales de EfficientNet-B2 sobre el MISMO split y el MISMO evaluador
# (training/evaluate.py::compute_metrics). ResNet-50 (timm) y YOLOv8-cls (Ultralytics):
uv run python -m training.train --config training/config.resnet50.yaml --tag resnet50
uv run python -m training.evaluate --checkpoint checkpoints/resnet50_iga.pt --tag resnet50 --no-primary
uv run python -m training.export_yolo_cls          # data/yolo_cls/{train,val,test}/<clase> (hard links)
uv run python -m training.train_yolo_cls            # entrena y escribe metrics-test-yolov8s-cls.json
uv run python -m training.train_yolo_cls --folds 5  # CV con los mismos folds que cross_validate
uv run python -m training.compare                   # tabla + ganador (F1 macro en test; empate → EfficientNet-B2)
# Un modelo solo se sirve con MODEL_ARCH=timm|yolo_cls y MODEL_CHECKPOINT_PATH; los tres a la par con
# MODEL_ARCH=ensemble y ENSEMBLE_MEMBERS=tag:arch:path:peso (peso = F1 en test). Evaluar el ensemble
# con la regla de .env o con --rule mas_confiable|voto_ponderado (escribe metrics-test-ensemble-max|avg.json):
uv run python -m training.evaluate_ensemble
```

### Detector de lesiones (segundo componente) — pendiente de `Detection.tar`

YOLOv8 (detección) y Faster R-CNN ResNet-50 (línea base) se entrenan sobre la mitad *Detection* de
ACNE04 (cajas Pascal VOC, una por lesión), que **no está en `archive/`**. Descárguela del repositorio
del paper (Wu et al., 2019 — `github.com/xpwu95/ldl`, enlace a Google Drive; también en Kaggle y
Roboflow) y descomprímala en `archive/ACNE04_Detection/` con `JPEGImages/` y `Annotations/`.

```bash
uv run python -m training.detection.ingest_acne04_detection   # VOC → YOLO, MISMO test congelado que el clasificador
uv run python -m training.detection.train_yolo_det            # YOLOv8s, mAP + conteo → runs/v1.0/detection/
uv run python -m training.detection.train_fasterrcnn          # línea base, mismo evaluador
uv run python -m training.detection.evaluate_detection --model yolo|fasterrcnn
```

El evaluador común (`training/detection/common.py`) reporta mAP50 / mAP50-95, precisión y recall al
umbral de operación, MAE y correlación del conteo, y la concordancia del grado Hayashi derivado del
conteo. La integración al motor (`/infer/single` con cajas, conteo y severidad por conteo) se hace
cuando exista un checkpoint entrenado.

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
