# Changelog — acne-ai-engine

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/);
versionado con [SemVer](https://semver.org/lang/es/).

## [No publicado]

### Added — segunda opinión antes de graduar (22/10/2026)

- **`app/models/condition_check.py`** (`GeminiConditionChecker`): para cada toma que el filtro local
  deja pasar como acné, `POST /infer/single` pregunta a Gemini si muestra `acne`, `piel_sana`,
  `otra_condicion` o `indeterminado`, con la lista de condiciones distintas del acné (sin pecas,
  lunares ni poros) y una frase de explicación. Pregunta revisión `c1`; versión
  `gemini/<modelo>/c1`. Motivo: una foto de celular de vitíligo con pecas dio `p(acné) = 0,995` en el
  filtro (ningún umbral la detiene) y el ensemble la graduó «casi sin lesiones» en las tres tomas;
  el filtro aprendió en parte la cámara (acné = acercamientos de ACNE04, sana = retratos CFD, otra =
  atlas de Fitzpatrick17k).
- Protocolo `SkinConditionChecker` y `SecondOpinion` (`app/models/base.py`);
  `get_second_opinion()` en `registry.py` (`None` si está desactivada o no se puede activar).
- Corre **en paralelo** con los modelos IGA (un hilo, `ThreadPoolExecutor`), así que casi no añade
  latencia. Si falla, se agota el tiempo o no hay clave, se gradúa como antes, sin advertencia.
- **Regla: otra condición no se gradúa.** Si la segunda opinión dice `otra_condicion`, la toma
  responde `verdict = no_acne` con `prediction` y `gags_secondary` nulos (la lectura calculada en
  paralelo se descarta), igual que el `no_acne` del filtro. Piel sana o acné junto a otra condición
  sí se gradúan, con advertencia en el portal. `SECOND_OPINION_STOPS_GRADING=false` la deja solo
  informativa.
- Contrato (`InferSingleResponse`): `second_opinion {label, conditions[], explanation, version} |
  null` (`SecondOpinionOut`) y **`verdict_source`** (`filtro` | `segunda_opinion`): quién decidió
  `verdict`.
- Variables nuevas: `SECOND_OPINION_BACKEND=none|gemini` (por defecto `none`),
  `SECOND_OPINION_TIMEOUT_SECONDS=20`, `SECOND_OPINION_STOPS_GRADING=true`; usa `GEMINI_API_KEY` y
  `GEMINI_MODEL`. Documentadas en `.env.example` (sin la clave).
- No es reproducible (RNF-03): el modelo remoto puede cambiar sin aviso; ninguna métrica de la tesis
  depende de ella. Solo se consulta desde la inferencia, que no se invoca en pretest (RF-TR-01).
- Pruebas: `tests/test_condition_check.py` con un verificador simulado; `conftest.py` fuerza
  `SECOND_OPINION_BACKEND=none` para que ninguna prueba envíe imágenes a Google. Total: **139**
  pruebas (pytest).

### Changed — umbral del filtro de acné a 0,1 (22/10/2026)

- `ACNE_GATE_THRESHOLD=0.1` (antes 0,009, el elegido en validación). Una foto de pecas (efélides)
  dio `p(acné) = 0,055`, pasaba como acné y el ensemble la graduó «severo». Coste en el test del
  modelo 2: sensibilidad sobre acné 0,964 (ACNE04 1,000; Fitzpatrick17k 0,863 frente a 0,900) y
  especificidad 0,949 (frente a 0,917). Se fija por entorno (`.env` del motor y
  `acne-infra/docker-compose.yml`); el checkpoint no cambia. Detalle en
  `acne-infra/docs/05_evaluation.md` §9.

### Security — privacidad de la segunda opinión (22/10/2026)

- Desactivada por defecto. Cuando está activa, sale del equipo solo una copia **reducida** (lado
  mayor 1 024 px), re-codificada en JPEG y **sin EXIF** (sin GPS, dispositivo ni fecha); no viaja el
  nombre, el código del caso ni la anamnesis.
- Requisitos aceptados por el equipo: clave de un proyecto de Google con facturación (el plan
  gratuito puede usar el contenido para entrenar) y consentimiento informado que mencione el análisis
  por un servicio externo. La clave vive solo en `.env`, nunca en `.env.example`.

### Added — filtro acné / piel sana / otra condición antes de graduar (01/10–19/10/2026)

- **`app/models/acne_gate.py`** (`TorchAcneGate`): EfficientNet-B0 de tres clases con recorte de
  rostro (YuNet). Decide en dos pasos: acné si `p(acné) ≥ umbral`; si no, la más probable entre piel
  sana y otra condición. `acne` → gradúa el ensemble IGA; `piel_sana` → lectura `sin_lesiones` con la
  probabilidad del filtro, sin ensemble; `no_acne` → no se gradúa. Sin checkpoint falla abierto
  (`gate_checked: false`). Vive en `/infer/single`, nunca en el cribado (RF-TR-01).
- Contrato de `/infer/single`: `verdict`, `gate_checked`, `acne_probability`, `gate_threshold`,
  `gate_version`, `gate_probabilities`; `prediction` y `gags_secondary` nulos con `no_acne`.
- **`training/gate/`**: construcción del dataset (ACNE04 + Fitzpatrick17k, Chicago Face Database,
  Fitzpatrick17k; partición por persona), entrenamiento, evaluación por fuente/fototipo/etnia/
  luminancia, reporte y `validate_clinic.py` para fotos de la clínica. Cada corrida deja
  `runs/gate/<run_id>/` y un reporte en `acne-infra/docs/entrenamientos/`.
- **`training/audit_fairness_cfd.py`**: auditoría de equidad de los modelos IGA sobre caras sanas.
- Activo desde el 19/10/2026: modelo 2 (`gate3_efficientnet_b0_20261019T154808Z`), acné de otras
  cámaras 0,413 → 0,900 frente al modelo 1. Detalle en `05_evaluation.md` §9–§10.

### Added — orientación de la cabeza en el cribado (01/10/2026)

- `LocalScreener` mide la pose con los landmarks de YuNet:
  `yaw_ratio = (x_nariz − x_medio_ojos) / ancho_rostro`; `frontal` si |r| < 0,04, girado si
  |r| ≥ 0,06, `indeterminado` con confianza < 0,6 o en la franja intermedia. `POST /image/screen`
  devuelve `pose` y `yaw_ratio` (solo geometría, nada clínico). `training/calibrate_pose.py` fija
  los cortes.

### Changed — imagen Docker con los modelos reales (23/09/2026)

- `WITH_TORCH=1` por defecto. Instala `torch==2.11.0`/`torchvision==0.26.0` desde `TORCH_INDEX`
  (CPU por defecto; `cu128` para NVIDIA) y `timm`, `captum`, `scikit-learn` y `ultralytics` en las
  versiones con que se entrenó. **Ya no usa el extra `.[torch]`**, que arrastraba `facenet-pytorch`
  (fija `torch<2.3`).
- Se añaden `libgl1` y `libglib2.0-0`, que OpenCV (cribado local) necesita en la imagen slim.
- `start-period` del healthcheck de 20 a 90 s (carga de tres modelos en CPU).

### Changed — regla del ensemble: gana el más seguro (18/09/2026)

- **`ENSEMBLE_RULE=mas_confiable`** (nuevo, por defecto): `prediction` es la lectura íntegra del
  miembro con mayor confianza —clase, vector y Grad-CAM—, con empate al de mayor peso; ya no se
  promedia nada. Motivo: el porcentaje de la sugerencia debe cuadrar con lo que un modelo leyó
  (petición del especialista). `voto_ponderado` conserva el promedio ponderado por F1. `version`
  pasa a `ensemble-max[…]` / `ensemble-avg[…]`. Test (n = 142): 0,746 / 0,668 / 0,916 frente a
  0,746 / 0,661 / 0,928 del voto ponderado —mismo número de aciertos (106 de 142), distinto reparto por clase—
  (`training/evaluate_ensemble.py --rule`, `runs/v1.0/metrics-test-ensemble-{max,avg}.json`).
  `tests/test_ensemble_backend.py` cubre ambas reglas.

### Added — cribado local de admisión (17/09/2026)

- **`LocalScreener`** (`app/models/local_screener.py`, `SCREENING_BACKEND=local` por defecto). Hasta
  ahora `POST /image/screen` solo cribaba con Gemini; con la CNN en servicio respondía
  `checked: false` y el registro admitía cualquier imagen — un retrato a lápiz se archivó como
  fotografía clínica. El cribado local corre en la máquina (OpenCV YuNet + tono de piel +
  saturación) y rechaza lo que no sea la fotografía de un rostro humano; calibrado sobre ACNE04
  (0,43 % de fotos reales rechazadas). Límite: ilustraciones coloreadas con piel realista.
  `get_screener()` en `registry.py` desacopla el cribado del clasificador. `tests/test_local_screener.py`.

### Added — selección de modelo y segundo componente (17/09/2026)

- **Comparativa de clasificadores sobre el mismo test congelado.** `training/train.py` acepta
  `--model` (cualquier timm) y `--tag`; el checkpoint guarda `model_name`. `training/evaluate.py`
  acepta `--checkpoint`, `--tag`, `--no-primary` y expone `compute_metrics()` (una sola definición
  del conjunto de métricas de la Sección 8.5, con la nueva exactitud a ±1 grado). Nuevos:
  `training/config.resnet50.yaml`, `training/export_yolo_cls.py` (split por *hard links*),
  `training/train_yolo_cls.py` (YOLOv8-cls con hue/saturación a 0 y RandAugment apagado;
  `--folds` para la CV con los mismos folds que `cross_validate`) y `training/compare.py` (tabla y
  regla de selección: F1 macro en test, empate < 0,01 → EfficientNet-B2).
- **Resultado (test, n = 142):** YOLOv8s-cls F1 0,676 / exactitud 0,739 / AUC 0,925 (CV 5-fold
  0,753 ± 0,055); EfficientNet-B2 0,633 / 0,704 / 0,896; ResNet-50 0,627 / 0,697 / 0,905.
- **Ensemble servido (`MODEL_ARCH=ensemble`, `app/models/ensemble_backend.py`).** Los tres modelos
  se ejecutan a la par sobre cada imagen; `prediction` es el promedio de probabilidades ponderado
  por el F1 de cada uno (`ENSEMBLE_MEMBERS=tag:arch:path:peso`), y la respuesta lleva `members[]`
  (clase, confianza, vector, Grad-CAM propio, versión, checksum, latencia) y `agreement`
  (`unanime|mayoria|discrepancia`). Un miembro que no carga tira abajo el backend (no se sirve un
  ensemble a medias). `/audit/iia-check` añade `ensemble_members`. Test: 0,747 / 0,662 / 0,928
  (`training/evaluate_ensemble.py`). `tests/test_ensemble_backend.py`.
- **Backend torch multi-arquitectura.** `MODEL_ARCH=timm|yolo_cls`: `torch_backend.py` carga
  cualquier timm (`model_name` del checkpoint, Grad-CAM sobre la última capa convolucional
  encontrada) o un YOLOv8-cls de Ultralytics (normalización 0-1, cabeza `(probs, logits)`, mapeo de
  las clases del checkpoint a `IGA_CLASSES`; `sin_lesiones` → 0). `tests/test_torch_backend.py`.
- **Pipeline de detección de lesiones (`training/detection/`), pendiente de datos.** Ingesta
  VOC → YOLO con el MISMO test congelado que el clasificador, entrenadores YOLOv8-det y Faster
  R-CNN ResNet-50 FPN, y un evaluador común (mAP50/50-95, precisión/recall, MAE y correlación del
  conteo, concordancia Hayashi por conteo). Probado con un fixture sintético
  (`tests/test_detection_pipeline.py`; `ACNE_SMOKE_DETECTORS=1` entrena una época de cada uno).
  La mitad *Detection* de ACNE04 no está en disco: ver README.
- Dependencias: `ultralytics` (AGPL-3.0, declarado), `torchmetrics`, `pycocotools`.

### Added

- **Primera CNN real entrenada (17/09/2026).** `MODEL_BACKEND=torch` deja de ser una promesa:
  `checkpoints/efficientnet_b2_iga.pt` (run `20260917T031206Z`, v1.1) se entrenó sobre ACNE04 y
  `GET /health` responde `model_operational: true`. Sobre el test congelado (n = 142): exactitud
  0,704, F1 macro 0,633, AUC-ROC macro 0,896; **no alcanza** el 0,752 de la Fase 1 y todos los
  errores son entre grados adyacentes. Metodología y números en
  `acne-infra/docs/02_data_understanding.md` … `05_evaluation.md`.
- `training/ingest_acne04.py` — lleva `archive/acne_1024/all_1024` al layout
  `data/raw/<clase_iga>/` que espera `preprocess.py`, con el mapeo Hayashi → IGA
  (`levle0→casi_sin_lesiones`, `1→leve`, `2→moderado`, `3→severo`; `sin_lesiones` queda sin
  muestras porque ACNE04 no tiene piel sana).
- `training/cross_validate.py` — validación cruzada estratificada k-fold sobre train + val con la
  misma receta que `train.py` (`fit()` compartido); el test congelado nunca participa. Escribe
  `runs/<dataset_version>/cv-<run_id>.json` con media ± desviación.
- `train.py` implementa lo que `config.yaml` ya declaraba y se ignoraba: `warmup_epochs`,
  `mixed_precision` (bfloat16), `cutmix_probability`; y añade `unfreeze_blocks: all`,
  `backbone_learning_rate`, `balanced_sampler`, `dropout` y `random_resized_crop_scale`. El
  checkpoint guarda además `input_size`.

### Changed

- **Receta v1.1** en `config.yaml`: entrada 320 px (antes 260), backbone completo con lr 3e-5 y
  cabeza 1e-4, sampler balanceado en lugar de pesos en la pérdida, CutMix, 60 épocas con paciencia 8.
  La v1.0 literal de la Sección 8.1 sobreajustaba (F1 test 0,560) y queda registrada como
  `runs/v1.0/20260917T024935Z/` y `checkpoints/efficientnet_b2_iga.v1.0-baseline.pt`.
- `.env`: `MODEL_BACKEND=torch`, `MODEL_INPUT_SIZE=320`.
- `evaluate.py` calcula el AUC-ROC solo sobre las clases presentes en el split; con `sin_lesiones`
  vacía `roc_auc_score` lanzaba y la métrica salía `null`.
- `facenet-pytorch` no se instala: fija `torch < 2.3`, incompatible con el wheel cu128 que exige la
  RTX 5060. Preprocesado e inferencia usan el recorte cuadrado centrado ya previsto como alternativa.

- **Los grados GAGS pasan a español**: `none` → `sin_acne`, `mild` → `leve`, `moderate` →
  `moderado`, `severe` → `severo`, `very_severe` → `muy_severo`. La escala IGA ya estaba en español
  y llevar las dos escalas clínicas en idiomas distintos dentro de la misma fila era la clase de
  incoherencia que se nota primero en la defensa. Espejo exacto de
  `acne-api/src/common/domain/iga.ts`, cambiado en el mismo commit; `GAGS_LABELS` se añade aquí para
  que las dos copias sigan siendo comparables línea a línea.

### Added

- **Cribado de admisión** — `POST /image/screen` decide si una imagen es una fotografía facial
  evaluable. **No emite ningún juicio clínico**: la respuesta no tiene campo para gravedad,
  lesiones ni clase IGA, y el prompt lo prohíbe. Corre en la admisión, que ocurre en las dos
  condiciones del experimento, así que revelar algo de la enfermedad metería la intervención en el
  pretest (RF-TR-01); una prueba comprueba la forma del tipo, no solo su comportamiento.
  **No busca acné**: `sin_lesiones` es una clase IGA válida y una cara limpia en un control es un
  caso resuelto. Un rostro sano se admite.
  Antes de gastar una llamada, una guardia local rechaza en ~10 ms lo que no tiene detalle alguno
  —lienzo en blanco, color plano— midiendo la desviación típica con PIL.
  El veredicto se **deriva** de `subject`, no se toma del campo `acceptable`: el modelo puede
  contradecirse, y creerle anularía la comprobación.
- `ImageScreener`, capacidad opcional separada de `Classifier`. Un backend que no sabe examinar
  imágenes responde `checked: false` en vez de fabricar un veredicto.
- 19 pruebas del cribado, ninguna llama a la API.
- **Lectura clínica razonada** (revisión de prompt `p2`). El modelo ya no devuelve solo una clase y
  cinco probabilidades: emite su lectura del caso en prosa, las lesiones elementales que distingue,
  las zonas comprometidas, **por qué esa clase IGA y no la contigua**, diagnósticos diferenciales,
  aspectos de manejo a valorar y las banderas que aporta la historia clínica. Una clase sin
  fundamento solo se puede creer o ignorar, y ninguna de las dos cosas es juicio clínico.
- **La anamnesis entra como contexto.** `POST /infer/single` acepta `clinical_context` con edad,
  sexo, fototipo y las respuestas del cuestionario. El prompt impone que **la severidad la fija la
  fotografía**: la historia contextualiza y levanta banderas, pero no gradúa — si no, un paciente
  que declara «acné severo» podría inducir la clase que mide OE2.
- `ClinicalContext` y `ClinicalAssessment` en el protocolo `Classifier`. El stub y el backend torch
  aceptan el contexto y lo descartan explícitamente: una CNN ve píxeles, y fingir lo contrario
  fabricaría una lectura. Su `assessment` es `None`, que es el valor honesto.
- `gags_secondary` usa ahora el GAGS que estima el propio modelo cuando lo hay, en lugar del centro
  de la banda IGA, que era una conveniencia de presentación.
- 8 pruebas más del validador y del turno que acompaña a la imagen. Ninguna llama a la API.
- **Backend `gemini`** (`app/models/gemini_backend.py`) — clasificador respaldado por un LLM
  multimodal externo, para poder medir el TPD antes de que exista una CNN entrenada. Transporte con
  `httpx` contra `generateContent`, `responseSchema` para salida estructurada y `temperature: 0`.
  Reporta `operational: false` **aunque funcione**: no es la CNN que fija la Sección 8.1.
- `model_kind` ∈ `stub | cnn | llm_externo` en `GET /audit/iia-check`, con dos avisos propios del
  caso Gemini. Dos backends pueden fallar la verificación de IIA por razones que no se parecen en
  nada, y hasta ahora la auditoría no permitía distinguirlas.
- **Regiones señaladas por el modelo** en lugar de Grad-CAM: se le piden los rectángulos de las
  zonas más afectadas y se dibujan sobre la imagen. Es lo que el modelo dice haber mirado, no una
  atribución de gradiente, y viaja por el mismo campo `heatmap_png`.
- 17 pruebas del validador de respuestas, **ninguna llama a la API**.

### Changed

- `MODEL_BACKEND` admite `"gemini"`; `HealthResponse`, `InferSingleResponse` e `IiaCheckResponse`
  amplían su unión `model_backend`.

### Fixed

- `tests/conftest.py` fija `MODEL_BACKEND=stub` antes de importar `app`. Sin eso, un `.env` con
  `MODEL_BACKEND=gemini` hacía que las pruebas de contrato llamaran a la API de verdad —gastando
  cuota, dependiendo de la red y, peor, afirmando el determinismo de un modelo remoto que nadie
  garantiza.
- `GET /health` devolvía 500 con el backend Gemini activo: `model_backend` seguía declarado como
  `Literal["stub", "torch"]` y la serialización fallaba. La prueba ahora comprueba contra
  `get_args(ModelBackend)` en vez de una lista escrita a mano.

### Notes

- Medición del 03/09/2026 sobre los 30 casos sintéticos con `gemini-2.5-flash`: **TPD 73,33 %**
  (22/30 exactos, **30/30 dentro de ±1 clase**), mediana 8 712 ms y p95 11 748 ms. La latencia
  **excede** el techo de 8 000 ms que RNF-01 fija para CPU, y el propio `/audit/iia-check` lo
  advierte. Las imágenes son sintéticas: la cifra mide la integración, no precisión clínica.


## [0.1.0] — 2026-08-31

### Added

- Aplicación FastAPI con los contratos exactos de la Sección 9.1: `POST /infer/single`,
  `POST /model/eval`, `GET /audit/iia-check` y `GET /health`, más el router `/stats/*`.
- **Backend `stub` determinista** — deriva una predicción estable del SHA-256 de la imagen y
  genera una superposición tipo Grad-CAM. Reporta `operational: false`, de modo que la tesis no
  puede declarar IIA = 1 sobre un marcador de posición.
- **Backend `torch`** — EfficientNet-B2 de `timm`, Grad-CAM con Captum, detección facial con
  crop, normalización ImageNet y entrada 260×260. Las importaciones son perezosas: el motor
  arranca en máquinas sin PyTorch.
- **Análisis estadístico con SciPy** — Shapiro-Wilk sobre las diferencias pareadas y luego t de
  Student para muestras relacionadas o Wilcoxon; diseño de una muestra para HE₃; tamaños de
  efecto Cohen's *dz* y *r* rank-biserial.
- **Telemetría** de latencia y tasa de error en ventana de 24 h, que alimenta el checklist IIA y
  el presupuesto de latencia de RNF-01.
- **Pipeline de entrenamiento** (fases 3–5 de CRISP-DM): `training/preprocess.py`,
  `training/dataset.py`, `training/train.py`, `training/evaluate.py` y `training/config.yaml`.
- 27 pruebas: determinismo del stub con imagen sintética y semilla fija, regla de decisión
  estadística y contrato HTTP.

### Notes

- Dos casos límite del contraste estadístico se resuelven explícitamente: diferencias todas cero
  (se rechaza el contraste) y diferencia constante no nula (varianza cero → Wilcoxon). Para el
  segundo se usa una tolerancia **relativa**, porque en punto flotante una diferencia
  «constante» conserva una dispersión de ~1e-16 que produciría una t de ~1e16 y un p-valor de
  cero: un resultado espurio fabricado por el redondeo.
