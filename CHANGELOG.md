# Changelog — acne-ai-engine

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/);
versionado con [SemVer](https://semver.org/lang/es/).

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
