"""Local admission screening: is this a photograph of a real human face?

Covers: seccion 9.3 (captura) and RF-TR-01. Activated with ``SCREENING_BACKEND=local`` (default).

Until 17/09/2026 the screening step only existed for the Gemini backend, so with the CNN in
service ``POST /image/screen`` answered ``checked: false`` and the record accepted anything — a
pencil portrait was filed as a clinical photograph. This screener runs entirely on the machine
(no image leaves it, RNF-02) and decides three things, in order:

1. **Is there content at all?** A flat colour, a blank canvas or a lens cap has no detail
   (mean per-channel standard deviation below ``BLANK_STDDEV_THRESHOLD``).
2. **Is there a human face?** OpenCV's YuNet detector (``face_detection_yunet_2023mar.onnx``,
   Apache-2.0, vendored under ``assets/``) at two input sizes; the best score has to reach
   ``MIN_FACE_SCORE``. Calibrated on the 1 406 ACNE04 photographs: 4 (0,28 %) are missed at this
   threshold, while pencil sketches, line drawings, noise and flat images produce no face at all.
3. **Is it a photograph, not an illustration?** Inside the face box, the share of skin-toned
   pixels (YCrCb) and the mean saturation. A stylised cartoon of a real face keeps its geometry —
   YuNet still finds it — but loses the colour of skin (skin 0,22, saturation 0,07 in the probe);
   real photographs sit at skin ≥ 0,4 or saturation ≥ 0,12 in all but 2 of 1 406 cases.

What it says nothing about, on purpose: lesions, severity, grade. It runs at intake, in both
experimental conditions, and must leave no clinical trace (see ``routers/screening.py``).
Quality remarks (framing, focus, lighting) are warnings that never block.

It also reports the **head pose** (frontal or turned, and towards which side of the image), read
from the five YuNet landmarks, so acne-api can tell a frontal shot from a profile filed in the wrong
slot. Geometry only — the same in both conditions, silent about the skin. Calibrated on 400 ACNE04
photographs (``training/calibrate_pose.py``): frontal faces sit at |yaw| < 0,03, a face turned
~30° already reaches 0,06, profiles spread to 0,30; cheek close-ups that YuNet half-recognises come
with scores 0,30-0,51 and landmarks that point nowhere, hence ``POSE_MIN_SCORE``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.models.base import HeadPose, ScreeningResult

ASSETS = Path(__file__).resolve().parent / "assets"
YUNET_MODEL = ASSETS / "face_detection_yunet_2023mar.onnx"

BLANK_STDDEV_THRESHOLD = 4.0
#: Detector input sizes tried, longest side. 640 recovers small or tilted faces; 320 is cheap.
DETECTOR_SIZES = (640, 320)
MIN_FACE_SCORE = 0.30
#: A detection narrower than this share of the image is noise, not a face: on random textures
#: YuNet emits boxes under 1 % of the width with scores up to 0,36, while the smallest real face
#: in ACNE04 spans 11 %.
MIN_DETECTION_FRACTION = 0.05
#: Below BOTH, the face is an illustration or a colourless render, not skin under a camera.
MIN_SKIN_FRACTION = 0.40
MIN_SATURATION = 0.12
#: Warnings only.
MIN_FACE_WIDTH_FRACTION = 0.15
MIN_SHARPNESS = 5.0
DARK_MEAN = 40.0
BRIGHT_MEAN = 235.0
#: A second face is reported only when it is confident and comparable in size to the main one;
#: YuNet also fires on ears and cheek edges next to a large face, at scores up to 0,9 but small.
SECOND_FACE_SCORE = 0.8
SECOND_FACE_AREA_RATIO = 0.25
#: Head pose. Below this detection score the landmarks are guesses (cheek close-ups): no pose.
POSE_MIN_SCORE = 0.6
#: |yaw| under this is frontal; at or above TURNED it is a turned head. In between: undecided.
FRONTAL_MAX_YAW = 0.04
TURNED_MIN_YAW = 0.06


@dataclass(frozen=True)
class Face:
    """A detection in the coordinates of the analysed image."""

    x: int
    y: int
    width: int
    height: int
    score: float
    #: YuNet's five points as (x, y): right eye, left eye, nose tip, right and left mouth corner
    #: ("right" is the subject's). Empty when the detector did not provide them.
    landmarks: tuple[tuple[float, float], ...] = ()


def estimate_pose(face: Face) -> tuple[HeadPose, float | None]:
    """Which way the head points, from the nose's offset to the midpoint of the eyes.

    In a frontal face the nose tip sits between the eyes; as the head turns, it slides towards the
    side it points to while the eyes stay roughly put. The offset is scaled by the face width so it
    does not depend on resolution or distance.
    """
    if face.score < POSE_MIN_SCORE or len(face.landmarks) < 3 or face.width <= 0:
        return "indeterminado", None
    (right_eye_x, _), (left_eye_x, _), (nose_x, _) = face.landmarks[:3]
    yaw = (nose_x - (right_eye_x + left_eye_x) / 2) / face.width
    yaw = round(yaw, 3)
    if abs(yaw) < FRONTAL_MAX_YAW:
        return "frontal", yaw
    if abs(yaw) >= TURNED_MIN_YAW:
        return ("nariz_derecha" if yaw > 0 else "nariz_izquierda"), yaw
    return "indeterminado", yaw


class LocalScreener:
    """``ImageScreener`` that needs no network and no clinical model."""

    backend: str = "torch"
    version: str = "local-yunet-2023mar+heuristicas-v1"

    def __init__(self, model_path: Path = YUNET_MODEL) -> None:
        try:
            import cv2
        except ImportError as error:  # pragma: no cover - cv2 ships with ultralytics
            raise RuntimeError("El cribado local requiere opencv-python.") from error
        if not model_path.is_file():
            raise RuntimeError(f"No se encontró el detector de rostros YuNet en {model_path}.")
        self._cv2 = cv2
        self._detectors = {
            size: cv2.FaceDetectorYN.create(str(model_path), "", (size, size), 0.1, 0.3, 5000)
            for size in DETECTOR_SIZES
        }

    def screen(self, image_bytes: bytes, mime_type: str) -> ScreeningResult:
        del mime_type
        cv2 = self._cv2
        import numpy as np

        array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            return ScreeningResult(
                acceptable=False,
                subject="otro",
                issues=("no_es_fotografia",),
                reason="No se pudo leer la imagen. Envíe una fotografía JPEG o PNG.",
            )

        if self._is_blank(image):
            return ScreeningResult(
                acceptable=False,
                subject="otro",
                issues=("no_es_fotografia",),
                reason="La imagen está vacía o no tiene contenido. Tome la fotografía de nuevo.",
            )

        faces, analysed = self._detect(image)
        faces = _merge_overlapping(faces)
        plausible = [
            f
            for f in faces
            if f.width >= MIN_DETECTION_FRACTION * analysed.shape[1]
            and f.height >= MIN_DETECTION_FRACTION * analysed.shape[0]
        ]
        best = max(plausible, key=lambda f: f.score, default=None)
        if best is None or best.score < MIN_FACE_SCORE:
            return ScreeningResult(
                acceptable=False,
                subject="otro",
                issues=(),
                reason=(
                    "No se detecta un rostro humano. Tome la fotografía directamente a la cara "
                    "del paciente, de frente o de perfil, con buena luz."
                ),
            )

        crop = analysed[best.y : best.y + best.height, best.x : best.x + best.width]
        skin, saturation, sharpness, brightness = self._face_statistics(crop)
        if skin < MIN_SKIN_FRACTION and saturation < MIN_SATURATION:
            return ScreeningResult(
                acceptable=False,
                subject="otro",
                issues=("no_es_fotografia",),
                reason=(
                    "La imagen parece un dibujo, una ilustración o una imagen sin el color real de "
                    "la piel. Se necesita una fotografía tomada con la cámara."
                ),
            )

        issues: list[str] = []
        if best.width / analysed.shape[1] < MIN_FACE_WIDTH_FRACTION:
            issues.append("encuadre")
        main_area = best.width * best.height
        others = [
            f
            for f in plausible
            if f is not best
            and f.score >= SECOND_FACE_SCORE
            and f.width * f.height >= SECOND_FACE_AREA_RATIO * main_area
        ]
        if others:
            issues.append("encuadre")
        if sharpness < MIN_SHARPNESS:
            issues.append("enfoque")
        if brightness < DARK_MEAN or brightness > BRIGHT_MEAN:
            issues.append("iluminacion")

        pose, yaw = estimate_pose(best)
        return ScreeningResult(
            acceptable=True,
            subject="rostro_humano",
            issues=tuple(dict.fromkeys(issues)),
            reason="",
            pose=pose,
            yaw_ratio=yaw,
        )

    def best_face(self, image: Any) -> Face | None:
        """The most confident plausible face of a BGR image, in THAT image's coordinates.

        Same detection and plausibility rules as ``screen``; ``None`` when nothing reaches
        ``MIN_FACE_SCORE``. Used by the acne gate to crop the face before judging the skin.
        """
        faces, analysed = self._detect(image)
        plausible = [
            f
            for f in _merge_overlapping(faces)
            if f.width >= MIN_DETECTION_FRACTION * analysed.shape[1]
            and f.height >= MIN_DETECTION_FRACTION * analysed.shape[0]
        ]
        best = max(plausible, key=lambda f: f.score, default=None)
        if best is None or best.score < MIN_FACE_SCORE:
            return None
        scale = image.shape[1] / analysed.shape[1]
        return Face(
            x=int(best.x * scale),
            y=int(best.y * scale),
            width=int(best.width * scale),
            height=int(best.height * scale),
            score=best.score,
            landmarks=tuple((x * scale, y * scale) for x, y in best.landmarks),
        )

    # --- pieces, separated so tests can exercise the rules without a real face ----------------

    def _is_blank(self, image: Any) -> bool:
        small = self._cv2.resize(image, (64, 64))
        return float(small.reshape(-1, 3).std(axis=0).mean()) < BLANK_STDDEV_THRESHOLD

    def _detect(self, image: Any) -> tuple[list[Face], Any]:
        """Best detections over the configured sizes, plus the resized image they refer to.

        Faces from every size are mapped back to the largest analysed size so the caller reads one
        coordinate system.
        """
        cv2 = self._cv2
        height, width = image.shape[:2]
        largest = DETECTOR_SIZES[0]
        scale_largest = largest / max(height, width)
        analysed = cv2.resize(
            image, (max(1, int(width * scale_largest)), max(1, int(height * scale_largest)))
        )

        faces: list[Face] = []
        for size, detector in self._detectors.items():
            scale = size / max(height, width)
            resized = (
                analysed
                if size == largest
                else cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))
            )
            detector.setInputSize((resized.shape[1], resized.shape[0]))
            _, detections = detector.detect(resized)
            if detections is None:
                continue
            factor = scale_largest / scale
            for row in detections:
                x, y, w, h = (float(v) for v in row[:4])
                points = [float(v) * factor for v in row[4:14]]
                faces.append(
                    Face(
                        x=max(0, int(x * factor)),
                        y=max(0, int(y * factor)),
                        width=max(1, int(w * factor)),
                        height=max(1, int(h * factor)),
                        score=float(row[-1]),
                        landmarks=tuple(zip(points[0::2], points[1::2], strict=True)),
                    )
                )
        return faces, analysed

    def _face_statistics(self, crop: Any) -> tuple[float, float, float, float]:
        """(skin fraction, mean saturation 0-1, Laplacian variance, mean brightness 0-255)."""
        cv2 = self._cv2
        if crop.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        cr, cb = ycrcb[..., 1], ycrcb[..., 2]
        skin = float(((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)).mean())
        saturation = float(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[..., 1].mean() / 255.0)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        return skin, saturation, sharpness, brightness


def _merge_overlapping(faces: list[Face], iou_threshold: float = 0.5) -> list[Face]:
    """One box per face: the two detector sizes report the same face twice; keep the best score."""
    kept: list[Face] = []
    for face in sorted(faces, key=lambda f: f.score, reverse=True):
        if all(_iou(face, other) < iou_threshold for other in kept):
            kept.append(face)
    return kept


def _iou(a: Face, b: Face) -> float:
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2, y2 = min(a.x + a.width, b.x + b.width), min(a.y + a.height, b.y + b.height)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0
