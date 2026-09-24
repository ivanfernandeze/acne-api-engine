"""Lesion detection and counting: the second component of the two-model architecture.

The classifier (``training/train.py``, ``training/train_yolo_cls.py``) grades severity from the
whole face; the detector locates every lesion and counts them, which is how the Hayashi scale
that labels ACNE04 is defined in the first place (0-5 / 6-20 / 21-50 / >50 lesions).

Requires the *Detection* half of ACNE04 (Pascal VOC boxes), which is NOT in ``archive/`` — see
``ingest_acne04_detection.py`` for where to get it and where to put it.
"""
