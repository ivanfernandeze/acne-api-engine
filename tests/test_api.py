"""HTTP contract of the engine (section 9.1)."""

from __future__ import annotations

import base64
from typing import get_args

from fastapi.testclient import TestClient

from app.config import ModelBackend
from app.main import app

client = TestClient(app)


def test_health_reports_the_backend() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "acne-ai-engine"
    # Tracks the declared union instead of a hand-written list, so adding a backend without
    # widening the response schema fails here rather than at runtime with a 500.
    assert payload["model_backend"] in set(get_args(ModelBackend))


def test_infer_single_returns_the_contract_of_section_9_1(sample_image: bytes) -> None:
    response = client.post(
        "/infer/single",
        json={
            "image_base64": base64.b64encode(sample_image).decode("ascii"),
            "mime_type": "image/png",
            "case_code": "ACN-2026-0001",
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["prediction"]["iga_class"] in {
        "sin_lesiones",
        "casi_sin_lesiones",
        "leve",
        "moderado",
        "severo",
    }
    assert 0.0 <= payload["prediction"]["confidence"] <= 1.0
    assert len(payload["prediction"]["class_probs"]) == 5
    assert payload["gags_secondary"]["grade"] is not None
    assert payload["latency_ms"] >= 0
    assert payload["model_version"]
    assert payload["seed"] == 42


def test_infer_single_is_deterministic_over_http(sample_image: bytes) -> None:
    body = {
        "image_base64": base64.b64encode(sample_image).decode("ascii"),
        "mime_type": "image/png",
        "case_code": "ACN-2026-0001",
    }

    first = client.post("/infer/single", json=body).json()
    second = client.post("/infer/single", json=body).json()

    assert first["prediction"] == second["prediction"]


def test_infer_rejects_invalid_base64() -> None:
    response = client.post(
        "/infer/single", json={"image_base64": "esto no es base64 !!!", "mime_type": "image/png"}
    )
    assert response.status_code == 422


def test_iia_check_reports_not_operational_with_the_stub() -> None:
    """The instrument that validates IIA = 1 must refuse to do so on a placeholder."""
    response = client.get("/audit/iia-check")

    assert response.status_code == 200
    payload = response.json()
    if payload["model_backend"] == "stub":
        assert payload["operational"] is False
        assert any("stub" in warning.lower() for warning in payload["warnings"])


def test_model_eval_explains_the_absence_of_metrics() -> None:
    response = client.post("/model/eval", json={"dataset_version": "v1.0", "split": "test"})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["confusion_matrix"]) == 5
    assert payload["note"] is None or "CRISP-DM" in payload["note"]


def test_paired_test_endpoint_applies_the_decision_rule() -> None:
    response = client.post(
        "/stats/paired-test",
        json={
            "design": "paired",
            "alpha": 0.05,
            "alternative": "less",
            "pretest": [9.0, 8.5, 9.4, 8.8, 9.1, 8.7, 9.3, 8.9],
            "postest": [6.1, 5.4, 6.6, 5.8, 6.2, 5.6, 6.4, 5.9],
            "seed": 42,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["normality"]["test"] == "shapiro-wilk"
    assert payload["test_applied"] in {"ttest_rel", "wilcoxon"}
    assert payload["p_value"] < 0.05


def test_paired_test_rejects_insufficient_data() -> None:
    response = client.post(
        "/stats/paired-test",
        json={
            "design": "paired",
            "alpha": 0.05,
            "alternative": "less",
            "pretest": [1.0, 2.0],
            "postest": [0.5, 1.0],
            "seed": 42,
        },
    )
    assert response.status_code == 422


def test_openapi_document_is_served() -> None:
    """acne-api generates its typed client from this document (section 4)."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert "/infer/single" in schema["paths"]
    assert "/audit/iia-check" in schema["paths"]
    assert "/stats/paired-test" in schema["paths"]
