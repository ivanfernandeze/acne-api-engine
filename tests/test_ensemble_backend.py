"""The ensemble runs every member, lets the surest speak (or averages by F1), reports agreement."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from app.domain.iga import IGA_CLASSES  # noqa: E402
from app.models.ensemble_backend import (  # noqa: E402
    EnsembleClassifier,
    MemberSpec,
    agreement_of,
    parse_members,
)


def _checkpoint(path: Path, seed: int) -> Path:
    torch.manual_seed(seed)
    model = timm.create_model("resnet18", pretrained=False, num_classes=len(IGA_CLASSES))
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": "resnet18",
            "version": f"r18_{seed}",
            "input_size": 64,
        },
        path,
    )
    return path


@pytest.fixture
def members(tmp_path: Path) -> list[MemberSpec]:
    a = _checkpoint(tmp_path / "a.pt", seed=1)
    b = _checkpoint(tmp_path / "b.pt", seed=2)
    return [MemberSpec("a", "timm", str(a), 0.7), MemberSpec("b", "timm", str(b), 0.3)]


def test_parse_members_reads_tag_arch_path_weight_and_windows_paths() -> None:
    specs = parse_members("y:yolo_cls:C:/w/y.pt:0.676, e:timm:./e.pt:0.633")
    assert [s.tag for s in specs] == ["y", "e"]
    assert specs[0].checkpoint_path == "C:/w/y.pt"
    assert specs[0].weight == pytest.approx(0.676)


@pytest.mark.parametrize(
    "spec",
    [
        "solo:timm:./a.pt:1.0",
        "a:timm:./a.pt:0.5,a:timm:./b.pt:0.5",
        "a:onnx:./a.pt:1,b:timm:./b.pt:1",
        "a:timm:./a.pt:x,b:timm:./b.pt:1",
    ],
)
def test_parse_members_rejects_malformed_specs(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_members(spec)


def test_default_rule_takes_the_most_confident_member_whole(
    members: list[MemberSpec], sample_image: bytes
) -> None:
    ensemble = EnsembleClassifier(members, seed=42, input_size=64, enable_gradcam=True)
    result = ensemble.predict(sample_image, "image/png")

    assert len(result.members) == 2
    assert [m.tag for m in result.members] == ["a", "b"]
    surest = max(result.members, key=lambda m: (m.confidence, m.weight))
    # Class, confidence, vector and Grad-CAM are the surest member's, untouched: no averaging.
    assert result.iga_class == surest.iga_class
    assert result.confidence == pytest.approx(surest.confidence)
    assert result.class_probs == surest.class_probs
    assert result.heatmap_png == surest.heatmap_png
    assert result.agreement in ("unanime", "discrepancia")
    assert all(m.heatmap_png is not None for m in result.members)


def test_weighted_vote_rule_averages_member_probabilities(
    members: list[MemberSpec], sample_image: bytes
) -> None:
    ensemble = EnsembleClassifier(
        members, seed=42, input_size=64, enable_gradcam=True, rule="voto_ponderado"
    )
    result = ensemble.predict(sample_image, "image/png")

    expected = {
        c: (0.7 * result.members[0].class_probs[c] + 0.3 * result.members[1].class_probs[c])
        for c in IGA_CLASSES
    }
    for c in IGA_CLASSES:
        assert result.class_probs[c] == pytest.approx(expected[c], abs=1e-3)
    assert result.iga_class == max(expected, key=expected.__getitem__)
    assert result.confidence == pytest.approx(expected[result.iga_class], abs=1e-3)
    # The lead Grad-CAM is the heaviest member's; each member carries its own.
    assert result.heatmap_png == result.members[0].heatmap_png


def test_identity_of_the_ensemble(members: list[MemberSpec]) -> None:
    ensemble = EnsembleClassifier(members, seed=42, input_size=64, enable_gradcam=False)
    assert ensemble.kind == "cnn" and ensemble.operational
    assert ensemble.version == "ensemble-max[r18_1+r18_2]"
    averaged = EnsembleClassifier(
        members, seed=42, input_size=64, enable_gradcam=False, rule="voto_ponderado"
    )
    assert averaged.version == "ensemble-avg[r18_1+r18_2]"
    assert len(ensemble.weights_checksum) == 64
    assert [spec.tag for spec, _ in ensemble.members] == ["a", "b"]


def test_a_missing_member_fails_the_whole_backend(
    tmp_path: Path, members: list[MemberSpec]
) -> None:
    broken = [*members, MemberSpec("c", "timm", str(tmp_path / "missing.pt"), 0.5)]
    with pytest.raises(RuntimeError, match="checkpoint"):
        EnsembleClassifier(broken, seed=42, input_size=64, enable_gradcam=False)


def test_agreement_levels() -> None:
    assert agreement_of(["leve", "leve", "leve"]) == "unanime"
    assert agreement_of(["leve", "leve", "severo"]) == "mayoria"
    assert agreement_of(["leve", "moderado", "severo"]) == "discrepancia"
    assert agreement_of(["leve", "severo"]) == "discrepancia"
