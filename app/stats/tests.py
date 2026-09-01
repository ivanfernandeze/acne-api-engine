"""Hypothesis contrasts for HE1-HE4.

Covers: RF-OE1-04, RF-OE2-05, RF-OE3-05, RF-OE4-03.

Decision rule mandated by the thesis (Hernandez, 2023): Shapiro-Wilk over the paired differences;
if normality is not rejected apply the paired t-Student, otherwise the Wilcoxon signed-rank test.
Alpha defaults to 0,05. Everything here is pure and side-effect free, so the same inputs always
produce the same output — which is what makes an external evaluator able to reproduce a p-value
six months later (RNF-05).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from app.schemas import (
    Alternative,
    EffectSize,
    NormalityResult,
    PairedTestRequest,
    PairedTestResponse,
    StatisticalTest,
)

#: Version stamped on every run, so a changed procedure is visible in the audit log.
FUNCTION_VERSION = "acne-stats/1.0.0"

#: Shapiro-Wilk is undefined below three observations.
MIN_SHAPIRO_N = 3


class InsufficientDataError(ValueError):
    """Raised when the sample cannot support any test."""


def run_paired_test(request: PairedTestRequest) -> PairedTestResponse:
    """Execute the contrast prescribed by the thesis for one hypothesis."""
    if request.design == "one_sample":
        return _run_one_sample(request)
    return _run_paired(request)


# --- Paired designs (HE1, HE2, HE4) ------------------------------------------


def _run_paired(request: PairedTestRequest) -> PairedTestResponse:
    if request.postest is None:
        raise InsufficientDataError("Un diseño pareado requiere las mediciones de postest.")

    pretest = np.asarray(request.pretest, dtype=float)
    postest = np.asarray(request.postest, dtype=float)

    if pretest.size != postest.size:
        raise InsufficientDataError(
            f"Las muestras pareadas deben tener el mismo tamaño: {pretest.size} vs {postest.size}."
        )
    if pretest.size < MIN_SHAPIRO_N:
        raise InsufficientDataError(
            f"Se requieren al menos {MIN_SHAPIRO_N} pares; se recibieron {pretest.size}."
        )

    differences = postest - pretest
    _reject_if_degenerate(differences)
    normality = _shapiro(differences)

    if normality.normal:
        result = stats.ttest_rel(postest, pretest, alternative=request.alternative)
        test_applied: StatisticalTest = "ttest_rel"
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        effect = _cohens_dz(differences)
    else:
        result = stats.wilcoxon(
            postest,
            pretest,
            alternative=request.alternative,
            zero_method="wilcox",
        )
        test_applied = "wilcoxon"
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        effect = _rank_biserial(differences)

    return PairedTestResponse(
        normality=normality,
        test_applied=test_applied,
        statistic=round(statistic, 6),
        p_value=round(p_value, 8),
        effect_size=effect,
        function_version=FUNCTION_VERSION,
    )


# --- One-sample design (HE3, RF-OE3-05) --------------------------------------


def _run_one_sample(request: PairedTestRequest) -> PairedTestResponse:
    """Contrast the observed TSP against a declared baseline.

    The Likert questionnaire is applied only in postest, so there is no paired sample. The thesis
    resolves this by comparing against the satisfaction reported for the traditional workflow, or
    failing that against a neutral theoretical value of 60 %.
    """
    values = np.asarray(request.postest or request.pretest, dtype=float)
    baseline = request.baseline if request.baseline is not None else 60.0

    if values.size < MIN_SHAPIRO_N:
        raise InsufficientDataError(
            f"Se requieren al menos {MIN_SHAPIRO_N} observaciones; se recibieron {values.size}."
        )

    differences = values - baseline
    _reject_if_degenerate(differences)
    normality = _shapiro(differences)

    if normality.normal:
        result = stats.ttest_1samp(values, popmean=baseline, alternative=request.alternative)
        test_applied: StatisticalTest = "ttest_1samp"
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        effect = _cohens_dz(differences)
    else:
        result = stats.wilcoxon(differences, alternative=request.alternative, zero_method="wilcox")
        test_applied = "wilcoxon_1samp"
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        effect = _rank_biserial(differences)

    return PairedTestResponse(
        normality=normality,
        test_applied=test_applied,
        statistic=round(statistic, 6),
        p_value=round(p_value, 8),
        effect_size=effect,
        function_version=FUNCTION_VERSION,
    )


# --- Building blocks ---------------------------------------------------------


def _has_negligible_variance(differences: np.ndarray) -> bool:
    """Whether the spread of the differences is indistinguishable from zero.

    The tolerance is RELATIVE to the magnitude of the differences, not absolute, because floating
    point makes "constant" samples only nearly constant: 8.2 − 5.2 is not bit-identical to
    8.0 − 5.0. Left unchecked, that dust becomes a standard deviation of ~1e-16, a t statistic of
    ~1e16 and a p-value of zero — a spurious "overwhelmingly significant" result manufactured
    entirely by rounding. Treating it as zero variance routes the sample to Wilcoxon instead.
    """
    scale = max(1.0, float(np.max(np.abs(differences))))
    return bool(np.ptp(differences) <= 1e-9 * scale)


def _reject_if_degenerate(differences: np.ndarray) -> None:
    """Reject only the sample that carries no information at all.

    All differences exactly zero means nothing changed between the two observations, and no test
    can speak to that. A CONSTANT non-zero difference is the opposite case — a perfect, uniform
    effect — and it must not be discarded: the paired t-test is undefined there (zero variance in
    the denominator), but Wilcoxon handles it, and ``_shapiro`` routes it there by reporting the
    normality test as not applicable.
    """
    if bool(np.all(np.abs(differences) < 1e-12)):
        raise InsufficientDataError(
            "Todas las diferencias pareadas son cero: no hay ningún cambio entre las dos "
            "observaciones que pueda contrastarse."
        )


def _shapiro(differences: np.ndarray) -> NormalityResult:
    """Shapiro-Wilk over the differences, with alpha fixed at 0,05 for the normality decision.

    Note the alpha here is the NORMALITY test's, not the hypothesis's: the choice of test must not
    change just because the researcher tightened the significance level of the contrast itself.
    """
    if differences.size < MIN_SHAPIRO_N:
        return NormalityResult(
            statistic=0.0,
            p_value=0.0,
            normal=False,
            applicable=False,
            note=(
                f"Shapiro-Wilk requiere al menos {MIN_SHAPIRO_N} observaciones. "
                "Se asume no normalidad y se aplica la prueba no paramétrica, que es la opción "
                "conservadora."
            ),
        )

    if _has_negligible_variance(differences):
        return NormalityResult(
            statistic=0.0,
            p_value=0.0,
            normal=False,
            applicable=False,
            note=(
                "Todas las diferencias son idénticas, por lo que su varianza es cero y "
                "Shapiro-Wilk no es aplicable. Se aplica la prueba de rangos con signo de "
                "Wilcoxon, que sí admite este caso."
            ),
        )

    statistic, p_value = stats.shapiro(differences)
    note: str | None = None
    if differences.size > 5000:
        note = (
            "Con más de 5000 observaciones Shapiro-Wilk detecta desviaciones triviales de la "
            "normalidad; interprete el resultado junto con la inspección gráfica."
        )

    return NormalityResult(
        statistic=round(float(statistic), 6),
        p_value=round(float(p_value), 8),
        normal=bool(p_value > 0.05),
        applicable=True,
        note=note,
    )


def _cohens_dz(differences: np.ndarray) -> EffectSize | None:
    """Cohen's dz — the effect size that accompanies a paired t-test."""
    sd = float(np.std(differences, ddof=1))
    if sd == 0:
        return None
    value = float(np.mean(differences)) / sd
    return EffectSize(kind="cohens_dz", value=round(value, 6), magnitude=_dz_magnitude(abs(value)))


def _rank_biserial(differences: np.ndarray) -> EffectSize | None:
    """Matched-pairs rank-biserial correlation — the effect size for Wilcoxon.

    r = (W+ − W−) / (W+ + W−), computed from the ranks of the absolute non-zero differences.
    """
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return None

    ranks = stats.rankdata(np.abs(nonzero))
    w_plus = float(np.sum(ranks[nonzero > 0]))
    w_minus = float(np.sum(ranks[nonzero < 0]))
    total = w_plus + w_minus
    if total == 0:
        return None

    value = (w_plus - w_minus) / total
    return EffectSize(
        kind="rank_biserial", value=round(value, 6), magnitude=_r_magnitude(abs(value))
    )


def _dz_magnitude(value: float) -> str:
    if value < 0.2:
        return "trivial"
    if value < 0.5:
        return "small"
    if value < 0.8:
        return "medium"
    return "large"


def _r_magnitude(value: float) -> str:
    if value < 0.1:
        return "trivial"
    if value < 0.3:
        return "small"
    if value < 0.5:
        return "medium"
    return "large"


def describe_alternative(alternative: Alternative) -> str:
    """Human-readable reading of the directional alternative, for the reports."""
    return {
        "two-sided": "bilateral",
        "less": "unilateral izquierda (postest menor que pretest)",
        "greater": "unilateral derecha (postest mayor que pretest)",
    }[alternative]
