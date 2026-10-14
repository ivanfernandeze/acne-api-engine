"""Statistical decision rule.

Covers RF-OE1-04, RF-OE2-05, RF-OE3-05, RF-OE4-03: the tests assert that the mandated rule is
actually the one implemented — Shapiro-Wilk first, then t-Student or Wilcoxon accordingly.
"""

from __future__ import annotations

import pytest

from app.schemas import PairedTestRequest
from app.stats.tests import InsufficientDataError, run_paired_test


def test_normal_differences_select_the_paired_t_test() -> None:
    """A clean, symmetric reduction passes Shapiro-Wilk, so the parametric test applies."""
    pretest = [9.0, 8.5, 9.4, 8.8, 9.1, 8.7, 9.3, 8.9, 9.2, 8.6]
    postest = [6.1, 5.4, 6.6, 5.8, 6.2, 5.6, 6.4, 5.9, 6.3, 5.5]

    result = run_paired_test(
        PairedTestRequest(
            design="paired", alpha=0.05, alternative="less", pretest=pretest, postest=postest
        )
    )

    assert result.normality.applicable is True
    assert result.normality.normal is True
    assert result.test_applied == "ttest_rel"
    assert result.p_value < 0.05  # HE1: the postest time is significantly lower.
    assert result.effect_size is not None
    assert result.effect_size.kind == "cohens_dz"


def test_non_normal_differences_select_wilcoxon() -> None:
    """A heavily skewed difference distribution fails Shapiro-Wilk and falls back to Wilcoxon."""
    pretest = [10.0] * 12
    postest = [9.9, 9.9, 9.9, 9.9, 9.9, 9.9, 9.9, 9.9, 9.9, 9.9, 9.8, 1.0]

    result = run_paired_test(
        PairedTestRequest(
            design="paired", alpha=0.05, alternative="less", pretest=pretest, postest=postest
        )
    )

    assert result.normality.normal is False
    assert result.test_applied == "wilcoxon"
    assert result.effect_size is not None
    assert result.effect_size.kind == "rank_biserial"


def test_binary_paired_data_lands_on_wilcoxon() -> None:
    """HE2 pairs correctness as 0/1, which is never normal — the rule must handle it."""
    pretest = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
    postest = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]

    result = run_paired_test(
        PairedTestRequest(
            design="paired",
            alpha=0.05,
            alternative="greater",
            pretest=[float(v) for v in pretest],
            postest=[float(v) for v in postest],
        )
    )

    assert result.normality.normal is False
    assert result.test_applied == "wilcoxon"
    assert result.p_value < 0.05


def test_one_sample_design_contrasts_against_the_baseline() -> None:
    """HE3: TSP of each respondent against the declared 60 % baseline (RF-OE3-05)."""
    result = run_paired_test(
        PairedTestRequest(
            design="one_sample",
            alpha=0.05,
            alternative="greater",
            pretest=[],
            postest=[92.5, 87.5, 95.0, 90.0, 85.0, 97.5],
            baseline=60.0,
        )
    )

    assert result.test_applied in {"ttest_1samp", "wilcoxon_1samp"}
    assert result.p_value < 0.05


def test_direction_matters() -> None:
    """A one-sided alternative pointing the wrong way must not reject."""
    pretest = [5.0, 5.2, 4.9, 5.1, 5.3, 4.8, 5.0, 5.2]
    postest = [8.0, 8.4, 7.7, 8.3, 8.1, 7.9, 8.2, 8.5]

    wrong_way = run_paired_test(
        PairedTestRequest(
            design="paired", alpha=0.05, alternative="less", pretest=pretest, postest=postest
        )
    )
    right_way = run_paired_test(
        PairedTestRequest(
            design="paired", alpha=0.05, alternative="greater", pretest=pretest, postest=postest
        )
    )

    assert wrong_way.p_value > 0.05
    assert right_way.p_value < 0.05


def test_identical_samples_are_rejected_with_a_clear_message() -> None:
    """Nothing changed between O₁ and O₂, so there is nothing to contrast."""
    with pytest.raises(InsufficientDataError, match="cero"):
        run_paired_test(
            PairedTestRequest(
                design="paired",
                alpha=0.05,
                alternative="two-sided",
                pretest=[1.0, 2.0, 3.0],
                postest=[1.0, 2.0, 3.0],
            )
        )


def test_constant_non_zero_difference_falls_back_to_wilcoxon() -> None:
    """A uniform improvement is a perfect effect, not missing data.

    Its variance is zero, so the paired t-test is undefined; the rule must route it to Wilcoxon
    rather than discard a result that is, if anything, maximally significant.
    """
    result = run_paired_test(
        PairedTestRequest(
            design="paired",
            alpha=0.05,
            alternative="greater",
            pretest=[5.0, 5.2, 4.9, 5.1, 5.3, 4.8, 5.0, 5.2],
            postest=[8.0, 8.2, 7.9, 8.1, 8.3, 7.8, 8.0, 8.2],
        )
    )

    assert result.normality.applicable is False
    assert result.test_applied == "wilcoxon"
    assert result.p_value < 0.05


def test_too_few_pairs_are_rejected() -> None:
    with pytest.raises(InsufficientDataError, match="al menos 3"):
        run_paired_test(
            PairedTestRequest(
                design="paired",
                alpha=0.05,
                alternative="less",
                pretest=[1.0, 2.0],
                postest=[0.5, 1.0],
            )
        )


def test_mismatched_sample_sizes_are_rejected() -> None:
    with pytest.raises(InsufficientDataError, match="mismo tamaño"):
        run_paired_test(
            PairedTestRequest(
                design="paired",
                alpha=0.05,
                alternative="less",
                pretest=[1.0, 2.0, 3.0],
                postest=[0.5, 1.0],
            )
        )


def test_results_are_reproducible() -> None:
    """RNF-05: the same inputs must give the same p-value, bit for bit."""
    request = PairedTestRequest(
        design="paired",
        alpha=0.05,
        alternative="less",
        pretest=[9.0, 8.5, 9.4, 8.8, 9.1, 8.7],
        postest=[6.1, 5.4, 6.6, 5.8, 6.2, 5.6],
    )

    first = run_paired_test(request)
    second = run_paired_test(request)

    assert first.p_value == second.p_value
    assert first.statistic == second.statistic
    assert first.function_version == second.function_version
