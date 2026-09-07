"""Unit tests for the geometric-model fitting code.

Run with:  python -m pytest -q   (from the repository root)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from fit_geometric import chi2_gof, fit_geometric, mle_p, prob_within, wilson_ci


# ---------------------------------------------------------------- MLE

def test_mle_is_reciprocal_mean():
    plays = [6, 6, 6, 6, 6, 6]
    assert mle_p(6, sum(plays)) == pytest.approx(1 / 6)


def test_mle_with_censoring_counts_all_plays():
    # 5 successes, 30 total plays (25 completed + 5 abandoned) => 5/30
    assert mle_p(5, 30) == pytest.approx(1 / 6)


def test_mle_rejects_degenerate_input():
    with pytest.raises(ValueError):
        mle_p(0, 0)
    with pytest.raises(ValueError):
        mle_p(7, 5)


# ---------------------------------------------------------------- Wilson CI

def test_wilson_ci_contains_proportion():
    lo, hi = wilson_ci(50, 300)
    assert lo < 50 / 300 < hi


def test_wilson_ci_width_shrinks_with_sample_size():
    w1 = wilson_ci(50, 100)[1] - wilson_ci(50, 100)[0]
    w2 = wilson_ci(50, 400)[1] - wilson_ci(50, 400)[0]
    assert w2 < w1


def test_wilson_ci_respects_boundaries():
    lo, hi = wilson_ci(0, 30)
    assert lo == 0.0 and hi > 0.0
    lo, hi = wilson_ci(30, 30)
    assert hi == 1.0 and lo < 1.0


# ---------------------------------------------------------------- probabilities

def test_prob_within():
    p = 1 / 6
    assert prob_within(1, p) == pytest.approx(p)
    assert prob_within(3, p) == pytest.approx(1 - (5 / 6) ** 3)
    assert prob_within(0, p) == 0.0


# ---------------------------------------------------------------- end-to-end fit

def test_fit_geometric_end_to_end():
    plays = [6] * 60
    grabbed = [1] * 60
    fit = fit_geometric(plays, grabbed)
    assert fit.p == pytest.approx(1 / 6)
    assert fit.n_plays == 360
    assert fit.n_censored == 0
    assert fit.expected_tries == pytest.approx(6.0)
    assert fit.expected_cost == pytest.approx(6.0)


def test_fit_with_censored_sessions():
    plays = [6] * 5 + [3] * 2
    grabbed = [1] * 5 + [0] * 2
    fit = fit_geometric(plays, grabbed)
    assert fit.p == pytest.approx(5 / 36)
    assert fit.n_success == 5
    assert fit.n_censored == 2


def test_fit_validates_input():
    with pytest.raises(ValueError):
        fit_geometric([0, 1], [1, 1])          # a session with zero plays
    with pytest.raises(ValueError):
        fit_geometric([1, 2], [1])             # mismatched lengths


# ---------------------------------------------------------------- goodness of fit

def test_chi2_gof_accepts_data_from_the_model():
    # Deterministic seed: data drawn exactly from the fitted distribution
    # must NOT be rejected at a reasonable level.
    rng = np.random.default_rng(0)
    t = rng.geometric(0.1667, size=2000)
    chi2, dof, p_value, ok = chi2_gof(t, 0.1667)
    assert ok
    assert dof >= 2
    assert p_value > 0.001


def test_chi2_gof_rejects_clearly_wrong_model():
    # Try counts that are far too concentrated (p too small in reality) should
    # be rejected when tested against a geometric with p = 0.5.
    rng = np.random.default_rng(1)
    t = rng.geometric(0.05, size=500)   # mostly big counts
    chi2, dof, p_value, ok = chi2_gof(t, 0.5)
    assert p_value < 0.001


def test_chi2_gof_requires_data():
    with pytest.raises(ValueError):
        chi2_gof(np.array([], dtype=int), 0.2)
