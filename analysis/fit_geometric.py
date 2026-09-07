"""Core statistical machinery for the claw-machine analysis.

Model
-----
Every play is an independent Bernoulli trial with the same success
probability ``p`` (the machine's effective grab probability).  Let ``T``
be the number of plays needed until (and including) the first successful
grab.  Then

    T ~ Geometric(p),    P(T = t) = (1-p)^(t-1) * p,    E[T] = 1/p.

The geometric distribution is the discrete analogue of the exponential
distribution and is the *only* discrete distribution with the memoryless
property, so it is the natural baseline model whenever the machine's
per-play behaviour is constant.

Maximum-likelihood estimation
-----------------------------
Data come in sessions.  A session either ends with a successful grab after
``t`` plays, or the player gives up after ``u`` failed plays (right
censoring).  With n successes and N = sum(t) + sum(u) total plays, the
likelihood is

    L(p) = p^n * (1-p)^(N-n)

and the MLE is simply

    p_hat = n / N,                    successes / total plays.

For uncensored data this reduces to p_hat = 1 / mean(t).  Because the MLE
is a binomial proportion, the standard error is sqrt(p_hat (1-p_hat) / N)
and confidence intervals use the Wilson score interval.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class GeometricFit:
    """Results of fitting a geometric model to observed claw-machine data."""

    p: float                          # MLE of the per-play success probability
    n_success: int                    # completed sessions (a tub was grabbed)
    n_censored: int                   # sessions where the player gave up
    n_plays: int                      # total plays across all sessions
    mean_tries: float                 # mean plays-to-success, completed sessions
    se: float                         # standard error of p_hat (binomial view)
    ci_wilson: tuple[float, float]    # Wilson score interval for p
    expected_tries: float             # E[T] = 1/p
    expected_cost: float              # expected spend per successful grab
    prob_within: dict[int, float]     # P(T <= k) for a set of k values

    def as_dict(self) -> dict:
        d = asdict(self)
        d["ci_wilson"] = list(d["ci_wilson"])
        return d


def mle_p(successes: int, plays: int) -> float:
    """MLE of p: successes / total plays.  Raises on degenerate input."""
    if plays <= 0:
        raise ValueError("total plays must be positive")
    if not (0 <= successes <= plays):
        raise ValueError("require 0 <= successes <= plays")
    return successes / plays


def wilson_ci(successes: int, plays: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (no continuity fix)."""
    if plays == 0:
        return (0.0, 1.0)
    p = successes / plays
    z2 = z * z
    denom = 1.0 + z2 / plays
    centre = (p + z2 / (2.0 * plays)) / denom
    half = z * math.sqrt(p * (1.0 - p) / plays + z2 / (4.0 * plays * plays)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def prob_within(t: int, p: float) -> float:
    """P(T <= t) = 1 - (1-p)^t for the geometric distribution."""
    if t < 1:
        return 0.0
    return 1.0 - (1.0 - p) ** t


def fit_geometric(
    plays: Sequence[int],
    grabbed: Sequence[bool | int],
    cost_per_play: float = 1.0,
    tub_price: float = 6.0,
) -> GeometricFit:
    """Fit the geometric model to per-session data.

    Parameters
    ----------
    plays : number of plays in each session (the winning play is included
        when the session ended with a grab).
    grabbed : 1 if the session ended with a successful grab, else 0.
    """
    plays = np.asarray(plays, dtype=int)
    grabbed = np.asarray(grabbed, dtype=int)
    if plays.ndim != 1 or grabbed.ndim != 1 or len(plays) != len(grabbed):
        raise ValueError("plays and grabbed must be 1-D arrays of equal length")
    if np.any(plays < 1):
        raise ValueError("each session must contain at least one play")

    success_mask = grabbed == 1
    n_success = int(success_mask.sum())
    n_censored = int((~success_mask).sum())
    n_plays = int(plays.sum())

    p = mle_p(n_success, n_plays)
    se = math.sqrt(p * (1.0 - p) / n_plays)
    lo, hi = wilson_ci(n_success, n_plays)
    mean_tries = float(plays[success_mask].mean()) if n_success else float("nan")

    expected_tries = 1.0 / p if p > 0 else float("inf")
    expected_cost = cost_per_play * expected_tries
    prob_within_k = {k: round(prob_within(k, p), 4) for k in (1, 2, 3, 5, 6, 10, 12)}

    return GeometricFit(
        p=p,
        n_success=n_success,
        n_censored=n_censored,
        n_plays=n_plays,
        mean_tries=mean_tries,
        se=se,
        ci_wilson=(lo, hi),
        expected_tries=expected_tries,
        expected_cost=expected_cost,
        prob_within=prob_within_k,
    )


def chi2_gof(t: np.ndarray, p_hat: float, min_expected: float = 5.0):
    """Pearson chi-square goodness-of-fit of try counts vs Geometric(p_hat).

    Cells are t = 1 .. K-1 individually plus one tail cell t >= K.  K is the
    largest value such that every *interior* cell still has expected count
    >= ``min_expected``; everything beyond is merged into the tail.  At
    least three cells are kept so that the test has positive degrees of
    freedom.  One degree of freedom is lost for estimating p.

    Returns ``(chi2, dof, p_value, cells_valid)`` where ``cells_valid`` is
    False when the minimum-expected-count rule could not be satisfied (the
    p-value is then approximate).  Returns None when no test is possible
    (dof < 1, e.g. all try counts are 1).
    """
    from scipy import stats  # lazy import keeps the core numpy-only

    n = len(t)
    if n == 0:
        raise ValueError("need at least one observed try count")
    max_t = int(t.max())
    p = float(p_hat)

    def exp_cell(k: int) -> float:
        return n * (1.0 - p) ** (k - 1) * p

    # Largest interior index whose individual cell still meets the rule.
    interior = 0
    while interior + 1 <= max_t and exp_cell(interior + 1) >= min_expected:
        interior += 1

    cells_valid = True
    if interior < 2:                      # keep at least 3 cells for df >= 1
        interior = min(2, max_t)
        cells_valid = False

    ks = list(range(1, interior + 1))
    probs = [exp_cell(k) / n for k in ks]
    tail = 1.0 - sum(probs)
    expected = np.array(probs + [tail]) * n
    observed = np.array([(t == k).sum() for k in ks] + [(t >= interior + 1).sum()])

    chi2 = float(((observed - expected) ** 2 / expected).sum())
    dof = len(observed) - 2  # cells - 1, minus 1 for the estimated parameter
    if dof < 1:
        return None
    p_value = float(stats.chi2.sf(chi2, dof))
    return chi2, dof, p_value, cells_valid
