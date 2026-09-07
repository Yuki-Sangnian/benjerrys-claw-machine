"""Simulate claw-machine sessions and Monte-Carlo-validate the estimator.

Generate reproducible example data (simulated, clearly NOT observed):

    python analysis/simulate.py --p 0.1667 --n 60 --seed 42 \
        --out data/example_observations.csv

Monte-Carlo validation of the MLE (bias, standard error, CI coverage):

    python analysis/simulate.py --monte-carlo --p 0.1667 --n 60 --reps 5000 \
        --seed 7 --figures figures
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import svgfig


def generate_example(p: float, n: int, seed: int, out_path: Path) -> None:
    """Write ``n`` completed sessions drawn from Geometric(p) to a CSV."""
    rng = np.random.default_rng(seed)
    plays = rng.geometric(p, size=n)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["plays", "grabbed"])
        for t in plays:
            writer.writerow([int(t), 1])
    print(f"wrote {n} simulated sessions (p={p:.4f}) to {out_path}")


def monte_carlo(p: float, n: int, reps: int, seed: int, figdir: Path | None) -> None:
    """Repeat ``n``-session experiments, estimate p each time, assess the MLE."""
    rng = np.random.default_rng(seed)
    estimates = np.empty(reps)
    coverage = 0
    for i in range(reps):
        t = rng.geometric(p, size=n)
        p_hat = 1.0 / t.mean()
        estimates[i] = p_hat
        se = np.sqrt(p_hat * (1.0 - p_hat) / t.sum())
        coverage += (p_hat - 1.96 * se <= p <= p_hat + 1.96 * se)

    mean_hat = estimates.mean()
    sd_hat = estimates.std(ddof=1)
    print(f"true p = {p}, sessions per experiment n = {n}, replicates = {reps}")
    print(f"mean p_hat          = {mean_hat:.5f}   (bias = {mean_hat - p:+.5f})")
    print(f"sd  p_hat           = {sd_hat:.5f}")
    print(f"95% CI coverage     = {coverage / reps:.4f} (nominal 0.95)")

    if figdir is not None:
        figdir.mkdir(parents=True, exist_ok=True)
        (figdir / "estimator_simulation.svg").write_text(
            svgfig.estimator_histogram(
                estimates.tolist(), p, float(mean_hat), float(sd_hat), n),
            encoding="utf-8")
        print(f"figure written to {figdir / 'estimator_simulation.svg'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p", type=float, default=1 / 6, help="true per-play success probability")
    ap.add_argument("--n", type=int, default=60, help="sessions per experiment")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", help="CSV path for the simulated example data")
    ap.add_argument("--monte-carlo", action="store_true", help="run the Monte-Carlo validation")
    ap.add_argument("--reps", type=int, default=5000, help="Monte-Carlo replicates")
    ap.add_argument("--figures", default="figures", help="directory for the Monte-Carlo figure")
    args = ap.parse_args()

    if args.out:
        generate_example(args.p, args.n, args.seed, Path(args.out))
    if args.monte_carlo:
        monte_carlo(args.p, args.n, args.reps, args.seed, Path(args.figures))
    if not args.out and not args.monte_carlo:
        ap.print_help()


if __name__ == "__main__":
    main()
