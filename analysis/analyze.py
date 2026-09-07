"""Fit a geometric distribution to claw-machine data and produce a report.

Usage
-----
    python analysis/analyze.py --data data/example_observations.csv \
        --figures figures --results results

The input CSV must have a header row with at least two columns:

    plays   : number of plays in the session (include the winning play if the
              session ended with a grab)
    grabbed : 1 if the session ended with a successful grab, 0 otherwise

Outputs
-------
* a printed report and ``results/report.json``
* ``figures/observed_vs_fitted.svg`` - histogram vs fitted geometric PMF
* ``figures/survival_check.svg``     - empirical vs theoretical survival curve
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

from fit_geometric import chi2_gof, fit_geometric
import svgfig


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="path to the observations CSV")
    ap.add_argument("--figures", default="figures", help="directory for output figures")
    ap.add_argument("--results", default="results", help="directory for report.json")
    ap.add_argument("--tub-price", type=float, default=6.0, help="retail value of a small tub (USD)")
    ap.add_argument("--cost-per-play", type=float, default=1.0, help="price of one play (USD)")
    return ap.parse_args()


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    plays, grabbed = [], []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        if "plays" not in fields or "grabbed" not in fields:
            raise SystemExit(f"{path}: CSV must have columns 'plays' and 'grabbed'")
        for row in reader:
            try:
                plays.append(int(row["plays"]))
                grabbed.append(1 if int(row["grabbed"]) else 0)
            except ValueError as exc:
                raise SystemExit(f"{path}: unparseable row {row!r}: {exc}") from exc
    return np.asarray(plays, dtype=int), np.asarray(grabbed, dtype=int)


def write_figures(plays, grabbed, fit, figdir: Path) -> None:
    """Write the two diagnostic SVGs via the dependency-free svgfig module."""
    t = plays[grabbed == 1]
    figdir.mkdir(parents=True, exist_ok=True)
    (figdir / "observed_vs_fitted.svg").write_text(
        svgfig.observed_vs_fitted(t.tolist(), fit.p), encoding="utf-8")
    (figdir / "survival_check.svg").write_text(
        svgfig.survival_check(t.tolist(), fit.p), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    figdir = Path(args.figures)
    resdir = Path(args.results)
    figdir.mkdir(parents=True, exist_ok=True)
    resdir.mkdir(parents=True, exist_ok=True)

    plays, grabbed = load_data(data_path)
    fit = fit_geometric(plays, grabbed, cost_per_play=args.cost_per_play, tub_price=args.tub_price)

    p_star = args.cost_per_play / args.tub_price  # break-even win probability
    binom = stats.binomtest(fit.n_success, fit.n_plays, p_star)
    gof = chi2_gof(plays[grabbed == 1], fit.p)

    lines = [
        "=" * 66,
        "Ben & Jerry's claw machine - geometric model fit",
        "=" * 66,
        f"data file            : {data_path}",
        f"sessions             : {fit.n_success + fit.n_censored} "
        f"({fit.n_success} completed, {fit.n_censored} censored)",
        f"total plays          : {fit.n_plays}",
        f"mean tries / success : {fit.mean_tries:.3f}",
        "",
        f"MLE win probability  : p_hat = {fit.n_success}/{fit.n_plays} = {fit.p:.5f}",
        f"standard error       : SE = {fit.se:.5f}",
        f"95% Wilson CI for p  : [{fit.ci_wilson[0]:.5f}, {fit.ci_wilson[1]:.5f}]",
        f"expected tries       : E[T] = 1/p_hat = {fit.expected_tries:.3f} plays",
        f"expected spend       : ${fit.expected_cost:.2f} per successful grab",
        f"break-even p*        : {p_star:.4f} (tub ${args.tub_price:.2f}, play ${args.cost_per_play:.2f})",
        f"H0: p = p* (binomial): two-sided p-value = {binom.pvalue:.4f}",
        "",
        "P(grab within k tries) = 1 - (1 - p_hat)^k",
    ]
    for k, pr in fit.prob_within.items():
        lines.append(f"    k = {k:>2} : {pr * 100:.1f}%")
    if gof is not None:
        chi2, dof, pv, ok = gof
        lines.append("")
        lines.append(f"Goodness of fit (Pearson chi-square vs fitted Geo): "
                     f"chi2 = {chi2:.3f}, df = {dof}, p = {pv:.3f}")
        lines.append("  min-expected-cell rule met: " + ("yes" if ok else "no - p-value approximate"))
    lines.append("")
    lines.append(f"figures written to {figdir.resolve()}")

    print("\n".join(lines))

    report = {
        "data_file": str(data_path),
        "p": round(fit.p, 5),
        "n_success": fit.n_success,
        "n_censored": fit.n_censored,
        "n_plays": fit.n_plays,
        "mean_tries": round(fit.mean_tries, 3),
        "se": round(fit.se, 5),
        "ci_wilson": [round(fit.ci_wilson[0], 5), round(fit.ci_wilson[1], 5)],
        "expected_tries": round(fit.expected_tries, 3),
        "expected_cost": round(fit.expected_cost, 3),
        "prob_within": fit.prob_within,
        "break_even_p": round(p_star, 5),
        "test_H0_p_equals_break_even_pvalue": round(float(binom.pvalue), 5),
        "gof": None if gof is None else {
            "chi2": round(gof[0], 4), "df": gof[1], "p": round(gof[2], 4), "cells_valid": gof[3],
        },
    }
    with open(resdir / "report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    write_figures(plays, grabbed, fit, figdir)


if __name__ == "__main__":
    main()
