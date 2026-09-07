# Collecting real data

This repository ships with **simulated** example data so the pipeline runs
out of the box.  To answer the question with *actual* observations from the
UTown Ben & Jerry's machine, follow the protocol below and drop the file in
as `data/observations.csv` (git-ignored so your raw data is never
committed).

## Session protocol

* **A session** is one attempt to get a tub: play repeatedly until you
  either grab a tub **or** you stop.
* **Record** for every session:
  * `plays` — the number of plays used.  If the session ended with a grab,
    include the winning play (so a first-try win is `plays = 1`).  If you
    gave up, record how many failed plays you made.
  * `grabbed` — `1` if the session ended with a successful grab, `0`
    otherwise.
* **Pre-commit to your stopping rule** (e.g. "I will spend at most \$5, or
  until I win, whichever comes first").  This keeps the censoring
  *independent* of the outcome, which is what makes the MLE
  $\hat p = \text{wins} / \text{plays}$ valid.

CSV format (header required):

```csv
plays,grabbed
4,1
7,0
1,1
```

## How many sessions do you need?

The standard error is $\mathrm{SE}(\hat p) = \sqrt{\hat p(1-\hat p)/N}$,
where $N$ is the **total number of plays**.  At the break-even value
$p = 1/6$ the 95% half-width is approximately

| Total plays $N$ | Half-width of 95% CI for $p$ |
|---|---|
| 60   | ±0.093 |
| 150  | ±0.060 |
| 300  | ±0.042 |
| 600  | ±0.030 |
| 1200 | ±0.021 |

Roughly 300–600 plays (≈ 50–100 sessions) narrow the estimate enough to
distinguish, say, $p = 0.13$ from the break-even $p^{*} = 1/6$.  Recording
the try count of every play (not just the wins) is what buys you this
precision.

## Practical notes

* **Machine settings change.**  Operators routinely adjust claw strength
  and payout ratios, and the machine refills/repacks the tubs.  Treat the
  estimate as valid for the observation period only, and record the date
  and time of day per session if you can.
* **Don't tilt or shake the machine.**  Apart from being against the
  rules, it changes the game and invalidates the i.i.d. assumption.
* **The geometric assumption can fail.**  If the claw visibly gets
  stronger after each failed attempt, $p$ is not constant and the geometric
  model underfits; the survival-curve diagnostic in
  `figures/survival_check.svg` will show curvature on the log scale.  The
  MLE $\hat p = \text{wins}/\text{plays}$ is still a sensible summary of
  the *average* win rate, but the distribution of try counts will not be
  geometric.
* **Honesty about the "in-person" bonus.**  The author could not operate
  the physical machine, so the shipped example data is simulated (see
  `analysis/simulate.py` for the exact seed and parameters).  Every number
  in `README.md` marked "example data" is therefore a reproducible
  illustration, not an observation.  The pipeline is ready the moment real
  data lands in `data/observations.csv`.
