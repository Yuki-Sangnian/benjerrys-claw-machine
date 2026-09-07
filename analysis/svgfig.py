"""Minimal dependency-free SVG figure writer for the analysis pipeline.

The committed figures in ``figures/`` are produced by this module rather
than by matplotlib so that they are small, deterministic, and easy to
diff.  All text is plain ASCII (``p_hat`` instead of math markup).
"""

from __future__ import annotations

from math import ceil, floor, log10

W, H = 640, 400
ML, MR, MT, MB = 62, 16, 36, 46
PW, PH = W - ML - MR, H - MT - MB

FONT = "font-family='DejaVu Sans, Arial, sans-serif'"


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _frame() -> str:
    return (
        f'<rect x="{ML}" y="{MT}" width="{PW}" height="{PH}" '
        f'fill="none" stroke="#333333" stroke-width="1"/>'
    )


def _x(v: float, x_lo: float, x_hi: float) -> float:
    return ML + (v - x_lo) / (x_hi - x_lo) * PW


def _y(v: float, y_lo: float, y_hi: float, ylog: bool = False) -> float:
    if ylog:
        return MT + PH - (log10(v) - log10(y_lo)) / (log10(y_hi) - log10(y_lo)) * PH
    return MT + PH - (v - y_lo) / (y_hi - y_lo) * PH


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    span = hi - lo
    if span <= 0:
        return [lo]
    mag = 10 ** floor(log10(span / n))
    for m in (1, 2, 5, 10):
        step = m * mag
        if span / n <= step:
            break
    start = ceil(lo / step) * step
    ticks: list[float] = []
    v = start
    while v <= hi + 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks


def _fmt(v: float) -> str:
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def _axes(
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    xlabel: str,
    ylabel: str,
    ylog: bool = False,
    xticks: list[float] | None = None,
    yticks: list[float] | None = None,
) -> str:
    xticks = xticks if xticks is not None else _nice_ticks(x_lo, x_hi)
    yticks = yticks if yticks is not None else _nice_ticks(y_lo, y_hi)
    out = [_frame()]
    for v in xticks:
        if not (x_lo - 1e-9 <= v <= x_hi + 1e-9):
            continue
        px = _x(v, x_lo, x_hi)
        out.append(f'<line x1="{px:.1f}" y1="{MT}" x2="{px:.1f}" y2="{MT + PH}" '
                   f'stroke="#dddddd" stroke-width="0.7"/>')
        out.append(f'<line x1="{px:.1f}" y1="{MT + PH}" x2="{px:.1f}" y2="{MT + PH + 4}" '
                   f'stroke="#333333" stroke-width="1"/>')
        out.append(f'<text x="{px:.1f}" y="{MT + PH + 16}" {FONT} font-size="10" '
                   f'text-anchor="middle">{_fmt(v)}</text>')
    for v in yticks:
        if not (y_lo - 1e-9 <= v <= y_hi + 1e-9):
            continue
        py = _y(v, y_lo, y_hi, ylog)
        out.append(f'<line x1="{ML}" y1="{py:.1f}" x2="{ML + PW}" y2="{py:.1f}" '
                   f'stroke="#dddddd" stroke-width="0.7"/>')
        out.append(f'<line x1="{ML - 4}" y1="{py:.1f}" x2="{ML}" y2="{py:.1f}" '
                   f'stroke="#333333" stroke-width="1"/>')
        out.append(f'<text x="{ML - 6}" y="{py + 3:.1f}" {FONT} font-size="10" '
                   f'text-anchor="end">{_fmt(v)}</text>')
    out.append(f'<text x="{ML + PW / 2}" y="{H - 8}" {FONT} font-size="11" '
               f'text-anchor="middle">{_esc(xlabel)}</text>')
    out.append(f'<text x="14" y="{MT + PH / 2}" {FONT} font-size="11" '
               f'text-anchor="middle" transform="rotate(-90 14 {MT + PH / 2})">'
               f'{_esc(ylabel)}</text>')
    return "".join(out)


def _svg(children: str, title: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}">\n'
        f'<title>{_esc(title)}</title>\n'
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>\n'
        + children
        + "</svg>\n"
    )


def _legend(items: list[tuple[str, str]]) -> str:
    """items: list of (label, style-snippet-for-a-short-line)."""
    out = []
    lx, ly = ML + 10, MT + 16
    for label, style in items:
        out.append(f'<line x1="{lx}" y1="{ly - 4}" x2="{lx + 22}" y2="{ly - 4}" {style}/>')
        out.append(f'<text x="{lx + 28}" y="{ly}" {FONT} font-size="10">{_esc(label)}</text>')
        ly += 15
    return "".join(out)


def observed_vs_fitted(t: list[int], p_hat: float) -> str:
    """Bar chart of observed relative frequencies vs fitted geometric PMF."""
    n = len(t)
    max_t = max(t)
    xs = list(range(1, max_t + 1))
    rel = [t.count(k) / n for k in xs]
    pmf = [(1.0 - p_hat) ** (k - 1) * p_hat for k in xs]
    mean_t = sum(k * r for k, r in zip(xs, rel))

    x_lo, x_hi = 0.5, max_t + 0.5
    y_hi = max(max(rel), max(pmf)) * 1.18
    bw = PW / (x_hi - x_lo) * 0.82

    body = [_axes(x_lo, x_hi, 0.0, y_hi,
                  "plays until first successful grab  t",
                  "probability",
                  xticks=_nice_ticks(1, max_t, 8))]
    for k, r in zip(xs, rel):
        px = _x(k, x_lo, x_hi)
        py = _y(r, 0.0, y_hi)
        body.append(f'<rect x="{px - bw / 2:.1f}" y="{py:.1f}" width="{bw:.1f}" '
                    f'height="{MT + PH - py:.1f}" fill="#5b8ff9" opacity="0.75"/>')
    pts = " ".join(f"{_x(k, x_lo, x_hi):.1f},{_y(p, 0.0, y_hi):.1f}" for k, p in zip(xs, pmf))
    body.append(f'<polyline points="{pts}" fill="none" stroke="#e8684a" stroke-width="1.6"/>')
    for k, p in zip(xs, pmf):
        body.append(f'<circle cx="{_x(k, x_lo, x_hi):.1f}" cy="{_y(p, 0.0, y_hi):.1f}" '
                    f'r="2.6" fill="#e8684a"/>')
    body.append(_legend([
        (f"observed (n={n}, mean={mean_t:.2f})", 'stroke="#5b8ff9" stroke-width="6" opacity="0.75"'),
        (f"fitted Geo(p_hat={p_hat:.3f})", 'stroke="#e8684a" stroke-width="1.6"'),
    ]))
    body.append(f'<text x="{W / 2}" y="20" {FONT} font-size="13" font-weight="bold" '
                f'text-anchor="middle">Observed try counts vs fitted geometric distribution</text>')
    return _svg("".join(body), "Observed vs fitted geometric")


def survival_check(t: list[int], p_hat: float) -> str:
    """Empirical vs theoretical survival on a log scale (memoryless check)."""
    n = len(t)
    max_t = max(t)
    xs = list(range(1, max_t + 1))
    y_lo, y_hi = 0.01, 1.0
    yticks = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

    def clip(v: float) -> float:
        return max(v, y_lo)          # keep log10 well-defined and in range

    emp = [clip(sum(1 for v in t if v > x) / n) for x in xs]
    theo = [clip((1.0 - p_hat) ** x) for x in xs]

    body = [_axes(1, max_t, y_lo, y_hi, "t (plays)", "P(T > t)  [log scale]",
                  ylog=True, xticks=_nice_ticks(1, max_t, 8), yticks=yticks)]
    # empirical step curve
    pts = [f"{_x(xs[0], 1, max_t):.1f},{_y(emp[0], y_lo, y_hi, True):.1f}"]
    for i in range(1, len(xs)):
        pts.append(f"{_x(xs[i], 1, max_t):.1f},{_y(emp[i - 1], y_lo, y_hi, True):.1f}")
        pts.append(f"{_x(xs[i], 1, max_t):.1f},{_y(emp[i], y_lo, y_hi, True):.1f}")
    pts.append(f"{_x(max_t + 1, 1, max_t):.1f},{_y(emp[-1], y_lo, y_hi, True):.1f}")
    body.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#5b8ff9" '
                f'stroke-width="1.6" stroke-linejoin="round"/>')
    pts_t = " ".join(f"{_x(x, 1, max_t):.1f},{_y(p, y_lo, y_hi, True):.1f}" for x, p in zip(xs, theo))
    body.append(f'<polyline points="{pts_t}" fill="none" stroke="#e8684a" '
                f'stroke-width="1.4" stroke-dasharray="5,4"/>')
    body.append(_legend([
        ("empirical S_hat(t)", 'stroke="#5b8ff9" stroke-width="1.6"'),
        (f"geometric (1 - p_hat)^t, p_hat={p_hat:.3f}", 'stroke="#e8684a" stroke-width="1.4" stroke-dasharray="5,4"'),
    ]))
    body.append(f'<text x="{W / 2}" y="20" {FONT} font-size="13" font-weight="bold" '
                f'text-anchor="middle">Memoryless check: survival curves are log-linear</text>')
    return _svg("".join(body), "Survival check")


def estimator_histogram(estimates: list[float], p_true: float,
                        p_mean: float, p_sd: float, n_sessions: int) -> str:
    """Histogram of Monte-Carlo MLE replicates with the true p marked."""
    lo, hi = min(estimates), max(estimates)
    nbins = 36
    width = (hi - lo) / nbins
    counts = [0] * nbins
    for v in estimates:
        i = min(int((v - lo) / width), nbins - 1)
        counts[i] += 1
    dens = [c / (len(estimates) * width) for c in counts]
    y_hi = max(dens) * 1.18
    x_lo, x_hi = lo - width, hi + width

    body = [_axes(x_lo, x_hi, 0.0, y_hi, "p_hat (MLE replicates)",
                  "density", xticks=_nice_ticks(lo, hi, 6))]
    bw = PW / (x_hi - x_lo) * width * 0.9   # bar width in pixels
    for i, d in enumerate(dens):
        px = _x(lo + (i + 0.5) * width, x_lo, x_hi)
        py = _y(d, 0.0, y_hi)
        body.append(f'<rect x="{px - bw / 2:.1f}" y="{py:.1f}" width="{bw:.1f}" '
                    f'height="{MT + PH - py:.1f}" fill="#5b8ff9" opacity="0.8"/>')
    for v, color, dash in ((p_true, "#e8684a", ""), (p_mean, "#333333", "4,3")):
        px = _x(v, x_lo, x_hi)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        body.append(f'<line x1="{px:.1f}" y1="{MT}" x2="{px:.1f}" y2="{MT + PH}" '
                    f'stroke="{color}" stroke-width="1.5"{dash_attr}/>')
    body.append(_legend([
        (f"true p = {p_true:.4f}", 'stroke="#e8684a" stroke-width="1.5"'),
        (f"mean p_hat = {p_mean:.4f} (bias {p_mean - p_true:+.4f})",
         'stroke="#333333" stroke-width="1.5" stroke-dasharray="4,3"'),
        (f"sd = {p_sd:.4f}", 'stroke="none"'),
    ]))
    body.append(f'<text x="{W / 2}" y="20" {FONT} font-size="13" font-weight="bold" '
                f'text-anchor="middle">Monte Carlo: MLE of p, n={n_sessions} sessions per replicate</text>')
    return _svg("".join(body), "Monte Carlo estimator distribution")
