"""Forecast GTA VI trophy earn rates on Exophase at the case's target.

Model (per trophy, all work on the logit scale, rates in percent):

  * Level: anchored on the trophy's LAST non-null read.  Exophase refreshes
    its counts in discrete jumps, so the most recent read is the only level
    we trust; fitting a line and reading the line (as the house curve
    baseline does) sits above the last read whenever the last refresh
    pushed rates down.
  * Slope: a weighted least-squares line of logit(rate) against
    x = ln(1 + hours since 2026-11-19T00:00Z), over the last 36 hours of
    reads, with exponential recency weights (24 h half-life).  The slope is
    then shrunk toward its tier's median slope (platinum / gold / silver /
    bronze), because a single trophy's early reads are coarse (launch
    night's 638 tracked players quantise every rate to ~0.16%) and noisy.
    With fewer than 3 usable reads, or no movement at all, the slope is 0.
  * Median at target: last logit + shrunk slope * (x_target - x_last),
    with the logit move capped at +/-3 (a rate cannot credibly move more
    than that in one horizon, and the cap stops one noisy trophy from
    exploding).
  * Spread: the baseline random-walk term (0.5 * sqrt(horizon in days))
    combined in quadrature with (a) slope uncertainty -- the standard
    error of the fitted slope times the log-time distance to the target,
    plus the distance the shrinkage itself moved the slope -- and (b) a
    refresh-jump term, the RMS of the logit steps between successive
    DISTINCT reads in the case's history, scaled by the square root of the
    expected number of refreshes before the target (the practice showed
    refresh jumps are the dominant error: the flat baseline's 90% band
    covered 0 of 55 trophies).  Floored at 0.5 logit, capped at 2.5.

Quantiles are logistic(median + z_tau * spread), rounded to 4 decimals
half-up, clamped to [0, 100] and forced non-decreasing.  No network, no
randomness, no wall-clock use: two runs on the same case write identical
bytes.  Standard library only.

Input: every *.in in $MAKEFUN_INPUT_DIR (else /input), one JSON case each.
Output: <id>.out beside it in $MAKEFUN_OUTPUT_DIR (else /output):
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Clock origin the house curve baseline uses: the release day, 2026-11-19 00:00 UTC.
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Fitting window and recency decay for the slope fit.
FIT_WINDOW_HOURS = 24.0
HALF_LIFE_HOURS = 24.0
# How strongly a trophy's own slope is trusted over its tier's median.
# weight = n_eff / (n_eff + SHRINK_K); a trophy with few usable reads leans
# on its tier, one with many stands on its own.
SHRINK_K = 8.0
# Dampening of the extrapolated move: logit curves saturate (a trophy's
# rate cannot keep rising on a straight line in log-time), so the move is
# divided by (1 + MOVE_DAMP * dx).  0 disables it.
MOVE_DAMP = 0.0
# Where the median is anchored: "last" (the trophy's last read),
# "line" (the fitted line's value at the last read's time) or "blend"
# (the mean of the two).  The last read is the freshest truth, but a page
# frozen since the last refresh lags the underlying rates; the fitted line
# smooths that lag out but sits high after a downward refresh jump.
ANCHOR = "last"
# Caps on the forecast move and spread, in logit units.
MAX_MOVE = 3.0
SPREAD_FLOOR = 0.5
SPREAD_CAP = 2.5
RW_SIGMA = 0.5  # random-walk spread per sqrt(day), as the house baseline.
# Refresh-jump allowance: the RMS logit step between distinct reads in the
# history, squared, times at most REFRESH_CAP -- the jumps are the page
# catching up with movement the slope already extrapolates, so they are
# not independent shocks on top of it.  The holdout (tests/holdout.py)
# found any positive allowance costs loss in both the smooth and the noisy
# synthetic worlds, so the cap is 0: the allowance is off, and the random
# walk plus the fit residual carry the spread.
REFRESH_CAP = 0.0
# Weight on the slope-uncertainty term in the spread (0 disables it).
SLOPE_VAR_WEIGHT = 1.0
# Overall scale on the combined spread.  The random-walk term assumes the
# logit wanders with daily vol 0.5, but the observed trajectories are
# smooth trends, not random walks: the residual around a good fit is far
# smaller, so the honest spread is narrower than the random walk alone.
# Calibrated on the holdout (tests/holdout.py, 15 synthetic finals in each
# of three worlds -- smooth, noisy, very-noisy): 0.65 beats the trend
# baseline in all three (smooth 0.135 vs 0.182, noisy 0.199 vs 0.228,
# very-noisy 0.278 vs 0.288) and has the best worst case of the grid
# 0.6-0.75.  The 0.5 floor still binds on short horizons, so the practice
# cases keep a refresh-sized band.
SPREAD_SCALE = 0.65


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(y):
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (the house baseline's rounding)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_launch(moment):
    return (moment - LAUNCH).total_seconds() / 3600.0


def log_time(moment):
    """x = ln(1 + hours since launch); guarded for reads just before launch."""
    return math.log(1.0 + max(-0.99, hours_since_launch(moment)))


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def weighted_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x.  Returns (a, b, se_b, resid_var).

    Returns None when the fit is degenerate (one distinct x, zero weight,
    or fewer than 2 points).
    """
    total_w = sum(ws)
    if total_w <= 0 or len(xs) < 2:
        return None
    mean_x = sum(w * x for x, w in zip(xs, ws)) / total_w
    mean_y = sum(w * y for y, w in zip(ys, ws)) / total_w
    sxx = sum(w * (x - mean_x) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mean_x) * (y - mean_y) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = mean_y - b * mean_x
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    resid_var = sum(w * r * r for r, w in zip(resid, ws)) / total_w
    # Standard error of the slope under the weighted fit.
    dof = max(1, len(xs) - 2)
    se_b = math.sqrt(resid_var * len(xs) / dof / sxx) if resid_var > 0 else 0.0
    return a, b, se_b, resid_var


def trophy_series(case, index):
    """(times, logits, weights) for one trophy, plus its last (x, logit).

    Only reads inside the fitting window count for the slope; the level
    always comes from the last non-null read in the whole history.
    """
    history = case["history"]
    if not history:
        return [], [], [], None
    last_at = parse_time(history[-1]["at"])
    xs, ys, ws = [], [], []
    last_y = None
    last_x = None
    for read in history:
        rate = read["rates"][index]
        if rate is None:
            continue
        moment = parse_time(read["at"])
        y = logit(rate)
        last_y = y
        last_x = log_time(moment)
        age_h = (last_at - moment).total_seconds() / 3600.0
        if age_h <= FIT_WINDOW_HOURS:
            xs.append(log_time(moment))
            ys.append(y)
            ws.append(0.5 ** (age_h / HALF_LIFE_HOURS))
    if last_y is None:
        return [], [], [], None
    return xs, ys, ws, (last_x, last_y)


def refresh_stats(case):
    """RMS logit step between successive DISTINCT reads, and the median gap
    in hours between such reads.  A read that repeats the previous page
    exactly carries no refresh information."""
    history = case["history"]
    steps = []
    gaps = []
    prev_rates = None
    prev_at = None
    for read in history:
        rates = read["rates"]
        if prev_rates is not None and rates != prev_rates:
            moment = parse_time(read["at"])
            delta = 0.0
            count = 0
            for old, new in zip(prev_rates, rates):
                if old is None or new is None:
                    continue
                delta += (logit(new) - logit(old)) ** 2
                count += 1
            if count:
                steps.append(math.sqrt(delta / count))
                gaps.append((moment - prev_at).total_seconds() / 3600.0)
        prev_rates = rates
        prev_at = parse_time(read["at"])
    rms = math.sqrt(sum(s * s for s in steps) / len(steps)) if steps else 0.0
    gap = median(gaps) if gaps else 24.0
    return rms, max(1.0, gap)


def forecast_case(case):
    rows = []
    for med, spread, trophy in forecast_parts(case):
        qs = []
        prev = -1.0
        for z in Z:
            q = round4(inv_logit(med + z * spread))
            q = max(0.0, min(100.0, q))
            q = max(q, prev)  # keep the row non-decreasing after rounding
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def forecast_parts(case):
    """Per trophy: (median logit, spread in logit, trophy dict).

    Split out of forecast_case so tests can ablate the median against the
    spread: the median comes from the shrunk slope anchored on the last
    read, the spread from the random walk, the slope's uncertainty and the
    refresh jumps seen in the case's history.
    """
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    x_target = log_time(target)
    refresh_rms, refresh_gap_h = refresh_stats(case)
    # Expected refreshes between the last read and the target.
    if case["history"]:
        last_at = parse_time(case["history"][-1]["at"])
    else:
        last_at = target
    refreshes = max(0.0, (target - last_at).total_seconds() / 3600.0 / refresh_gap_h)
    refresh_var = (refresh_rms ** 2) * min(refreshes, REFRESH_CAP)

    # First pass: each trophy's own slope, then tier medians for shrinkage.
    # own[i] is (a, b, se_b, resid_var, last, n_eff); the no-fit entries
    # carry a = 0 and b = 0 with last set, or last None when never observed.
    own = []
    for i, trophy in enumerate(trophies):
        xs, ys, ws, last = trophy_series(case, i)
        if last is None:
            own.append((0.0, 0.0, 0.0, 0.0, None, 0.0))
            continue
        fit = weighted_line(xs, ys, ws)
        if fit is None:
            own.append((0.0, 0.0, 0.0, 0.0, last, 0.0))
        else:
            a, b, se_b, resid_var = fit
            n_eff = sum(ws)
            own.append((a, b, se_b, resid_var, last, n_eff))

    parts = []
    for i, trophy in enumerate(trophies):
        a, b, se_b, resid_var, last, n_eff = own[i]
        if last is None:
            # No read ever listed this trophy: near-zero rate, wide band.
            parts.append((logit(0.01), SPREAD_CAP, trophy))
            continue
        last_x, last_y = last
        # Leave-one-out prior: the trophy's own slope must not set its
        # own prior, or shrinkage does nothing -- platinum is a singleton
        # tier, so its prior would otherwise be its own slope exactly.
        others = [o[1] for j, o in enumerate(own)
                  if j != i and trophies[j]["tier"] == trophy["tier"]
                  and o[4] is not None and o[5] > 0]
        if others:
            prior = median(others)
        else:
            others_all = [o[1] for j, o in enumerate(own)
                          if j != i and o[4] is not None and o[5] > 0]
            prior = median(others_all) if others_all else 0.0
        w = n_eff / (n_eff + SHRINK_K) if n_eff > 0 else 0.0
        b_star = w * b + (1.0 - w) * prior
        dx = x_target - last_x
        move = max(-MAX_MOVE, min(MAX_MOVE, b_star * dx))
        move = move / (1.0 + MOVE_DAMP * abs(dx))
        line_level = a + b * last_x if n_eff > 0 else last_y
        if ANCHOR == "line":
            base = line_level
        elif ANCHOR == "blend":
            base = 0.5 * (last_y + line_level)
        else:
            base = last_y
        med = base + move
        med = max(logit(0.01), min(logit(99.99), med))
        slope_var = SLOPE_VAR_WEIGHT * (
            ((se_b * dx) ** 2) + (((b - b_star) * dx) ** 2))
        spread = math.sqrt(
            (RW_SIGMA * math.sqrt(horizon_days)) ** 2
            + slope_var
            + resid_var
            + refresh_var
        )
        spread = max(SPREAD_FLOOR, min(SPREAD_CAP, spread * SPREAD_SCALE))
        parts.append((med, spread, trophy))
    return parts


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
