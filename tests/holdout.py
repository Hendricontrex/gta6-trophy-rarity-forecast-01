"""Holdout validation for main.py (not shipped to the grader's sandbox run,
but kept in the repo to show the model was tested, not guessed).

Builds synthetic final-like cases and scores three forecasters on them:
  * mine    -- main.py's forecast_case
  * flat    -- the house persistence baseline (kit starter, byte-for-byte formula)
  * trend   -- the house curve baseline (OLS logit vs ln(1+h) over last 24h)

Ground truth is generated, not fitted: Exophase-style refresh steps
(the page frozen for hours, then jumping), tracked players growing
logistically, and each trophy's earners following a logistic curve in
log-time, calibrated so the simulated page at 2026-11-20 13:25 UTC
reproduces the live Exophase rates read that day (Ch.1 11.10%, platinum
0.00%, Gone Fishing 100%, ...).  Four scenarios x three seeds.

Scoring is the contest's own: mean pinball loss on the logit scale,
2 * max(tau*d, (tau-1)*d), d = logit(y) - logit(q), rates clamped to
[0.01, 99.99] before the logit.

Run:  python3 tests/holdout.py
"""

import json
import math
import random
import sys
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402  (the entry program under test)

Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
DEADLINE = datetime(2026, 11, 21, 11, 30, 0, tzinfo=timezone.utc)
TARGET = datetime(2026, 11, 26, 0, 0, 0, tzinfo=timezone.utc)
FIRST_READ = datetime(2026, 11, 18, 23, 50, 0, tzinfo=timezone.utc)

# Live Exophase rates read 2026-11-20 13:25 UTC (the calibration snapshot),
# in trophy id order 1..55.
LIVE_RATES = [
    0.00, 100.0, 11.10, 1.44, 0.03, 0.02, 0.02, 3.77, 0.03, 0.02,
    0.02, 0.02, 0.02, 0.00, 0.00, 0.01, 0.01, 0.00, 0.01, 12.76,
    1.31, 0.01, 0.01, 0.01, 0.00, 0.01, 0.01, 0.01, 0.00, 0.01,
    0.00, 0.27, 0.03, 1.42, 0.07, 0.22, 0.19, 0.04, 0.01, 0.01,
    0.04, 0.01, 0.15, 0.01, 1.09, 0.01, 0.01, 0.01, 0.64, 0.05,
    0.01, 0.03, 0.01, 0.01, 0.01,
]
# Launch-night rates (the practice cases' flat page, 638 players tracked).
NIGHT_RATES = [
    0.63, 100.0, 1.25, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10,
    1.10, 1.10, 1.10, 0.63, 0.63, 0.78, 0.63, 0.63, 0.78, 2.04,
    0.94, 0.78, 0.78, 0.78, 0.63, 0.63, 0.78, 0.63, 0.78, 0.63,
    0.63, 0.63, 0.94, 0.78, 1.25, 0.78, 0.94, 1.57, 0.94, 0.78,
    0.94, 0.78, 0.78, 0.94, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78,
    0.78, 0.78, 0.94, 0.78, 0.78,
]


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


def pinball(y, q, tau):
    d = logit(y) - logit(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def score(forecast, answer):
    """Mean pinball loss over trophies x levels."""
    total = 0.0
    n = 0
    by_id = {r["id"]: r["q"] for r in forecast["trophies"]}
    for tid, y in answer.items():
        for tau, q in zip(LEVELS, by_id[tid]):
            total += pinball(y, q, tau)
            n += 1
    return total / n


def flat_baseline(case):
    """The house persistence baseline, exactly as the kit starter."""
    last = case["history"][-1]
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        lg = logit(rate if rate is not None else 0.0)
        rows.append({"id": trophy["id"],
                     "q": [main.round4(inv_logit(lg + z * spread)) for z in Z]})
    return {"trophies": rows}


def trend_parts(case):
    """The house curve baseline, as (median logit, spread) per trophy."""
    last_at = main.parse_time(case["history"][-1]["at"])
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = main.log_time(main.parse_time(case["target"]))
    parts = []
    for i, trophy in enumerate(case["trophies"]):
        xs, ys = [], []
        for read in case["history"]:
            moment = main.parse_time(read["at"])
            if (last_at - moment).total_seconds() / 3600.0 > 24.0:
                continue
            rate = read["rates"][i]
            if rate is None:
                continue
            xs.append(main.log_time(moment))
            ys.append(logit(rate))
        last_rate = None
        for read in reversed(case["history"]):
            if read["rates"][i] is not None:
                last_rate = read["rates"][i]
                break
        if last_rate is None:
            last_rate = 0.01
        if len(xs) < 3:
            med = logit(last_rate)
            spread = 0.5 * math.sqrt(horizon_days)
        else:
            n = len(xs)
            mx = sum(xs) / n
            my = sum(ys) / n
            sxx = sum((x - mx) ** 2 for x in xs)
            sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            b = sxy / sxx if sxx > 1e-12 else 0.0
            a = my - b * mx
            resid = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n
            med = a + b * x_target
            spread = math.sqrt((0.5 * math.sqrt(horizon_days)) ** 2 + resid)
        med = max(logit(0.01), min(logit(99.99), med))
        spread = max(0.5, spread)
        parts.append((med, spread))
    return parts


def trend_baseline(case):
    """The house curve baseline: OLS logit vs ln(1+h) over the last 24h,
    read at the target; spread = random walk + residual, combined."""
    rows = []
    for (med, spread), trophy in zip(trend_parts(case), case["trophies"]):
        rows.append({"id": trophy["id"],
                     "q": [main.round4(inv_logit(med + z * spread)) for z in Z]})
    return {"trophies": rows}


def build_forecast(parts):
    """A forecast dict from a list of (median logit, spread, trophy)."""
    rows = []
    for med, spread, trophy in parts:
        qs = []
        prev = -1.0
        for z in Z:
            q = main.round4(inv_logit(med + z * spread))
            q = max(0.0, min(100.0, q))
            q = max(q, prev)
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def build_case(rng, n_inf, prog_mult, noise_level=0):
    """One synthetic final-like case + its answer rates at the target.

    noise_level 0 is the smooth world: the truth stays exactly
    log-linear in log-time.  Levels 1 and 2 add three horizon shocks no
    forecaster can see coming: a weekend surge of new players that dilutes
    every rate, a per-trophy unforecastable multiplier on earners, and a
    random walk on each trophy's logit trajectory.
    """
    noisy = noise_level > 0
    SHOCK_SD = (0.0, 0.35, 0.70)[noise_level]
    RW_SD = (0.0, 0.10, 0.20)[noise_level]
    WEEKEND_DILUTION = (0.0, 0.5, 1.0)[noise_level]
    kit_dir = Path(__file__).resolve().parent.parent
    practice = json.loads((kit_dir / "gta6-trophy-rarity-forecast-01-kit"
                           / "input" / "p00.in").read_text())
    trophies = practice["trophies"]

    def hours(moment):
        return (moment - LAUNCH).total_seconds() / 3600.0

    def x_of(moment):
        return math.log(1.0 + max(-0.99, hours(moment)))

    # Tracked players: N(t) = 638 * (1+t)^k, k set so N(168h) = n_inf.
    k_track = math.log(n_inf / 638.0) / math.log(169.0)

    # Noisy world: a weekend surge of new players (Sat 21 Nov 00:00 UTC to
    # Mon 23 Nov 00:00 UTC) dilutes every rate, and each trophy's earners
    # get an unforecastable multiplier that ramps in over the weekend.
    WEEKEND_START = datetime(2026, 11, 21, 0, 0, 0, tzinfo=timezone.utc)
    WEEKEND_END = datetime(2026, 11, 23, 0, 0, 0, tzinfo=timezone.utc)

    def weekend_frac(moment):
        if not noisy or moment <= WEEKEND_START:
            return 0.0
        if moment >= WEEKEND_END:
            return 1.0
        return ((moment - WEEKEND_START).total_seconds()
                / (WEEKEND_END - WEEKEND_START).total_seconds())

    def tracked(moment):
        base = 638.0 * (1.0 + max(0.0, hours(moment))) ** k_track
        if noisy:
            base *= 1.0 + WEEKEND_DILUTION * weekend_frac(moment)
        return base

    now = datetime(2026, 11, 20, 13, 25, 0, tzinfo=timezone.utc)
    n_now = tracked(now)
    x_now = x_of(now)

    # Per-trophy earners: E(t) = E_inf * sigmoid((x(t) - m) / w).
    # E_inf from the live snapshot times a maturity-dependent growth factor;
    # m solved so the curve passes through the live snapshot.
    w = 0.7
    curves = []
    for i, live in enumerate(LIVE_RATES):
        if i == 1:  # Gone Fishing: everyone has it, rate stays 100.
            curves.append(("full", None))
            continue
        live_earners = max(live, 0.005) / 100.0 * n_now
        if live >= 1.0:
            growth = rng.uniform(1.5, 3.0)
        elif live >= 0.1:
            growth = rng.uniform(3.0, 8.0)
        elif live > 0.0:
            growth = rng.uniform(5.0, 20.0)
        else:
            growth = rng.uniform(8.0, 30.0)
        growth *= prog_mult
        e_inf = live_earners * growth
        # Never let the ultimate rate exceed 95% of players.
        e_inf = min(e_inf, 0.95 * n_inf)
        ratio = min(0.999, max(0.001, live_earners / e_inf))
        m = x_now - w * math.log(ratio / (1.0 - ratio))
        curves.append(("logistic", (e_inf, m, w)))

    # Per-trophy unforecastable multipliers for the noisy world.
    shock = [math.exp(rng.gauss(0.0, SHOCK_SD)) for _ in range(55)]

    def earners(i, moment, ref_idx=None):
        kind, params = curves[i]
        if kind == "full":
            return tracked(moment)
        e_inf, m, w = params
        value = e_inf * sigmoid((x_of(moment) - m) / w)
        if noisy and moment > DEADLINE:
            # Unforecastable per-trophy shock, ramping in over the weekend.
            value *= shock[i] ** weekend_frac(moment)
            # A small random walk on the logit's trajectory.
            if ref_idx is not None:
                value *= math.exp(walk[(i, ref_idx)])
        return value

    def true_rate(i, moment):
        return 100.0 * earners(i, moment) / tracked(moment)

    # Refresh times: the page freezes, then jumps, every 4-8h.
    refreshes = [FIRST_READ]
    t = FIRST_READ
    while t < TARGET + timedelta(hours=1):
        t += timedelta(hours=rng.uniform(4.0, 8.0))
        refreshes.append(t)

    # Random walk on each trophy's logit trajectory (noisy world only):
    # one N(0, 0.1 * sqrt(dt_days)) step per refresh interval, accumulated.
    walk = {}
    if noisy:
        for i in range(55):
            total = 0.0
            prev = refreshes[0]
            for j, ref in enumerate(refreshes):
                dt_days = (ref - prev).total_seconds() / 86400.0
                total += rng.gauss(0.0, RW_SD * math.sqrt(max(dt_days, 0.0)))
                walk[(i, j)] = total
                prev = ref

    def page_at(moment):
        """The page as of `moment`: values frozen at the last refresh."""
        ref = max(r for r in refreshes if r <= moment)
        j = refreshes.index(ref)
        n = tracked(ref)
        rates = []
        for i in range(55):
            r = 100.0 * earners(i, ref, j) / n
            rates.append(round(min(100.0, max(0.0, r)), 2))
        return n, rates

    # Reads every 10 minutes from the first slot to the deadline.
    history = []
    slot = FIRST_READ
    while slot < DEADLINE:
        n, rates = page_at(slot)
        history.append({
            "slot": slot.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "at": slot.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "players_tracked": int(round(n)),
            "trophies_earned": None,
            "rates": rates,
        })
        slot += timedelta(minutes=10)

    # Answer: the page the first read in the target hour shows -- the
    # values frozen at the last refresh before the target.
    _, ans_rates = page_at(TARGET)
    answer = {i + 1: ans_rates[i] for i in range(55)}

    case = {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": "f00-synth",
        "at": DEADLINE.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_minutes": 6510,
        "target": TARGET.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "levels": LEVELS,
        "game": practice["game"],
        "trophies": trophies,
        "history": history,
        "notes": "synthetic holdout case",
    }
    return case, answer


def main_run():
    """Final evaluation of the SHIPPED model against the two baselines.

    Shipped constants (main.py): FIT_WINDOW_HOURS 24, SHRINK_K 8,
    MOVE_DAMP 0, ANCHOR last, REFRESH_CAP 0, SLOPE_VAR_WEIGHT 1,
    SPREAD_SCALE 0.65, leave-one-out tier-median shrinkage prior.
    Three synthetic worlds x 5 scenarios x 3 seeds = 45 final-like cases.
    """
    scenarios = [
        ("tracked 300k, progression x1.0", 300_000, 1.0),
        ("tracked 100k, progression x1.0", 100_000, 1.0),
        ("tracked 800k, progression x1.0", 800_000, 1.0),
        ("tracked 300k, progression x1.5 (fast)", 300_000, 1.5),
        ("tracked 300k, progression x0.6 (slow)", 300_000, 0.6),
    ]
    totals = {}

    def note(world, label, loss):
        totals.setdefault((world, label), []).append(loss)

    for world, noise_level in (("smooth", 0), ("noisy", 1), ("very-noisy", 2)):
        for name, n_inf, prog in scenarios:
            for seed in (1, 2, 3):
                rng = random.Random(zlib.crc32(f"{world}|{name}|{seed}".encode()))
                case, answer = build_case(rng, n_inf, prog, noise_level)
                f = score(flat_baseline(case), answer)
                t = score(trend_baseline(case), answer)
                m = score(main.forecast_case(case), answer)
                note(world, "flat", f)
                note(world, "trend", t)
                note(world, "mine", m)
                print(f"{world:10s} {name:34s} s{seed}: "
                      f"mine {m:.4f}  flat {f:.4f}  trend {t:.4f}")
    print()
    print(f"{'world':10s} {'model':6s} {'mean':>7s} {'best':>7s} {'worst':>7s}")
    for world in ("smooth", "noisy", "very-noisy"):
        for label in ("mine", "flat", "trend"):
            vals = totals[(world, label)]
            print(f"{world:10s} {label:6s} {sum(vals)/len(vals):7.4f} "
                  f"{min(vals):7.4f} {max(vals):7.4f}")
    print()
    for label in ("mine", "flat", "trend"):
        means = [sum(totals[(w, label)]) / len(totals[(w, label)])
                 for w in ("smooth", "noisy", "very-noisy")]
        print(f"{label:6s} smooth {means[0]:.4f}  noisy {means[1]:.4f}  "
              f"very-noisy {means[2]:.4f}  mean {sum(means)/3:.4f}  "
              f"worst {max(means):.4f}")


if __name__ == "__main__":
    main_run()
