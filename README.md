[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Hendricontrex/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Hendricontrex/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast

Forecast, for every one of Grand Theft Auto VI's 55 PlayStation 5
trophies, the share of the players Exophase tracks who will have earned
it at 2026-11-26 00:00 UTC — a week after launch — as five quantiles
(5%, 25%, 50%, 75%, 95%) per trophy. Entry for the
[make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
of the same name.

## The program

`main.py` (Python 3.13, standard library only, no network) reads every
`/input/<id>.in` case and writes `/output/<id>.out`. It runs in about
0.05 s per case, deterministically — two runs on the same case write
byte-identical output.

## The model

Earn rates are cumulative shares, so the work happens on the logit
scale, against log-time since launch (2026-11-19 00:00 UTC):

- **Median:** per trophy, a recency-weighted line of logit(rate) against
  ln(1 + hours since launch) over the last 24 h of reads (24 h half-life
  weights), anchored on the last observed rate and extrapolated to the
  target. The slope is shrunk toward the leave-one-out tier-median slope
  (shrinkage weight n_eff / (n_eff + 8)) — trophies with thin or noisy
  reads borrow strength from their tier, and a trophy never sets its own
  prior (platinum is a singleton tier, so this matters). The logit move
  is capped at ±3.
- **Spread:** the square root of a random-walk term
  (0.5 · √horizon in days) plus the slope's standard error, the
  shrinkage gap, and the fit residual — then scaled by 0.65 and floored
  at 0.5. The scale was calibrated on the holdout (below): the house's
  random-walk spread is far wider than the errors a good median actually
  makes, and a narrower band wins in every synthetic world tested.
- **Deliberately not included:** a refresh-jump allowance. Exophase
  refreshes the page in discrete jumps (launch night: frozen for hours,
  then 638 → 3,868 players tracked in one jump), but those jumps are the
  page catching up with movement the slope already extrapolates, not
  independent shocks — every positive allowance cost loss on the holdout,
  so it is off.
- **Edges:** a trophy never observed gets a near-zero median with the
  widest band; a trophy at ~100% stays pinned near 100%; rows are
  clamped to [0, 100] and forced non-decreasing after rounding to four
  decimals.

## Why it should beat the baselines

The flat baseline (rates stay where they are) loses the moment any
trophy moves — and launch-week trophies move by logit units a day. The
trend baseline extrapolates each trophy's recent curve with a
random-walk spread, but it has no shrinkage (thin trophies chase noise)
and its spread is set by a 0.5-daily-vol random walk that is far wider
than the residual error of a fitted trend. This entry keeps the trend's
median where the data is rich, borrows the tier's slope where it is
thin, and carries a spread calibrated to the errors a good median
actually makes.

## Holdout results

`tests/holdout.py` builds synthetic final-like cases — reads every
10 minutes from launch night to the deadline, a page frozen between
refreshes every 4–8 h, tracked players growing to a ceiling, each
trophy's earners logistic in log-time and calibrated so the simulated
page at 20 Nov 13:25 UTC reproduces the live Exophase rates — and
scores this entry against the flat and trend baselines with the
contest's own pinball loss. Three worlds, 5 scenarios × 3 seeds = 15
cases each (45 total):

| world | this entry | flat | trend |
| --- | --- | --- | --- |
| smooth (truth stays log-linear) | **0.135** | 0.212 | 0.182 |
| noisy (weekend surge, per-trophy shocks, logit random walk) | **0.199** | 0.276 | 0.228 |
| very-noisy (shocks doubled) | **0.278** | 0.356 | 0.288 |
| mean across worlds | **0.204** | 0.281 | 0.233 |

The entry beats the trend baseline in 44 of the 45 cases and the flat
baseline in all 45. The one loss is a very-noisy fast-progression case
(0.368 vs trend 0.355), where an unforecastable per-trophy shock
dominates everything.

## Verification

- Practice cases `p00` / `p01` (kit): two runs byte-identical,
  55 rows, monotone, in range, 2,693 bytes each, ~0.05 s per run.
- Edge cases (synthetic, kit format): all-null history, single read,
  empty history, flat reads with one jump, 0.01% / 99.99% trophies —
  all produce valid 55-row output. The jump case caught a real bug
  (singleton-tier prior = own slope), fixed by leave-one-out shrinkage.
- `python3 tests/holdout.py` reproduces the table above.

## Files

- `main.py` — the entry program
- `tests/holdout.py` — synthetic holdout harness + both baselines
  (re-implemented from the kit's RULES.txt)
- `LICENSE` — MIT, from the kit starter
