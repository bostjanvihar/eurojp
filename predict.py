"""
Predictive approaches for the next Eurojackpot draw + a self-scoring ledger.

Three deterministic methods, recomputed from the FULL history on every run:

  1. equalize  - the ticket that makes the all-time count distribution as even
                 as possible after the next draw (minimises the variance of
                 the 50 counts). Provably: the least-drawn numbers; ties are
                 broken by the longest time since last seen. Implemented via
                 the explicit variance-delta objective so it stays transparent.

  2. maintain  - the ticket that keeps the all-time *shape* (normalised
                 frequency distribution) as unchanged as possible after the
                 next draw (minimises squared distance between the before and
                 after share vectors). Provably: the most-drawn numbers; ties
                 broken by the shortest time since last seen.

  3. trend     - Holt double exponential smoothing (level + slope) on each
                 number's per-draw hit indicator; forecasts every number's
                 next-draw probability one step ahead, i.e. it "continues the
                 lines" of the timeline chart for all numbers at once. The
                 memory is set by HALF_LIFE below.

A scorecard keeps every prediction in data/predictions.csv; as soon as the
targeted draw is scraped, each ticket is scored (number of hits) so the
dashboard can compare methods against pure chance over time.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------ configuration ------------------------------ #
HALF_LIFE = 20        # draws; weight of a draw halves every HALF_LIFE draws.
                      # This replaces a hard rolling window (a raw 15-draw
                      # window would be far too noisy for 10%-probability
                      # events); 20 keeps the trend responsive but stable.
TREND_HALF_LIFE = 40  # slower half-life for the slope component (Holt beta)
PROB_FLOOR, PROB_CEIL = 0.005, 0.60   # clip Holt forecasts to sane bounds
# --------------------------------------------------------------------------- #

DATA_DIR = Path(__file__).parent / "data"
LEDGER = DATA_DIR / "predictions.csv"
LEDGER_COLS = ["made_after_draw", "method", "pool", "numbers", "target_date", "hits"]

MAIN_COLS = ["n1", "n2", "n3", "n4", "n5"]
EURO_COLS = ["e1", "e2"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _counts_and_gaps(df: pd.DataFrame, cols, pool):
    """All-time counts and draws-since-last-seen for every number in pool."""
    counts = {n: 0 for n in pool}
    last_seen = {n: -1 for n in pool}
    for idx, row in enumerate(df[cols].itertuples(index=False)):
        for n in row:
            counts[n] += 1
            last_seen[n] = idx
    total = len(df)
    gaps = {n: total - 1 - last_seen[n] for n in pool}
    return counts, gaps


# --------------------------------------------------------------------------- #
# method 1 - equalize the all-time distribution
# --------------------------------------------------------------------------- #
def pick_equalize(df, cols, pool, k):
    """
    Choose k numbers minimising the variance of the count vector after the
    next draw. With fixed total, variance is minimised by minimising
    sum((c_i + x_i)^2); adding a hit to number i raises that sum by
    delta_i = 2*c_i + 1, so we take the k smallest deltas (= smallest counts).
    """
    counts, gaps = _counts_and_gaps(df, cols, pool)
    delta = {n: 2 * counts[n] + 1 for n in pool}
    # smallest delta first; tie-break: longest unseen first
    order = sorted(pool, key=lambda n: (delta[n], -gaps[n], n))
    return sorted(order[:k])


# --------------------------------------------------------------------------- #
# method 2 - maintain the all-time distribution
# --------------------------------------------------------------------------- #
def pick_maintain(df, cols, pool, k):
    """
    Choose k numbers minimising the squared distance between the normalised
    frequency vectors before and after the next draw. The change term for
    number i works out to (T*x_i - k*c_i)^2, so including i costs
    delta_i = T^2 - 2*k*T*c_i (up to constants): minimised by the largest
    counts. Tie-break: most recently seen first.
    """
    counts, gaps = _counts_and_gaps(df, cols, pool)
    T = max(sum(counts.values()) // k, 1)  # number of draws
    delta = {n: T * T - 2 * k * T * counts[n] for n in pool}
    order = sorted(pool, key=lambda n: (delta[n], gaps[n], n))
    return sorted(order[:k])


# --------------------------------------------------------------------------- #
# method 3 - Holt double exponential smoothing per number
# --------------------------------------------------------------------------- #
def holt_forecast(df, cols, pool, k):
    """
    For every number: series y_t = 1 if the number appeared in draw t else 0.
    Holt smoothing tracks a level and a slope; the one-step-ahead forecast
    level+slope is the number's predicted probability for the NEXT draw.
    Probabilities are clipped and normalised so they sum to k (a draw
    contains exactly k numbers). Returns (top-k pick, prob Series, level history
    of the top-k numbers for charting).
    """
    alpha = 1 - 0.5 ** (1 / HALF_LIFE)
    beta = 1 - 0.5 ** (1 / TREND_HALF_LIFE)
    base = k / len(pool)

    forecasts, levels_hist = {}, {}
    for n in pool:
        y = df[cols].eq(n).any(axis=1).astype(float).to_numpy()
        level, slope = base, 0.0
        hist = np.empty(len(y))
        for t, obs in enumerate(y):
            new_level = alpha * obs + (1 - alpha) * (level + slope)
            slope = beta * (new_level - level) + (1 - beta) * slope
            level = new_level
            hist[t] = level
        forecasts[n] = level + slope
        levels_hist[n] = hist

    probs = pd.Series(forecasts).clip(PROB_FLOOR, PROB_CEIL)
    probs = probs / probs.sum() * k          # normalise: probabilities sum to k
    top = sorted(probs.nlargest(k).index.tolist())
    return top, probs.sort_index(), {n: levels_hist[n] for n in top}


# --------------------------------------------------------------------------- #
# prediction ledger / scorecard
# --------------------------------------------------------------------------- #
def _load_ledger() -> pd.DataFrame:
    if LEDGER.exists():
        return pd.read_csv(LEDGER, dtype=str)
    return pd.DataFrame(columns=LEDGER_COLS)


def update_ledger(df: pd.DataFrame, picks: dict) -> pd.DataFrame:
    """
    1. Score every pending prediction whose targeted draw has now happened
       (the first draw AFTER the draw the prediction was made after).
    2. Record fresh predictions for the upcoming draw (idempotent: reruns
       before a new draw simply overwrite the pending rows).
    """
    ledger = _load_ledger()
    dates = df["date"].dt.strftime("%Y-%m-%d").tolist()
    latest = dates[-1]

    # ---- score pending rows ------------------------------------------------
    for i, row in ledger[ledger["target_date"].isna() | (ledger["target_date"] == "")].iterrows():
        later = [d for d in dates if d > row["made_after_draw"]]
        if not later:
            continue
        target = later[0]
        drawn_row = df[df["date"] == target].iloc[0]
        cols = MAIN_COLS if row["pool"] == "main" else EURO_COLS
        drawn = {int(drawn_row[c]) for c in cols}
        predicted = {int(x) for x in str(row["numbers"]).split()}
        ledger.loc[i, "target_date"] = target
        ledger.loc[i, "hits"] = str(len(drawn & predicted))

    # ---- write fresh pending predictions -----------------------------------
    pending_mask = (ledger["made_after_draw"] == latest) & (
        ledger["target_date"].isna() | (ledger["target_date"] == "")
    )
    ledger = ledger[~pending_mask]
    new_rows = [
        {"made_after_draw": latest, "method": method, "pool": pool,
         "numbers": " ".join(map(str, nums)), "target_date": "", "hits": ""}
        for (method, pool), nums in picks.items()
    ]
    ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)

    DATA_DIR.mkdir(exist_ok=True)
    ledger.to_csv(LEDGER, index=False)
    return ledger


def scorecard(ledger: pd.DataFrame) -> pd.DataFrame:
    """Average hits per method vs the chance expectation."""
    scored = ledger[ledger["hits"] != ""].dropna(subset=["hits"]).copy()
    if scored.empty:
        return pd.DataFrame()
    scored["hits"] = scored["hits"].astype(int)
    out = (
        scored.groupby(["method", "pool"])
        .agg(predictions=("hits", "size"), total_hits=("hits", "sum"),
             avg_hits=("hits", "mean"))
        .reset_index()
    )
    out["avg_hits"] = out["avg_hits"].round(3)
    out["chance_expectation"] = np.where(out["pool"] == "main", 5 * 5 / 50, 2 * 2 / 12).round(3)
    return out.sort_values(["pool", "avg_hits"], ascending=[True, False])


# --------------------------------------------------------------------------- #
# entry point used by analysis.py
# --------------------------------------------------------------------------- #
def run(df: pd.DataFrame, era: pd.DataFrame) -> dict:
    """
    df  : full history (main-number methods)
    era : current-rules subset (euro-number methods, pool 1-12)
    Returns everything the dashboard needs.
    """
    main_pool, euro_pool = range(1, 51), range(1, 13)

    picks = {
        ("equalize", "main"): pick_equalize(df, MAIN_COLS, main_pool, 5),
        ("maintain", "main"): pick_maintain(df, MAIN_COLS, main_pool, 5),
        ("equalize", "euro"): pick_equalize(era, EURO_COLS, euro_pool, 2),
        ("maintain", "euro"): pick_maintain(era, EURO_COLS, euro_pool, 2),
    }
    trend_main, probs_main, levels_main = holt_forecast(df, MAIN_COLS, main_pool, 5)
    trend_euro, probs_euro, _ = holt_forecast(era, EURO_COLS, euro_pool, 2)
    picks[("trend", "main")] = trend_main
    picks[("trend", "euro")] = trend_euro

    ledger = update_ledger(df, picks)
    return {
        "picks": picks,
        "probs_main": probs_main,
        "probs_euro": probs_euro,
        "levels_main": levels_main,
        "scorecard": scorecard(ledger),
    }
