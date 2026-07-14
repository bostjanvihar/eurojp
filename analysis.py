"""
Eurojackpot statistical analysis.

Reads data/eurojackpot.csv and produces, inside docs/ :
  charts/*.png                      all charts
  index.html                        self-contained dashboard (GitHub Pages)
  main_number_frequency.csv         frequency table, main numbers
  euro_number_frequency.csv         frequency table, euro numbers
  top_pairs.csv / top_triplets.csv  most common combinations
  eurojackpot.xlsx                  full data + all stat tables as a workbook

Run: python analysis.py
"""

from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import predict

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "eurojackpot.csv"
DOCS = ROOT / "docs"
CHARTS = DOCS / "charts"

MAIN_POOL = range(1, 51)
EURO_POOL = range(1, 13)
CURRENT_ERA = "2022-03-25"  # euro numbers extended to 1-12, Tuesday draws added
HOT_WINDOW = 52             # ~6 months of draws (2 per week)
TIMELINE_WINDOW = 50        # rolling window (in draws) for the timeline chart

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

ACCENT = "#c8a12c"   # gold
DARK = "#1f2937"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df["year"] = df.date.dt.year
    return df


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def frequency(df: pd.DataFrame, cols, pool) -> pd.DataFrame:
    counts = Counter()
    for c in cols:
        counts.update(df[c].tolist())
    out = pd.DataFrame({"number": list(pool)})
    out["count"] = out.number.map(counts).fillna(0).astype(int)
    out["percent_of_draws"] = (out["count"] / len(df) * 100).round(2)
    return out.sort_values("count", ascending=False).reset_index(drop=True)


def cooccurrence(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    matrix = np.zeros((50, 50), dtype=int)
    pair_counter, triplet_counter = Counter(), Counter()
    for row in df[["n1", "n2", "n3", "n4", "n5"]].itertuples(index=False):
        nums = sorted(row)
        for a, b in combinations(nums, 2):
            matrix[a - 1, b - 1] += 1
            matrix[b - 1, a - 1] += 1
            pair_counter[(a, b)] += 1
        for t in combinations(nums, 3):
            triplet_counter[t] += 1

    pairs = pd.DataFrame(
        [{"pair": f"{a} & {b}", "count": c} for (a, b), c in pair_counter.most_common(25)]
    )
    triplets = pd.DataFrame(
        [{"triplet": " & ".join(map(str, t)), "count": c}
         for t, c in triplet_counter.most_common(15)]
    )
    return matrix, pairs, triplets


def gap_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Draws since each main number last appeared + its historical average gap."""
    last_seen, gaps = {}, {n: [] for n in MAIN_POOL}
    for idx, row in enumerate(df[["n1", "n2", "n3", "n4", "n5"]].itertuples(index=False)):
        for n in row:
            if n in last_seen:
                gaps[n].append(idx - last_seen[n])
            last_seen[n] = idx
    total = len(df)
    rows = []
    for n in MAIN_POOL:
        avg = np.mean(gaps[n]) if gaps[n] else np.nan
        current = total - 1 - last_seen.get(n, -1)
        rows.append({
            "number": n,
            "draws_since_seen": current,
            "avg_gap": round(avg, 1) if gaps[n] else None,
            "overdue_ratio": round(current / avg, 2) if gaps[n] and avg else None,
        })
    return pd.DataFrame(rows).sort_values("draws_since_seen", ascending=False)


def hot_cold(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    recent = frequency(df.tail(HOT_WINDOW), ["n1", "n2", "n3", "n4", "n5"], MAIN_POOL)
    return recent.head(10), recent.tail(10).iloc[::-1].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def save(fig, name):
    fig.tight_layout()
    fig.savefig(CHARTS / name, dpi=130)
    plt.close(fig)


def chart_frequency(freq, title, name, highlight=10):
    freq = freq.sort_values("number")
    colors = [ACCENT if c >= freq["count"].nlargest(highlight).min() else "#9ca3af"
              for c in freq["count"]]
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.bar(freq.number, freq["count"], color=colors)
    ax.set_xticks(list(freq.number))
    ax.set_xlabel("Number")
    ax.set_ylabel("Times drawn")
    ax.set_title(title)
    save(fig, name)


def chart_timeline(df, freq_main):
    top = freq_main.head(6).number.tolist()
    window = TIMELINE_WINDOW
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for n in top:
        hit = df[["n1", "n2", "n3", "n4", "n5"]].eq(n).any(axis=1).astype(int)
        ax.plot(df.date, hit.rolling(window, min_periods=20).mean() * 100,
                label=f"#{n}", linewidth=1.6)
    ax.axhline(10, color="black", ls="--", lw=1, label="expected (10%)")
    ax.set_ylabel(f"% of last {window} draws containing the number")
    ax.set_title(f"Top {len(top)} main numbers - rolling {window}-draw hit rate over time")
    ax.legend(ncol=4, fontsize=9)
    save(fig, "timeline_top_numbers.png")


def chart_year_heatmap(df):
    years = sorted(df.year.unique())
    grid = np.zeros((50, len(years)))
    for j, y in enumerate(years):
        sub = df[df.year == y]
        counts = Counter()
        for c in ["n1", "n2", "n3", "n4", "n5"]:
            counts.update(sub[c].tolist())
        # normalise by number of draws that year so eras are comparable
        for n in MAIN_POOL:
            grid[n - 1, j] = counts.get(n, 0) / max(len(sub), 1) * 100
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(grid, aspect="auto", cmap="YlOrBr")
    ax.set_xticks(range(len(years)), years, rotation=45)
    ax.set_yticks(range(0, 50, 2), range(1, 51, 2))
    ax.set_ylabel("Main number")
    ax.set_title("How often each main number was drawn, per year (% of that year's draws)")
    fig.colorbar(im, label="% of draws")
    save(fig, "year_heatmap.png")


def chart_cooccurrence(matrix):
    fig, ax = plt.subplots(figsize=(10.5, 9.5))
    im = ax.imshow(matrix, cmap="YlOrBr")
    ticks = range(0, 50, 2)
    ax.set_xticks(ticks, range(1, 51, 2), fontsize=8)
    ax.set_yticks(ticks, range(1, 51, 2), fontsize=8)
    ax.set_title("Main-number co-occurrence: how often two numbers land in the same draw")
    fig.colorbar(im, label="Draws together")
    save(fig, "cooccurrence_heatmap.png")


def chart_sum_distribution(df):
    sums = df[["n1", "n2", "n3", "n4", "n5"]].sum(axis=1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(sums, bins=40, color=ACCENT, edgecolor="white")
    ax.axvline(sums.mean(), color=DARK, ls="--", label=f"mean = {sums.mean():.0f}")
    ax.set_xlabel("Sum of the five main numbers")
    ax.set_ylabel("Draws")
    ax.set_title("Distribution of the main-number sum")
    ax.legend()
    save(fig, "sum_distribution.png")


def chart_odd_even(df):
    odd = df[["n1", "n2", "n3", "n4", "n5"]].apply(lambda r: (r % 2).sum(), axis=1)
    counts = odd.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([f"{o} odd / {5-o} even" for o in counts.index], counts.values, color=ACCENT)
    ax.set_ylabel("Draws")
    ax.set_title("Odd/even split of the five main numbers")
    plt.setp(ax.get_xticklabels(), rotation=20)
    save(fig, "odd_even.png")


def chart_holt_forecast(probs, k, expected_pct, title, name):
    """Bar chart of Holt-predicted next-draw probability for every number."""
    top_cut = probs.nlargest(k).min()
    colors = ["#6366f1" if p >= top_cut else "#9ca3af" for p in probs]
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.bar(probs.index, probs * 100, color=colors)
    ax.axhline(expected_pct, color="black", ls="--", lw=1,
               label=f"uniform expectation ({expected_pct:.1f}%)")
    ax.set_xticks(list(probs.index))
    ax.set_xlabel("Number")
    ax.set_ylabel("Predicted probability (%)")
    ax.set_title(title)
    ax.legend()
    save(fig, name)


def chart_trend_continuation(df, levels_main, probs):
    """Holt level curves for the trend-picked numbers vs their raw rolling rate,
    with the one-step-ahead forecast marked - literally continuing fig 3."""
    fig, ax = plt.subplots(figsize=(13, 4.5))
    cmap = plt.get_cmap("tab10")
    tail = min(300, len(df))          # ~last 3 years; full history is in fig 3
    dates = df.date.iloc[-tail:]
    for i, (n, hist) in enumerate(levels_main.items()):
        c = cmap(i)
        raw = df[["n1", "n2", "n3", "n4", "n5"]].eq(n).any(axis=1).astype(float)
        ax.plot(dates, raw.rolling(TIMELINE_WINDOW, min_periods=10).mean().iloc[-tail:] * 100,
                color=c, alpha=0.25, lw=1)
        ax.plot(dates, hist[-tail:] * 100, color=c, lw=1.8, label=f"#{n}")
        ax.scatter([df.date.iloc[-1]], [probs[n] * 100], color=c, marker="D", zorder=5)
    ax.axhline(10, color="black", ls="--", lw=1, label="expected (10%)")
    ax.set_ylabel("Hit rate (%)")
    ax.set_title(f"Trend method: Holt-smoothed hit rate (bold) vs rolling "
                 f"{TIMELINE_WINDOW}-draw rate (faint); diamonds = next-draw forecast")
    ax.legend(ncol=4, fontsize=9)
    save(fig, "trend_continuation.png")


# --------------------------------------------------------------------------- #
# HTML dashboard
# --------------------------------------------------------------------------- #
def html_table(df, max_rows=None):
    if max_rows:
        df = df.head(max_rows)
    return df.to_html(index=False, border=0, classes="tbl")


def build_html(ctx: dict) -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Eurojackpot Tracker</title>
<style>
 body{{font-family:Georgia,serif;max-width:1080px;margin:0 auto;padding:24px;color:#1f2937;background:#fdfcf9}}
 h1{{border-bottom:3px solid {ACCENT};padding-bottom:8px}}
 h2{{margin-top:44px;color:#374151}}
 img{{max-width:100%;border:1px solid #e5e7eb;border-radius:6px;margin:8px 0}}
 .meta{{color:#6b7280;font-size:.9em}}
 .grid{{display:flex;gap:24px;flex-wrap:wrap}}
 .grid>div{{flex:1;min-width:300px}}
 table.tbl{{border-collapse:collapse;font-size:.88em;margin:10px 0}}
 table.tbl th{{background:{ACCENT};color:white;padding:5px 12px;text-align:left}}
 table.tbl td{{padding:4px 12px;border-bottom:1px solid #e5e7eb}}
 .warn{{background:#fef3c7;border-left:4px solid {ACCENT};padding:12px 16px;border-radius:4px}}
 .picks{{background:#eef2ff;border-left:4px solid #6366f1;padding:12px 16px;border-radius:4px}}
 code{{background:#f3f4f6;padding:1px 5px;border-radius:3px}}
</style></head><body>
<h1>Eurojackpot Tracker &amp; Statistics</h1>
<p class="meta">Last updated: {ctx['updated']} &nbsp;|&nbsp; Draws analysed: <b>{ctx['n_draws']}</b>
 ({ctx['first_date']} &rarr; {ctx['last_date']}) &nbsp;|&nbsp;
 <a href="eurojackpot.xlsx">Download spreadsheet (.xlsx)</a> &nbsp;|&nbsp;
 <a href="../data/eurojackpot.csv">Raw CSV</a></p>

<h2>Latest draw &mdash; {ctx['last_date']} ({ctx['last_weekday']})</h2>
<p style="font-size:1.5em"><b>{ctx['last_main']}</b> &nbsp;+&nbsp; Euro numbers <b>{ctx['last_euro']}</b></p>

<h2>1 &middot; Most frequently drawn main numbers (all time)</h2>
<img src="charts/main_frequency.png" alt="Main number frequency">
<div class="grid">
 <div><h3>Top 15</h3>{ctx['top_main']}</div>
 <div><h3>Bottom 10 (coldest all-time)</h3>{ctx['bottom_main']}</div>
</div>

<h2>2 &middot; Euro numbers</h2>
<img src="charts/euro_frequency.png" alt="Euro number frequency">
<p>Frequencies are shown for the current rules era (since 25&nbsp;March&nbsp;2022, pool 1&ndash;12).
All-time counts are in the spreadsheet, but note that numbers 11&ndash;12 only entered the pool in 2022
and 9&ndash;10 in 2014, so all-time euro comparisons are not apples-to-apples.</p>
{ctx['euro_table']}

<h2>3 &middot; Frequency over time</h2>
<img src="charts/timeline_top_numbers.png" alt="Timeline of top numbers">
<img src="charts/year_heatmap.png" alt="Per-year heatmap">

<h2>4 &middot; Numbers drawn together</h2>
<img src="charts/cooccurrence_heatmap.png" alt="Co-occurrence heatmap">
<div class="grid">
 <div><h3>Most common pairs</h3>{ctx['pairs']}</div>
 <div><h3>Most common triplets</h3>{ctx['triplets']}</div>
</div>

<h2>5 &middot; Pattern statistics</h2>
<div class="grid">
 <div><img src="charts/sum_distribution.png" alt="Sum distribution"></div>
 <div><img src="charts/odd_even.png" alt="Odd even split"></div>
</div>

<h2>6 &middot; Hot, cold and overdue</h2>
<div class="grid">
 <div><h3>Hot &mdash; last {ctx['hot_window']} draws</h3>{ctx['hot']}</div>
 <div><h3>Cold &mdash; last {ctx['hot_window']} draws</h3>{ctx['cold']}</div>
 <div><h3>Most overdue (vs own avg gap)</h3>{ctx['overdue']}</div>
</div>

<h2>7 &middot; Informed picks for the next draw</h2>
<div class="picks">
<p><b>Statistically "typical" ticket profile:</b> five numbers summing to roughly
{ctx['sum_low']}&ndash;{ctx['sum_high']}, with a 3/2 or 2/3 odd&ndash;even split
(that pattern covers ~{ctx['oe_pct']}% of all historical draws).</p>
<p><b>Frequency-leaning pick:</b> {ctx['pick_hot']} &nbsp;+&nbsp; euro {ctx['pick_hot_euro']}<br>
<b>Overdue-leaning pick:</b> {ctx['pick_due']} &nbsp;+&nbsp; euro {ctx['pick_due_euro']}<br>
<b>Balanced pick</b> (frequent + overdue + typical sum/odd-even profile): {ctx['pick_mix']} &nbsp;+&nbsp; euro {ctx['pick_mix_euro']}</p>
</div>

<h2>8 &middot; Three algorithmic predictions for the next draw</h2>
<p>All three are recomputed from the complete history after every draw.</p>
<div class="picks">
<p><b>1 &middot; Equalizer</b> &mdash; the ticket that would push the all-time number
distribution closest to perfectly even (minimises the variance of the 50 counts;
ties broken by longest absence):<br>
<span style="font-size:1.25em"><b>{ctx['pred_eq']}</b> + euro <b>{ctx['pred_eq_euro']}</b></span></p>
<p><b>2 &middot; Maintainer</b> &mdash; the ticket that would keep the current all-time
distribution shape most unchanged (minimises the shift of the normalised
frequency vector; ties broken by most recent appearance):<br>
<span style="font-size:1.25em"><b>{ctx['pred_mt']}</b> + euro <b>{ctx['pred_mt_euro']}</b></span></p>
<p><b>3 &middot; Trend continuation</b> &mdash; Holt double-exponential smoothing
(level + slope, half-life {ctx['half_life']} draws) of every number's hit rate,
extrapolated one draw ahead; the five highest forecast probabilities win:<br>
<span style="font-size:1.25em"><b>{ctx['pred_tr']}</b> + euro <b>{ctx['pred_tr_euro']}</b></span></p>
</div>
<img src="charts/holt_forecast_main.png" alt="Holt forecast, main numbers">
<img src="charts/holt_forecast_euro.png" alt="Holt forecast, euro numbers">
<img src="charts/trend_continuation.png" alt="Trend continuation">

<h3>Prediction scorecard</h3>
{ctx['scorecard']}
<p class="meta">Every prediction is stored before the draw and scored automatically once
the targeted draw is in. "avg_hits" is the average number of correct numbers per
ticket; pure chance expects <b>0.5</b> hits for main numbers (5 picks from 50) and
<b>0.333</b> for euro numbers (2 picks from 12). Watch this table to see whether any
method ever separates itself from luck.</p>

<div class="warn"><b>Honesty note:</b> Eurojackpot draws are independent random events.
No past pattern changes the odds of any future combination &mdash; every ticket has the same
1 in 139,838,160 jackpot chance. The picks above simply summarise historical tendencies;
their only practical edge is avoiding <i>popular human patterns</i> (dates, sequences),
which reduces the chance of <i>sharing</i> a prize, not of winning one.</div>

<p class="meta" style="margin-top:40px">Data source: euro-jackpot.net results archive.
Auto-updated by GitHub Actions after every Tuesday and Friday draw.</p>
</body></html>"""


# --------------------------------------------------------------------------- #
def informed_picks(freq_main, freq_euro, gaps, df):
    hot5 = sorted(freq_main.head(8).number.sample(5, random_state=len(df)).tolist())
    due5 = sorted(gaps.head(8).number.sample(5, random_state=len(df)).tolist())
    mix_pool = list(dict.fromkeys(
        freq_main.head(6).number.tolist() + gaps.head(6).number.tolist()))
    # pick a mix that lands in the central sum band with 2-3 odd numbers
    sums = df[["n1", "n2", "n3", "n4", "n5"]].sum(axis=1)
    lo, hi = int(sums.quantile(0.25)), int(sums.quantile(0.75))
    rng = np.random.default_rng(len(df))
    best = None
    for _ in range(400):
        cand = sorted(rng.choice(mix_pool, 5, replace=False).tolist())
        s, odd = sum(cand), sum(n % 2 for n in cand)
        if lo <= s <= hi and odd in (2, 3):
            best = cand
            break
    mix5 = best or sorted(mix_pool[:5])
    hot_e = sorted(freq_euro.head(2).number.tolist())
    due_e = sorted(freq_euro.tail(2).number.tolist())
    mix_e = sorted([freq_euro.head(1).number.iloc[0], freq_euro.tail(1).number.iloc[0]])
    return dict(pick_hot=hot5, pick_due=due5, pick_mix=mix5,
                pick_hot_euro=hot_e, pick_due_euro=due_e, pick_mix_euro=mix_e,
                sum_low=lo, sum_high=hi)


def main():
    CHARTS.mkdir(parents=True, exist_ok=True)
    df = load()
    era = df[df.date >= CURRENT_ERA]

    freq_main = frequency(df, ["n1", "n2", "n3", "n4", "n5"], MAIN_POOL)
    freq_euro_era = frequency(era, ["e1", "e2"], EURO_POOL)
    freq_euro_all = frequency(df, ["e1", "e2"], EURO_POOL)
    matrix, pairs, triplets = cooccurrence(df)
    gaps = gap_analysis(df)
    hot, cold = hot_cold(df)

    # charts
    chart_frequency(freq_main, "Main numbers 1-50 - times drawn (all draws since 2012)",
                    "main_frequency.png")
    chart_frequency(freq_euro_era,
                    f"Euro numbers 1-12 - times drawn since {CURRENT_ERA} (current rules)",
                    "euro_frequency.png", highlight=4)
    chart_timeline(df, freq_main)
    chart_year_heatmap(df)
    chart_cooccurrence(matrix)
    chart_sum_distribution(df)
    chart_odd_even(df)

    # tables to CSV
    freq_main.to_csv(DOCS / "main_number_frequency.csv", index=False)
    freq_euro_all.to_csv(DOCS / "euro_number_frequency.csv", index=False)
    pairs.to_csv(DOCS / "top_pairs.csv", index=False)
    triplets.to_csv(DOCS / "top_triplets.csv", index=False)

    # spreadsheet
    with pd.ExcelWriter(DOCS / "eurojackpot.xlsx", engine="openpyxl") as xl:
        df.drop(columns="year").to_excel(xl, sheet_name="Draws", index=False)
        freq_main.to_excel(xl, sheet_name="Main frequency", index=False)
        freq_euro_all.to_excel(xl, sheet_name="Euro freq (all time)", index=False)
        freq_euro_era.to_excel(xl, sheet_name="Euro freq (since 2022)", index=False)
        pairs.to_excel(xl, sheet_name="Top pairs", index=False)
        triplets.to_excel(xl, sheet_name="Top triplets", index=False)
        gaps.to_excel(xl, sheet_name="Gap analysis", index=False)

    odd = df[["n1", "n2", "n3", "n4", "n5"]].apply(lambda r: (r % 2).sum(), axis=1)
    oe_pct = round(odd.isin([2, 3]).mean() * 100)

    # ---- algorithmic predictions + scorecard ----
    pred = predict.run(df, era)
    picks3 = pred["picks"]
    chart_holt_forecast(pred["probs_main"], 5, 10.0,
                        "Trend method - predicted probability of each main number "
                        "appearing in the NEXT draw (Holt forecast)",
                        "holt_forecast_main.png")
    chart_holt_forecast(pred["probs_euro"], 2, 100 * 2 / 12,
                        "Trend method - predicted probability of each euro number "
                        "appearing in the NEXT draw (current era)",
                        "holt_forecast_euro.png")
    chart_trend_continuation(df, pred["levels_main"], pred["probs_main"])
    sc = pred["scorecard"]
    scorecard_html = (html_table(sc) if not sc.empty else
                      "<p><i>No scored predictions yet - the first scores appear "
                      "after the next draw.</i></p>")

    last = df.iloc[-1]
    picks = informed_picks(freq_main, freq_euro_era, gaps, df)
    ctx = dict(
        updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        n_draws=len(df),
        first_date=df.date.min().date(), last_date=df.date.max().date(),
        last_weekday=last.weekday,
        last_main=" - ".join(str(last[c]) for c in ["n1", "n2", "n3", "n4", "n5"]),
        last_euro=f"{last.e1} & {last.e2}",
        top_main=html_table(freq_main, 15),
        bottom_main=html_table(freq_main.tail(10).iloc[::-1]),
        euro_table=html_table(freq_euro_era),
        pairs=html_table(pairs, 20),
        triplets=html_table(triplets, 10),
        hot=html_table(hot), cold=html_table(cold),
        overdue=html_table(gaps.head(10)),
        hot_window=HOT_WINDOW,
        oe_pct=oe_pct,
        pred_eq=" - ".join(map(str, picks3[("equalize", "main")])),
        pred_eq_euro=" & ".join(map(str, picks3[("equalize", "euro")])),
        pred_mt=" - ".join(map(str, picks3[("maintain", "main")])),
        pred_mt_euro=" & ".join(map(str, picks3[("maintain", "euro")])),
        pred_tr=" - ".join(map(str, picks3[("trend", "main")])),
        pred_tr_euro=" & ".join(map(str, picks3[("trend", "euro")])),
        half_life=predict.HALF_LIFE,
        scorecard=scorecard_html,
        **picks,
    )
    (DOCS / "index.html").write_text(build_html(ctx), encoding="utf-8")
    print(f"Analysis complete: {len(df)} draws -> docs/index.html")


if __name__ == "__main__":
    main()
