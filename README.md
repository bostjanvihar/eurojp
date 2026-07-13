# Eurojackpot Tracker

Fully automated, free-to-host pipeline that:

1. **Scrapes every Eurojackpot draw ever played** (since 23 March 2012) from the
   euro-jackpot.net results archive into `data/eurojackpot.csv`.
2. **Auto-updates twice a week** via GitHub Actions, right after the Tuesday and
   Friday draws (with a next-morning backup run in case results are published late).
3. **Re-runs the statistical analysis** on every update and publishes a dashboard
   with charts, tables and a downloadable spreadsheet via GitHub Pages.

Everything runs on GitHub's free tier — no server, no credit card.

## What the analysis includes

| Section | Output |
|---|---|
| Most frequent main numbers (all time) | Bar chart + written top/bottom tables |
| Euro number frequencies | Chart + table (current-rules era, all-time in xlsx) |
| Frequency over time | Rolling hit-rate lines + per-year heatmap |
| Numbers drawn together | 50×50 co-occurrence heatmap, top pairs & triplets |
| Pattern stats | Sum distribution, odd/even split |
| Hot / cold / overdue | Last-52-draw hot & cold lists, gap analysis |
| Informed picks | Three example tickets (frequency-, overdue- and balance-based) |

Outputs land in `docs/`: `index.html` (dashboard), `charts/*.png`,
`eurojackpot.xlsx` (draws + all stat tables), and CSV versions of every table.

## Setup (one time, ~5 minutes)

1. **Create a repository.** On github.com click **New repository**, name it
   (e.g. `eurojackpot-tracker`), keep it **Public** (required for free Pages),
   then upload the contents of this folder (drag & drop via *Add file → Upload
   files* works fine — make sure the `.github/workflows/update.yml` path is kept).

2. **Allow the workflow to push.** Repo → *Settings → Actions → General →
   Workflow permissions* → select **Read and write permissions** → Save.

3. **Run it once manually.** *Actions* tab → **Update Eurojackpot data** →
   **Run workflow**. The first run downloads the complete 2012→today history
   (takes a minute or two) and commits `data/eurojackpot.csv` plus the `docs/`
   dashboard.

4. **Turn on GitHub Pages.** *Settings → Pages* → Source: **Deploy from a
   branch** → Branch: `main`, folder **/docs** → Save. A couple of minutes later
   your dashboard is live at `https://<your-username>.github.io/<repo-name>/`.

That's it. Every Tuesday and Friday night the workflow scrapes the new draw,
updates the CSV/XLSX and regenerates all statistics automatically.

> **Note:** GitHub disables scheduled workflows in repos with no activity for
> 60 days. Any commit (or clicking "Run workflow") re-enables them; the bot's
> own twice-weekly commits normally keep it alive indefinitely.

## Running locally

```bash
pip install -r requirements.txt
python scraper.py      # add --full to force a complete re-download
python analysis.py
open docs/index.html
```

## Honest disclaimer

Lottery draws are independent random events. Historical frequencies, streaks
and "overdue" numbers have **no predictive power** — every combination has an
identical 1 in 139,838,160 jackpot probability. The statistics here describe
the past; the "informed picks" are for fun. The one real, if modest, use of
such analysis: avoiding *popular human patterns* (birthdays ≤31, straight
sequences) lowers the risk of splitting a prize with many co-winners.

## Notes on the data

- Source: euro-jackpot.net yearly archive pages (scraped politely, 2 requests/week
  after bootstrap).
- Rule changes handled: euro-number pool grew 8→10 (Oct 2014) and 10→12
  (Mar 2022); Tuesday draws began 29 Mar 2022. Euro-number stats are therefore
  shown for the current era, with all-time counts in the spreadsheet.
- If the source site ever changes its HTML, the parser has a fallback mode; if
  scraping fails entirely the workflow keeps existing data intact and simply
  reports the error in the Actions log.
