"""
Eurojackpot results scraper.

Downloads winning numbers from the yearly archive pages on euro-jackpot.net
(complete history since the first draw on 23 March 2012) and maintains
data/eurojackpot.csv.

- First run (no CSV yet): fetches every year 2012 -> current year.
- Later runs: fetches only the current year (and January runs also fetch the
  previous year to be safe) and appends any draws not yet in the CSV.

Run: python scraper.py [--full]   (--full forces a complete re-download)
"""

import csv
import re
import sys
import time
from datetime import date
from pathlib import Path

import requests

FIRST_YEAR = 2012
ARCHIVE_URL = "https://www.euro-jackpot.net/results-archive-{year}"
DATA_FILE = Path(__file__).parent / "data" / "eurojackpot.csv"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
FIELDNAMES = ["date", "weekday", "n1", "n2", "n3", "n4", "n5", "e1", "e2"]

DATE_RE = re.compile(r"/results/(\d{2})-(\d{2})-(\d{4})")
ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
LI_NUM_RE = re.compile(r"<li[^>]*>\s*(\d{1,2})\s*</li>", re.S | re.I)
ANY_NUM_RE = re.compile(r">\s*(\d{1,2})\s*<")


def fetch(url: str, retries: int = 3) -> str:
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.text
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def parse_archive(html: str) -> list[dict]:
    """Parse one yearly archive page into a list of draw dicts."""
    draws = {}
    # Row-based parsing first; fall back to date-anchored splitting if the
    # site ever drops the <table> markup.
    rows = ROW_RE.findall(html)
    if not rows:
        parts = DATE_RE.split(html)
        # re.split with 3 groups yields [pre, d, m, y, chunk, d, m, y, chunk...]
        rows = []
        for i in range(1, len(parts) - 3, 4):
            d, m, y, chunk = parts[i], parts[i + 1], parts[i + 2], parts[i + 3]
            rows.append(f'/results/{d}-{m}-{y}"{chunk}')

    for row in rows:
        m = DATE_RE.search(row)
        if not m:
            continue
        dd, mm, yyyy = m.groups()
        iso = f"{yyyy}-{mm}-{dd}"

        nums = [int(n) for n in LI_NUM_RE.findall(row)]
        if len(nums) < 7:
            nums = [int(n) for n in ANY_NUM_RE.findall(row)]
        if len(nums) < 7:
            continue

        main = sorted(nums[:5])
        euro = sorted(nums[5:7])
        if not (
            all(1 <= n <= 50 for n in main)
            and len(set(main)) == 5
            and all(1 <= e <= 12 for e in euro)
            and len(set(euro)) == 2
        ):
            continue

        d = date.fromisoformat(iso)
        draws[iso] = {
            "date": iso,
            "weekday": d.strftime("%A"),
            "n1": main[0], "n2": main[1], "n3": main[2],
            "n4": main[3], "n5": main[4],
            "e1": euro[0], "e2": euro[1],
        }
    return list(draws.values())


def load_existing() -> dict:
    if not DATA_FILE.exists():
        return {}
    with DATA_FILE.open(newline="", encoding="utf-8") as f:
        return {row["date"]: row for row in csv.DictReader(f)}


def save(draws: dict) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with DATA_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for key in sorted(draws):
            writer.writerow(draws[key])


def main() -> None:
    full = "--full" in sys.argv
    existing = load_existing()
    this_year = date.today().year

    if full or not existing:
        years = range(FIRST_YEAR, this_year + 1)
    elif date.today().month == 1:
        years = [this_year - 1, this_year]
    else:
        years = [this_year]

    added = 0
    for year in years:
        print(f"Fetching {year} ...")
        try:
            html = fetch(ARCHIVE_URL.format(year=year))
        except RuntimeError as e:
            print(f"  WARNING: {e}", file=sys.stderr)
            continue
        parsed = parse_archive(html)
        print(f"  found {len(parsed)} draws")
        for draw in parsed:
            if draw["date"] not in existing:
                added += 1
            existing[draw["date"]] = {k: str(v) for k, v in draw.items()}
        time.sleep(2)  # be polite

    if not existing:
        sys.exit("No draws scraped and no existing data - aborting.")

    save(existing)
    print(f"Done. {added} new draw(s) added, {len(existing)} total in {DATA_FILE}.")


if __name__ == "__main__":
    main()
