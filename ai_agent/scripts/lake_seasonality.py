"""Seasonal view of every lake: what is NORMAL for each month, and which lakes
are unusually low now. Reads the results of lake_water_timeseries.py.

Writes to ai_agent/reports/lakes/:
    lake_yearly.csv           per lake and year: pre-monsoon minimum and
                              post-monsoon maximum open water (ha)
    lake_seasonal_normal.csv  per lake and month: typical open water (median
                              over years), and how many years back it
    lake_below_normal.csv     newest measurement of each lake vs its normal,
                              biggest shortfall first
    lake_seasonality.png      key lakes: one line per year, by month

Rules (kept simple on purpose):
  * "Pre-monsoon minimum" = lowest measured value in Mar-May of a year.
    "Post-monsoon maximum" = highest measured value in Oct-Jan, labelled by
    the year the season STARTS (Oct 2018 - Jan 2019 counts as 2018).
  * Months with fewer than 2 years of data have no normal.
  * Rainfall is not used: a dry year looks like lake loss.
Run from the project root:   python ai_agent/scripts/lake_seasonality.py
"""

import argparse
import csv
import datetime as dt
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "lakes"
SERIES = OUT_DIR / "lake_water_timeseries.csv"
KEY_NAMES = ["Bellandur", "Varthur", "Hebbal", "Ulsoor"]
MIN_YEARS = 2
BELOW = 0.15
MONTH_LABELS = "J F M A M J J A S O N D".split()


def load(skip_dates=()):
    if not SERIES.exists():
        raise SystemExit(f"{SERIES} not found. Run lake_water_timeseries.py.")
    lakes = {}
    with open(SERIES, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "ok":
                continue
            day = dt.datetime.strptime(row["date"][:10], "%Y-%m-%d").date()
            if row["date"][:10] in skip_dates:
                continue
            lake = lakes.setdefault(row["lake_id"], {
                "name": row["lake"], "outline_ha": float(row["outline_ha"] or 0),
                "obs": []})
            lake["obs"].append((day, float(row["water_ha"]), row["source"]))
    for lake in lakes.values():
        lake["obs"].sort()
    return lakes


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def yearly(lakes):
    rows = []
    for lake_id, lake in lakes.items():
        pre, post = defaultdict(list), defaultdict(list)
        for day, water, _ in lake["obs"]:
            if day.month in (3, 4, 5):
                pre[day.year].append(water)
            if day.month >= 10:
                post[day.year].append(water)
            elif day.month == 1:
                post[day.year - 1].append(water)
        for year in sorted(set(pre) | set(post)):
            rows.append({
                "lake_id": lake_id, "lake": lake["name"],
                "outline_ha": lake["outline_ha"], "year": year,
                "pre_monsoon_min_ha": round(min(pre[year]), 2) if pre[year] else "",
                "pre_monsoon_median_ha": round(float(np.median(pre[year])), 2) if pre[year] else "",
                "n_pre": len(pre[year]),
                "post_monsoon_max_ha": round(max(post[year]), 2) if post[year] else "",
                "n_post": len(post[year]),
            })
    return rows


def normals(lakes):
    """lake -> month -> (median, n_years)"""
    result, rows = {}, []
    for lake_id, lake in lakes.items():
        by_month = defaultdict(lambda: defaultdict(list))
        for day, water, _ in lake["obs"]:
            by_month[day.month][day.year].append(water)
        result[lake_id] = {}
        for month in range(1, 13):
            years = by_month[month]
            if len(years) >= MIN_YEARS:
                per_year = [float(np.mean(v)) for v in years.values()]
                result[lake_id][month] = (float(np.median(per_year)), len(years))
                rows.append({"lake_id": lake_id, "lake": lake["name"],
                             "month": month,
                             "normal_ha": round(result[lake_id][month][0], 2),
                             "n_years": len(years)})
    return result, rows


def below_normal(lakes, normal):
    rows = []
    newest = max(l["obs"][-1][0] for l in lakes.values())
    for lake_id, lake in lakes.items():
        day, water, source = lake["obs"][-1]
        if (newest - day).days > 400 or lake["outline_ha"] < 5:
            continue        # stale or tiny: not judged
        # the normal must not include the year being judged
        same = defaultdict(list)
        for d, w, _ in lake["obs"]:
            gap = abs(d.month - day.month)
            if d.year != day.year and min(gap, 12 - gap) <= 1:
                same[d.year].append(w)
        if len(same) < MIN_YEARS:
            continue
        expected = float(np.median([np.mean(v) for v in same.values()]))
        if expected < 5.0:      # tiny normal: a shortfall means little
            continue
        deficit = (expected - water) / expected
        rows.append({
            "lake_id": lake_id, "lake": lake["name"],
            "outline_ha": lake["outline_ha"], "date": day.isoformat(),
            "source": source, "observed_ha": round(water, 2),
            "normal_ha": round(expected, 2), "n_years": len(same),
            "deficit_pct": round(100 * deficit, 1),
            "status": ("below normal" if deficit >= BELOW else
                       "above normal" if deficit <= -BELOW else "normal"),
        })
    rows.sort(key=lambda r: -r["deficit_pct"])
    return rows


def date_qc(lakes, min_lakes=30, low=0.5, high=2.0):
    """Whole-image check: on one date, do MOST lakes sit far from their own
    normal at once? That points at the image (haze, sun glint, a failed
    threshold), not at the lakes. Returns rows, one per date."""
    per_lake = {}
    for lake_id, lake in lakes.items():
        by_month = defaultdict(lambda: defaultdict(list))
        for day, water, _ in lake["obs"]:
            by_month[day.month][day.year].append(water)
        per_lake[lake_id] = by_month
    ratios = defaultdict(list)
    for lake_id, lake in lakes.items():
        for day, water, _ in lake["obs"]:
            others = []
            for m in (day.month - 1, day.month, day.month + 1):
                m = (m - 1) % 12 + 1
                for year, vals in per_lake[lake_id][m].items():
                    if year != day.year:
                        others.append(float(np.mean(vals)))
            if len(others) >= 2:
                expected = float(np.median(others))
                if expected >= 5.0:          # ignore tiny / dry normals
                    ratios[day].append(water / expected)
    rows = []
    for day in sorted(ratios):
        r = np.array(ratios[day])
        if len(r) < min_lakes:
            continue
        med = float(np.median(r))
        rows.append({"date": day.isoformat(), "lakes_compared": len(r),
                     "median_vs_normal": round(med, 2),
                     "share_below_half": round(float((r < 0.5).mean()), 2),
                     "flag": "CHECK IMAGE" if (med < low or med > high) else ""})
    return rows


def chart(lakes):
    keys = [(i, l) for i, l in lakes.items()
            if any(k.lower() in l["name"].lower() for k in KEY_NAMES)]
    keys.sort(key=lambda kv: -kv[1]["outline_ha"])
    keys = keys[:4]
    if not keys:
        return None
    years = sorted({d.year for _, l in keys for d, _, _ in l["obs"]})
    cmap = plt.get_cmap("Blues")
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.8),
                             squeeze=False)
    for ax, (lake_id, lake) in zip(axes[0], keys):
        for year in years:
            pts = sorted((d.month, w) for d, w, _ in lake["obs"]
                         if d.year == year)
            if not pts:
                continue
            shade = 0.3 + 0.7 * (year - years[0]) / max(1, years[-1] - years[0])
            latest = year == years[-1]
            xs, ys = [], []
            for k, (m, w) in enumerate(pts):
                if k and m - pts[k - 1][0] > 2:      # gap (cloudy months)
                    xs.append(np.nan)
                    ys.append(np.nan)
                xs.append(m)
                ys.append(w)
            ax.plot(xs, ys, marker="o",
                    ms=3.5, lw=2.2 if latest else 1.2,
                    color=cmap(shade), zorder=3 if latest else 2)
            if year in (years[0], years[-1]):
                ax.annotate(str(year), pts[-1], xytext=(3, 0),
                            textcoords="offset points", fontsize=7,
                            color="#444", va="center")
        ax.set_title(lake["name"], fontsize=10)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_LABELS, fontsize=8)
        ax.set_xlim(0.7, 13.3)
        ax.set_ylim(bottom=0)
        ax.axvspan(6, 9.5, color="#999", alpha=0.10, lw=0)
        ax.grid(axis="y", color="#e5e5e5", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel("open water (ha)")
    fig.suptitle("Open water by month, one line per year (darker = newer)\n"
                 "grey band = monsoon, mostly cloudy, lines break there",
                 fontsize=10)
    fig.tight_layout()
    path = OUT_DIR / "lake_seasonality.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-dates", default="",
                        help="comma-separated YYYY-MM-DD dates to leave out "
                             "(use for images the date check flags)")
    parser.add_argument("--exclude-flagged", action="store_true",
                        help="leave out dates where most lakes read far BELOW "
                             "normal at once, and remember them")
    parser.add_argument("--save-exclusions", action="store_true",
                        help="remember the --skip-dates list for the AI "
                             "service (excluded_dates.txt)")
    args = parser.parse_args()
    skip = {d.strip() for d in args.skip_dates.split(",") if d.strip()}
    lakes = load(skip)
    if args.exclude_flagged:
        low = {r["date"] for r in date_qc(lakes)
               if r["median_vs_normal"] < 0.5}
        skip |= low
        print(f"Excluding {len(low)} flagged dates (most lakes far below "
              "normal at once).")
        lakes = load(skip)
    if args.save_exclusions or args.exclude_flagged:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / "excluded_dates.txt", "w", encoding="utf-8") as h:
            h.write("# image dates left out (checked by hand)\n")
            h.write("\n".join(sorted(skip)) + "\n")
        print(f"Saved {len(skip)} excluded dates to "
              f"{OUT_DIR / 'excluded_dates.txt'}")
    if skip:
        print(f"Leaving out dates: {', '.join(sorted(skip))}")
    print(f"Lakes with measurements: {len(lakes)}")
    y = yearly(lakes)
    write(OUT_DIR / "lake_yearly.csv", y)
    normal, nrows = normals(lakes)
    if nrows:
        write(OUT_DIR / "lake_seasonal_normal.csv", nrows)
    below = below_normal(lakes, normal)
    if below:
        write(OUT_DIR / "lake_below_normal.csv", below)
    png = chart(lakes)
    qc_rows = date_qc(lakes)
    if qc_rows:
        write(OUT_DIR / "lake_date_qc.csv", qc_rows)
        flagged = [r for r in qc_rows if r["flag"]]
        print(f"\nWhole-image check: {len(flagged)} of {len(qc_rows)} dates "
              "have most lakes far from their normal at once.")
        for r in flagged:
            print(f"  {r['date']}  median lake is {r['median_vs_normal']:.2f}"
                  f" x normal ({r['lakes_compared']} lakes)  <- CHECK IMAGE")
        if flagged:
            print("  Look at those dates (real drought or flood years can "
                  "also do this). If an image is bad, re-run with "
                  "--skip-dates DATE,DATE")

    print(f"\nNewest measurement vs normal, {len(below)} lakes comparable:")
    for status in ("below normal", "normal", "above normal"):
        print(f"  {status:<13}{sum(1 for r in below if r['status'] == status):>5}")
    print("\nLowest vs normal (check these first):")
    for r in below[:10]:
        print(f"  {r['lake'] or r['lake_id']:<28}{r['date']}  "
              f"{r['observed_ha']:>7.1f} ha vs normal {r['normal_ha']:>7.1f} "
              f"({abs(r['deficit_pct']):.0f}% "
              f"{'below' if r['deficit_pct'] >= 0 else 'above'})  "
              f"[{r['n_years']} yrs]")
    print("\nKey lakes, pre-monsoon (Mar-May) MEDIAN open water by year (ha):")
    for lake_id, lake in lakes.items():
        if any(k.lower() in lake["name"].lower() for k in KEY_NAMES):
            vals = {r["year"]: r["pre_monsoon_median_ha"] for r in y
                    if r["lake_id"] == lake_id and r["pre_monsoon_median_ha"] != ""}
            print(f"  {lake['name']:<22}" + "  ".join(
                f"{k}:{v:.0f}" for k, v in sorted(vals.items())))
    print(f"\nSaved in {OUT_DIR}: lake_yearly.csv, lake_seasonal_normal.csv, "
          "lake_below_normal.csv, lake_date_qc.csv" + (f", {png.name}" if png else ""))
    print("Reminder: rainfall is not used, so dry years look like lake loss; "
          "small lakes (under ~5 ha) are the least reliable.")


if __name__ == "__main__":
    main()
