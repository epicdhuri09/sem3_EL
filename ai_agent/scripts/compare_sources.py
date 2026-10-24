"""Do your own scenes (2015-2026, 17-20 m) and the archive images (10 m) agree?

Takes every lake measured by BOTH sources on the same day (or within 5 days)
and compares the open-water numbers. If they agree, the 2015-2016 scenes can be
read together with the archive years; if not, the report must say they are on a
different footing.

Writes to ai_agent/reports/lakes/:
    source_comparison.csv   one row per date pair: total ha each, ratio,
                            correlation, typical difference
    source_agreement.png    scatter of local vs archive per lake
Run from the project root:   python ai_agent/scripts/compare_sources.py
"""

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
MAX_GAP_DAYS = 5
MIN_LAKE_HA = 5.0      # compare lakes with at least this much outline


def main():
    if not SERIES.exists():
        raise SystemExit(f"{SERIES} not found.")
    local, archive = defaultdict(dict), defaultdict(dict)
    with open(SERIES, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "ok" or float(row["outline_ha"] or 0) < MIN_LAKE_HA:
                continue
            day = dt.datetime.strptime(row["date"][:10], "%Y-%m-%d").date()
            target = local if row["source"] == "local" else archive
            target[day][row["lake_id"]] = float(row["water_ha"])

    pairs = []
    for ld in sorted(local):
        near = [(abs((ad - ld).days), ad) for ad in archive
                if abs((ad - ld).days) <= MAX_GAP_DAYS]
        if near:
            pairs.append((ld, min(near)[1]))
    if not pairs:
        raise SystemExit("No local scene has an archive image within "
                         f"{MAX_GAP_DAYS} days.")

    rows, points = [], []
    for ld, ad in pairs:
        common = sorted(set(local[ld]) & set(archive[ad]))
        if len(common) < 5:
            continue
        a = np.array([local[ld][k] for k in common])
        b = np.array([archive[ad][k] for k in common])
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        rows.append({
            "local_date": ld.isoformat(), "archive_date": ad.isoformat(),
            "days_apart": abs((ad - ld).days), "lakes": len(common),
            "local_total_ha": round(float(a.sum()), 1),
            "archive_total_ha": round(float(b.sum()), 1),
            "local_over_archive": round(float(a.sum() / b.sum()), 2) if b.sum() else "",
            "correlation": round(corr, 3),
            "median_abs_diff_ha": round(float(np.median(np.abs(a - b))), 2),
        })
        points.append((f"{ld} vs {ad}", a, b))

    if not rows:
        raise SystemExit("Pairs found but fewer than 5 common lakes each.")
    print(f"{'local':<12}{'archive':<12}{'days':>5}{'lakes':>7}{'local ha':>10}"
          f"{'archive ha':>12}{'ratio':>7}{'corr':>7}{'median |diff|':>15}")
    for r in rows:
        print(f"{r['local_date']:<12}{r['archive_date']:<12}{r['days_apart']:>5}"
              f"{r['lakes']:>7}{r['local_total_ha']:>10.0f}"
              f"{r['archive_total_ha']:>12.0f}{r['local_over_archive']:>7}"
              f"{r['correlation']:>7}{r['median_abs_diff_ha']:>13.1f} ha")
    ratios = [r["local_over_archive"] for r in rows if r["local_over_archive"] != ""]
    corrs = [r["correlation"] for r in rows if r["correlation"] == r["correlation"]]
    print(f"\nTypical local/archive total ratio: {np.median(ratios):.2f} "
          f"(1.00 = same). Typical correlation: {np.median(corrs):.2f}.")
    print("Reading: ratio within about 0.85-1.15 and correlation above 0.9 "
          "= the two sources can be read together. Pairs several days apart "
          "also differ because water changes; same-day pairs are the fair test.")

    with open(OUT_DIR / "source_comparison.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    cmap = plt.get_cmap("Blues")
    for i, (label, a, b) in enumerate(points):
        ax.scatter(b, a, s=14, color=cmap(0.35 + 0.6 * i / max(1, len(points) - 1)),
                   label=label, alpha=0.85)
    top = max(max(a.max(), b.max()) for _, a, b in points) * 1.05
    ax.plot([0, top], [0, top], color="#999", lw=1, ls="--")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("archive (10 m): open water, ha")
    ax.set_ylabel("your scene (17-20 m): open water, ha")
    ax.set_title("Same lake, same time: do the two sources agree?", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "source_agreement.png", dpi=130)
    print(f"\nSaved: {OUT_DIR / 'source_comparison.csv'}, source_agreement.png")


if __name__ == "__main__":
    main()
