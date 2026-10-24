# Accuracy and limitations (AI part)

Written to be pasted into the report. Numbers marked **[pending]** must be
filled from the final run; do not invent them.

## 1. Data
| Source | Used for | Licence / credit |
|---|---|---|
| Sentinel-2 L2A (ESA), your downloaded scenes 2015-2021, 2026 | land cover, lake water 2015-16, 2018-21, 2026 | Copernicus data |
| Sentinel-2 L2A archive (AWS Earth Search), 2017-2026 | lake water time series, 10 m, cloud mask | Copernicus data |
| ESA WorldCover 2020 (v100) / 2021 (v200), 10 m | training labels and the accuracy test | CC BY 4.0 |
| OpenStreetMap lake outlines | which lakes exist and their shape | (c) OpenStreetMap contributors, ODbL |

No Sentinel-2 L2A images of Bengaluru exist in the archive for 2015-2016 and
only partly for 2017; those dates come from the project's own scenes (17-20 m,
no cloud mask).

## 2. Land-cover model (4 classes: water, vegetation, bare land, built-up)
* U-Net on 6 bands + NDVI/NDWI/NDBI, fixed reflectance scaling (same input
  for the same surface on every date).
* **First model (RGB, rule-made labels):** looked good against its own labels
  (78.6%) but scored about **37% accuracy / 23% mean IoU** against ESA WorldCover.
  Rule labels are not ground truth, so that first score was not meaningful.
* **Current model (trained on WorldCover):** tested on the right-hand 30% of
  each scene, never trained on, with a 128 px gap: **85.7% accuracy,
  mean IoU 51.8%**.

| Class | Precision | Recall | Note |
|---|---|---|---|
| Water | 37% | 82% | over-predicts water when judged by WorldCover |
| Vegetation | 92% | 93% | reliable |
| Built-up | 87% | 72% | under-counted |
| Bare land | 31% | 46% | weakest class |

* City-wide area bias against WorldCover: water about +12%, built-up about
  -17%, bare land about +50%. **Do not quote city-wide land-cover change as a
  precise number.**
* Only 2020 and 2021 have WorldCover labels. Other years are predictions of a
  model tested on those two; accuracy on other years is not measured.
* WorldCover is itself a model product (about 74-77% global accuracy), so a
  score against it measures agreement with it.

## 3. Lake water measurement (the main result)
Method for each lake and date: take the OpenStreetMap outline; mask clouds and
shadow (archive: scene classification layer; 2015-16 scenes: no cloud mask);
compute MNDWI; choose a threshold for that lake and date (Otsu, limited to
-0.10..0.30; a fixed 0.0 if the image is not clearly two-class); water = above
threshold **and** near-infrared below 0.20; count only inside the outline
(+30 m). If under 80% of the outline is clear the date is reported as
"cloudy", never as a number.

What was checked by eye (quality pictures, real scenes):
* **Bellandur:** open water 111-115 ha inside a 315 ha outline, on four dates
  Oct 2015 - Jan 2016. The rest of the outline is floating weeds and marsh,
  so "open water" is correctly far smaller than the outline.
* **Varthur:** April 2018 about 10 ha (a thin channel through vegetation);
  April 2021 0 ha (dry bed, confirmed in the image); March 2026 87 ha (full
  lake). April series: 35, 10, 12, 9, 0 ha for 2016, 2018, 2019, 2020, 2021.
* Not yet checked: smaller lakes, 2017-2025 archive scenes, agreement between
  the 2015-16 scenes and the archive. **[pending: overlap comparison]**

Known limits:
* Thresholds are starting values, not tuned against hand-drawn outlines.
* Lakes under about 5 ha have few pixels and are unreliable; the final run
  covers lakes of 5 ha and over.
* OpenStreetMap is volunteer-made: lakes can be missing or mis-drawn.
* Floating weeds hide water: results are **open water**, not lake size.
* Monsoon months are mostly cloudy; totals are not comparable between dates
  unless the same lakes were measured.
* Local scenes (17-20 m) and archive scenes (10 m) can differ slightly.

## 4. Seasonality (why months cannot be compared directly)
Bengaluru's lakes fill after the monsoon (Jun-Nov) and shrink towards April-May.
Comparing different months shows the season, not change. The analysis therefore
(a) compares each lake with its own typical value for the same month (+-1) in
other years, (b) tracks one season across years (pre-monsoon Feb-May level), and
(c) flags only lakes at least 15% below their normal. Rainfall is not used, so
a dry year also looks like loss; with 5-10 years of data a "normal" is a rough
guide, not a statistic.

## 5. Not implemented / not claimed
* Structures / encroachment counting: the model has no building or road class.
* Causes: only simple rule-based hypotheses, marked unconfirmed.
* Priority score: a weighted sum with project-chosen weights (severity 40%,
  persistence 25%, ecology 20%, population 15%); ecology and population inputs
  are placeholders (50) until real data is added.
* Explanations (occlusion) describe what the model relies on, not the ground.

## 6. Final-run numbers  **[pending]**
Lakes measured; cloud-free rate; lakes below normal; Bellandur / Varthur /
Hebbal / Ulsoor series; overlap comparison; chosen threshold settings.
