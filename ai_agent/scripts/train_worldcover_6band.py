"""Train the all-bands model on ESA WorldCover labels, with an honest test.

Inputs : the 6 Sentinel-2 bands (+ NDVI/NDWI/NDBI) of the scenes that have a
         WorldCover reference (made by evaluate_vs_worldcover.py).
Labels : WorldCover, converted to water / vegetation / bare land / built-up.
Split  : the RIGHT 30% of every scene is kept aside and never trained on. A
         128-pixel gap separates it from the training area so nearby pixels
         don't leak across. The score is measured on that held-out strip only.

Run from the project root:
    python ai_agent/scripts/train_worldcover_6band.py
Optional:  --steps 1200  --base 24  --batch 16  --crop 128
"""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch import nn

from landcover6 import (
    CLASS_NAMES, IGNORE, IN_CHANNELS, MODEL_PATH, NUM_CLASSES, ROOT,
    SCALING_VERSION,
    UNet, WC_DIR, DATA_DIR, confusion, load_scene, metrics_from,
    predict_scene, print_metrics,
)

SEED = 42
TEST_FRACTION = 0.30
GAP = 128
REPORT_DIR = ROOT / "reports" / "evaluation"


def find_reference_scenes():
    names = []
    for path in sorted(WC_DIR.glob("reference_*_worldcover.tif")):
        names.append(path.name[len("reference_"):-len("_worldcover.tif")])
    return names


def load_training_scenes():
    scenes = {}
    for name in find_reference_scenes():
        features, valid, _, reason = load_scene(DATA_DIR / name)
        if features is None:
            print(f"SKIP {name}: {reason}")
            continue
        with rasterio.open(WC_DIR / f"reference_{name}_worldcover.tif") as src:
            labels = src.read(1)
        if labels.shape != valid.shape:
            print(f"SKIP {name}: reference does not match the scene grid")
            continue
        labels = labels.copy()
        labels[~valid] = IGNORE
        height, width = labels.shape
        test_start = int(width * (1 - TEST_FRACTION))
        scenes[name] = {
            "features": features, "labels": labels, "valid": valid,
            "test_start": test_start, "train_end": test_start - GAP,
        }
        print(f"{name}: {height}x{width} px | train columns 0-"
              f"{test_start - GAP} | held-out columns {test_start}-{width}")
    return scenes


def class_weights(scenes, cap=12.0):
    counts = np.zeros(NUM_CLASSES)
    for scene in scenes.values():
        region = scene["labels"][:, :scene["train_end"]]
        counts += np.bincount(region[region < NUM_CLASSES],
                              minlength=NUM_CLASSES)
    freq = counts / counts.sum()
    weights = 1.0 / np.sqrt(freq + 1e-6)
    weights = np.clip(weights / weights.min(), 1.0, cap)
    print("Training label shares: " + ", ".join(
        f"{n} {f * 100:.1f}%" for n, f in zip(CLASS_NAMES, freq)))
    print("Class weights: " + ", ".join(
        f"{n} {w:.1f}" for n, w in zip(CLASS_NAMES, weights)))
    return torch.tensor(weights, dtype=torch.float32)


def sample_batch(scenes, batch, crop, rng):
    xs, ys = [], []
    names = list(scenes)
    while len(xs) < batch:
        scene = scenes[rng.choice(names)]
        height = scene["labels"].shape[0]
        for _ in range(20):
            y = rng.randint(0, height - crop)
            x = rng.randint(0, scene["train_end"] - crop)
            label = scene["labels"][y:y + crop, x:x + crop]
            if (label < NUM_CLASSES).mean() >= 0.5:
                break
        feat = scene["features"][:, y:y + crop, x:x + crop]
        k = rng.randint(0, 3)
        feat, label = np.rot90(feat, k, (1, 2)), np.rot90(label, k)
        if rng.random() < 0.5:
            feat, label = feat[:, :, ::-1], label[:, ::-1]
        xs.append(np.ascontiguousarray(feat))
        ys.append(np.ascontiguousarray(label))
    return (torch.from_numpy(np.stack(xs)),
            torch.from_numpy(np.stack(ys).astype(np.int64)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--base", type=int, default=24)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--crop", type=int, default=128)
    parser.add_argument("--max-weight", type=float, default=12.0,
                        help="cap on the class weight of rare classes (water). "
                             "High values find more water but also flag more "
                             "false water; try 12, 6, 3, 1.5")
    parser.add_argument("--out", default="",
                        help="save the model here instead of the default path "
                             "(use this when comparing several settings)")
    args = parser.parse_args()

    rng = random.Random(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    scenes = load_training_scenes()
    if not scenes:
        raise SystemExit(
            "No WorldCover reference found. Run "
            "evaluate_vs_worldcover.py first (it creates them)."
        )

    weights = class_weights(scenes, args.max_weight)
    model = UNet(base=args.base)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-3, total_steps=args.steps
    )

    print(f"\nTraining {args.steps} steps on CPU "
          f"(batch {args.batch}, crop {args.crop}px)...")
    start, running = time.time(), 0.0
    model.train()
    for step in range(1, args.steps + 1):
        x, y = sample_batch(scenes, args.batch, args.crop, rng)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running += loss.item()

        if step == 20 or step % 100 == 0:
            elapsed = time.time() - start
            left = elapsed / step * (args.steps - step) / 60
            avg = running / (20 if step == 20 else 100)
            print(f"step {step}/{args.steps} | loss {avg:.4f} | "
                  f"about {left:.0f} min left")
            running = 0.0

    save_path = Path(args.out) if args.out else MODEL_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_channels": IN_CHANNELS,
        "base": args.base,
        "classes": dict(enumerate(CLASS_NAMES)),
        "scaling": SCALING_VERSION,
        "trained_on": sorted(scenes),
        "note": "Trained on ESA WorldCover labels; right 30% held out.",
        "max_weight": args.max_weight,
    }, save_path)
    print(f"\nSaved model: {save_path}")

    # ---- honest score on the held-out right-hand strip -------------------
    print(f"\n{'=' * 64}\nHELD-OUT TEST (right 30% of each scene, never "
          f"trained on)\n{'=' * 64}")
    model.eval()
    total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    rows = []
    for name, scene in scenes.items():
        prediction = predict_scene(model, scene["features"], scene["valid"])
        t = scene["test_start"]
        matrix = confusion(scene["labels"][:, t:], prediction[:, t:])
        total += matrix
        print_metrics(f"{name} (held-out strip)", matrix)
        precision, recall, iou, accuracy, miou = metrics_from(matrix)
        rows.append({
            "scene": name, "accuracy": round(float(accuracy), 4),
            "mean_iou": round(float(miou), 4),
            **{f"{n}_recall": round(float(recall[i]), 4)
               for i, n in enumerate(CLASS_NAMES)},
            **{f"{n}_precision": round(float(precision[i]), 4)
               for i, n in enumerate(CLASS_NAMES)},
        })

    print(f"\n{'=' * 64}\nCOMBINED HELD-OUT RESULT\n{'=' * 64}")
    print_metrics("All held-out strips", total)
    precision, recall, iou, accuracy, miou = metrics_from(total)
    p, r = float(precision[0]), float(recall[0])
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    print(f"\nWATER SUMMARY (max-weight {args.max_weight}): precision "
          f"{p * 100:.1f}%  recall {r * 100:.1f}%  F1 {f1 * 100:.1f}%  "
          f"IoU {float(iou[0]) * 100:.1f}%  | overall accuracy "
          f"{float(accuracy) * 100:.1f}%  mean IoU {float(miou) * 100:.1f}%")
    print("\nFor comparison, the old RGB model scored about 37% accuracy / "
          "23% mean IoU against WorldCover\n(measured on whole scenes, so "
          "the comparison is approximate).")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORT_DIR / "worldcover_6band_heldout_metrics.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
