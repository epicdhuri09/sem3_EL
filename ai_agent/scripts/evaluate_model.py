"""Evaluate the trained land-cover model.

Reports per-class precision, recall and IoU plus a confusion matrix, for:
  * the VALIDATION scenes (never seen in training) - the number that matters
  * every individual scene (to spot scenes where the model struggles)

IMPORTANT: the "truth" here is the heuristic pseudo-label masks, not ground
truth. These scores show how well the model reproduces those rules, and
whether small classes such as water are being missed. They do NOT prove
real-world accuracy.

Run from the project root:
    python ai_agent/scripts/evaluate_model.py
"""

import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Reuse the exact model, dataset and helpers from training so nothing drifts.
from train_model import (
    BATCH_SIZE,
    IMAGE_DIR,
    MASK_DIR,
    NUM_CLASSES,
    SEED,
    SmallUNet,
    TileDataset,
    scene_name,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "landcover_unet.pth"
REPORT_DIR = ROOT / "reports" / "evaluation"
CLASS_NAMES = ["water", "vegetation", "bare_land", "built_up"]


def validation_scenes(all_scenes):
    """Recreate the exact train/validation split used by train_model.py."""
    scenes = sorted(all_scenes)
    random.seed(SEED)
    random.shuffle(scenes)
    return set(scenes[-2:])


def confusion_for(model, image_paths):
    """Return a 4x4 matrix: rows = pseudo-label (truth), cols = prediction."""
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    loader = DataLoader(TileDataset(image_paths), batch_size=BATCH_SIZE,
                        shuffle=False)
    with torch.inference_mode():
        for x, y in loader:
            pred = model(x).argmax(dim=1).numpy().ravel()
            true = y.numpy().ravel()
            keep = (true >= 0) & (true < NUM_CLASSES)
            idx = true[keep] * NUM_CLASSES + pred[keep]
            matrix += np.bincount(
                idx, minlength=NUM_CLASSES * NUM_CLASSES
            ).reshape(NUM_CLASSES, NUM_CLASSES)
    return matrix


def metrics_from(matrix):
    """Per-class precision / recall / IoU, plus overall accuracy and mIoU."""
    tp = np.diag(matrix).astype(float)
    truth_total = matrix.sum(axis=1).astype(float)
    pred_total = matrix.sum(axis=0).astype(float)

    def safe(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0)

    precision = safe(tp, pred_total)
    recall = safe(tp, truth_total)
    iou = safe(tp, truth_total + pred_total - tp)
    accuracy = tp.sum() / matrix.sum() if matrix.sum() else float("nan")
    return precision, recall, iou, accuracy, np.nanmean(iou)


def fmt(value):
    return "  n/a " if np.isnan(value) else f"{value * 100:5.1f}%"


def print_report(title, matrix):
    precision, recall, iou, accuracy, miou = metrics_from(matrix)
    truth_share = matrix.sum(axis=1) / max(matrix.sum(), 1)
    majority = truth_share.max()

    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")
    print(f"Pixels evaluated: {matrix.sum():,}")
    print(f"{'class':<12}{'in labels':>10}{'precision':>11}{'recall':>9}"
          f"{'IoU':>8}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<12}{fmt(truth_share[i]):>10}{fmt(precision[i]):>11}"
              f"{fmt(recall[i]):>9}{fmt(iou[i]):>8}")
    print(f"Overall pixel accuracy: {fmt(accuracy).strip()}  "
          f"(always guessing the biggest class would give "
          f"{fmt(majority).strip()})")
    print(f"Mean IoU: {fmt(miou).strip()}")

    print("\nConfusion matrix (rows = pseudo-label, columns = model):")
    print(f"{'':<12}" + "".join(f"{n:>12}" for n in CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<12}" + "".join(f"{v:>12,}" for v in matrix[i]))


def main():
    if not CHECKPOINT.exists():
        raise SystemExit(f"No checkpoint at {CHECKPOINT}. Run train_model.py")

    images = sorted(
        p for p in IMAGE_DIR.glob("*.png")
        if (MASK_DIR / f"{p.stem}.png").exists()
    )
    if not images:
        raise SystemExit("No image/mask pairs found. Run the earlier steps.")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    model = SmallUNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    by_scene = {}
    for path in images:
        by_scene.setdefault(scene_name(path), []).append(path)

    val_scenes = validation_scenes(by_scene.keys())
    print("HELD-OUT validation scenes (never used for training): "
          + ", ".join(sorted(val_scenes)))

    val_paths = [p for s in sorted(val_scenes) for p in by_scene[s]]
    val_matrix = confusion_for(model, val_paths)
    print_report("VALIDATION SCENES (the honest number)", val_matrix)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene in sorted(by_scene):
        matrix = confusion_for(model, by_scene[scene])
        precision, recall, iou, accuracy, miou = metrics_from(matrix)
        rows.append({
            "scene": scene,
            "split": "validation" if scene in val_scenes else "train",
            "tiles": len(by_scene[scene]),
            "accuracy": round(float(accuracy), 4),
            "mean_iou": round(float(miou), 4),
            **{f"{n}_recall": round(float(recall[i]), 4)
               for i, n in enumerate(CLASS_NAMES)},
            **{f"{n}_precision": round(float(precision[i]), 4)
               for i, n in enumerate(CLASS_NAMES)},
        })

    print(f"\n{'=' * 64}\nPER-SCENE SUMMARY\n{'=' * 64}")
    print(f"{'scene':<12}{'split':<12}{'tiles':>6}{'accuracy':>10}"
          f"{'water recall':>14}")
    for r in rows:
        print(f"{r['scene']:<12}{r['split']:<12}{r['tiles']:>6}"
              f"{r['accuracy'] * 100:>9.1f}%"
              f"{r['water_recall'] * 100:>13.1f}%")

    out_csv = REPORT_DIR / "per_scene_metrics.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    np.savetxt(REPORT_DIR / "validation_confusion_matrix.csv", val_matrix,
               fmt="%d", delimiter=",",
               header=",".join(CLASS_NAMES), comments="")

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {REPORT_DIR / 'validation_confusion_matrix.csv'}")
    print("\nREMINDER: 'truth' here = heuristic pseudo-labels, not verified "
          "ground truth.")


if __name__ == "__main__":
    main()
