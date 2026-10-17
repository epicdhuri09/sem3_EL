
from pathlib import Path
import csv

import numpy as np
from PIL import Image
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "training" / "images"
MODEL_PATH = ROOT / "models" / "landcover_unet.pth"
OUTPUT_DIR = ROOT / "reports" / "predictions" / "all_scenes"

BATCH_SIZE = 8
NAMES = ["water", "vegetation", "bare_land", "built_up"]
COLORS = np.array([
    [0, 100, 255],     # Water: blue
    [20, 180, 60],     # Vegetation: green
    [220, 180, 80],    # Bare land: tan
    [230, 65, 65],     # Built-up: red
], dtype=np.uint8)


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.enc1 = block(3, 16)
        self.enc2 = block(16, 32)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = block(32, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = block(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = block(32, 16)
        self.out = nn.Conv2d(16, 4, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def tile_position(path):
    # Filenames follow scene_row_column.png
    parts = path.stem.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Unexpected tile filename: {path.name}")

    scene, row, column = parts
    if not row.isdigit() or not column.isdigit():
        raise ValueError(f"Cannot read tile coordinates: {path.name}")

    return scene, int(row), int(column)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    paths = sorted(IMAGE_DIR.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No tiles in {IMAGE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model exactly once.
    model = SmallUNet()
    checkpoint = torch.load(
        MODEL_PATH, map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Each scene stores its predicted tile masks and pixel positions.
    scenes = {}
    global_counts = np.zeros(4, dtype=np.int64)

    print(f"Tiles: {len(paths)}")
    print("Loading model once; predicting in batches...")

    with torch.inference_mode():
        for start in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[start:start + BATCH_SIZE]
            images = []

            for path in batch_paths:
                scene, row, column = tile_position(path)
                image = Image.open(path).convert("RGB")
                array = np.asarray(image, dtype=np.float32) / 255.0
                images.append(array)

            batch = torch.from_numpy(
                np.stack(images)
            ).permute(0, 3, 1, 2)

            predictions = model(batch).argmax(dim=1).numpy().astype(
                np.uint8
            )

            for path, prediction in zip(batch_paths, predictions):
                scene, row, column = tile_position(path)

                if scene not in scenes:
                    scenes[scene] = []

                scenes[scene].append((row, column, prediction))
                global_counts += np.bincount(
                    prediction.ravel(), minlength=4
                )

            completed = min(start + len(batch_paths), len(paths))
            print(f"Predicted {completed}/{len(paths)} tiles")

    csv_path = OUTPUT_DIR / "scene_summary.csv"
    csv_rows = []

    # Stitch tiles using their recorded pixel offsets.
    for scene, tiles in sorted(scenes.items()):
        height = max(row + mask.shape[0] for row, _, mask in tiles)
        width = max(col + mask.shape[1] for _, col, mask in tiles)

        # 255 means no tile covers this position.
        canvas = np.full((height, width), 255, dtype=np.uint8)
        scene_counts = np.zeros(4, dtype=np.int64)

        for row, col, mask in tiles:
            h, w = mask.shape
            canvas[row:row + h, col:col + w] = mask
            scene_counts += np.bincount(
                mask.ravel(), minlength=4
            )

        colored = np.zeros((height, width, 3), dtype=np.uint8)
        valid = canvas < 4
        colored[valid] = COLORS[canvas[valid]]

        map_path = OUTPUT_DIR / f"{scene}_landcover.png"
        Image.fromarray(colored).save(map_path)

        total = int(scene_counts.sum())
        row_data = {
            "scene": scene,
            "tiles": len(tiles),
            "pixels": total,
        }

        for class_id, name in enumerate(NAMES):
            pct = (
                100 * scene_counts[class_id] / total
                if total else 0
            )
            row_data[f"{name}_percent"] = round(pct, 2)

        csv_rows.append(row_data)
        print(f"Saved {map_path.name} ({len(tiles)} tiles)")

    fields = [
        "scene", "tiles", "pixels",
        "water_percent", "vegetation_percent",
        "bare_land_percent", "built_up_percent",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    total = int(global_counts.sum())
    print("\nAll-tile prediction summary:")
    for class_id, name in enumerate(NAMES):
        pct = 100 * global_counts[class_id] / total if total else 0
        print(f"{name}: {pct:.2f}%")

    print(f"\nScene maps: {OUTPUT_DIR}")
    print(f"Summary CSV: {csv_path}")
    print("Note: predictions use heuristic training labels.")


if __name__ == "__main__":
    main()
