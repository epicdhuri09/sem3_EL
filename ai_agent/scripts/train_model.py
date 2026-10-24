
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "training" / "images"
MASK_DIR = ROOT / "data" / "training" / "masks"
OUTPUT_DIR = ROOT / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 8
EPOCHS = 30
NUM_CLASSES = 4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def scene_name(path):
    # Tile filenames are expected to start with their scene folder name.
    return path.stem.split("_", 1)[0]


class TileDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = MASK_DIR / f"{image_path.stem}.png"

        image = np.asarray(
            Image.open(image_path).convert("RGB"), dtype=np.float32
        ) / 255.0
        mask = np.asarray(Image.open(mask_path), dtype=np.int64)

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        return image, mask


class SmallUNet(nn.Module):
    def __init__(self, classes=NUM_CLASSES):
        super().__init__()

        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
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
        self.out = nn.Conv2d(16, classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def main():
    images = sorted(IMAGE_DIR.glob("*.png"))
    valid_images = [
        p for p in images
        if (MASK_DIR / f"{p.stem}.png").exists()
    ]

    if len(valid_images) < 10:
        raise RuntimeError(
            f"Only {len(valid_images)} matching image/mask pairs found."
        )

    # Split by scene, not random tiles, to reduce spatial leakage.
    scenes = sorted({scene_name(p) for p in valid_images})
    random.shuffle(scenes)
    val_scenes = set(scenes[-2:])
    train_images = [p for p in valid_images if scene_name(p) not in val_scenes]
    val_images = [p for p in valid_images if scene_name(p) in val_scenes]

    if not train_images or not val_images:
        raise RuntimeError("Could not create train/validation scene split.")

    print(f"Device: CPU")
    print(f"Scenes: {len(scenes)} | Train tiles: {len(train_images)}"
          f" | Validation tiles: {len(val_images)}")
    print("Labels: 0=water, 1=vegetation, 2=bare land, 3=built-up")
    print("WARNING: masks are heuristic pseudo-labels, not ground truth.")

    train_loader = DataLoader(
        TileDataset(train_images), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TileDataset(val_images), batch_size=BATCH_SIZE, shuffle=False
    )

    device = torch.device("cpu")
    model = SmallUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([7.0, 1.0, 1.0, 1.0]))

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item() * x.size(0)

        train_loss /= len(train_images)
        val_loss /= len(val_images)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train loss={train_loss:.4f} | val loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": {
                        0: "water",
                        1: "vegetation",
                        2: "bare_land",
                        3: "built_up",
                    },
                    "input_channels": 3,
                    "tile_size": 256,
                    "warning": "Trained on heuristic pseudo-labels.",
                },
                OUTPUT_DIR / "landcover_unet.pth",
            )

    print(f"Saved best checkpoint: {OUTPUT_DIR / 'landcover_unet.pth'}")


if __name__ == "__main__":
    main()
