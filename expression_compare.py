"""Build a legacy-vs-affect comparison sheet for the 7 expression presets."""

import sys
from pathlib import Path

import cv2

PRESETS = ["happy", "sad", "angry", "surprised", "thinking", "shy", "wronged"]
ROOT = Path(__file__).resolve().parent / "data"
LEGACY = ROOT / "expression_shots_legacy"
AFFECT = ROOT / "expression_shots_affect"
OUT = ROOT / "expression_shots" / "affect_vs_legacy.jpg"

CROP = (slice(40, 760), slice(500, 1300))
TILE = (200, 180)


def tile(image, label):
    img = cv2.resize(image[CROP], TILE)
    cv2.putText(img, label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    return img


def main():
    rows = []
    for folder, tag in ((LEGACY, "legacy"), (AFFECT, "affect")):
        tiles = []
        for name in PRESETS:
            img = cv2.imread(str(folder / f"{name}.jpg"))
            if img is None:
                img = cv2.imread(str(LEGACY / f"{name}.jpg"))
            tiles.append(tile(img, f"{tag}:{name}"))
        rows.append(cv2.hconcat(tiles))
    sheet = cv2.vconcat(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), sheet)
    print("saved", OUT)


if __name__ == "__main__":
    sys.exit(main())
