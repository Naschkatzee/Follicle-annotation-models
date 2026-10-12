"""
Prepare a 2D nnU-Net v2 dataset from BiomedParse pseudo-labels.

Input:
    /content/drive/MyDrive/biomedparse_project/images/
    /content/drive/MyDrive/biomedparse_project/outputs/

Output:
    /content/drive/MyDrive/biomedparse_project/nnunet_raw/Dataset001_Follicles/

Images are converted to grayscale PNG.
Masks are converted to binary labels:
0 = background
1 = follicle

These are pseudo-labels / weak labels, not expert ground truth.
"""

from pathlib import Path
import csv
import json
import re

import numpy as np
from PIL import Image

BASE = Path("/content/drive/MyDrive/biomedparse_project")
IMAGE_ROOT = BASE / "images"
MASK_ROOT = BASE / "outputs"

DATASET_DIR = BASE / "nnunet_raw" / "Dataset001_Follicles"
IMAGES_TR = DATASET_DIR / "imagesTr"
LABELS_TR = DATASET_DIR / "labelsTr"

CATEGORIES = ["PCO", "Normal", "Dominant_Follicle"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


def natural_key(path: Path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def category_prefix(category: str) -> str:
    return {
        "PCO": "pco",
        "Normal": "normal",
        "Dominant_Follicle": "dominant",
    }[category]


def find_images(folder: Path):
    return sorted(
        [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=natural_key,
    )


IMAGES_TR.mkdir(parents=True, exist_ok=True)
LABELS_TR.mkdir(parents=True, exist_ok=True)

prepared = 0
missing_masks = []
shape_mismatches = []
mapping_rows = []

for category in CATEGORIES:
    image_dir = IMAGE_ROOT / category
    mask_dir = MASK_ROOT / category

    print("\n" + "=" * 60)
    print(f"Category: {category}")
    print("=" * 60)

    if not image_dir.exists():
        print(f"WARNING: image folder not found: {image_dir}")
        continue

    if not mask_dir.exists():
        print(f"WARNING: mask folder not found: {mask_dir}")
        continue

    image_files = find_images(image_dir)
    print(f"Images found: {len(image_files)}")

    prefix = category_prefix(category)

    for idx, image_path in enumerate(image_files, start=1):
        mask_path = mask_dir / f"{image_path.stem}_biomedparse_mask.png"

        if not mask_path.exists():
            missing_masks.append(str(mask_path))
            print(f"  MISSING MASK: {image_path.name}")
            continue

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if image.size != mask.size:
            shape_mismatches.append(
                f"{image_path} | image={image.size}, mask={mask.size}"
            )
            print(
                f"  SIZE MISMATCH: {image_path.name} "
                f"image={image.size}, mask={mask.size}"
            )
            continue

        mask_arr = np.asarray(mask)
        binary_mask = (mask_arr > 0).astype(np.uint8)
        mask_binary_img = Image.fromarray(binary_mask)

        case_id = f"{prefix}_{idx:04d}"

        out_image = IMAGES_TR / f"{case_id}_0000.png"
        out_label = LABELS_TR / f"{case_id}.png"

        image.save(out_image)
        mask_binary_img.save(out_label)

        mapping_rows.append({
            "case_id": case_id,
            "category": category,
            "source_image": str(image_path),
            "source_mask": str(mask_path),
            "nnunet_image": str(out_image),
            "nnunet_label": str(out_label),
        })

        prepared += 1
        print(f"  OK: {image_path.name} -> {case_id}")

dataset_json = {
    "channel_names": {
        "0": "ultrasound"
    },
    "labels": {
        "background": 0,
        "follicle": 1
    },
    "numTraining": prepared,
    "file_ending": ".png"
}

with open(DATASET_DIR / "dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset_json, f, indent=2)

with open(
    DATASET_DIR / "case_mapping.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "case_id",
            "category",
            "source_image",
            "source_mask",
            "nnunet_image",
            "nnunet_label",
        ],
    )
    writer.writeheader()
    writer.writerows(mapping_rows)

if missing_masks:
    with open(
        DATASET_DIR / "missing_masks.txt",
        "w",
        encoding="utf-8",
    ) as f:
        for item in missing_masks:
            f.write(item + "\n")

if shape_mismatches:
    with open(
        DATASET_DIR / "shape_mismatches.txt",
        "w",
        encoding="utf-8",
    ) as f:
        for item in shape_mismatches:
            f.write(item + "\n")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"Prepared cases: {prepared}")
print(f"Missing masks: {len(missing_masks)}")
print(f"Shape mismatches: {len(shape_mismatches)}")
print(f"Dataset folder: {DATASET_DIR}")
