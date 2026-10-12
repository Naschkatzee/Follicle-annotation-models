import os
import sys
import json
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

BIOMEDPARSE_DIR = Path("/content/BiomedParse")
sys.path.insert(0, str(BIOMEDPARSE_DIR))
os.chdir(BIOMEDPARSE_DIR)

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE = Path("/content/drive/MyDrive/biomedparse_project")

IMAGE_ROOT = BASE / "images"
OUTPUT_ROOT = BASE / "outputs"

CATEGORIES = [
    "PCO",
    "Normal",
    "Dominant_Follicle",
]

PROMPTS = ["ovarian follicle"]

MASK_THRESHOLD = 0.5
MIN_COMPONENT_AREA = 30


# --------------------------------------------------
# Load model once
# --------------------------------------------------

print("Loading BiomedParse...")

opt = load_opt_from_config_files(
    ["configs/biomedparse_inference.yaml"]
)

opt = init_distributed(opt)

pretrained_pth = "hf_hub:microsoft/BiomedParse"

model = (
    BaseModel(opt, build_model(opt))
    .from_pretrained(pretrained_pth)
    .eval()
    .cuda()
)

with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
        BIOMED_CLASSES + ["background"],
        is_eval=True,
    )

print("Model loaded successfully.")


# --------------------------------------------------
# Process one image
# --------------------------------------------------

def process_image(image_path: Path, output_dir: Path):

    stem = image_path.stem

    json_path = output_dir / f"{stem}_biomedparse.json"
    mask_path = output_dir / f"{stem}_biomedparse_mask.png"
    boxes_path = output_dir / f"{stem}_biomedparse_boxes.jpg"

    # Skip if all outputs already exist
    if json_path.exists() and mask_path.exists() and boxes_path.exists():
        print(f"SKIP: {image_path.name}")
        return "skipped"

    image = Image.open(image_path).convert("RGB")

    print(f"Processing: {image_path}")

    with torch.no_grad():
        pred_masks = interactive_infer_image(
            model,
            image,
            PROMPTS,
        )

    mask = pred_masks[0]

    # Convert to numpy if necessary
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    mask = np.squeeze(mask)

    binary = (mask > MASK_THRESHOLD).astype(np.uint8)

    # Save mask
    Image.fromarray(binary * 255).save(mask_path)

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    boxes = []

    for component_id in range(1, num_labels):

        x = stats[component_id, cv2.CC_STAT_LEFT]
        y = stats[component_id, cv2.CC_STAT_TOP]
        w = stats[component_id, cv2.CC_STAT_WIDTH]
        h = stats[component_id, cv2.CC_STAT_HEIGHT]
        area = stats[component_id, cv2.CC_STAT_AREA]

        if area < MIN_COMPONENT_AREA:
            continue

        boxes.append(
            {
                "label": "follicle",
                "bbox": [
                    int(x),
                    int(y),
                    int(x + w),
                    int(y + h),
                ],
                "area": int(area),
            }
        )

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(boxes, f, indent=2)

    # Draw boxes
    visual = image.copy()
    draw = ImageDraw.Draw(visual)

    for i, item in enumerate(boxes, start=1):

        x1, y1, x2, y2 = item["bbox"]

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3,
        )

        draw.text(
            (x1, max(0, y1 - 15)),
            f"follicle {i}",
            fill="red",
        )

    visual.save(boxes_path)

    print(
        f"  -> follicles: {len(boxes)} | "
        f"saved to {output_dir}"
    )

    return "processed"


# --------------------------------------------------
# Batch loop
# --------------------------------------------------

processed = 0
skipped = 0
failed = 0

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

for category in CATEGORIES:

    input_dir = IMAGE_ROOT / category
    output_dir = OUTPUT_ROOT / category

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 60)
    print(f"Category: {category}")
    print("=" * 60)

    if not input_dir.exists():
        print(f"WARNING: folder not found: {input_dir}")
        continue

    image_files = sorted(
        [
            p
            for p in input_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in extensions
        ]
    )

    print(f"Images found: {len(image_files)}")

    for idx, image_path in enumerate(
        image_files,
        start=1,
    ):

        print(
            f"\n[{idx}/{len(image_files)}] "
            f"{image_path.name}"
        )

        try:

            result = process_image(
                image_path,
                output_dir,
            )

            if result == "processed":
                processed += 1
            else:
                skipped += 1

        except Exception as e:

            failed += 1

            print(
                f"ERROR on {image_path.name}: "
                f"{type(e).__name__}: {e}"
            )


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print("Processed:", processed)
print("Skipped:", skipped)
print("Failed:", failed)