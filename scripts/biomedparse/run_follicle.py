import sys
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

BIOMEDPARSE_DIR = Path("/content/BiomedParse")

sys.path.insert(0, str(BIOMEDPARSE_DIR))
os.chdir(BIOMEDPARSE_DIR)

from PIL import Image, ImageDraw
import json
import numpy as np
import torch
import cv2

from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image

BASE = Path("/content/drive/MyDrive/biomedparse_project")

IMAGE_PATH = BASE / "images" / "PCO" / "PCO_1.jpg"
OUTPUT_DIR = BASE / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


print("Model loaded")


# --------------------------
# Load image
# --------------------------

image = Image.open(IMAGE_PATH).convert("RGB")

width, height = image.size

print("Image size:", width, height)


# --------------------------
# Text prompt
# --------------------------

prompts = [
    "ovarian follicle"
]


# --------------------------
# Inference
# --------------------------

print("Running inference...")

with torch.no_grad():
    pred_masks = interactive_infer_image(
        model,
        image,
        prompts,
    )


mask = pred_masks[0]

print(
    "Mask range:",
    float(mask.min()),
    float(mask.max())
)


# --------------------------
# Binary segmentation
# --------------------------

threshold = 0.5

binary = (mask > threshold).astype(np.uint8)


# save mask
mask_path = OUTPUT_DIR / "PCO_1_biomedparse_mask.png"

Image.fromarray(
    binary * 255
).save(mask_path)


print("Mask saved:", mask_path)


# --------------------------
# Connected components
# → separate follicles
# --------------------------

num_labels, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
)


boxes = []


for component_id in range(1, num_labels):

    x = stats[
        component_id,
        cv2.CC_STAT_LEFT
    ]

    y = stats[
        component_id,
        cv2.CC_STAT_TOP
    ]

    w = stats[
        component_id,
        cv2.CC_STAT_WIDTH
    ]

    h = stats[
        component_id,
        cv2.CC_STAT_HEIGHT
    ]

    area = stats[
        component_id,
        cv2.CC_STAT_AREA
    ]


    # ignore tiny noise
    if area < 30:
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


print(
    "Detected components:",
    len(boxes)
)


# --------------------------
# Save JSON
# --------------------------

json_path = (
    OUTPUT_DIR /
    "PCO_1_biomedparse.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        boxes,
        f,
        indent=2,
    )


print(
    "JSON saved:",
    json_path
)


# --------------------------
# Visualization
# --------------------------

visual = image.copy()

draw = ImageDraw.Draw(visual)


for i, item in enumerate(
    boxes,
    start=1,
):

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


visual_path = (
    OUTPUT_DIR /
    "PCO_1_biomedparse_boxes.jpg"
)

visual.save(visual_path)


print(
    "Visualization saved:",
    visual_path
)