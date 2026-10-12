"""
Batch annotation of ovarian ultrasound images with a free multimodal
OpenRouter model.

Run from the project root, for example:

    python .\follicle-annotation-models\scripts\qwen\run_vlm_batch.py

Before running in CMD:

    set OPENROUTER_API_KEY=your_key

Before running in PowerShell:

    $env:OPENROUTER_API_KEY="your_key"
"""

import os
import base64
import mimetypes
import time
import json
import re
from pathlib import Path

from openai import OpenAI, RateLimitError, APIError
from PIL import Image, ImageDraw

MODEL_ID = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

BASE = Path(r"D:\Uni_Bamberg\8_Semester\Bachelor thesis\Annotating Borna dataset")
IMAGE_ROOT = BASE / "images"
OUTPUT_ROOT = BASE / "outputs_nemotron"

CATEGORIES = [
    "PCO",
    "Normal",
    "Dominant_Follicle",
]

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_RETRIES = 5
REQUEST_DELAY_SECONDS = 2

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY is not set.\n"
        "CMD: set OPENROUTER_API_KEY=your_key\n"
        "PowerShell: $env:OPENROUTER_API_KEY=\"your_key\""
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

PROMPT = """
You are annotating an ovarian ultrasound image.

Detect ONLY clearly visible ovarian follicles.

A follicle should appear as a round or oval dark fluid-filled structure
inside the ovary.

For every visible follicle, return one bounding box.

Return ONLY valid JSON in this exact structure:

[
  {
    "label": "follicle",
    "bbox": [x1, y1, x2, y2]
  }
]

Rules:
- The only allowed label is "follicle".
- Do NOT annotate the entire ultrasound image.
- Do NOT annotate the whole ovary.
- Do NOT annotate text.
- Do NOT annotate borders.
- Do NOT annotate measurement markers.
- Do NOT annotate interface elements.
- Do NOT annotate black background.
- Do NOT annotate other anatomical structures.
- Create one separate bounding box for every visible follicle.
- bbox must have the format [x1, y1, x2, y2].
- Use absolute pixel coordinates relative to the original image.
- If no follicle is clearly visible, return [].
- Return JSON only.
- Do not use Markdown.
- Do not include explanations.
"""


def image_to_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        mime_type = "image/jpeg"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def natural_key(path: Path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end >= start:
        text = text[start:end + 1]
    return text.strip()


def validate_annotations(data, image_width: int, image_height: int):
    if not isinstance(data, list):
        raise ValueError("Response is not a JSON list.")

    cleaned = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("label") != "follicle":
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = map(int, bbox)
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
            continue
        cleaned.append({"label": "follicle", "bbox": [x1, y1, x2, y2]})
    return cleaned


def draw_boxes(image_path: Path, boxes, output_path: Path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for i, item in enumerate(boxes, start=1):
        x1, y1, x2, y2 = item["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 15)), f"follicle {i}", fill="red")
    image.save(output_path)


def annotate_image(image_path: Path):
    image_data = image_to_data_url(image_path)
    with Image.open(image_path) as img:
        image_width, image_height = img.size

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            print(f"  Request attempt {attempt + 1}/{MAX_RETRIES}...")

            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data},
                            },
                            {
                                "type": "text",
                                "text": PROMPT,
                            },
                        ],
                    }
                ],
                temperature=0,
            )

            if (
                response is None
                or not response.choices
                or response.choices[0].message is None
                or response.choices[0].message.content is None
            ):
                raise RuntimeError("Model returned empty content.")

            raw_text = response.choices[0].message.content
            if not isinstance(raw_text, str) or not raw_text.strip():
                raise RuntimeError("Model returned blank text content.")

            cleaned_text = clean_json_text(raw_text)
            data = json.loads(cleaned_text)

            annotations = validate_annotations(
                data,
                image_width=image_width,
                image_height=image_height,
            )

            return annotations, raw_text

        except RateLimitError as e:
            last_error = e
            wait_time = 30 * (attempt + 1)
            print(f"  Rate limit. Waiting {wait_time} seconds...")
            time.sleep(wait_time)

        except APIError as e:
            last_error = e
            wait_time = 15 * (attempt + 1)
            print(f"  API error: {e}")
            print(f"  Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

        except (RuntimeError, json.JSONDecodeError, TypeError, ValueError) as e:
            last_error = e
            wait_time = 10 * (attempt + 1)
            print(f"  Bad response: {e}")
            print(f"  Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    raise RuntimeError(
        f"Model failed after all retry attempts. Last error: {last_error}"
    )


processed = 0
skipped = 0
failed = 0
failed_images = []

for category in CATEGORIES:
    input_dir = IMAGE_ROOT / category
    output_dir = OUTPUT_ROOT / category
    output_dir.mkdir(parents=True, exist_ok=True)

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
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ],
        key=natural_key,
    )

    print(f"Images found: {len(image_files)}")

    for idx, image_path in enumerate(image_files, start=1):
        stem = image_path.stem
        json_path = output_dir / f"{stem}_nemotron.json"
        boxes_path = output_dir / f"{stem}_nemotron_boxes.jpg"
        raw_path = output_dir / f"{stem}_nemotron_raw.txt"

        print(f"\n[{idx}/{len(image_files)}] {image_path.name}")

        if json_path.exists() and boxes_path.exists() and raw_path.exists():
            print("  SKIP: already processed")
            skipped += 1
            continue

        try:
            annotations, raw_text = annotate_image(image_path)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(annotations, f, indent=2)

            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(raw_text)

            draw_boxes(image_path, annotations, boxes_path)

            print(f"  -> follicles: {len(annotations)}")
            processed += 1

            if REQUEST_DELAY_SECONDS > 0:
                time.sleep(REQUEST_DELAY_SECONDS)

        except Exception as e:
            failed += 1
            error_message = (
                f"{category}/{image_path.name}: "
                f"{type(e).__name__}: {e}"
            )
            failed_images.append(error_message)
            print(f"  ERROR: {error_message}")


OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
failed_log = OUTPUT_ROOT / "failed_images.txt"

with open(failed_log, "w", encoding="utf-8") as f:
    for item in failed_images:
        f.write(item + "\n")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print("Processed:", processed)
print("Skipped:", skipped)
print("Failed:", failed)
print("Failure log:", failed_log)
