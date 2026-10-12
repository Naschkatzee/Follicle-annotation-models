"""
python test_qwen.py
"""

import os
import base64
import mimetypes
import time

from openai import OpenAI, RateLimitError


MODEL_ID = "qwen/qwen3.8-27b:free"

image_path = r"D:\Uni_Bamberg\8_Semester\Bachelor thesis\Annotating Borna dataset\images\Normal\Normal_41.jpg"


# -------------------------
# OpenRouter client
# -------------------------

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY is not set.\n"
        "In CMD run:\n"
        "set OPENROUTER_API_KEY=your_key"
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)


# -------------------------
# Convert local image to Base64
# -------------------------

def image_to_data_url(path):
    mime_type, _ = mimetypes.guess_type(path)

    if mime_type is None:
        mime_type = "image/jpeg"

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


image_data = image_to_data_url(image_path)


# -------------------------
# Annotation prompt
# -------------------------

prompt = """
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


# -------------------------
# Send request with retries
# -------------------------

response = None

for attempt in range(5):
    try:
        print(f"Request attempt {attempt + 1}/5...")

        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            temperature=0,
        )

        break

    except RateLimitError:
        wait_time = 10 * (attempt + 1)

        print(
            f"Rate limit. Waiting {wait_time} seconds..."
        )

        time.sleep(wait_time)


if response is None:
    raise RuntimeError(
        "Qwen is still rate-limited after 5 attempts."
    )


# -------------------------
# Print result
# -------------------------

result = response.choices[0].message.content

print("\nQwen result:\n")
print(result)