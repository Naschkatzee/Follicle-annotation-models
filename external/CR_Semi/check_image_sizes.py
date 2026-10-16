"""
python external\CR_Semi\check_image_sizes.py

Target: 768 × 1024
=> original image
→ resize with preserved aspect ratio
→ pad to fixed size
→ tensor
"""

from pathlib import Path
from PIL import Image
from collections import Counter

images_dir = Path(
    r"D:\Uni_Bamberg\8_Semester\Bachelor thesis\Annotating Borna dataset\follicle-annotation-models\nnunet_data\Dataset001_Follicles\imagesTr"
)

sizes = []

for path in sorted(images_dir.glob("*_0000.png")):
    with Image.open(path) as img:
        sizes.append(img.size)  # (W, H)

print("Number of images:", len(sizes))
print("Unique sizes:", len(set(sizes)))

counter = Counter(sizes)

print("\nMost common sizes:")
for size, count in counter.most_common(10):
    print(size, count)

widths = [w for w, h in sizes]
heights = [h for w, h in sizes]

print("\nWidth min/max:", min(widths), max(widths))
print("Height min/max:", min(heights), max(heights))