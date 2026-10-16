"""
python external\CR_Semi\test_dataset_2d.py
"""

from pathlib import Path

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import ImageOps


def resize_and_pad(image, mask, target_size=(1024, 768)):
    """
    target_size = (target_height, target_width)
    """

    target_h, target_w = target_size

    orig_w, orig_h = image.size

    scale = min(
        target_w / orig_w,
        target_h / orig_h
    )

    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    image = image.resize(
        (new_w, new_h),
        resample=Image.BILINEAR
    )

    mask = mask.resize(
        (new_w, new_h),
        resample=Image.NEAREST
    )

    pad_w = target_w - new_w
    pad_h = target_h - new_h

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    image = ImageOps.expand(
        image,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=0
    )

    mask = ImageOps.expand(
        mask,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=0
    )

    return image, mask


class FollicleDataset2D(Dataset):
    def __init__(self, images_dir, labels_dir):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)

        self.image_paths = sorted(self.images_dir.glob("*_0000.png"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        case_id = image_path.name.replace("_0000.png", "")
        label_path = self.labels_dir / f"{case_id}.png"

        image = Image.open(image_path).convert("L")
        mask = Image.open(label_path).convert("L")

        image, mask = resize_and_pad(
            image,
            mask,
            target_size=(1024, 768)
        )

        image = np.array(image, dtype=np.float32)
        mask = np.array(mask, dtype=np.uint8)

        # Normalize image to [0, 1]
        image = image / 255.0

        # Ensure binary mask: 0 = background, 1 = follicle
        mask = (mask > 0).astype(np.int64)

        # [H, W] -> [1, H, W]
        image = torch.from_numpy(image).unsqueeze(0)

        # mask stays [H, W]
        mask = torch.from_numpy(mask)

        return image, mask, case_id