"""
start from cd external\CR_Semi
python trainer\trainer_CRSemi_2D.py
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataop.dataset_2d import FollicleDataset2D
from models.asym_backbone_2D import asym_model_2D
from models.Rend_CRSemi_2D import PointRend, PointHead
from utils.sampling_point2 import point_sample


def dice_score_from_logits(logits, target, eps=1e-6):
    pred = torch.argmax(logits, dim=1)

    pred_fg = (pred == 1).float()
    target_fg = (target == 1).float()

    intersection = (pred_fg * target_fg).sum(dim=(1, 2))

    denominator = (
        pred_fg.sum(dim=(1, 2))
        + target_fg.sum(dim=(1, 2))
    )

    dice = (2.0 * intersection + eps) / (denominator + eps)

    return dice.mean()


def build_model(device):
    backbone = asym_model_2D(
        n_channels=1,
        n_classes=2
    )

    head = PointHead(
        in_c=98,
        num_classes=2
    )

    model = PointRend(
        backbone=backbone,
        head=head
    )

    return model.to(device)


def train():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    images_dir = Path(
        r"D:\Uni_Bamberg\8_Semester\Bachelor thesis\Annotating Borna dataset\follicle-annotation-models\nnunet_data\Dataset001_Follicles\imagesTr"
    )

    labels_dir = Path(
        r"D:\Uni_Bamberg\8_Semester\Bachelor thesis\Annotating Borna dataset\follicle-annotation-models\nnunet_data\Dataset001_Follicles\labelsTr"
    )

    output_dir = Path(
        r"D:\Uni_Bamberg\8_Semester\Bachelor thesis\Annotating Borna dataset\follicle-annotation-models\external\CR_Semi\results_2d"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset = FollicleDataset2D(
        images_dir=images_dir,
        labels_dir=labels_dir
    )

    generator = torch.Generator().manual_seed(42)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print("Train cases:", len(train_dataset))
    print("Validation cases:", len(val_dataset))

    model = build_model(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    n_epochs = 5

    best_dice = -1.0

    for epoch in range(n_epochs):

        # -------------------------
        # TRAIN
        # -------------------------
        model.train()

        train_losses = []
        train_dices = []

        train_bar = tqdm(
            train_loader,
            desc=f"Train {epoch + 1}/{n_epochs}"
        )

        for images, masks, case_ids in train_bar:

            images = images.to(
                device,
                non_blocking=True
            )

            masks = masks.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad()

            result = model(images)

            coarse = result["coarse"]

            coarse_up = F.interpolate(
                coarse,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            seg_loss = F.cross_entropy(
                coarse_up,
                masks
            )

            gt_points = point_sample(
                masks.float().unsqueeze(1),
                result["points"],
                mode="nearest",
                align_corners=False
            )

            gt_points = (
                gt_points
                .squeeze(1)
                .long()
            )

            points_loss = F.cross_entropy(
                result["rend"],
                gt_points
            )

            loss = seg_loss + points_loss

            loss.backward()

            optimizer.step()

            dice = dice_score_from_logits(
                coarse_up.detach(),
                masks
            )

            train_losses.append(
                loss.item()
            )

            train_dices.append(
                dice.item()
            )

            train_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                dice=f"{dice.item():.4f}"
            )

        # -------------------------
        # VALIDATION
        # -------------------------
        model.eval()

        val_losses = []
        val_dices = []

        with torch.no_grad():

            val_bar = tqdm(
                val_loader,
                desc=f"Val {epoch + 1}/{n_epochs}"
            )

            for images, masks, case_ids in val_bar:

                images = images.to(device)
                masks = masks.to(device)

                # Validation uses coarse prediction only.
                # This avoids PointRend inference complications
                # during the first baseline test.
                backbone_result = model.backbone(images)

                coarse = backbone_result["coarse"]

                coarse_up = F.interpolate(
                    coarse,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

                loss = F.cross_entropy(
                    coarse_up,
                    masks
                )

                dice = dice_score_from_logits(
                    coarse_up,
                    masks
                )

                val_losses.append(
                    loss.item()
                )

                val_dices.append(
                    dice.item()
                )

                val_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    dice=f"{dice.item():.4f}"
                )

        mean_train_loss = sum(train_losses) / len(train_losses)
        mean_train_dice = sum(train_dices) / len(train_dices)

        mean_val_loss = sum(val_losses) / len(val_losses)
        mean_val_dice = sum(val_dices) / len(val_dices)

        print()
        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={mean_train_loss:.4f}, "
            f"train_dice={mean_train_dice:.4f}, "
            f"val_loss={mean_val_loss:.4f}, "
            f"val_dice={mean_val_dice:.4f}"
        )

        latest_path = output_dir / "checkpoint_latest.pth"

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": mean_train_loss,
                "train_dice": mean_train_dice,
                "val_loss": mean_val_loss,
                "val_dice": mean_val_dice,
            },
            latest_path
        )

        if mean_val_dice > best_dice:

            best_dice = mean_val_dice

            best_path = output_dir / "checkpoint_best.pth"

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": mean_train_loss,
                    "train_dice": mean_train_dice,
                    "val_loss": mean_val_loss,
                    "val_dice": mean_val_dice,
                },
                best_path
            )

            print(
                f"New best model saved: "
                f"Dice={best_dice:.4f}"
            )

        print()

    print("Training finished.")
    print("Best validation Dice:", best_dice)


if __name__ == "__main__":
    train()