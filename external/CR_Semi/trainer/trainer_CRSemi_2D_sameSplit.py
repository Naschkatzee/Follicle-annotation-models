"""
start from cd external\CR_Semi
python -m trainer.trainer_CRSemi_2D
!python -m trainer.trainer_CRSemi_2D_sameSplit
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataop.dataset_2d import FollicleDataset2D
from models.asym_backbone_2D import asym_model_2D
from models.Rend_CRSemi_2D import PointRend, PointHead
from utils.sampling_point2 import point_sample

import json

from torch.utils.data import DataLoader, Subset

import csv


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
        "/content/drive/MyDrive/biomedparse_project/nnunet_raw/Dataset001_Follicles/imagesTr"
    )

    labels_dir = Path(
        "/content/drive/MyDrive/biomedparse_project/nnunet_raw/Dataset001_Follicles/labelsTr"
    )
    

    output_dir = Path(
        "/content/drive/MyDrive/biomedparse_project/crsemi_2d_fold0_final"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset = FollicleDataset2D(
        images_dir=images_dir,
        labels_dir=labels_dir
    )

    split_path = Path(
    "/content/drive/MyDrive/biomedparse_project/nnunet_preprocessed/Dataset001_Follicles/splits_final.json"
)

    with open(split_path, "r") as f:
        splits = json.load(f)

    train_ids = set(splits[0]["train"])
    val_ids = set(splits[0]["val"])

    all_case_ids = []

    for i in range(len(dataset)):
        _, _, case_id = dataset[i]
        all_case_ids.append(case_id)

    train_indices = [
        i for i, case_id in enumerate(all_case_ids)
        if case_id in train_ids
    ]

    val_indices = [
        i for i, case_id in enumerate(all_case_ids)
        if case_id in val_ids
    ]

    print("Matched train:", len(train_indices))
    print("Matched val:", len(val_indices))

    missing_train = train_ids - set(all_case_ids)
    missing_val = val_ids - set(all_case_ids)

    print("Missing train IDs:", missing_train)
    print("Missing val IDs:", missing_val)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)


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

    n_epochs = 300

    best_dice = -1.0

    start_epoch = 0

    history_path = output_dir / "training_history.csv"

    if not history_path.exists():
        with open(history_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "train_dice",
                "val_loss",
                "val_dice",
            ])

    latest_path = output_dir / "checkpoint_latest.pth"
    best_path = output_dir / "checkpoint_best.pth"

    if latest_path.exists():
        checkpoint = torch.load(
            latest_path,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = checkpoint["epoch"]

        best_dice = checkpoint.get("best_dice")

        if best_dice is None and best_path.exists():
            best_checkpoint = torch.load(
                best_path,
                map_location=device,
                weights_only=False
            )

            best_dice = best_checkpoint.get(
                "val_dice",
                -1.0
            )

        if best_dice is None:
            best_dice = checkpoint.get(
                "val_dice",
                -1.0
            )

        print(
            f"Resuming from epoch {start_epoch}, "
            f"best Dice={best_dice:.4f}"
        )


    for epoch in range(start_epoch, n_epochs):

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

        with open(history_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                mean_train_loss,
                mean_train_dice,
                mean_val_loss,
                mean_val_dice,
                ])

        print()
        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={mean_train_loss:.4f}, "
            f"train_dice={mean_train_dice:.4f}, "
            f"val_loss={mean_val_loss:.4f}, "
            f"val_dice={mean_val_dice:.4f}"
        )

        latest_path = output_dir / "checkpoint_latest.pth"

        if mean_val_dice > best_dice:
            best_dice = mean_val_dice

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": mean_train_loss,
                    "train_dice": mean_train_dice,
                    "val_loss": mean_val_loss,
                    "val_dice": mean_val_dice,
                    "best_dice": best_dice,
                },
                best_path
            )

            print(f"New best model saved: Dice={best_dice:.4f}")


        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": mean_train_loss,
                "train_dice": mean_train_dice,
                "val_loss": mean_val_loss,
                "val_dice": mean_val_dice,
                "best_dice": best_dice,
            },
            latest_path
        )
        
        if (epoch + 1) % 25 == 0:
            periodic_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pth"

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": mean_train_loss,
                    "train_dice": mean_train_dice,
                    "val_loss": mean_val_loss,
                    "val_dice": mean_val_dice,
                    "best_dice": best_dice,
                },
                periodic_path
                )

            print(f"Periodic checkpoint saved: epoch {epoch + 1}")
        
        print ()

    print("Training finished.")
    print("Best validation Dice:", best_dice)


if __name__ == "__main__":
    train()