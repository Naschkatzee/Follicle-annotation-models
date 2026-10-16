import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sampling_point2 import sampling_points, point_sample


class PointHead(nn.Module):
    def __init__(self, in_c=98, num_classes=2, k=3, beta=0.75, num_points=70000):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Conv1d(in_c, 64, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(True),
            nn.Conv1d(64, 32, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(True),
            nn.Conv1d(32, num_classes, 1)
        )

        self.k = k
        self.beta = beta
        self.num_points = num_points

    def forward(self, x, p2_1, p2_2, out):

        if not self.training:
            return self.inference(x, p2_1, p2_2, out)

        points = sampling_points(
            out,
            (x.shape[-2] // 4) * (x.shape[-1] // 4),
            self.k,
            self.beta
        )

        coarse = point_sample(out, points, align_corners=False)
        fine_1 = point_sample(p2_1, points, align_corners=False)
        fine_2 = point_sample(p2_2, points, align_corners=False)

        feature_representation = torch.cat(
            [coarse, fine_1, fine_2],
            dim=1
        )

        rend = self.mlp(feature_representation)

        return {
            "rend": rend,
            "points": points
        }

    @torch.no_grad()
    def inference(self, x, p2_1, p2_2, out):

        while out.shape[-2:] != x.shape[-2:]:

            out = F.interpolate(
                out,
                scale_factor=2,
                mode="bilinear",
                align_corners=True
            )

            points_idx, points = sampling_points(
                out,
                self.num_points,
                training=self.training
            )

            coarse = point_sample(
                out,
                points,
                align_corners=False
            )

            fine_1 = point_sample(
                p2_1,
                points,
                align_corners=False
            )

            fine_2 = point_sample(
                p2_2,
                points,
                align_corners=False
            )

            feature_representation = torch.cat(
                [coarse, fine_1, fine_2],
                dim=1
            )

            rend = self.mlp(feature_representation)

            B, C, H, W = out.shape

            points_idx = points_idx.unsqueeze(1).expand(
                -1,
                C,
                -1
            )

            out = (
                out.reshape(B, C, -1)
                .scatter_(2, points_idx, rend)
                .view(B, C, H, W)
            )

        return {"fine": out}


class PointRend(nn.Module):
    def __init__(self, backbone, head, ema=False):
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.ema = ema

    def forward(self, x):

        result = self.backbone(x)

        if self.ema:
            return result

        result.update(
            self.head(
                x,
                result["p2_1"],
                result["p2_2"],
                result["coarse"]
            )
        )

        return result