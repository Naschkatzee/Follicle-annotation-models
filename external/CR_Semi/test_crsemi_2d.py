"""
python test_crsemi_2d.py
"""

import torch

from models.asym_backbone_2D import asym_model_2D
from models.Rend_CRSemi_2D import PointRend, PointHead

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

x = torch.randn(
    1,
    1,
    512,
    512
)

model.train()

with torch.no_grad():
    result = model(x)

for name, value in result.items():
    if torch.is_tensor(value):
        print(name, value.shape)
    else:
        print(name, type(value))