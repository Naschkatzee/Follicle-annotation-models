"""
python test_backbone_2d.py
"""

import torch

from models.asym_backbone_2D import asym_model_2D

model = asym_model_2D(
    n_channels=1,
    n_classes=2
)

x = torch.randn(
    1,
    1,
    512,
    512
)

with torch.no_grad():
    result = model(x)

for name, tensor in result.items():
    print(name, tensor.shape)