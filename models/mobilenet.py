# models/mobilenet.py

import torch.nn as nn
from torchvision import models


def get_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")

    for param in model.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Linear(1280, num_classes)

    return model