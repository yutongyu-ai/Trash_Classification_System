from torchvision import models
import torch.nn as nn


def get_model(hidden_size=64, num_classes=6):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, hidden_size),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(hidden_size, num_classes)
    )
    return model


