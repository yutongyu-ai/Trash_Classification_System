import torch

from models.resnet18 import get_model


def test_get_model_output_shape():
    model = get_model(num_classes=6)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 224, 224))
    assert output.shape == (1, 6)


def test_get_model_respects_num_classes():
    model = get_model(num_classes=10)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 224, 224))
    assert output.shape == (2, 10)
