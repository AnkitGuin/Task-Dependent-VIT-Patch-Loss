"""Evaluation with no backward or test-set selection."""

import torch
from .metrics import classification_metrics


def evaluate(model, x, y, device="cpu", batch_size=128):
    model.eval()
    outputs = []
    with torch.no_grad():
        for chunk in x.split(batch_size):
            outputs.append(model(chunk.to(device)).cpu())
    return classification_metrics(torch.cat(outputs).numpy(), y.cpu().numpy())
