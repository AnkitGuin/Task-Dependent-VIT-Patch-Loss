"""Checkpoint files record plain tensor state dictionaries, no pickle reload by default."""

import torch


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
