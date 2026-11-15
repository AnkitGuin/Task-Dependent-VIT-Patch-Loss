"""Early stopping selects best validation cross-entropy, not test performance."""

import copy
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .evaluate import evaluate
from .metrics import count_params
from ..seed import seed_all
from ..theory.linear_algebra import alignment


def fit_model(model, train, validation, test, config, seed, device="cpu", signal=None):
    seed_all(seed)
    device = torch.device(device)
    model.to(device)
    xtr, ytr = train
    xv, yv = validation
    xt, yt = test
    if int(config.get("epochs", 8)) < 1:
        raise ValueError("At least one training epoch is required")
    for name, (images, labels) in zip(("train", "validation", "test"), (train, validation, test)):
        if len(images) == 0 or len(images) != len(labels) or not np.isfinite(images).all():
            raise ValueError(f"Invalid or nonfinite {name} data")

    def to_t(x, y):
        return torch.from_numpy(x).float(), torch.from_numpy(y).long()

    xtr, ytr = to_t(xtr, ytr)
    xv, yv = to_t(xv, yv)
    xt, yt = to_t(xt, yt)
    loader = DataLoader(
        TensorDataset(xtr, ytr),
        batch_size=int(config.get("batch_size", 32)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(config.get("lr", 0.001)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    steps = []
    best = float("inf")
    best_weights = None
    stale = 0
    best_epoch = 0
    start = time.perf_counter()
    for epoch in range(int(config.get("epochs", 8))):
        model.train()
        losses = []
        for images, labels in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = torch.nn.functional.cross_entropy(logits, labels.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        val = evaluate(model, xv, yv, device)
        if not np.isfinite(val["negative_log_likelihood"]):
            raise FloatingPointError("Nonfinite validation loss")
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "val_loss": val["negative_log_likelihood"],
            "val_acc": val["accuracy"],
        }
        if signal is not None and hasattr(model, "tokenizer"):
            v = np.asarray(signal).reshape(-1)
            if len(v) == model.tokenizer.matrix.shape[1]:
                row["row_alignment"] = alignment(v, model.tokenizer.matrix)
        steps.append(row)
        if row["val_loss"] < best - 1e-7:
            best = row["val_loss"]
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if stale >= int(config.get("patience", 5)):
            break
    model.load_state_dict(best_weights)
    metrics = evaluate(model, xt, yt, device)
    metrics.update(
        {
            "model_params": count_params(model),
            "trained_epochs": len(steps),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "optimizer_steps": len(loader) * len(steps),
            "best_validation_epoch": best_epoch,
            "wallclock_sec": time.perf_counter() - start,
        }
    )
    return metrics, steps, model
