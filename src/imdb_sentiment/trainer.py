"""Small explicit PyTorch training loop with restored best checkpoints."""

import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from .evaluation import compute_metrics, save_json


def evaluate(model, loader, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, count, truths, logits = 0.0, 0, [], []
    criterion = nn.BCEWithLogitsLoss()
    with torch.inference_mode():
        for ids, lengths, labels in loader:
            labels = labels.to(device)
            scores = model(ids.to(device), lengths)
            total_loss += criterion(scores, labels).item() * len(labels)
            count += len(labels)
            truths.extend(labels.cpu().tolist())
            logits.extend(scores.cpu().tolist())
    if count == 0:
        raise ValueError("Cannot evaluate an empty loader")
    return total_loss / count, np.asarray(truths), np.asarray(logits)


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.bad_epochs = float("inf"), 0

    def update(self, value: float) -> tuple[bool, bool]:
        if not np.isfinite(value):
            raise RuntimeError("Non-finite validation loss")
        improved = value < self.best - self.min_delta
        if improved:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


def train_neural(model, loaders: dict, config: dict, directory, device):
    from .visualization import plot_learning
    net = config["neural"]
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=net["learning_rate"], weight_decay=net["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()
    stopping = EarlyStopping(net["early_stopping_patience"], net["early_stopping_min_delta"])
    history, best_epoch = [], 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for epoch in range(1, net["epochs"] + 1):
        model.train()
        total_loss, count = 0.0, 0
        for ids, lengths, labels in loaders["train"]:
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(ids.to(device), lengths), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), net["grad_clip"])
            optimizer.step()
            total_loss += loss.item() * len(labels)
            count += len(labels)
        val_loss, truth, logits = evaluate(model, loaders["val"], device)
        metrics = compute_metrics(truth, logits >= 0)
        row = {"epoch": epoch, "train_loss": total_loss / count, "val_loss": val_loss,
               **{f"val_{key}": value for key, value in metrics.items() if key != "confusion_matrix"}}
        history.append(row)
        improved, stop = stopping.update(val_loss)
        if improved:
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "config": config, "epoch": epoch,
                        "vocab_size": model.embedding.num_embeddings}, directory / "best_model.pt")
        print(f"epoch {epoch}: train_loss={row['train_loss']:.4f}, val_loss={val_loss:.4f}, val_f1={metrics['f1']:.4f}", flush=True)
        if stop:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    checkpoint = torch.load(directory / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    loss, truth, logits = evaluate(model, loaders["val"], device)
    save_json(directory / "validation_metrics.json", {**compute_metrics(truth, logits >= 0), "loss": loss})
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(directory / "history.csv", index=False)
    plot_learning(history_frame, directory / "learning_curves.png")
    return model, {"best_epoch": best_epoch, "checkpoint_restored": True, "training_seconds": seconds,
                   "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
                   "timing_scope": "Training, validation, early stopping and checkpoint writes; excludes feature preparation and final test"}
