"""Optional scheduling/regularization without changing the benchmark trainer."""

import time
import pandas as pd
import torch
from torch import nn
from .evaluation import compute_metrics, save_json
from .trainer import EarlyStopping, evaluate
from .visualization import plot_learning


def train_optimized(model, loaders: dict, config: dict, directory, device) -> dict:
    net = config["neural"]
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=net["learning_rate"], weight_decay=net["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=net["scheduler_gamma"]) if net["scheduler"] == "exponential" else None
    stopping = EarlyStopping(net["early_stopping_patience"], net["early_stopping_min_delta"])
    criterion = nn.BCEWithLogitsLoss()
    history, best_epoch = [], 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for epoch in range(1, net["epochs"] + 1):
        model.train()
        total_loss, count = 0.0, 0
        learning_rate = optimizer.param_groups[0]["lr"]
        for ids, lengths, labels in loaders["train"]:
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(ids.to(device), lengths), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), net["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item() * len(labels)
            count += len(labels)
        val_loss, truth, logits = evaluate(model, loaders["val"], device)
        metrics = compute_metrics(truth, logits >= 0)
        history.append({"epoch": epoch, "learning_rate": learning_rate, "train_loss": total_loss / count, "val_loss": val_loss,
                        **{f"val_{k}": v for k, v in metrics.items() if k != "confusion_matrix"}})
        improved, stop = stopping.update(val_loss)
        if improved:
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "config": config, "epoch": epoch,
                        "vocab_size": model.embedding.num_embeddings, "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict() if scheduler else None,
                        "validation_metrics": {**metrics, "loss": val_loss}}, directory / "best_model.pt")
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        print(f"epoch {epoch}/{net['epochs']}: lr={learning_rate:.6g}, train_loss={total_loss/count:.4f}, val_loss={val_loss:.4f}, val_accuracy={metrics['accuracy']:.4f}, val_f1={metrics['f1']:.4f}", flush=True)
        if stop:
            break
        if scheduler:
            scheduler.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    checkpoint = torch.load(directory / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    loss, truth, logits = evaluate(model, loaders["val"], device)
    save_json(directory / "validation_metrics.json", {**compute_metrics(truth, logits >= 0), "loss": loss})
    plot_learning(pd.DataFrame(history), directory / "learning_curves.png")
    return {"best_epoch": best_epoch, "epochs_trained": len(history), "checkpoint_restored": True,
            "training_seconds": seconds, "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "timing_scope": "Training, validation, early stopping and checkpoint/history writes; excludes preprocessing and final evaluation"}
