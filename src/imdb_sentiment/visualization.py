"""Static, exportable experiment plots; no interactive display requirement."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion(matrix, path):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    ax.set(xlabel="Predicted", ylabel="True", title="Test confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_learning(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for key in ["train_loss", "val_loss"]:
        axes[0].plot(history.epoch, history[key], marker="o", label=key)
    axes[0].set(xlabel="Epoch", ylabel="BCE loss")
    axes[0].legend()
    axes[1].plot(history.epoch, history.val_f1, marker="o", label="Validation F1")
    axes[1].set(xlabel="Epoch", ylabel="F1", ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_lengths(bundle, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, frame in bundle.splits():
        ax.hist(frame.token_count, bins=50, alpha=0.4, label=name, density=True)
    ax.set(xlabel="Word tokens before truncation", ylabel="Density", title="Review length distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_comparison(table, histories: dict, directory):
    fig, ax = plt.subplots(figsize=(9, 4))
    table.set_index("model")[["accuracy", "precision", "recall", "f1", "macro_f1"]].plot.bar(ax=ax, ylim=(0, 1), rot=0)
    ax.set(ylabel="Test metric", title="Measured model comparison")
    fig.tight_layout()
    fig.savefig(directory / "metrics_bar_chart.png", dpi=140)
    plt.close(fig)
    for metric, filename in [("val_loss", "rnn_lstm_loss_curves.png"), ("val_f1", "rnn_lstm_f1_curves.png")]:
        if not histories:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, history in histories.items():
            ax.plot(history.epoch, history[metric], marker="o", label=f"{name}: {metric}")
            if metric == "val_loss":
                ax.plot(history.epoch, history.train_loss, linestyle="--", label=f"{name}: train_loss")
        ax.set(xlabel="Epoch", ylabel=metric)
        if metric == "val_f1":
            ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / filename, dpi=140)
        plt.close(fig)
