from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_losses(histories: dict, output: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, history in histories.items():
        ax.plot(history["train"], label=f"{name} train")
        ax.plot(history["val"], linestyle="--", label=f"{name} val")
    ax.set(xlabel="Époque", ylabel="MSE normalisée", title="Courbes d'apprentissage")
    ax.legend(); ax.grid(alpha=0.2); fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def plot_forecasts(targets, predictions: dict, output: Path, sample: int = 0):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(targets[sample], color="black", linewidth=2, label="Réel")
    for name in ["Transformer", "MLP", "Naive", "SeasonalNaive"]:
        if name in predictions:
            ax.plot(predictions[name][sample], label=name, alpha=0.8)
    ax.set(xlabel="Pas futur", ylabel="OT", title="Prévision sur une origine de test")
    ax.legend(ncol=3); ax.grid(alpha=0.2); fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def plot_errors(table, output: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, rows in table.groupby("model"):
        ax.plot(rows["horizon"], rows["MAE"], marker="o", label=model)
    ax.set(xlabel="Horizon", ylabel="MAE", title="Erreur par horizon")
    ax.legend(ncol=2, fontsize=8); ax.grid(alpha=0.2); fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)

