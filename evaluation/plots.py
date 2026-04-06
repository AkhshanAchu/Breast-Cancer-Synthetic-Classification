from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, classes, tag, output_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {tag}")
    plt.tight_layout()
    plt.savefig(str(Path(output_dir) / f"cm_{tag}.png"))
    plt.close()


def plot_training_curves(h_real, h_aug, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for h, label, color in [
        (h_real, "Real Only", "steelblue"),
        (h_aug, "Real+Synthetic", "darkorange"),
    ]:
        axes[0].plot(h["val_acc"], label=label, color=color)
        axes[1].plot(h["val_loss"], label=label, color=color)
    axes[0].set_title("Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(str(Path(output_dir) / "training_curves.png"))
    plt.close()


def plot_accuracy_bar(acc_real, acc_aug, output_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        ["Real Only", "Real + Synthetic (GAN)"],
        [acc_real, acc_aug],
        color=["steelblue", "darkorange"],
        width=0.4,
    )
    for bar, acc in zip(bars, [acc_real, acc_aug]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{acc:.4f}",
            ha="center", va="bottom", fontweight="bold",
        )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Best Validation Accuracy")
    ax.set_title("GAN Augmentation — Accuracy Impact")
    plt.tight_layout()
    plt.savefig(str(Path(output_dir) / "accuracy_comparison.png"))
    plt.close()
