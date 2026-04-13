"""Create a single comparison plot for training curves (static vs dynamic).

Reads:
    - results/latest/history_static.json
    - results/latest/history_dynamic.json

Writes:
    - results/latest/training_compare.png

Usage:
  python plot_compare.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from config import LATEST_RESULTS_DIR


def _load_history(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_training_comparison(history_static: dict, history_dynamic: dict, save_path: Path) -> None:
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss ---
    epochs_s = list(range(1, len(history_static.get("train_loss", [])) + 1))
    epochs_d = list(range(1, len(history_dynamic.get("train_loss", [])) + 1))

    ax_loss.plot(epochs_s, history_static.get("train_loss", []), label="Static Train Loss")
    ax_loss.plot(epochs_s, history_static.get("val_loss", []), label="Static Val Loss")
    ax_loss.plot(epochs_d, history_dynamic.get("train_loss", []), label="Dynamic Train Loss")
    ax_loss.plot(epochs_d, history_dynamic.get("val_loss", []), label="Dynamic Val Loss")

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training & Validation Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    # --- Accuracy ---
    ax_acc.plot(epochs_s, history_static.get("train_acc", []), label="Static Train Acc")
    ax_acc.plot(epochs_s, history_static.get("val_acc", []), label="Static Val Acc")
    ax_acc.plot(epochs_d, history_dynamic.get("train_acc", []), label="Dynamic Train Acc")
    ax_acc.plot(epochs_d, history_dynamic.get("val_acc", []), label="Dynamic Val Acc")

    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Training & Validation Accuracy")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    history_static_path = LATEST_RESULTS_DIR / "history_static.json"
    history_dynamic_path = LATEST_RESULTS_DIR / "history_dynamic.json"

    if not history_static_path.exists():
        raise FileNotFoundError(
            f"Missing {history_static_path}. Train the static model first to generate history_static.json."
        )
    if not history_dynamic_path.exists():
        raise FileNotFoundError(
            f"Missing {history_dynamic_path}. Train the dynamic model first to generate history_dynamic.json."
        )

    history_static = _load_history(history_static_path)
    history_dynamic = _load_history(history_dynamic_path)

    out_path = LATEST_RESULTS_DIR / "training_compare.png"
    plot_training_comparison(history_static, history_dynamic, out_path)
    print(f"Comparison plot saved to {out_path}")


if __name__ == "__main__":
    main()
