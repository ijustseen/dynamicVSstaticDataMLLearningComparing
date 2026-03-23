"""
Evaluation script: compare static and dynamic models on the test set.
Produces metrics (accuracy, precision, recall, F1), confusion matrices, and comparison table.

Usage:
    python evaluate.py
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BATCH_SIZE_DYNAMIC,
    BATCH_SIZE_STATIC,
    CHECKPOINTS_DIR,
    DEVICE,
    EMOTIONS,
    PROCESSED_DIR,
    RESULTS_DIR,
)
from dataset import DynamicEmotionDataset, StaticEmotionDataset
from models import get_model


def predict(model, dataloader, device):
    """Run inference and return all predictions and true labels."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="  Predicting", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def measure_fps(model, dataloader, device, num_batches=20):
    """Measure inference speed (FPS)."""
    model.eval()
    total_samples = 0
    start_time = time.time()

    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            inputs = inputs.to(device)
            _ = model(inputs)
            total_samples += inputs.size(0)

    elapsed = time.time() - start_time
    fps = total_samples / elapsed if elapsed > 0 else 0
    return fps


def plot_confusion_matrix(cm, class_names, title, save_path):
    """Plot and save a confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=14)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate_model(model_type, model, dataloader, device, class_names):
    """Evaluate a single model and return metrics dict."""
    print(f"\nEvaluating {model_type.upper()} model...")

    preds, labels = predict(model, dataloader, device)
    fps = measure_fps(model, dataloader, device)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    cm = confusion_matrix(labels, preds)

    report = classification_report(labels, preds, target_names=class_names, zero_division=0)

    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print(f"  FPS:       {fps:.1f}")
    print(f"\n  Classification Report:\n{report}")

    # Save confusion matrix plot
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        cm,
        class_names,
        f"Confusion Matrix — {model_type.capitalize()} Model",
        RESULTS_DIR / f"confusion_{model_type}.png",
    )

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fps": fps,
        "confusion_matrix": cm.tolist(),
        "report": report,
    }


def main():
    print(f"\n{'=' * 60}")
    print("Model Evaluation & Comparison")
    print(f"Device: {DEVICE}")
    print(f"{'=' * 60}")

    class_names = [EMOTIONS[i] for i in range(len(EMOTIONS))]
    splits_file = PROCESSED_DIR / "splits.txt"

    results = {}

    # --- Static model ---
    static_ckpt = CHECKPOINTS_DIR / "best_static.pth"
    if static_ckpt.exists():
        model_static = get_model("static", pretrained=False).to(DEVICE)
        checkpoint = torch.load(static_ckpt, map_location=DEVICE, weights_only=True)
        model_static.load_state_dict(checkpoint["model_state_dict"])

        test_ds = StaticEmotionDataset(PROCESSED_DIR / "static", splits_file, split="test")
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE_STATIC, shuffle=False, num_workers=4)
        print(f"\nStatic test samples: {len(test_ds)}")

        results["static"] = evaluate_model("static", model_static, test_loader, DEVICE, class_names)
    else:
        print(f"\nWARNING: Static model checkpoint not found: {static_ckpt}")

    # --- Dynamic model ---
    dynamic_ckpt = CHECKPOINTS_DIR / "best_dynamic.pth"
    if dynamic_ckpt.exists():
        model_dynamic = get_model("dynamic", pretrained=False).to(DEVICE)
        checkpoint = torch.load(dynamic_ckpt, map_location=DEVICE, weights_only=True)
        model_dynamic.load_state_dict(checkpoint["model_state_dict"])

        test_ds = DynamicEmotionDataset(PROCESSED_DIR / "dynamic", splits_file, split="test")
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE_DYNAMIC, shuffle=False, num_workers=4
        )
        print(f"\nDynamic test samples: {len(test_ds)}")

        results["dynamic"] = evaluate_model(
            "dynamic", model_dynamic, test_loader, DEVICE, class_names
        )
    else:
        print(f"\nWARNING: Dynamic model checkpoint not found: {dynamic_ckpt}")

    # --- Comparison table ---
    if "static" in results and "dynamic" in results:
        print(f"\n{'=' * 60}")
        print("COMPARISON")
        print(f"{'=' * 60}")
        print(f"{'Metric':<15} {'Static':>10} {'Dynamic':>10} {'Diff':>10}")
        print("-" * 45)
        for metric in ["accuracy", "precision", "recall", "f1", "fps"]:
            s = results["static"][metric]
            d = results["dynamic"][metric]
            diff = d - s
            sign = "+" if diff >= 0 else ""
            fmt = ".4f" if metric != "fps" else ".1f"
            print(f"{metric:<15} {s:>10{fmt}} {d:>10{fmt}} {sign}{diff:>9{fmt}}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_results = {}
    for key, val in results.items():
        save_results[key] = {k: v for k, v in val.items() if k != "report"}
    with open(RESULTS_DIR / "evaluation_results.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
