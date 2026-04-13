"""
Evaluation script: compare static and dynamic models on the test set.
Produces metrics (accuracy, precision, recall, F1), confusion matrices, and comparison table.

Usage:
    python evaluate.py
    python evaluate.py --variant opt
    python evaluate.py --static-ckpt path/to/static.pth --dynamic-ckpt path/to/dynamic.pth --tag my_tag

Outputs:
    - without --tag: writes to results/latest/ (overwrites "latest")
    - with --tag: writes to results/experiments/<tag>/
"""

import json
import time
from datetime import datetime
from pathlib import Path
import argparse

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
    LABEL_MAP_FILE,
    PROCESSED_DIR,
    EXPERIMENTS_DIR,
    LATEST_RESULTS_DIR,
)
from dataset import DynamicEmotionDataset, StaticEmotionDataset, load_label_map
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


def evaluate_model(model_type, model, dataloader, device, class_names, output_suffix, output_dir: Path):
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
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        cm,
        class_names,
        f"Confusion Matrix — {model_type.capitalize()}{output_suffix} Model",
        output_dir / f"confusion_{model_type}{output_suffix}.png",
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
    parser = argparse.ArgumentParser(description="Evaluate models on the test set")
    parser.add_argument(
        "--variant",
        choices=["base", "opt"],
        default="base",
        help="Which checkpoints/results to use: base (best_static.pth) or opt (best_static_opt.pth)",
    )
    parser.add_argument(
        "--static-ckpt",
        type=str,
        default=None,
        help="Optional explicit path to static checkpoint (.pth). Overrides --variant static checkpoint.",
    )
    parser.add_argument(
        "--dynamic-ckpt",
        type=str,
        default=None,
        help="Optional explicit path to dynamic checkpoint (.pth). Overrides --variant dynamic checkpoint.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional experiment tag. If provided, outputs go to results/experiments/<tag>/.",
    )
    args = parser.parse_args()
    output_suffix = "" if args.variant == "base" else "_opt"

    if args.tag:
        output_dir = EXPERIMENTS_DIR / args.tag
    else:
        output_dir = LATEST_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Model Evaluation & Comparison")
    print(f"Device: {DEVICE}")
    print(f"Variant: {args.variant}")
    if args.tag:
        print(f"Tag: {args.tag}")
        meta_path = output_dir / "meta.json"
        meta = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "variant": args.variant,
            "device": str(DEVICE),
            "static_ckpt": args.static_ckpt,
            "dynamic_ckpt": args.dynamic_ckpt,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Meta saved to {meta_path}")
    print(f"{'=' * 60}")

    if not LABEL_MAP_FILE.exists():
        print("ERROR: label_map.json not found. Run prepare_data.py first.")
        return

    label_to_idx, idx_to_label = load_label_map(LABEL_MAP_FILE)
    class_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    num_classes = len(class_names)
    splits_file = PROCESSED_DIR / "splits.txt"

    results = {}

    # --- Static model ---
    static_ckpt = (
        Path(args.static_ckpt)
        if args.static_ckpt
        else CHECKPOINTS_DIR / f"best_static{'' if args.variant == 'base' else '_opt'}.pth"
    )
    if static_ckpt.exists():
        model_static = get_model("static", num_classes=num_classes, pretrained=False).to(DEVICE)
        checkpoint = torch.load(static_ckpt, map_location=DEVICE, weights_only=True)
        model_static.load_state_dict(checkpoint["model_state_dict"])

        test_ds = StaticEmotionDataset(
            PROCESSED_DIR / "static", splits_file, split="test", label_map_file=LABEL_MAP_FILE
        )
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE_STATIC, shuffle=False, num_workers=4)
        print(f"\nStatic test samples: {len(test_ds)}")

        results["static"] = evaluate_model(
            "static", model_static, test_loader, DEVICE, class_names, output_suffix, output_dir
        )
    else:
        print(f"\nWARNING: Static model checkpoint not found: {static_ckpt}")

    # --- Dynamic model ---
    dynamic_ckpt = (
        Path(args.dynamic_ckpt)
        if args.dynamic_ckpt
        else CHECKPOINTS_DIR / f"best_dynamic{'' if args.variant == 'base' else '_opt'}.pth"
    )
    if dynamic_ckpt.exists():
        model_dynamic = get_model("dynamic", num_classes=num_classes, pretrained=False).to(DEVICE)
        checkpoint = torch.load(dynamic_ckpt, map_location=DEVICE, weights_only=True)
        model_dynamic.load_state_dict(checkpoint["model_state_dict"])

        test_ds = DynamicEmotionDataset(
            PROCESSED_DIR / "dynamic", splits_file, split="test", label_map_file=LABEL_MAP_FILE
        )
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE_DYNAMIC, shuffle=False, num_workers=4
        )
        print(f"\nDynamic test samples: {len(test_ds)}")

        results["dynamic"] = evaluate_model(
            "dynamic", model_dynamic, test_loader, DEVICE, class_names, output_suffix, output_dir
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
    save_results = {}
    for key, val in results.items():
        save_results[key] = {k: v for k, v in val.items() if k != "report"}
    results_path = output_dir / f"evaluation_results{output_suffix}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
