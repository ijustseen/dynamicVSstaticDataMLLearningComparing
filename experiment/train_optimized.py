"""
Optimized training script for emotion recognition.
Adds anti-overfitting controls while keeping the original train.py untouched.

Usage:
    python train_optimized.py --model static --epochs 30 --lr 1e-4
    python train_optimized.py --model dynamic --epochs 30 --lr 1e-4
"""

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BATCH_SIZE_DYNAMIC,
    BATCH_SIZE_STATIC,
    CHECKPOINTS_DIR,
    DEVICE,
    LATEST_RESULTS_DIR,
    LABEL_MAP_FILE,
    LEARNING_RATE,
    NUM_EPOCHS,
    PROCESSED_DIR,
    RESULTS_DIR,
    WEIGHT_DECAY,
)
from dataset import DynamicEmotionDataset, StaticEmotionDataset, load_label_map
from models import get_model


def train_one_epoch(model, dataloader, criterion, optimizer, device, amp_enabled, scaler):
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="  Training", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device, amp_enabled):
    """Validate and return average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for inputs, labels in tqdm(dataloader, desc="  Validating", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def plot_training_history(history, save_path):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-", label="Train Accuracy")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Optimized training for emotion recognition")
    parser.add_argument("--model", type=str, required=True, choices=["static", "dynamic"])
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    model_type = args.model
    amp_enabled = DEVICE.type == "cuda"

    print(f"\n{'=' * 60}")
    print(f"Training OPTIMIZED {model_type.upper()} model")
    print(f"Device: {DEVICE}")
    print(f"AMP enabled: {amp_enabled}")
    print(f"{'=' * 60}\n")

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / "runs" / f"{run_id}_{model_type}_opt"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run artifacts will also be saved to: {run_dir}")

    splits_file = PROCESSED_DIR / "splits.txt"
    if not splits_file.exists():
        print("ERROR: splits.txt not found. Run prepare_data.py first.")
        return
    if not LABEL_MAP_FILE.exists():
        print("ERROR: label_map.json not found. Run prepare_data.py first.")
        return

    label_to_idx, idx_to_label = load_label_map(LABEL_MAP_FILE)
    num_classes = len(label_to_idx)
    print(f"Classes ({num_classes}): {', '.join(idx_to_label[i] for i in sorted(idx_to_label))}")

    if model_type == "static":
        data_dir = PROCESSED_DIR / "static"
        train_ds = StaticEmotionDataset(data_dir, splits_file, split="train", label_map_file=LABEL_MAP_FILE)
        val_ds = StaticEmotionDataset(data_dir, splits_file, split="val", label_map_file=LABEL_MAP_FILE)
        batch_size = BATCH_SIZE_STATIC
    else:
        data_dir = PROCESSED_DIR / "dynamic"
        train_ds = DynamicEmotionDataset(data_dir, splits_file, split="train", label_map_file=LABEL_MAP_FILE)
        val_ds = DynamicEmotionDataset(data_dir, splits_file, split="val", label_map_file=LABEL_MAP_FILE)
        batch_size = BATCH_SIZE_DYNAMIC

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")

    if len(train_ds) == 0:
        print("ERROR: No training samples found. Check dataset preparation.")
        return

    pin_memory = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    model = get_model(model_type, num_classes=num_classes, pretrained=True).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_acc_ckpt = -1.0
    no_improve_epochs = 0

    print(f"\nStarting optimized training for {args.epochs} epochs...\n")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, amp_enabled, scaler
        )
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE, amp_enabled)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        improved_loss = val_loss < (best_val_loss - args.min_delta)
        improved_acc = val_acc > best_val_acc_ckpt

        if improved_acc:
            best_val_acc_ckpt = val_acc
            checkpoint_path = CHECKPOINTS_DIR / f"best_{model_type}_opt.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "label_smoothing": args.label_smoothing,
                    "weight_decay": args.weight_decay,
                    "lr": args.lr,
                },
                checkpoint_path,
            )
            print(f"  Best-by-acc checkpoint saved (val_acc={val_acc:.4f}, val_loss={val_loss:.4f})")

        if improved_loss:
            best_val_loss = val_loss
            best_val_acc = max(best_val_acc, val_acc)
            no_improve_epochs = 0

            checkpoint_path = CHECKPOINTS_DIR / f"best_{model_type}_opt_loss.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "label_smoothing": args.label_smoothing,
                    "weight_decay": args.weight_decay,
                    "lr": args.lr,
                },
                checkpoint_path,
            )
            print(f"  Best-by-loss checkpoint saved (val_loss={val_loss:.4f}, val_acc={val_acc:.4f})")
        else:
            no_improve_epochs += 1
            print(f"  No val_loss improvement: {no_improve_epochs}/{args.early_stopping_patience}")

        if no_improve_epochs >= args.early_stopping_patience:
            print("  Early stopping triggered.")
            break

    elapsed = time.time() - start_time
    print(f"\nOptimized training completed in {elapsed:.1f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best validation accuracy (checkpoint): {best_val_acc_ckpt:.4f}")

    torch.save(
        {"model_state_dict": model.state_dict()},
        CHECKPOINTS_DIR / f"last_{model_type}_opt.pth",
    )

    history_path = LATEST_RESULTS_DIR / f"history_{model_type}_opt.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_path = LATEST_RESULTS_DIR / f"training_{model_type}_opt.png"
    plot_training_history(history, plot_path)
    print(f"History saved to {history_path}")

    best_ckpt_path = CHECKPOINTS_DIR / f"best_{model_type}_opt.pth"
    best_loss_ckpt_path = CHECKPOINTS_DIR / f"best_{model_type}_opt_loss.pth"
    last_ckpt_path = CHECKPOINTS_DIR / f"last_{model_type}_opt.pth"
    for artifact_path in (history_path, plot_path, best_ckpt_path, best_loss_ckpt_path, last_ckpt_path):
        if artifact_path.exists():
            shutil.copy2(artifact_path, run_dir / artifact_path.name)
        else:
            print(f"WARNING: artifact not found for run archive: {artifact_path}")


if __name__ == "__main__":
    main()
