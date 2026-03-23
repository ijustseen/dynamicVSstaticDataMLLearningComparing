"""
Training script for static and dynamic emotion recognition models.
Usage:
    python train.py --model static
    python train.py --model dynamic
"""

import argparse
import json
import time
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
    LEARNING_RATE,
    NUM_EPOCHS,
    PROCESSED_DIR,
    RESULTS_DIR,
    WEIGHT_DECAY,
)
from dataset import DynamicEmotionDataset, StaticEmotionDataset
from models import get_model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="  Training", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model. Returns average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="  Validating", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

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
    """Plot and save training/validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
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
    parser = argparse.ArgumentParser(description="Train emotion recognition model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["static", "dynamic"],
        help="Model type: 'static' (ResNet-18) or 'dynamic' (ResNet-18 + LSTM)",
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    args = parser.parse_args()

    model_type = args.model
    print(f"\n{'=' * 60}")
    print(f"Training {model_type.upper()} model")
    print(f"Device: {DEVICE}")
    print(f"{'=' * 60}\n")

    # Create directories
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets
    splits_file = PROCESSED_DIR / "splits.txt"
    if not splits_file.exists():
        print("ERROR: splits.txt not found. Run prepare_data.py first.")
        return

    if model_type == "static":
        data_dir = PROCESSED_DIR / "static"
        train_ds = StaticEmotionDataset(data_dir, splits_file, split="train")
        val_ds = StaticEmotionDataset(data_dir, splits_file, split="val")
        batch_size = BATCH_SIZE_STATIC
    else:
        data_dir = PROCESSED_DIR / "dynamic"
        train_ds = DynamicEmotionDataset(data_dir, splits_file, split="train")
        val_ds = DynamicEmotionDataset(data_dir, splits_file, split="val")
        batch_size = BATCH_SIZE_DYNAMIC

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")

    if len(train_ds) == 0:
        print("ERROR: No training samples found. Check dataset preparation.")
        return

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Create model
    model = get_model(model_type, pretrained=True).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Training loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"\nStarting training for {args.epochs} epochs...\n")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = CHECKPOINTS_DIR / f"best_{model_type}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Save last model
    torch.save(
        {"model_state_dict": model.state_dict()},
        CHECKPOINTS_DIR / f"last_{model_type}.pth",
    )

    # Save history
    with open(RESULTS_DIR / f"history_{model_type}.json", "w") as f:
        json.dump(history, f, indent=2)

    # Plot
    plot_training_history(history, RESULTS_DIR / f"training_{model_type}.png")


if __name__ == "__main__":
    main()
