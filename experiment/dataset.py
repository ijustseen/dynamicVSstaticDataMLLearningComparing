"""PyTorch datasets for static and dynamic emotion recognition."""

import json
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMG_SIZE, LABEL_MAP_FILE, SEQUENCE_LENGTH


def get_transforms(is_training=True):
    """Get image transforms for training/evaluation."""
    if is_training:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def load_splits(splits_file):
    """Load train/val/test split assignments from file."""
    splits = {}
    with open(splits_file, "r") as f:
        for line in f:
            key, split = line.strip().split("\t")
            splits[key] = split
    return splits


def load_label_map(label_map_file=LABEL_MAP_FILE):
    """Load label mappings saved during data preparation."""
    with open(label_map_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    label_to_idx = {str(k): int(v) for k, v in data["label_to_idx"].items()}
    idx_to_label = {int(k): str(v) for k, v in data["idx_to_label"].items()}
    return label_to_idx, idx_to_label


class StaticEmotionDataset(Dataset):
    """Dataset for static (single image) emotion recognition."""

    def __init__(self, data_dir, splits_file, split="train", label_map_file=LABEL_MAP_FILE):
        """
        Args:
            data_dir: Path to processed/static/ directory
            splits_file: Path to splits.txt
            split: 'train', 'val', or 'test'
        """
        self.data_dir = Path(data_dir)
        self.is_training = split == "train"
        self.transform = get_transforms(self.is_training)
        self.label_to_idx, self.idx_to_label = load_label_map(label_map_file)

        splits = load_splits(splits_file)

        self.samples = []  # List of (image_path, label)

        for emo_name, emo_idx in self.label_to_idx.items():
            emo_dir = self.data_dir / emo_name
            if not emo_dir.exists():
                continue
            for img_file in sorted(emo_dir.glob("*.png")):
                key = img_file.stem
                if splits.get(key) == split:
                    self.samples.append((str(img_file), emo_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        return image, label


class DynamicEmotionDataset(Dataset):
    """Dataset for dynamic (video sequence) emotion recognition."""

    def __init__(self, data_dir, splits_file, split="train", label_map_file=LABEL_MAP_FILE):
        """
        Args:
            data_dir: Path to processed/dynamic/ directory
            splits_file: Path to splits.txt
            split: 'train', 'val', or 'test'
        """
        self.data_dir = Path(data_dir)
        self.is_training = split == "train"
        self.transform = get_transforms(self.is_training)
        self.label_to_idx, self.idx_to_label = load_label_map(label_map_file)

        splits = load_splits(splits_file)

        self.samples = []  # List of (sequence_dir, label)

        for emo_name, emo_idx in self.label_to_idx.items():
            emo_dir = self.data_dir / emo_name
            if not emo_dir.exists():
                continue
            for seq_dir in sorted(emo_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                key = seq_dir.name
                if splits.get(key) == split:
                    self.samples.append((str(seq_dir), emo_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_dir, label = self.samples[idx]
        seq_dir = Path(seq_dir)

        frames = []
        frame_files = sorted(seq_dir.glob("frame_*.png"))

        for frame_file in frame_files[:SEQUENCE_LENGTH]:
            image = cv2.imread(str(frame_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transform(image)
            frames.append(image)

        # Pad if necessary
        while len(frames) < SEQUENCE_LENGTH:
            frames.insert(0, frames[0].clone())

        # Stack: (SEQUENCE_LENGTH, C, H, W)
        sequence = torch.stack(frames)
        return sequence, label
