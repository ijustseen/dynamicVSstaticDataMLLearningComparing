"""Prepare static and dynamic datasets from Urdu-Multimodal video files."""

import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from config import (
    IMG_SIZE,
    LABEL_MAP_FILE,
    MIN_FACE_SIZE,
    PROCESSED_DIR,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    TEST_RATIO,
    TRAIN_RATIO,
    URDU_DATASET_DIR,
    URDU_METADATA_FILE,
    URDU_VIDEO_DIR,
    VAL_RATIO,
)


def normalize_label(label):
    """Normalize class labels to safe lowercase directory names."""
    return str(label).strip().lower().replace(" ", "_")


def load_samples_from_csv(metadata_file):
    """Load (sample_id, label, video_path) from train.csv and keep existing videos only."""
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_file}\n"
            "Place Urdu-Multimodal-Emotion-Dataset in data/Urdu-Multimodal-Emotion-Dataset/"
        )

    samples = []
    with open(metadata_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"id", "video_path", "label"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"train.csv must contain columns {sorted(required)}; got {reader.fieldnames}"
            )

        for row in reader:
            sample_id = str(row["id"]).strip()
            label = normalize_label(row["label"])
            rel_video = str(row["video_path"]).strip()
            video_file = URDU_DATASET_DIR / rel_video

            if not video_file.exists() and rel_video.startswith("video/"):
                video_file = URDU_VIDEO_DIR / Path(rel_video).name

            if not video_file.exists():
                continue

            samples.append(
                {
                    "id": sample_id,
                    "label": label,
                    "video_path": str(video_file),
                }
            )

    return samples


def detect_and_crop_face(image, face_cascade):
    """Detect and crop largest face; fallback to resized full frame."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
    )

    if len(faces) == 0:
        return cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    margin = int(0.1 * max(w, h))

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(image.shape[1], x + w + margin)
    y2 = min(image.shape[0], y + h + margin)

    face = image[y1:y2, x1:x2]
    return cv2.resize(face, (IMG_SIZE, IMG_SIZE))


def sample_frame_indices(frame_count, seq_len):
    """Generate seq_len indices uniformly across a clip."""
    if frame_count <= 0:
        return []
    if frame_count == 1:
        return [0] * seq_len
    points = np.linspace(0, frame_count - 1, num=seq_len)
    return [int(round(p)) for p in points]


def extract_frames(video_path, frame_indices):
    """Read specific frame indices from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    frames = []
    for idx in frame_indices:
        idx = max(0, min(idx, total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frames.append(frame)

    cap.release()
    return frames


def create_label_map(samples):
    """Create stable label-to-index mapping from present samples."""
    labels = sorted({s["label"] for s in samples})
    label_to_idx = {label: i for i, label in enumerate(labels)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    return label_to_idx, idx_to_label


def create_splits(samples, output_file):
    """Create stratified train/val/test split by label."""
    rng = np.random.default_rng(RANDOM_SEED)
    by_label = {}
    for s in samples:
        by_label.setdefault(s["label"], []).append(s)

    split_map = {}
    counts = {"train": 0, "val": 0, "test": 0}

    for label, group in by_label.items():
        order = np.arange(len(group))
        rng.shuffle(order)
        group = [group[i] for i in order]

        n = len(group)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        for i, sample in enumerate(group):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            split_map[sample["id"]] = split
            counts[split] += 1

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for sample_id in sorted(split_map.keys()):
            f.write(f"{sample_id}\t{split_map[sample_id]}\n")

    print(
        f"Splits: train={counts['train']}, val={counts['val']}, test={counts['test']}"
    )
    return split_map


def prepare_static_data(samples, face_cascade, output_dir):
    """Save one representative frame per video to static/<label>/<id>.png."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted({s["label"] for s in samples})
    for label in labels:
        (output_dir / label).mkdir(exist_ok=True)

    saved = 0
    for sample in tqdm(samples, desc="Preparing static data"):
        cap = cv2.VideoCapture(sample["video_path"])
        if not cap.isOpened():
            continue
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if frame_count <= 0:
            continue

        # Static baseline takes the center frame from each clip.
        center_idx = frame_count // 2
        frames = extract_frames(sample["video_path"], [center_idx])
        if not frames:
            continue

        face = detect_and_crop_face(frames[0], face_cascade)
        out_path = output_dir / sample["label"] / f"{sample['id']}.png"
        cv2.imwrite(str(out_path), face)
        saved += 1

    print(f"Static data: {saved} images saved to {output_dir}")
    return saved


def prepare_dynamic_data(samples, face_cascade, output_dir):
    """Save SEQUENCE_LENGTH frames per video to dynamic/<label>/<id>/frame_xx.png."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted({s["label"] for s in samples})
    for label in labels:
        (output_dir / label).mkdir(exist_ok=True)

    saved = 0
    for sample in tqdm(samples, desc="Preparing dynamic data"):
        cap = cv2.VideoCapture(sample["video_path"])
        if not cap.isOpened():
            continue
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if frame_count <= 0:
            continue

        frame_indices = sample_frame_indices(frame_count, SEQUENCE_LENGTH)
        frames = extract_frames(sample["video_path"], frame_indices)
        if not frames:
            continue

        seq_dir = output_dir / sample["label"] / sample["id"]
        seq_dir.mkdir(exist_ok=True)

        ok = True
        for i, frame in enumerate(frames):
            face = detect_and_crop_face(frame, face_cascade)
            if face is None:
                ok = False
                break
            cv2.imwrite(str(seq_dir / f"frame_{i:02d}.png"), face)

        while ok and len(list(seq_dir.glob("frame_*.png"))) < SEQUENCE_LENGTH:
            existing = sorted(seq_dir.glob("frame_*.png"))
            if not existing:
                ok = False
                break
            shutil.copy(existing[-1], seq_dir / f"frame_{len(existing):02d}.png")

        if ok:
            saved += 1
        else:
            shutil.rmtree(seq_dir, ignore_errors=True)

    print(f"Dynamic data: {saved} sequences saved to {output_dir}")
    return saved


def save_label_map(label_to_idx, idx_to_label):
    """Save label mappings used by train/evaluate/realtime."""
    LABEL_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label_to_idx": label_to_idx,
        "idx_to_label": {str(k): v for k, v in idx_to_label.items()},
    }
    with open(LABEL_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Label map saved to {LABEL_MAP_FILE}")


def main():
    print("=" * 60)
    print("Urdu-Multimodal Dataset Preparation")
    print("=" * 60)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    print("\nLoading metadata...")
    samples = load_samples_from_csv(URDU_METADATA_FILE)
    print(f"Found {len(samples)} usable samples with existing videos")

    if len(samples) == 0:
        print("ERROR: No samples found. Check dataset path and train.csv.")
        return

    label_to_idx, idx_to_label = create_label_map(samples)
    print("\nLabels:")
    for idx in sorted(idx_to_label.keys()):
        print(f"  {idx}: {idx_to_label[idx]}")
    save_label_map(label_to_idx, idx_to_label)

    print("\nCreating train/val/test splits...")
    create_splits(samples, PROCESSED_DIR / "splits.txt")

    print("\nPreparing static (single-frame) data...")
    prepare_static_data(samples, face_cascade, PROCESSED_DIR / "static")

    print(f"\nPreparing dynamic (sequence of {SEQUENCE_LENGTH} frames) data...")
    prepare_dynamic_data(samples, face_cascade, PROCESSED_DIR / "dynamic")

    print("\n" + "=" * 60)
    print("Done! Data saved to:", PROCESSED_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
