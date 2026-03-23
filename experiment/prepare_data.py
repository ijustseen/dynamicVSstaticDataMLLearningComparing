"""
Data preparation: parse CK+ dataset into static (peak frames) and dynamic (sequences) formats.
"""

import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from config import (
    CK_EMOTIONS_DIR,
    CK_IMAGES_DIR,
    CK_LABEL_MAP,
    IMG_SIZE,
    PROCESSED_DIR,
    SEQUENCE_LENGTH,
)


def load_ck_plus_samples():
    """
    Parse CK+ directory structure and return list of samples.
    Each sample: (subject, sequence, emotion_label, image_paths)
    """
    samples = []

    if not CK_EMOTIONS_DIR.exists():
        raise FileNotFoundError(
            f"Emotion labels directory not found: {CK_EMOTIONS_DIR}\n"
            "Please download CK+ and place it in data/CK+/"
        )

    for subject_dir in sorted(CK_EMOTIONS_DIR.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject = subject_dir.name

        for seq_dir in sorted(subject_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            sequence = seq_dir.name

            # Read emotion label file (only last frame has emotion label)
            emotion_files = sorted(seq_dir.glob("*.txt"))
            if not emotion_files:
                continue  # No emotion label for this sequence

            with open(emotion_files[0], "r") as f:
                raw_label = int(float(f.read().strip()))

            if raw_label not in CK_LABEL_MAP:
                continue  # Skip neutral (0) or unknown labels

            emotion_label = CK_LABEL_MAP[raw_label]

            # Get corresponding image sequence
            img_dir = CK_IMAGES_DIR / subject / sequence
            if not img_dir.exists():
                continue

            image_paths = sorted(img_dir.glob("*.png"))
            if not image_paths:
                image_paths = sorted(img_dir.glob("*.jpg"))
            if not image_paths:
                continue

            samples.append(
                {
                    "subject": subject,
                    "sequence": sequence,
                    "emotion": emotion_label,
                    "image_paths": [str(p) for p in image_paths],
                }
            )

    return samples


def detect_and_crop_face(image, face_cascade):
    """
    Detect face in image and return cropped + resized face.
    Returns None if no face detected.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

    if len(faces) == 0:
        # Fallback: use the entire image
        return cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    # Use the largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Add margin
    margin = int(0.1 * max(w, h))
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(image.shape[1], x + w + margin)
    y2 = min(image.shape[0], y + h + margin)

    face = image[y1:y2, x1:x2]
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    return face


def prepare_static_data(samples, face_cascade, output_dir):
    """
    Extract peak frame (last frame) from each sequence.
    Save as: output_dir/<emotion>/<subject>_<sequence>.png
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from config import EMOTIONS

    for emo_idx, emo_name in EMOTIONS.items():
        (output_dir / emo_name).mkdir(exist_ok=True)

    count = 0
    for sample in tqdm(samples, desc="Preparing static data"):
        peak_frame_path = sample["image_paths"][-1]
        image = cv2.imread(peak_frame_path)
        if image is None:
            continue

        face = detect_and_crop_face(image, face_cascade)
        if face is None:
            continue

        emo_name = EMOTIONS[sample["emotion"]]
        filename = f"{sample['subject']}_{sample['sequence']}.png"
        cv2.imwrite(str(output_dir / emo_name / filename), face)
        count += 1

    print(f"Static data: {count} images saved to {output_dir}")
    return count


def prepare_dynamic_data(samples, face_cascade, output_dir):
    """
    Extract last N frames from each sequence.
    Save as: output_dir/<emotion>/<subject>_<sequence>/frame_00.png, frame_01.png, ...
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from config import EMOTIONS

    for emo_idx, emo_name in EMOTIONS.items():
        (output_dir / emo_name).mkdir(exist_ok=True)

    count = 0
    for sample in tqdm(samples, desc="Preparing dynamic data"):
        paths = sample["image_paths"]

        # Take last SEQUENCE_LENGTH frames; pad with first frame if too short
        if len(paths) >= SEQUENCE_LENGTH:
            selected_paths = paths[-SEQUENCE_LENGTH:]
        else:
            # Pad by repeating the first frame
            padding = [paths[0]] * (SEQUENCE_LENGTH - len(paths))
            selected_paths = padding + paths

        emo_name = EMOTIONS[sample["emotion"]]
        seq_dir = output_dir / emo_name / f"{sample['subject']}_{sample['sequence']}"
        seq_dir.mkdir(exist_ok=True)

        valid = True
        for i, frame_path in enumerate(selected_paths):
            image = cv2.imread(frame_path)
            if image is None:
                valid = False
                break
            face = detect_and_crop_face(image, face_cascade)
            if face is None:
                valid = False
                break
            cv2.imwrite(str(seq_dir / f"frame_{i:02d}.png"), face)

        if valid:
            count += 1
        else:
            # Clean up partial sequence
            shutil.rmtree(seq_dir, ignore_errors=True)

    print(f"Dynamic data: {count} sequences saved to {output_dir}")
    return count


def create_splits(samples, output_file):
    """
    Create subject-independent train/val/test splits.
    Save split assignments to a text file.
    """
    from config import RANDOM_SEED, TEST_RATIO, TRAIN_RATIO, VAL_RATIO

    np.random.seed(RANDOM_SEED)

    # Get unique subjects
    subjects = sorted(set(s["subject"] for s in samples))
    np.random.shuffle(subjects)

    n = len(subjects)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train : n_train + n_val])
    test_subjects = set(subjects[n_train + n_val :])

    splits = {}
    for sample in samples:
        key = f"{sample['subject']}_{sample['sequence']}"
        if sample["subject"] in train_subjects:
            splits[key] = "train"
        elif sample["subject"] in val_subjects:
            splits[key] = "val"
        else:
            splits[key] = "test"

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for key, split in sorted(splits.items()):
            f.write(f"{key}\t{split}\n")

    train_count = sum(1 for v in splits.values() if v == "train")
    val_count = sum(1 for v in splits.values() if v == "val")
    test_count = sum(1 for v in splits.values() if v == "test")
    print(f"Splits: train={train_count}, val={val_count}, test={test_count}")
    print(f"Subjects: train={len(train_subjects)}, val={len(val_subjects)}, test={len(test_subjects)}")

    return splits


def main():
    print("=" * 60)
    print("CK+ Dataset Preparation")
    print("=" * 60)

    # Load Haar cascade for face detection
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Parse CK+
    print("\nParsing CK+ dataset...")
    samples = load_ck_plus_samples()
    print(f"Found {len(samples)} labeled sequences")

    if len(samples) == 0:
        print("ERROR: No samples found. Check the CK+ dataset path.")
        return

    # Print emotion distribution
    from config import EMOTIONS

    emotion_counts = {}
    for s in samples:
        emo = EMOTIONS[s["emotion"]]
        emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
    print("\nEmotion distribution:")
    for emo, count in sorted(emotion_counts.items()):
        print(f"  {emo}: {count}")

    # Create splits
    print("\nCreating train/val/test splits...")
    splits = create_splits(samples, PROCESSED_DIR / "splits.txt")

    # Prepare static data
    print("\nPreparing static (peak frame) data...")
    prepare_static_data(samples, face_cascade, PROCESSED_DIR / "static")

    # Prepare dynamic data
    print(f"\nPreparing dynamic (sequence of {SEQUENCE_LENGTH} frames) data...")
    prepare_dynamic_data(samples, face_cascade, PROCESSED_DIR / "dynamic")

    print("\n" + "=" * 60)
    print("Done! Data saved to:", PROCESSED_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
