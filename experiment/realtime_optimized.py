"""
Optimized real-time emotion recognition.
Keeps original realtime.py untouched and adds speed-focused options.

Usage examples:
    python realtime_optimized.py --model static
    python realtime_optimized.py --model dynamic --seq-len 8 --infer-every 2 --detect-every 3
"""

import argparse
import time
from collections import deque

import cv2
import torch
from torchvision import transforms

from config import CHECKPOINTS_DIR, DEVICE, IMG_SIZE, LABEL_MAP_FILE, SEQUENCE_LENGTH
from dataset import load_label_map
from models import get_model


def get_inference_transform(img_size):
    """Transform for real-time inference."""
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class DynamicCachedPredictor:
    """Dynamic inference helper with feature caching for sliding windows."""

    def __init__(self, model, seq_len, device):
        self.model = model
        self.seq_len = seq_len
        self.device = device
        self.feature_buffer = deque(maxlen=seq_len)

    def clear(self):
        self.feature_buffer.clear()

    def predict(self, face_tensor, amp_enabled):
        """Predict emotion logits for one face tensor with cached features."""
        x = face_tensor.unsqueeze(0).to(self.device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            feat = self.model.feature_extractor(x).flatten(1).squeeze(0)
            self.feature_buffer.append(feat)

            if len(self.feature_buffer) < self.seq_len:
                first = self.feature_buffer[0]
                padded = [first] * (self.seq_len - len(self.feature_buffer)) + list(self.feature_buffer)
            else:
                padded = list(self.feature_buffer)

            seq = torch.stack(padded, dim=0).unsqueeze(0)
            lstm_out, _ = self.model.lstm(seq)
            logits = self.model.fc(lstm_out[:, -1, :])

        return logits


def select_checkpoint(model_type, prefer_optimized):
    """Choose checkpoint path while keeping backward compatibility."""
    candidates = []
    if prefer_optimized:
        candidates.append(CHECKPOINTS_DIR / f"best_{model_type}_opt.pth")
    candidates.append(CHECKPOINTS_DIR / f"best_{model_type}.pth")
    if not prefer_optimized:
        candidates.append(CHECKPOINTS_DIR / f"best_{model_type}_opt.pth")

    for ckpt in candidates:
        if ckpt.exists():
            return ckpt
    return None


def largest_face(faces):
    """Pick the largest detected face."""
    if len(faces) == 0:
        return None
    return max(faces, key=lambda b: b[2] * b[3])


def main():
    parser = argparse.ArgumentParser(description="Optimized real-time emotion recognition")
    parser.add_argument("--model", type=str, default="static", choices=["static", "dynamic"])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--infer-every", type=int, default=2)
    parser.add_argument("--detect-every", type=int, default=3)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--prefer-optimized-checkpoint", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (mainly for static model)")
    args = parser.parse_args()

    if not LABEL_MAP_FILE.exists():
        print("ERROR: label_map.json not found. Run prepare_data.py first.")
        return

    _, idx_to_label = load_label_map(LABEL_MAP_FILE)
    num_classes = len(idx_to_label)

    model_type = args.model
    seq_len = max(2, args.seq_len) if model_type == "dynamic" else SEQUENCE_LENGTH
    img_size = max(96, args.img_size)

    print(f"Loading optimized {model_type} model...")
    model = get_model(model_type, num_classes=num_classes, pretrained=False).to(DEVICE)

    ckpt_path = select_checkpoint(model_type, prefer_optimized=args.prefer_optimized_checkpoint)
    if ckpt_path is None:
        print("ERROR: checkpoint not found.")
        print(f"Checked: {CHECKPOINTS_DIR / f'best_{model_type}.pth'} and optional *_opt variant")
        return

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if args.compile and hasattr(torch, "compile") and model_type == "static":
        try:
            model = torch.compile(model)
            print("torch.compile enabled for static model")
        except Exception as exc:
            print(f"torch.compile skipped: {exc}")

    print(f"Model loaded from {ckpt_path}")
    print(f"Device: {DEVICE}")

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    transform = get_inference_transform(img_size)
    amp_enabled = DEVICE.type == "cuda"

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return

    print("\nRunning optimized real-time emotion recognition")
    print(f"Model: {model_type}, img_size={img_size}, infer_every={args.infer_every}, detect_every={args.detect_every}")
    if model_type == "dynamic":
        print(f"Dynamic sequence length: {seq_len} (original default: {SEQUENCE_LENGTH})")
    print("Press 'q' to quit\n")

    predictor = DynamicCachedPredictor(model, seq_len=seq_len, device=DEVICE) if model_type == "dynamic" else None

    fps_counter = []
    frame_idx = 0

    last_face_box = None
    last_emotion = ""
    last_conf = 0.0

    with torch.inference_mode():
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            start_time = time.time()
            frame_idx += 1

            should_detect = (frame_idx % max(1, args.detect_every) == 0) or last_face_box is None
            if should_detect:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(64, 64),
                )
                face = largest_face(faces)
                last_face_box = face
                if face is None and predictor is not None:
                    predictor.clear()

            if last_face_box is not None:
                x, y, w, h = last_face_box
                margin = int(0.1 * max(w, h))
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)

                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size > 0:
                    do_infer = frame_idx % max(1, args.infer_every) == 0
                    if do_infer:
                        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                        face_tensor = transform(face_rgb)

                        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                            if model_type == "static":
                                logits = model(face_tensor.unsqueeze(0).to(DEVICE, non_blocking=True))
                            else:
                                logits = predictor.predict(face_tensor, amp_enabled=amp_enabled)

                        probs = torch.softmax(logits, dim=1)
                        conf, pred = probs.max(1)
                        last_conf = conf.item()
                        emotion_idx = pred.item()
                        last_emotion = idx_to_label.get(emotion_idx, f"class_{emotion_idx}")

                    color = (0, 255, 0) if last_conf > 0.5 else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{last_emotion} ({last_conf:.0%})"
                    label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                    cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            elapsed = time.time() - start_time
            fps_counter.append(1.0 / elapsed if elapsed > 0 else 0.0)
            if len(fps_counter) > 30:
                fps_counter.pop(0)
            avg_fps = sum(fps_counter) / len(fps_counter)

            info = (
                f"Model: {model_type.upper()} | FPS: {avg_fps:.1f} | "
                f"InferEvery: {max(1, args.infer_every)}"
            )
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Emotion Recognition (Optimized)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
