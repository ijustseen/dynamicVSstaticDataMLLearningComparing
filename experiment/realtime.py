"""
Real-time emotion recognition using webcam.
Usage:
    python realtime.py --model static
    python realtime.py --model dynamic
"""

import argparse
import time

import cv2
import torch
from torchvision import transforms

from config import CHECKPOINTS_DIR, DEVICE, IMG_SIZE, LABEL_MAP_FILE, SEQUENCE_LENGTH
from dataset import load_label_map
from models import get_model


def get_inference_transform():
    """Transform for real-time inference."""
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Real-time emotion recognition")
    parser.add_argument(
        "--model",
        type=str,
        default="static",
        choices=["static", "dynamic"],
        help="Model type",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    if not LABEL_MAP_FILE.exists():
        print("ERROR: label_map.json not found. Run prepare_data.py first.")
        return

    _, idx_to_label = load_label_map(LABEL_MAP_FILE)
    num_classes = len(idx_to_label)

    model_type = args.model
    print(f"Loading {model_type} model...")

    # Load model
    model = get_model(model_type, num_classes=num_classes, pretrained=False).to(DEVICE)
    ckpt_path = CHECKPOINTS_DIR / f"best_{model_type}.pth"

    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        print("Train the model first: python train.py --model", model_type)
        return

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded from {ckpt_path}")

    # Face detector
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    transform = get_inference_transform()

    # Video capture
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return

    print(f"\nRunning real-time emotion recognition ({model_type} model)")
    print("Press 'q' to quit\n")

    # Buffer for dynamic model
    frame_buffer = []
    fps_counter = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))

        emotion_text = ""
        confidence = 0.0

        for x, y, w, h in faces:
            # Add margin
            margin = int(0.1 * max(w, h))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)

            face_roi = frame[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_tensor = transform(face_rgb)

            with torch.no_grad():
                if model_type == "static":
                    input_tensor = face_tensor.unsqueeze(0).to(DEVICE)
                    output = model(input_tensor)
                else:
                    # Dynamic: maintain frame buffer
                    frame_buffer.append(face_tensor)
                    if len(frame_buffer) > SEQUENCE_LENGTH:
                        frame_buffer.pop(0)

                    if len(frame_buffer) < SEQUENCE_LENGTH:
                        # Pad with copies of first frame
                        padded = [frame_buffer[0]] * (SEQUENCE_LENGTH - len(frame_buffer)) + list(
                            frame_buffer
                        )
                    else:
                        padded = list(frame_buffer)

                    sequence = torch.stack(padded).unsqueeze(0).to(DEVICE)
                    output = model(sequence)

                probs = torch.softmax(output, dim=1)
                confidence, pred = probs.max(1)
                confidence = confidence.item()
                emotion_idx = pred.item()
                emotion_text = idx_to_label.get(emotion_idx, f"class_{emotion_idx}")

            # Draw bounding box and label
            color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{emotion_text} ({confidence:.0%})"
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # FPS
        elapsed = time.time() - start_time
        fps_counter.append(1.0 / elapsed if elapsed > 0 else 0)
        if len(fps_counter) > 30:
            fps_counter.pop(0)
        avg_fps = sum(fps_counter) / len(fps_counter)

        # Display info
        info = f"Model: {model_type.upper()} | FPS: {avg_fps:.1f}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
