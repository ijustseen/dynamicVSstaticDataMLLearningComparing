"""Configuration constants for the experiment."""

from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
URDU_DATASET_DIR = DATA_DIR / "Urdu-Multimodal-Emotion-Dataset"
URDU_METADATA_FILE = URDU_DATASET_DIR / "train.csv"
URDU_VIDEO_DIR = URDU_DATASET_DIR / "video"
PROCESSED_DIR = DATA_DIR / "processed"
LABEL_MAP_FILE = PROCESSED_DIR / "label_map.json"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

# --- Results structure ---
# 'latest' keeps the most recent artifacts (overwritten each run).
# 'experiments' stores tagged evaluation outputs in separate folders.
LATEST_RESULTS_DIR = RESULTS_DIR / "latest"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"

# --- Image / Video ---
IMG_SIZE = 224
SEQUENCE_LENGTH = 16  # Number of frames for dynamic model
MIN_FACE_SIZE = 64

# --- Training ---
BATCH_SIZE_STATIC = 32
BATCH_SIZE_DYNAMIC = 8
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4

# --- Model ---
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 1
RESNET_FEATURE_DIM = 512  # ResNet-18 output before FC layer

# --- Split ---
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

# --- Device ---
import torch
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
