"""
Configuration constants for the experiment.
"""

import os
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CK_PLUS_DIR = DATA_DIR / "CK+"
CK_IMAGES_DIR = CK_PLUS_DIR / "cohn-kanade-images"
CK_EMOTIONS_DIR = CK_PLUS_DIR / "Emotion"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

# --- Dataset ---
EMOTIONS = {
    0: "anger",
    1: "contempt",
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise",
}
NUM_CLASSES = len(EMOTIONS)
# CK+ emotion labels mapping (original label -> our index)
# CK+ uses: 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise
CK_LABEL_MAP = {
    1: 0,  # anger -> 0
    2: 1,  # contempt -> 1
    3: 2,  # disgust -> 2
    4: 3,  # fear -> 3
    5: 4,  # happiness -> 4
    6: 5,  # sadness -> 5
    7: 6,  # surprise -> 6
}

# --- Image ---
IMG_SIZE = 224
SEQUENCE_LENGTH = 16  # Number of frames for dynamic model

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
