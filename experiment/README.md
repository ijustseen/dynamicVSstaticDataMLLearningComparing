# Experiment: Emotion Recognition — Static vs Dynamic

## Setup

```bash
cd experiment
pip install -r requirements.txt
```

## Dataset

1. Download the CK+ dataset and place it in `data/CK+/` with the following structure:

   ```
   data/CK+/
   ├── cohn-kanade-images/    # Image sequences
   ├── Emotion/               # Emotion labels
   └── Landmarks/             # (optional) Facial landmarks
   ```

2. Run preprocessing:
   ```bash
   python prepare_data.py
   ```
   This creates `data/processed/` with static (peak frames) and dynamic (sequences) datasets.

## Training

```bash
# Train static model (ResNet-18)
python train.py --model static

# Train dynamic model (ResNet-18 + LSTM)
python train.py --model dynamic
```

## Evaluation

```bash
python evaluate.py
```

## Real-time Demo

```bash
python realtime.py --model static    # or --model dynamic
```
