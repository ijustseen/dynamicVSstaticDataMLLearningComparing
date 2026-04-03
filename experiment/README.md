# Experiment: Emotion Recognition - Static vs Dynamic (Urdu-Multimodal)

## Setup

```bash
cd experiment
pip install -r requirements.txt
```

## Dataset

1. Download the Hugging Face dataset Maisum-Abbas-123/Urdu-Multimodal-Emotion-Dataset and place it in data/Urdu-Multimodal-Emotion-Dataset/ with this structure:

```text
data/Urdu-Multimodal-Emotion-Dataset/
|-- train.csv
|-- video/
`-- audio/                 # optional for future multimodal work
```

1. Run preprocessing:

```bash
python prepare_data.py
```

This creates data/processed/ with:

- static/ (one representative frame per clip)
- dynamic/ (uniformly sampled frame sequences)
- splits.txt (train/val/test split)
- label_map.json (class mapping used across scripts)

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

## Runbook

cd /Users/andrew/www/maturski/experiment
source .venv/bin/activate
pip install -r requirements.txt
find data/Urdu-Multimodal-Emotion-Dataset -maxdepth 2 -type d | head -n 40
python prepare_data.py
ls -la data/processed && wc -l data/processed/splits.txt && cat data/processed/label_map.json
python train.py --model static --epochs 20
python train.py --model dynamic --epochs 20
python evaluate.py
ls -la checkpoints results
