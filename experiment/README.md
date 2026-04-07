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
# Recommended baseline (stable training)
python train.py --model static --epochs 30 --lr 1e-4
python train.py --model dynamic --epochs 30 --lr 1e-4
```

Notes:
- Dynamic tends to achieve higher accuracy, but runs at lower FPS.
- Use `evaluate.py` for objective comparison; it saves metrics to `results/evaluation_results.json`.

## Evaluation

```bash
python evaluate.py
```

Optional (training curves on a single figure):

```bash
python plot_compare.py
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
python train.py --model static --epochs 30 --lr 1e-4
python train.py --model dynamic --epochs 30 --lr 1e-4
python evaluate.py
ls -la checkpoints results
