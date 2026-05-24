# Maturski — FER: static vs dynamic (ResNet-18 vs ResNet-18+LSTM)

This repository contains:

- the final paper text (see `FINAL-PAPER*.md`)
- a reproducible experiment for facial emotion recognition comparing **static** (single frame) vs **dynamic** (frame sequence)

The experiment code lives in `experiment/`.

## Structure

- `experiment/` — data preparation, training, evaluation, real-time demo
- `FINAL-PAPER.md`, `FINAL-PAPER-SR.md` — final paper text

## Quickstart (macOS / Linux)

### 0) Clone

```bash
git clone [<REPO_URL>](https://github.com/ijustseen/dynamicVSstaticDataMLLearningComparing.git)
cd dynamicVSstaticDataMLLearningComparing
```

### 1) Install dependencies (venv)

```bash
cd experiment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Environment check:

```bash
python -c "import sys; print(sys.version)"
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('mps', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())"
```

### 2) Dataset

Download the Hugging Face dataset `Maisum-Abbas-123/Urdu-Multimodal-Emotion-Dataset` and place it under:

```text
experiment/data/Urdu-Multimodal-Emotion-Dataset/
|-- train.csv
|-- video/
`-- audio/                 # optional
```

### 3) Prepare processed datasets (static + dynamic)

```bash
python prepare_data.py
```

This creates `experiment/data/processed/` with:

- `static/` — one representative frame per clip
- `dynamic/` — frame sequences (fixed window length by default)
- `splits.txt` — train/val/test split
- `label_map.json` — class mapping

Sanity checks:

```bash
ls -la data/processed
wc -l data/processed/splits.txt
cat data/processed/label_map.json
```

### 4) Training

Recommended baseline:

```bash
python train.py --model static --epochs 30 --lr 1e-4
python train.py --model dynamic --epochs 30 --lr 1e-4
```

Quick smoke test (to verify everything runs):

```bash
python train.py --model static --epochs 1
python train.py --model dynamic --epochs 1
```

Outputs:

- checkpoints: `experiment/checkpoints/best_static.pth`, `experiment/checkpoints/best_dynamic.pth`
- plots/history: `experiment/results/*`

### 5) Evaluation

```bash
python evaluate.py
```

Optional: plot training curves on a single figure

```bash
python plot_compare.py
```

### 6) Real-time webcam demo

```bash
python realtime.py --model static
# or
python realtime.py --model dynamic
```

Tip: press `q` to quit.

## Windows

For Windows/PowerShell there is a copy-paste runbook: `experiment/COMMANDS.md`.

## Notes

- More experiment details and the expected dataset structure: `experiment/README.md`.
- If you want a step-by-step reproduction log, see: `experiment/WORKLOG.md`.
