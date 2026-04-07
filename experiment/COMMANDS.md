# Experiment Commands (Windows / PowerShell)

This file is a copy-paste runbook for reproducing the experiment.

Assumptions:
- You are in a PowerShell terminal.
- You have Python 3.12+ installed.
- Dataset is placed under `experiment/data/Urdu-Multimodal-Emotion-Dataset/` (with `train.csv`, `video/`, optionally `audio/`).

## 0) Go to the experiment folder

```powershell
cd D:\www\maturski\experiment
```

## 1) Create and activate a virtual environment

```powershell
python -m venv .venv

# If activation is blocked by execution policy:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

.\.venv\Scripts\Activate.ps1
```

Quick check:

```powershell
python -c "import sys; print(sys.executable)"
```

## 2) Install dependencies

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the base requirements:

```powershell
pip install -r requirements.txt
```

### 2a) (Recommended) Switch PyTorch to NVIDIA CUDA build

If you have an NVIDIA GPU, install CUDA wheels (this downloads large packages):

```powershell
# Remove CPU wheels (if present)
pip uninstall -y torch torchvision

# Install CUDA build (try cu124 first)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# If cu124 is not available for your Python/version, fallback:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU is used:

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## 3) Prepare processed datasets (static + dynamic)

This creates `data/processed/` with:
- `static/` (one face-cropped frame per clip)
- `dynamic/` (sequences of 16 frames per clip)
- `splits.txt`
- `label_map.json`

Run:

```powershell
python prepare_data.py
```

Sanity checks:

```powershell
Test-Path .\data\processed\splits.txt
Test-Path .\data\processed\label_map.json
```

## 4) Training

Smoke tests (recommended before long runs):

```powershell
python train.py --model static --epochs 1
python train.py --model dynamic --epochs 1
```

Full runs (example):

```powershell
# Recommended baseline (more stable training):
python train.py --model static --epochs 30 --lr 1e-4
python train.py --model dynamic --epochs 30 --lr 1e-4

# If you want a faster run:
# python train.py --model static --epochs 15
# python train.py --model dynamic --epochs 15
```

Outputs:
- Checkpoints: `checkpoints/best_static.pth`, `checkpoints/best_dynamic.pth`
- Training history + plots: `results/history_*.json`, `results/training_*.png`

## 5) Evaluation and comparison

```powershell
python evaluate.py
```

Optional: generate a single comparison plot (static vs dynamic training curves):

```powershell
python plot_compare.py
```

Outputs (in `results/`):
- `evaluation_results.json`
- `confusion_static.png`
- `confusion_dynamic.png`
- `training_compare.png`

## 6) Real-time webcam demo

Static:

```powershell
python realtime.py --model static
```

Dynamic:

```powershell
python realtime.py --model dynamic
```

Notes:
- This is a qualitative demo (no ground-truth labels). Use `evaluate.py` for objective comparison.
- Press `q` to quit.

## Common issues

1) `Activate.ps1` cannot be loaded

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\.venv\Scripts\Activate.ps1
```

2) Training plots look empty after 1 epoch
- This can happen because the plots are drawn as lines without markers; with a single point the line is barely visible.
- Run 2+ epochs or adjust plotting to use markers.

3) CUDA not detected by PyTorch
- Check `nvidia-smi` first.
- Reinstall CUDA wheels for PyTorch (see section 2a).
