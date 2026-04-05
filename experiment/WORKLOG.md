# Журнал эксперимента (WORKLOG)

Цель: запустить и воспроизвести эксперимент по распознаванию эмоций (static vs dynamic) на датасете Urdu-Multimodal.

## Контекст

- Дата: 2026-04-05
- ОС: Windows (PowerShell)
- Python: 3.12.6
- Рабочая директория эксперимента: `experiment/`
- Датасет: `experiment/data/Urdu-Multimodal-Emotion-Dataset/` (train.csv + video/ + audio/)

## Что сделано

### 1) Подготовка окружения

- Создано виртуальное окружение:
  - `python -m venv .venv`
- Активация в PowerShell:
  - `.\.venv\Scripts\Activate.ps1`
  - Если блокируется политикой выполнения: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force`

### 2) Установка зависимостей

- Обновление pip:
  - `python -m pip install --upgrade pip`
- Установка зависимостей эксперимента:
  - `pip install -r requirements.txt`
- Проверки импорта:
  - `import cv2` (opencv-python) — OK
  - `import torch` — OK (установился CPU-вариант: `torch 2.11.0+cpu`, `cuda False`)

### 3) Исправления по ходу запуска

- Ошибка: `ModuleNotFoundError: No module named 'cv2'`
  - Причина: зависимости ещё не были установлены в venv.
  - Решение: установка `requirements.txt`.

- Ошибка: `ValueError: train.csv must contain columns ['id', 'label', 'video_path']; got ['\ufeffid', ...]`
  - Причина: UTF-8 BOM в заголовке CSV (первое поле читалось как `\ufeffid`).
  - Решение: в файле [prepare_data.py](prepare_data.py) чтение CSV переведено на `encoding="utf-8-sig"`.
  - Проверка: `load_samples_from_csv(...)` загружает 8283 сэмпла.

## Текущий статус

- `python prepare_data.py` завершился успешно.

Результат запуска `prepare_data.py`:

- Usable samples: 8283 (видео найдены)
- Labels (label_map.json):
  - 0: anger
  - 1: happy
  - 2: love
  - 3: neutral
  - 4: sad
- Splits:
  - train=6625, val=827, test=831
- Подготовка static:
  - 8283 изображений, ~11:16 (≈ 12.24 it/s)
- Подготовка dynamic:
  - 8283 последовательностей по 16 кадров, ~1:46:05 (≈ 1.30 it/s)
- Выход:
  - `experiment/data/processed/label_map.json`
  - `experiment/data/processed/splits.txt`
  - `experiment/data/processed/static/`
  - `experiment/data/processed/dynamic/`

## Следующие шаги (после завершения prepare_data)

1) Проверить, что созданы:
- `experiment/data/processed/static/`
- `experiment/data/processed/dynamic/`
- `experiment/data/processed/splits.txt`
- `experiment/data/processed/label_map.json`

2) Обучение моделей:
- `python train.py --model static --epochs 15`
- `python train.py --model dynamic --epochs 15`

3) Оценка:
- `python evaluate.py`

## Smoke-test (обучение)

- Команда: `python train.py --model static --epochs 1`
- Устройство: `cuda`
- Итог 1 эпохи:
  - Train Loss 1.6201 | Train Acc 0.2841
  - Val   Loss 1.5118 | Val   Acc 0.3253
- Артефакты:
  - `experiment/checkpoints/best_static.pth` (создан)
  - `experiment/results/history_static.json` (создан)

- Команда: `python train.py --model dynamic --epochs 1`
- Устройство: `cuda`
- Итог 1 эпохи:
  - Train Loss 1.6049 | Train Acc 0.2317
  - Val   Loss 1.5988 | Val   Acc 0.2503
- Артефакты:
  - `experiment/checkpoints/best_dynamic.pth` (создан)
  - `experiment/results/history_dynamic.json` (создан)

## Smoke-test (evaluate)

- Команда: `python evaluate.py`
- Артефакты:
  - `experiment/results/evaluation_results.json`
  - `experiment/results/confusion_static.png`
  - `experiment/results/confusion_dynamic.png`
- Итоги (test set, smoke-чекпоинты после 1 эпохи):
  - static: accuracy 0.3225 | precision 0.3552 | recall 0.3077 | f1 0.2692 | fps 24.6
  - dynamic: accuracy 0.2491 | precision 0.0501 | recall 0.2000 | f1 0.0802 | fps 5.7

Примечание: это smoke-оценка (после 1 эпохи). Финальные выводы нужно делать после полного обучения (например, 15+ эпох).

## Заметки

- Препроцессинг может занимать значительное время (обработка тысяч видео).
- Для ускорения обучения может понадобиться GPU/CUDA-установка PyTorch.

### GPU / CUDA (опционально)

- Видеокарта: NVIDIA GeForce RTX 4080 Super
- Драйвер установлен: `nvidia-smi` показывает Driver Version 591.86 (CUDA Version 13.1)
- Переключили PyTorch на CUDA (pip, cu124):
  - `torch 2.6.0+cu124`
  - `torchvision 0.21.0+cu124`
  - Проверка: `torch.cuda.is_available() == True`
  - `torch.version.cuda == 12.4`
  - `torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 4080 SUPER"`

Примечание: установка CUDA-варианта скачивает большой объём пакетов.

## Последнее обновление

- 2026-04-05: подготовили данные `prepare_data.py` → `data/processed/`.
- 2026-04-05: переключили PyTorch на GPU (CUDA, cu124), проверили `torch.cuda.is_available() == True`.
- 2026-04-05: сделали smoke-train (static 1 epoch и dynamic 1 epoch), получили `best_static.pth` и `best_dynamic.pth`.
- 2026-04-05: запустили `evaluate.py`, сохранили `evaluation_results.json` и confusion matrices в `experiment/results/`.
