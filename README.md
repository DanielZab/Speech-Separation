# Speech Separation with Conv-TasNet

This project implements a speech separation system based on **Conv-TasNet** using PyTorch and torchaudio. It trains a model to separate mixed audio signals (two speakers) into their individual sources using the LibriMix dataset.

---

## Overview

The training pipeline:

* Loads and preprocesses the **Libri2Mix (8kHz)** dataset
* Trains a Conv-TasNet model for **two-speaker separation**
* Uses **Permutation Invariant Training (PIT)** to handle source ambiguity
* Evaluates performance using:
  * SI-SDR (loss)
  * STOI (validation metric)
* Logs results with TensorBoard
* Saves model checkpoints

---

## Project Structure

```
.
├── main.py                # Training script
├── dataloader.py         # Custom dataset loader
├── util.py               # Utility functions (PIT, collate)
├── LibriMix/             # Dataset directory (not included)
├── checkpoints/          # Saved model weights
├── runs/                 # TensorBoard logs
└── images/               # Optional documentation images
```

---

## Requirements

Install dependencies:

```bash
pip install torch torchaudio torchmetrics tqdm tensorboard
```

---

## Dataset

This project uses the **Libri2Mix** dataset (8kHz version) for 2 audio sources.

Expected structure:

```
LibriMix/data/Libri2Mix/wav8k/max/
├── train-100/
├── train-360/
```
---

## Loss & Metrics

### Training Loss

* **SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)**

### Validation Metrics

* **STOI (Short-Time Objective Intelligibility)**

---

## Logging

TensorBoard logs are stored in:

```
runs/<run_id>/
```

Launch TensorBoard:

```bash
tensorboard --logdir=runs
```

---

## Checkpoints

Models are saved in:

```
checkpoints/<run_id>/
├── best_model.pt
└── final_model.pt
```

* `best_model.pt`: Highest validation performance
* `final_model.pt`: Model at end of training

---

## Notes

* Uses **Permutation Invariant Training (PIT)** to match predicted sources with ground truth
* Includes CUDA memory management to avoid excessive allocation

