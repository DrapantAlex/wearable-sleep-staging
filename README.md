# Wearable-Only Sleep Staging

3-class sleep staging (**Wake / NREM / REM**) from consumer wearable data (wrist accelerometry + heart rate), with cross-dataset external validation.

## Overview

Sleep staging normally requires polysomnography (PSG) in a lab. This project explores how well sleep stages can be estimated using **only** signals from a wrist-worn device.

**Pipeline:**
1. **Feature extraction** — 18 features per 30-second epoch from accelerometry (ENMO, arm angle, activity counts) and heart rate (mean, variability, MSSD), plus rolling-context features over neighbouring epochs.
2. **Models**
   - **Bidirectional LSTM** processing each night as a full sequence, trained with focal loss and square-root class weights to handle class imbalance.
   - **Random Forest** on the same epoch-level features.
   - **Soft-voting ensemble** of the two, with the weight tuned on internal validation only.
3. **Evaluation** — Leave-One-Subject-Out (LOSO) cross-validation on the Walch dataset, then external testing on DREAMT.
4. **Calibration & decision** — temperature scaling (ECE 0.093 → 0.028) and class-probability reweighting.
5. **Domain adaptation** — fine-tuning the classifier head on a subset of DREAMT subjects.

## Results

**Internal validation (Walch, LOSO, 31 subjects)**

| Model | Macro-F1 | Cohen's κ |
|---|---|---|
| Random Forest | 0.612 | 0.400 |
| BiLSTM | 0.652 | 0.479 |
| **Ensemble (BiLSTM + RF)** | **0.668** | **0.495** |

**External validation (DREAMT)**

| Model | Macro-F1 | Cohen's κ |
|---|---|---|
| BiLSTM (Walch only) | 0.547 | 0.352 |
| **BiLSTM + fine-tuning** | **0.580** | **0.388** |

The performance drop on DREAMT highlights the domain shift between datasets (different devices, populations and label distributions), and fine-tuning recovers part of it.

## Data

The datasets are **not included** in this repository.

- **Walch et al.** — Apple Watch sleep dataset, available on PhysioNet.
- **DREAMT** — multisensor wearable dataset, available on PhysioNet under its data use agreement.

## How to run

The notebook was developed on Kaggle. To run it:
- open it on Kaggle and attach the two datasets, or
- download the data locally and update the paths at the top of the notebook.

```bash
pip install -r requirements.txt
```

> Note: comments and some outputs in the notebook are in Italian.
