# HMH Molecular Property Prediction

This repository contains an efficient molecular property prediction pipeline for TDC ADMET classification datasets using HMH spectral-diffusion molecular features.

The project focuses on binary molecular property prediction tasks such as AMES toxicity, hERG inhibition, DILI, blood-brain barrier penetration, HIA, and CYP inhibition. Molecules are represented from SMILES strings with RDKit, converted into molecular graph structure, encoded with an HMH spectral/diffusion context vector, and classified with a lightweight neural network.

## Highlights

- TDC ADMET classification datasets through `pytdc`
- RDKit-based molecular graph construction from SMILES
- HMH graph-context feature extraction using topology, feature similarity, diffusion, and spectral eigenvectors
- Cached HMH features to avoid recomputing expensive graph features
- Efficient MLP classifier for fast CPU/GPU training
- Notebook for exploration and a script for reproducible cluster runs

## Supported Datasets

| Task key | TDC loader | Dataset |
| --- | --- | --- |
| `AMES` | `Tox` | `AMES` |
| `hERG` | `Tox` | `hERG` |
| `hERG_Karim` | `Tox` | `hERG_Karim` |
| `DILI` | `Tox` | `DILI` |
| `HIA` | `ADME` | `HIA_Hou` |
| `BBB` | `ADME` | `BBB_Martins` |
| `CYP2C9_inhibition` | `ADME` | `CYP2C9_Veith` |
| `CYP2D6_inhibition` | `ADME` | `CYP2D6_Veith` |
| `CYP3A4_inhibition` | `ADME` | `CYP3A4_Veith` |

## Repository Structure

```text
.
├── notebooks/
│   └── TDC_ADMET_HMH_classification.ipynb
├── scripts/
│   └── train_tdc_hmh.py
├── src/
│   └── hmh_molecular_prediction/
│       └── __init__.py
├── requirements.txt
└── README.md
```

## Setup

Python 3.10 or 3.11 is recommended. Python 3.14 is not recommended for `pytdc`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For Jupyter:

```bash
python -m ipykernel install --user --name tdc-py311 --display-name "Python 3.11 (TDC)"
```

## Run a Quick Experiment

```bash
python scripts/train_tdc_hmh.py --task DILI --max-rows 500 --epochs 20
```

Run a full dataset:

```bash
python scripts/train_tdc_hmh.py --task AMES --epochs 120
```

Run multiple independent seeds:

```bash
python scripts/train_tdc_hmh.py --task BBB --epochs 120 --seed 1
python scripts/train_tdc_hmh.py --task BBB --epochs 120 --seed 2
python scripts/train_tdc_hmh.py --task BBB --epochs 120 --seed 3
```

Outputs are saved under `runs/`, and cached HMH feature tensors are saved under `data/hmh_tdc_cache/`.

## Cluster Usage

On a Slurm cluster, use the script rather than the notebook. A typical job should request CPU cores for HMH feature generation and optionally a GPU for neural network training.

Example command inside a Slurm job:

```bash
python scripts/train_tdc_hmh.py \
  --task AMES \
  --epochs 120 \
  --n-jobs 8 \
  --cache-dir "$SCRATCH/hmh_tdc_cache" \
  --output-dir "$SCRATCH/hmh_tdc_runs"
```

The expensive feature computation is cached, so later runs with the same dataset and HMH parameters reuse the saved features.

## Model Summary

Input:

- SMILES string for each molecule

Feature construction:

- RDKit atom features
- molecular adjacency from bonds
- feature-neighborhood graph
- Markov/diffusion context
- augmented HMH Laplacian
- bottom spectral eigenvectors
- pooled graph-level HMH vector

Prediction:

- binary classification logit
- probability after sigmoid

Evaluation:

- ROC-AUC
- average precision
- accuracy at threshold 0.5

## Notes

This repository is intended for research prototyping and reproducible molecular property prediction experiments. For publication-quality results, run multiple random seeds, use scaffold splits where appropriate, and report mean plus standard deviation across seeds.
