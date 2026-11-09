---
license: apache-2.0
language:
- en
tags:
- chemistry
- drug-discovery
- ADMET
- molecular-properties
- blind-challenge
- computational-chemistry
- cytochrome-p450

pretty_name: OpenADMET CYP Inhibition Blind Challenge
size_categories:
- 10K<n<100K
task_categories:
- tabular-regression
- tabular-classification
annotations_creators:
- expert-generated
source_datasets:
- original
configs:
- config_name: default
  data_files:
  - split: train
    path: cyp-challenge-TRAIN_inhibition.csv
  - split: test
    path: cyp-challenge-TEST-UNBLINDED_inhibition.csv
  - split: test_blinded
    path: cyp-challenge-TEST-BLINDED.csv
- config_name: tdi
  data_files:
  - split: train
    path: cyp-challenge-TRAIN_TDI.csv
  - split: test
    path: cyp-challenge-TEST-UNBLINDED_TDI.csv
- config_name: single_concentration
  data_files:
  - split: train
    path: cyp-challenge-single-concentration-TRAIN.csv
  - split: full
    path: cyp-challenge-single-concentration-FULL.csv
- config_name: emax
  data_files:
  - split: train
    path: cyp-challenge-TRAIN_Emax.csv
- config_name: structure
  data_files:
  - split: test
    path: cyp-challenge-TEST-BLINDED_structures.csv
---
# CYP Challenge Train/Test Dataset

A high-quality experimental dataset for predicting inhibition of the major drug-metabolizing Cytochrome P450 enzymes (CYP1A2, CYP2C9, CYP2D6, CYP3A4), released as part of the OpenADMET CYP Inhibition Blind Challenge.

**Blog post:** [Announcing OpenADMET’s CYP inhibition blind challenge](https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/)

**Challenge Space:** [OpenADMET CYP Inhibition Blind Challenge](https://huggingface.co/spaces/openadmet/cyp-challenge)

**Challenge period:** August 17, 2026 - November 3, 2026

**Produced by:** OpenADMET

## CHANGELOG

* Updated 2026-11-05 Added the complete single-concentration screening set as the `full` split of the `single_concentration` config. Note: This is a *superset* of the `train` split, not a disjoint split.
* Updated 2026-11-04 **Unblinded the test set.** The direct-inhibition and TDI test labels are now public as `cyp-challenge-TEST-UNBLINDED_inhibition.csv` and `cyp-challenge-TEST-UNBLINDED_TDI.csv`. The `test` split of the `default` and `tdi` configs now resolves to these files; the original SMILES-only file is still available as the `test_blinded` split of `default`.
* Updated 2026-09-23 Added the structure track's blinded test set (SMILES for the co-folding/structure-prediction task; ground-truth CYP3A4 complex structures used for scoring are not disclosed yet).
* Updated 2026-08-06 Finalised ground truth: refreshed direct-inhibition and TDI training/test data with the finalised assay results, and added a new Emax (maximal effect) training set.
* Updated 2026-08-03 Initial release: direct inhibition training/blinded test data, time-dependent inhibition (TDI) training data, and single-concentration screening training data.

### Dataset contents

| Config | Split | File | Description |
|---|---|---|---|
| `default` | `train` | `cyp-challenge-TRAIN_inhibition.csv` | Primary direct-inhibition training set — 4,905 compounds with pIC50 (plus 95% CI and std) for CYP1A2, CYP2C9, CYP2D6, CYP3A4 |
| `default` | `test` | `cyp-challenge-TEST-UNBLINDED_inhibition.csv` | **Unblinded** 750-compound test set — the ground truth used to score the Direct Inhibition track: pIC50 (plus 95% CI and std) for CYP1A2, CYP2C9, CYP2D6, CYP3A4 |
| `default` | `test_blinded` | `cyp-challenge-TEST-BLINDED.csv` | The original blinded test set (`Molecule_Name`, `SMILES` only), kept so the challenge conditions can be reproduced |
| `tdi` | `train` | `cyp-challenge-TRAIN_TDI.csv` | Time-dependent inhibition (TDI) training set — 6,145 compounds with `is_TDI` classification labels for CYP2D6/CYP3A4, TDI-condition pIC50s, and paired direct-inhibition pIC50s for comparison |
| `tdi` | `test` | `cyp-challenge-TEST-UNBLINDED_TDI.csv` | **Unblinded** 750-compound TDI test set — the ground truth used to score the TDI track: `is_TDI` labels for CYP2D6/CYP3A4, TDI-condition pIC50s (with 95% CI and std) and paired direct-inhibition pIC50s |
| `single_concentration` | `train` | `cyp-challenge-single-concentration-TRAIN.csv` | Single-concentration screening data — 17,504 measurements across 4,376 compounds x 4 enzymes (log2 fold-change format) |
| `single_concentration` | `full` | `cyp-challenge-single-concentration-FULL.csv` | Complete single-concentration screening set — 45,428 measurements across 11,358 compounds x 4 enzymes. A **superset** of the `train` split containing every `train` row, 4 single-concentration test compounds, and a further 6,978 compounds that are in neither |
| `emax` | `train` | `cyp-challenge-TRAIN_Emax.csv` | Emax (maximal effect) training set — 6,146 compounds with `is_TDI` classification labels for all four CYPs, plus TDI-condition and direct-inhibition Emax values (with 95% CI) |
| `structure` | `test` | `cyp-challenge-TEST-BLINDED_structures.csv` | 20-compound blinded test set for the structure-prediction track — `Molecule_Name`, `SMILES` only; the ground-truth CYP3A4-ligand complex structures used for scoring are withheld for the challenge |

## Loading with Hugging Face `datasets`

```python
from datasets import load_dataset
# Default config (primary direct-inhibition assay)
ds = load_dataset("openadmet/cyp-challenge-train-test")
train = ds["train"]
test  = ds["test"]           # unblinded: SMILES + pIC50 ground truth
blind = ds["test_blinded"]   # challenge version: SMILES only
# TDI (time-dependent inhibition) config
ds_tdi = load_dataset("openadmet/cyp-challenge-train-test", "tdi")
train_tdi = ds_tdi["train"]
test_tdi  = ds_tdi["test"]   # unblinded
# Single-concentration config
ds_single = load_dataset("openadmet/cyp-challenge-train-test", "single_concentration")
train_single = ds_single["train"]
full_single  = ds_single["full"]
# Emax config
ds_emax = load_dataset("openadmet/cyp-challenge-train-test", "emax")
train_emax = ds_emax["train"]
# Structure config
ds_structure = load_dataset("openadmet/cyp-challenge-train-test", "structure")
test_structure = ds_structure["test"]
```

## Loading directly with pandas

```python
import pandas as pd
train        = pd.read_csv("hf://datasets/openadmet/cyp-challenge-train-test/cyp-challenge-TRAIN_inhibition.csv")
test         = pd.read_csv("hf://datasets/openadmet/cyp-challenge-train-test/cyp-challenge-TEST-UNBLINDED_inhibition.csv")
test_blinded = pd.read_csv("hf://datasets/openadmet/cyp-challenge-train-test/cyp-challenge-TEST-BLINDED.csv")
train_tdi    = pd.read_csv("hf://datasets/openadmet/cyp-challenge-train-test/cyp-challenge-TRAIN_TDI.csv")
test_tdi     = pd.read_csv("hf://datasets/openadmet/cyp-challenge-train-test/cyp-challenge-TEST-UNBLINDED_TDI.csv")
train_single = pd.read_csv("hf://datasets/openadmet/cyp-challenge-train-test/cyp-challenge-single-concentration-TRAIN.csv")
full_single  = pd.read_csv("hf://datasets/openadmet/cyp-challenge-train-test/cyp-challenge-single-concentration-FULL.csv")
train_emax   = pd.read_csv("hf://datasets/openadmet/cyp-challenge-train-test/cyp-challenge-TRAIN_Emax.csv")
test_structure = pd.read_csv("hf://datasets/openadmet/cyp-challenge-train-test/cyp-challenge-TEST-BLINDED_structures.csv")
```
