# CYP inhibition benchmark

An independent, post-challenge study using OpenADMET's released CYP inhibition data. This is **not a blind competition entry, an official leaderboard result, or an upstream OpenADMET contribution**.

The first experiment compares RDKit Morgan fingerprints with a random forest against a small, multitask molecular graph neural network implemented in PyTorch. It includes deterministic scaffold validation, structure-overlap checks, missing-label masking, and experimental-uncertainty-aware scoring.

## Results

| Model | Original holdout macro ST-RAE (lower better) | Macro MAE (pIC50) | Macro Spearman |
|---|---:|---:|---:|
| Training-mean baseline | 1.2737 | 1.3139 | 0.0000 |
| RDKit Morgan + random forest | 0.9341 | 1.0813 | 0.5481 |
| Small molecular GNN | 0.9457 | 1.0674 | 0.5247 |

The GNN does not beat the fingerprint model on the primary metric. Its slightly lower MAE is not enough to establish a better model. A paired 1,000-resample bootstrap gives a GNN-minus-RF ST-RAE difference interval of [-0.0165, 0.0417], which includes zero. These are one-seed baseline experiments, not a conclusion about GNNs in general. See [REPORT.md](REPORT.md) for methods, chronology, and limitations.

## Reproduce

Python 3.10 was used. A CPU-only PyTorch installation is sufficient. On other systems, install the appropriate PyTorch wheel using its official instructions, then install the remaining dependencies. No GPU service, account, or paid API is required by the code.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m unittest discover -s tests -v
python benchmark.py --epochs 40
python analyze.py
```

`dataset_manifest.json` pins the public Hugging Face dataset revision and both CSV SHA-256 checksums. The script downloads only these files, verifies checksums, and does not upload or submit anything. The `data/` directory is ignored by Git. `results/` contains fixed predictions, splits, metrics, and a training-row index so each evaluated prediction can be traced to its molecule ID. No model weights or raw assay tables are committed.

Training took roughly two minutes in the initial CPU environment; runtime varies. The code sets two PyTorch CPU threads and two random-forest workers, rather than launching a compute sweep.

## Scope and next steps

This first study covers direct inhibition regression for CYP1A2, CYP2C9, CYP2D6, and CYP3A4. It does not cover time-dependent inhibition classification, docking, clinical drug-drug interaction prediction, or wet-lab validation. A responsible extension is predeclared multi-seed evaluation on new development splits, stronger bond-aware models, and external evaluation data. Do not repeatedly tune on this now-observed original holdout and continue calling it untouched.

## Sources and attribution

- Dataset: https://huggingface.co/datasets/openadmet/cyp-challenge-train-test
- Official tutorial and scoring reference: https://github.com/OpenADMET/CYP-Challenge-Tutorial
- Dataset/assay construction: https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/
- Challenge wrap-up and release: https://openadmet.ghost.io/take-me-heme-binding-modes-wrapping-up-the-cyp-inhibition-challenge/

The dataset card declares Apache-2.0 at the pinned revision. Code is provided under Apache-2.0; see LICENSE and NOTICE. This independent project is not endorsed by OpenADMET. The implementation was AI-assisted; measured outputs, tests, and limitations are included for review.
