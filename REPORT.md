# Comparing fingerprints and a small molecular GNN for CYP inhibition

November 9, 2026

## Question and outcome

Can a small multitask molecular GNN improve on a simple fingerprint model for direct CYP inhibition? In this first, post-challenge experiment, the answer is not clearly. The GNN lowers macro MAE slightly, but has worse point-estimate macro ST-RAE, the challenge's primary metric. The paired bootstrap interval for their ST-RAE difference includes zero. This is a useful baseline and a record of what did not work, not a competitive-method claim.

## Data and chemistry checks

The dataset is OpenADMET's CYP challenge release, pinned in `dataset_manifest.json`. The original training file has 4,905 molecules and sparse labels across four enzymes. The original holdout has 750 molecules and 748, 750, 748, and 748 observed labels for CYP1A2, CYP2C9, CYP2D6, and CYP3A4 respectively. Two holdout rows lack labels for three endpoints, so not every endpoint has exactly 750 evaluated examples.

SMILES are parsed and canonicalized with RDKit, retaining stereochemistry. There are no invalid SMILES, canonical duplicates within either file, or identical canonical structures shared across the files. The implementation would exclude exact training matches to holdout structures based on structure only, but none were found. This is an exact-structure audit, not proof against analog similarity, different salts, tautomers, or related chemical-series leakage. No neutralization, desalting, or tautomer standardization was applied. Scaffold separation does not mean all chemical similarities disappear.

Only SMILES are model inputs. Molecule IDs, pIC50 confidence intervals, and assay standard deviations are not predictive features. pIC50 is modeled directly, without unit conversion or arbitrary clipping. No external/proprietary data or pretrained weights are used.

## Models and fixed training choices

- Training-mean baseline: separate observed training mean per endpoint.
- Morgan/RF: radius-2, 2,048-bit Morgan fingerprints, RDKit implementation. Separate random forest per endpoint; 200 trees, minimum leaf size 2, maximum-feature fraction 0.5, two workers, seed 20261109. These settings were fixed, not optimized on validation or holdout.
- Molecular GNN: atom features encode common elements, atomic number, degree, formal charge, aromaticity, hydrogen count, and ring membership. Two 64-dimensional atom-message updates use mean neighbor aggregation, with a graph-level sum divided by the square root of atom count and a shared four-output MLP. Bonds define adjacency; bond order and bond stereochemistry are not embedded. The model is intentionally small, not Chemprop, a foundation model, or a new proposed architecture.

The GNN uses Adam at learning rate 0.001, batches of 64, no pretraining, and a 40-epoch maximum. Per-endpoint means and standard deviations are estimated from training labels only. The loss averages squared standardized error over all observed labels; missing labels are masked. This means tasks contribute in proportion to observed labels, whereas the reported macro metric weights endpoints equally. Random forests drop missing labels independently per endpoint. There is no synthetic imputation of missing target values.

## Development splits and selection

The seed is 20261109 for the split generator, NumPy, PyTorch, and random forests. Both validation modes reserve 20% of groups using `GroupShuffleSplit`:

1. Canonical-molecule-grouped random split. With no duplicate structures in this dataset, this is effectively a seeded molecule split.
2. Bemis-Murcko scaffold split. All molecules sharing a scaffold stay on one side. All ring-free molecules share the empty-scaffold group. There is no label stratification, so label counts and target distributions may differ.

Exact index assignments and their matching molecule IDs are in `results/splits.json` and `results/training_index.csv`. The mean/RF baselines are evaluated on both development splits. The GNN is evaluated only on the scaffold split. Its selected checkpoint is epoch 37, chosen using lowest standardized validation MSE, not ST-RAE. The GNN is then reinitialized with the same seed and trained on the complete original training file for 37 epochs before evaluating the original holdout. This comparison does not equalize wall-clock time, hyperparameter search, number of trainable parameters, or tuning effort. All RF settings are fixed; GNN epoch selection gets validation access.

The reported GNN scaffold-validation result is also the model-selection set result and is therefore optimistic, not a separate unbiased test estimate.

## Primary metric and aggregation

For each endpoint and each labeled molecule, let its fitted experimental credible interval be [L, U]. The soft-threshold absolute error of prediction p is max(L - p, 0) + max(p - U, 0), which is zero for predictions inside the interval. ST-RAE is the sum of those errors divided by the sum of the same interval-distance errors for a constant prediction equal to the **evaluation subset's mean observed pIC50**. The denominator mean is part of the scoring definition only and is not supplied to the trained predictive models. If that denominator is zero, the score is undefined; this did not occur in this experiment.

Macro ST-RAE is the unweighted mean across the four endpoint ST-RAE values, not a pooled error over all observations. MAE, R2, and Spearman are also computed separately per endpoint and averaged. A constant predictor receives Spearman zero by convention. The training-mean baseline can score above 1 because the scorer's denominator uses the evaluation mean rather than the training mean.

This implementation matches `rae_soft_threshold_absolute_error` in the official tutorial's `evaluation/custom_scoring_functions.py`; 50 random-array parity checks were run against that reference. It uses the file's provided `*_conf_low` and `*_conf_high` bounds. No substitute MAE-only score is presented as the official primary metric.

## Evaluation chronology and results

The model architecture, RF settings, split seed, maximum epochs, and GNN checkpoint-selection rule were fixed before scoring the original holdout. Source data were already public and the public challenge wrap-up was read during project planning; therefore this was never a blind experiment.

The first run completed fitting and scoring but failed when serializing JSON because a numerically constant mean prediction produced undefined Spearman correlation. That reporting guard was fixed, and the identical experiment was rerun with the same seed and unchanged predictive models. There was no holdout-driven parameter change. The final exported predictions were then used for a separate bootstrap analysis without training or model selection. Results are not evidence that holdout labels were inaccessible: the script downloads both files and scores the holdout after model selection by its program order.

| Model | Macro ST-RAE | 95% bootstrap interval | Macro MAE | Macro Spearman |
|---|---:|---|---:|---:|
| Training-mean baseline | 1.2737 | [1.2425, 1.3124] | 1.3139 | 0.0000 |
| Morgan/RF | 0.9341 | [0.9000, 0.9743] | 1.0813 | 0.5481 |
| Small molecular GNN | 0.9457 | [0.9116, 0.9872] | 1.0674 | 0.5247 |

The bootstrap uses 1,000 shared compound-row resamples across models and endpoints, recomputing each endpoint's ST-RAE denominator each time. These percentile intervals describe sampling variability conditional on the fitted models. The GNN-minus-RF ST-RAE 95% interval is [-0.0165, 0.0417]. It does not establish that either model is superior. It does not cover training-seed variation, architecture-search uncertainty, or new chemical-series shift. Shared row bootstrap is not a chemical-series-grouped bootstrap.

On scaffold validation, Morgan/RF has macro ST-RAE 0.8640 and the selected GNN has 0.7584. Those development results do not transfer directly to the original holdout. In particular, CYP2D6 has a large target-distribution shift: 8.64% of observed training labels are below pIC50 4, versus 64.71% on holdout. Those fractions are a post-evaluation analysis, not features or a calibration rule used in training. They offer a concrete failure-analysis question rather than a reason to retune this holdout.

## Limits and next useful work

One seed, a small atom-only GNN, and a fixed RF baseline cannot establish a general ranking of molecular learning methods. No TDI labels, classification metrics, structure predictions, docking, kinetic interpretation, clinical DDI assessment, or prospective experimental validation are claimed. These independent scores are not official competition ranks, and no upstream pull request has been made.

A useful next study would predeclare multiple development splits and seeds, add bond-type-aware message passing or a documented Chemprop baseline, and investigate single-task versus multitask learning with the same data access. Any improvement should be selected on fresh development data and assessed on genuinely new evaluation data. This now-observed original holdout must not be repeatedly tuned and described as untouched.

## Sources

- OpenADMET dataset: https://huggingface.co/datasets/openadmet/cyp-challenge-train-test
- Official tutorial, scorer, and Apache-2.0 license: https://github.com/OpenADMET/CYP-Challenge-Tutorial
- Assay design and analog-expansion construction: https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/
- Public release and final results, November 5, 2026: https://openadmet.ghost.io/take-me-heme-binding-modes-wrapping-up-the-cyp-inhibition-challenge/
