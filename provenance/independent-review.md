# Independent reproduction check

November 9, 2026. Review performed before public release, without changing model settings or selecting against the original holdout.

- Downloaded both pinned CSVs and verified their manifest SHA-256 values. Fetched the pinned dataset card and confirmed Apache-2.0.
- Re-ran the seven original tests and added six invariant tests covering scalar interval-distance scoring, missing-label exclusion, masked gradients, disconnected/edgeless graph inputs, undefined denominator and reversed intervals.
- Reproduced deterministic split indices and verified no exact canonical-structure duplicates or train/holdout overlap. Checked all 5,655 molecular graphs for nonempty atom features, valid edge indices and bidirectional adjacency.
- Ran 100 parity checks against the official scorer at tutorial commit b3b7361e4cb808cec5021514ce39dbfbf1969ccf.
- Independently repeated scaffold GNN fitting: selected epoch 37 again. Independently refit full-training RF and GNN models; holdout predictions agreed with the frozen files to within 3e-15. GNN training was checkpointed between execution windows, preserving model, optimizer and random-generator state; no parameters changed.
- Recomputed all frozen point metrics and paired 1,000-row-resample bootstrap intervals; agreement to 1e-12. These are reproduction checks of fixed choices, not additional independent test sets or training seeds.

Limitations in REPORT.md still apply. This review does not make the evaluation blind, prove generalization to new chemical series, or establish model superiority. In particular, the code's sequence and this review cannot independently prove the absence of all prior human exposure to public holdout labels.
