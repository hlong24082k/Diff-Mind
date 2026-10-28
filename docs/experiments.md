# Experiment Log (model lifecycle)

One row per run. Never edit past rows; add a new one. Goal: every model is reproducible and every iteration learns from the last.

## Lifecycle loop
1. **Hypothesis** — what do we change and what metric should move (from the failure analysis of the previous run).
2. **Data** — dataset version on HF Hub (tag/revision), counts, class balance.
3. **Train** — config file + git commit; W&B run link.
4. **Evaluate** — `evaluate.py` on the *same* frozen 150-example test set.
5. **Analyse** — `notebooks/03_error_analysis.ipynb`; record root causes.
6. **Decide** — promote, iterate (data-centric first), or discard.

## Runs

| ID | Date | Base model | Dataset rev | Config / commit | W&B | JSON valid | FP rate | Recall | Macro F1 | Notes / next hypothesis |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| base-0shot | | Qwen2.5-Coder-3B-Instruct | | — | — | | | | | M1 baseline |
| base-2shot | | Qwen2.5-Coder-3B-Instruct | | — | — | | | | | M1 baseline |
| v1 | | Qwen2.5-Coder-3B-Instruct | | | | | | | | |

## Promoted model
Track the single model currently used in production (HF repo + revision + GGUF filename) here.
