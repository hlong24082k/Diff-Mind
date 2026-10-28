# Diff-Mind

## Overview

Diff-Mind is a QLoRA fine-tune of `Qwen/Qwen2.5-Coder-3B-Instruct` that acts as a **high-signal pull request reviewer**. It flags only genuine defects in four categories (`SECURITY`, `BUG`, `PERFORMANCE`, `RESOURCE_LEAK`), abstains on clean code, and emits strict JSON. It ships as a Q4_K_M GGUF that runs on a GitHub Actions CPU runner, so it needs no GPU in production. It targets developers who want automated PR review without style nitpicks. The owner is also using it to learn LoRA fine-tuning and MLOps, so simple, reproducible choices are preferred over clever ones.

## Goals

1. Beat the untuned base model on the frozen 150-example test set. Targets (from the plan §2): JSON schema compliance above 98%, false positive rate below 10%, defect recall above 80%, macro-F1 +20% over baseline.
2. Review a 100-line diff in under 15 seconds on a 4-vCPU GitHub Actions runner, with the model loading in under 5 seconds.
3. Keep every run reproducible: config, git commit, dataset revision and W&B link logged in `docs/experiments.md`.

## Core User Flow

1. A developer opens or updates a pull request.
2. The GitHub Action fetches the diff, drops non-code files, and numbers the diff lines.
3. The GGUF model reviews each file's diff, wrapped in `<git_diff>` tags.
4. The output is validated against the Pydantic schema, with one retry on failure.
5. If `pass` is false, inline comments are posted on the flagged lines.
6. If `pass` is true, a clean status goes in the Actions summary and nothing is posted to the PR thread.

## Features

### Model and data

- Diff dataset of 1,500 examples (60% defective / 40% clean), teacher-labeled, split 1,200 / 150 / 150 and published on the HF Hub.
- QLoRA training with completion-only loss on a Kaggle/Colab T4.
- Strict JSON output contract, validated by `src/schema.py`.

### Evaluation

- One harness for the baseline (zero-shot and 2-shot) and the fine-tuned model.
- Metrics: schema validity, false positive rate, defect recall, macro-F1.
- Adversarial check: prompt injection inside code comments.

### Delivery

- Merge, then quantize to Q4_K_M GGUF, published on the HF Hub.
- `review_pr.py` CLI plus a GitHub Actions workflow that caches the GGUF.

## Scope

### In Scope

- Python diffs.
- Four defect categories: `SECURITY`, `BUG`, `PERFORMANCE`, `RESOURCE_LEAK`.
- Milestones M1–M5: dataset and baseline, training, evaluation and error analysis, GGUF packaging, GitHub Action.

### Out of Scope

- Style, naming and formatting feedback.
- Languages other than Python (for now).
- Training on a local Mac (training runs on Kaggle/Colab).
- Executing or auto-applying model output.
- The 7B model (a later option, not part of the current plan).

## Success Criteria

1. The baseline evaluation runs end-to-end on all 150 test samples, outputs a full metrics table, and does not crash on malformed output (M1).
2. The fine-tuned model reaches over 98% JSON validity and under 10% false positive rate on the frozen test set (M3).
3. The GGUF loads in under 5 seconds and reviews a diff in under 15 seconds on CPU (M4).
4. Two live test PRs behave as expected: a clean refactor gets 0 comments, and a PR with a hardcoded secret plus a SQL injection gets 2 inline comments on the right lines (M5).
