# Architecture Context

## Stack

| Layer | Technology | Role |
| --- | --- | --- |
| Base model | `Qwen/Qwen2.5-Coder-3B-Instruct` | Model being adapted (behavioural alignment, not new knowledge) |
| Fine-tuning | Unsloth + TRL + PEFT + bitsandbytes | QLoRA training (4-bit nf4, double quant, fp16) on Kaggle/Colab T4 |
| Experiment tracking | Weights & Biases | Loss curves, run links logged in `docs/experiments.md` |
| Data / artifacts | Hugging Face Hub (`datasets`, `huggingface_hub`) | Versioned dataset, LoRA adapters, GGUF artifact |
| Teacher labeling | Gemini Flash Lite (`google-genai`) | Distills ground-truth labels for the dataset |
| Output contract | Pydantic (`src/schema.py`) | Single source of truth for model output validation |
| Local inference / eval | `llama-cpp-python` (GGUF) | Baseline and GGUF evaluation on the Mac; CLI in production |
| Quantization | `llama.cpp` / Unsloth `save_pretrained_gguf` | Merged model to Q4_K_M GGUF (~2 GB) |
| CI / delivery | GitHub Actions (CPU runner) + GitHub REST API | Runs the reviewer on PRs, posts inline comments |
| Config | YAML (`configs/qlora_config.yaml`) | All hyperparameters; no magic numbers in scripts |
| Language | Python 3.10+ (local venv on 3.12) | All scripts |

## System Boundaries

- `src/schema.py` — Pydantic output contract (`ReviewItem`, `PRReviewOutput`). Import it; never re-declare it.
- `src/data_pipeline/` — `curate_dataset.py` builds diffs and teacher labels; `format_chat.py` owns the prompt formatter (diff numbering, `<git_diff>` wrapping, chat template, EOS check). One formatter is shared by labeling, training, evaluation and inference.
- `src/training/` — `train_qlora.py`, Unsloth/TRL training. Runs on Kaggle/Colab GPU, not locally.
- `src/evaluation/` — `evaluate.py`, one harness for baseline (M1) and post-training (M3) evaluation on the frozen test set.
- `src/inference/` — `review_pr.py`, production CLI invoked by GitHub Actions.
- `configs/` — `qlora_config.yaml`, every hyperparameter.
- `notebooks/` — self-contained Kaggle/Colab entry points (curation, training, error analysis).
- `.github/workflows/` — CI definition for the PR reviewer.
- `docs/` — plan (source of truth for scope), experiment log, progress tracker, this file.
- `data/`, `outputs/` — local, git-ignored working data and run outputs.

## Storage Model

- **Hugging Face Hub (dataset repo)**: versioned train/validation/test splits (1,200 / 150 / 150). Published revisions are never overwritten; new data means a new revision.
- **Hugging Face Hub (model repos)**: LoRA adapters (~60 MB) and the Q4_K_M GGUF artifact, each pinned by explicit revision.
- **Local disk (git-ignored)**: `data/` for raw and labeled diffs (JSONL), `outputs/` for per-run predictions and metrics, `.venv/`, `.env`.
- **W&B**: training and eval curves.
- **Git repo**: code, config, docs. `docs/experiments.md` records config + git commit + dataset revision + W&B link for every run.

## Auth and Access Model

- Secrets (HF, W&B, teacher API, GitHub tokens) come from environment variables (`.env` locally, Kaggle secrets in notebooks). They are never committed.
- The GitHub Action runs with least privilege: `pull-requests: write`, `contents: read`.
- The teacher API key and HF token are used only in offline data/training steps, not in production inference.

## Invariants

1. Training uses fp16, never bf16 (T4/P100 have no bf16), 4-bit nf4 with double quant, `max_seq_length=2048`.
2. Loss is completion-only: prompt tokens have `labels == -100`, verified by tensor inspection. EOS `<|im_end|>` follows the closing `}`.
3. The test set is frozen (150 examples). Never train, tune prompts, or select checkpoints on it; use validation. Every model is compared on the same test set.
4. Output is raw JSON with no markdown fences; `pass` is true iff `reviews` is empty. It is always validated with `src/schema.py`.
5. Diffs are wrapped in `<git_diff>` tags, and model output is never executed.
6. The same formatter produces prompts for training, evaluation and inference (otherwise train/serve skew).
7. Evaluation must not crash on malformed model output; it counts it as a schema failure.
8. Hyperparameters live in the config file and everything is seeded (`seed: 42`).
9. A model is promoted only if it beats the previous one on the frozen test set across the plan §2 metrics.
