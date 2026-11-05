# Progress Tracker

Update this file after every meaningful implementation
change.

## Current Phase

- Milestone 2 (QLoRA Training, Kaggle T4): **done (training 2026-11-01, closed 2026-11-03)**: run `v1` on commit `6bf9ad6`, best `eval_loss` 0.3984 at step 125, peak VRAM 3.21 GiB, sanity 10/10 stopped and valid, adapter `HLeon/diff-mind-qwen2.5-coder-3b-lora` @ `a32dbbb5…`, W&B `ucuukj2r`. Details in `docs/experiments.md` ("v1 training notes"). Versions pinned in notebook cell 2 (6.2 done). Local `outputs/v1/` has `train_summary.json`, `rendered_example.txt`, `versions.txt` from v1; its `sanity.jsonl` is the smoke-run copy (the Hub root also holds stale smoke `train_summary.json`/`sanity.jsonl`), so the 10/10 sanity result comes from the Kaggle cell 6b output. Notebook cell 6 now deletes those smoke leftovers before the full run. To-do list: `docs/plans/milestone-2.md`.
- Milestone 1 (Dataset Engineering & Baseline Harness): Phases 0–3, 5, 6 done; Phase 4 done except 4.4 (HF push, needs a write token). Detailed to-do list: `docs/plans/milestone-1.md`.

## Current Goal

- Run Phase 3 for real: `--step label` on all 2,800 diffs, hand-review 30 labels, then `--step balance`.

## Completed

- Project scaffold and `src/schema.py` (self-check passes).
- M1 Phase 0.1: `docs/` removed from `.gitignore` so docs are tracked.
- M1 Phase 0.2 (partly): `google-genai` and `python-dotenv` added to `requirements.txt`; `.venv` created (Python 3.12) with all requirements installed; `.env` created with empty `GEMINI_API_KEY` and `HF_TOKEN`.
- M1 Phase 0.4: prompt format decided (see Architecture Decisions).
- M1 Phase 0.5: severity rubric decided (see Architecture Decisions).
- Context docs filled: `docs/architecture.md`, `docs/project-overview.md`.

- M1 Phase 2: `src/data_pipeline/curate_dataset.py` (step 1) writes `data/raw_diffs.jsonl` (2,800 diffs: 2,385 CommitPackFT + 415 CyberNative, all unique, 10–150 lines, ≤1,800 tokens by the Qwen tokenizer). Run: `python -m src.data_pipeline.curate_dataset`. `tokenizers` added to `requirements.txt`; tests in `tests/test_curate_dataset.py`.
- M1 Phase 0.2/0.3: keys work; teacher model `gemini-3.5-flash-lite` verified via `models.list()` (2026-10-28), pinned as `TEACHER_MODEL` (override with `--model`).
- M1 Phase 3 (code): `curate_dataset.py --step label|review|balance`. Smoke test on 5 diffs: 5/5 valid, ~540 input / ~20 output tokens per diff. Full run NOT done yet. Free-tier cap ~15 req/min and calls take 5-17 s, so labeling uses a shared rate limiter (`--rpm 14`) with `--workers 6`; 30-diff test (4 workers) took 3 min (~10/min) -> full run of ~2,750 remaining diffs ~3.5-4.5 h, resume-safe, stops cleanly on a daily-quota error (`PerDay`). Tests added for `check_label` and `balance`.
- M1 Phase 3 labeling v2: 10 diffs per request (`--batch`) and API-key rotation (`GEMINI_API_KEYS=k1,k2,...` in `.env`, falls back to `GEMINI_API_KEY`; a key is dropped on a `PerDay` 429). Free tier is 500 requests/day/key/model, so one key covers the remaining ~2,300 diffs (~230 requests, ~17 min at 14 req/min). Tests cover rotation and missing batch items. Not yet run live (key's daily quota was used up by the single-diff run; resets ~6 h later).
- M1 Phase 3.4: all 2,800 diffs labeled (2,793 valid). 30-label review done (findings in `docs/experiments.md` "Dataset notes"). `balance` now never uses CyberNative `pass:true` as clean and takes `--n-defective/--n-clean`.
- M1 Phase 3.5 prep: `--step fixes --target 1200` appended 1,200 reversed fix-commit diffs (`source: commitpackft_fix`: 61 leak, 177 perf, 200 security, 762 bug keyword groups) to `data/raw_diffs.jsonl` to fill the defective/PERFORMANCE/RESOURCE_LEAK shortfall. 200 labeled so far, **1,000 still unlabeled** (owner will run `--step label`), then `--step balance`.
- M1 Phase 3.5: `data/balanced.jsonl` written, 1,335 rows (801 defective / 534 clean, 60/40). Short of 1,500 and the category floor (PERFORMANCE 8, RESOURCE_LEAK 43); details in `docs/experiments.md`. Phase 3 complete. Clean pool = forward `commitpackft` only.
- M1 dataset analysis: `python -m src.data_pipeline.analyze_dataset` writes 7 charts + `data_tables.md` (numbers behind each chart) to `docs/figures/`. `matplotlib` added to `requirements.txt`.

## In Progress

- M1 Phase 6 done: `base-0shot` (JSON valid 0%: fenced + wrong schema) and `base-2shot` (valid 95.5%, FP 49.1%, recall 72.2%, macro-F1 0.375) on the 132-row test split; full report in `docs/experiments.md`. `requirements.txt` direct deps pinned. `python -m pytest` works; bare `pytest` fails with `No module named 'src'` (no conftest/pyproject).

- Phase 0.3: verify the Gemini Flash Lite model ID. Blocked on the key.

- Phase 1: `number_diff` and `build_messages` in `src/data_pipeline/format_chat.py`; tests in `tests/test_format_chat.py` (`pytest`, added to requirements).

## Done (recent)

- Phase 4.1-4.3, 4.5: `src/data_pipeline/publish_dataset.py` (split 1,071/132/132, leakage check, dataset columns, test hash); test in `tests/test_curate_dataset.py`.

- Phase 5: `src/evaluation/evaluate.py` + `tests/test_evaluate.py` (15 tests pass). Run: `python -m src.evaluation.evaluate --model <x.gguf|repo-id|fake> --mode zero|two-shot --split validation --limit 10`. Writes `outputs/<run_id>/{predictions.jsonl,baseline_results.json}`. `--model fake` (garbage stub) gives 0% validity without crashing. **Not yet run on a real model** (no GGUF downloaded; Phase 6.1).

## Next Up

- **M3 Phase 3 done (2026-11-05):** `bootstrap_ci` in `evaluate.py` (1,000 row resamples, seed 42, percentile 95% CI for valid / FP / recall / macro-F1 / line acc.; written to `metrics["ci95"]`). `src/evaluation/compare.py` recomputes everything from `predictions.jsonl` + `source` joined from `data/dataset/test.jsonl` and prints 3 markdown tables. On the baselines it reproduces M1 (2-shot 95.5 / 49.1 / 72.2 / 0.375; CyberNative 35/35, commit-based 22/44). Note: base-2shot macro-F1 CI is very wide [0.262, 0.631] because PERFORMANCE has 1 test row (it drops in and out of the macro average across resamples). Tests: `tests/test_evaluate.py::test_bootstrap_ci`, `tests/test_compare.py`. **Next:** Phase 4.1 headline run.
- **M3 Phase 2.1 implemented (2026-11-05):** `src/evaluation/check_template.py` (+ `tests/test_check_template.py`) renders the GGUF's embedded template through llama-cpp's own `Jinja2ChatFormatter` and diffs it against the prompt part of `outputs/v1/rendered_example.txt` (train row `cn-1`, since that is the row the reference was rendered from). On `v1-s125-q4_k_m`: byte-identical, `chat_format=chat_template.default`, eos `<|im_end|>`. 2.2 smoke (validation, 10 rows): 10/10 valid bare JSON, no truncation, recall 5/5 (all CyberNative), FP 2/5: both are BUG claims on commit-based clean rows that the code contradicts (a `None` case already guarded by `if ... is not None`; `format(x, '')` called invalid). Early hint of a *speculative-flag* cause for Phase 7; nothing tuned. 2.3 skipped. **Next:** Phase 3.
- **M3 Phase 0 and Phase 1 done (2026-11-05; 1.4 partly):** protocol + thresholds pre-registered; test sha256 `cbd859af…1cda` verified; baselines re-score to 95.5% / 49.1% / 72.2% / 0.375 (2-shot). `src/training/export_gguf.py` + `notebooks/04_export_gguf.ipynb` ran on Kaggle (T4 needed: without a GPU Unsloth falls back to CPU and cannot merge a 4-bit model; exports go to `/tmp/gguf`). Scratch repo `HLeon/diff-mind-v1-gguf` holds `v1-s125-q4_k_m` (1.80 GiB), `v1-s125-f16` (5.75 GiB), `v1-s201-q4_k_m` (1.80 GiB). `v1-s125-q4_k_m` downloaded and loads in llama-cpp. **Next:** Phase 2 (template parity vs `outputs/v1/rendered_example.txt`, then validation smoke) on that file; download the f16 and step-201 files before Phase 4.2/4.3.
- **M3 plan written (2026-11-03):** `docs/plans/milestone-3.md` (decisions from a grilling session: merge to Q4_K_M GGUF on Kaggle, pre-registered step-125 headline, 3 ablations, bootstrap CIs, ~8 adversarial probes, label-noise triage, decision gate instead of v2). Start at Phase 0.
- **M3**: merge the v1 adapter, convert to a Q4_K_M GGUF, and run `evaluate.py` on the frozen test split against the base-2shot baseline (FP 49.1%, recall 72.2%, macro-F1 0.375). Diff `outputs/v1/rendered_example.txt` against the GGUF's own chat template first (train/serve parity). Report recall by source and per category.

- Phase 6.1: smoke test on ~10 validation examples with `Qwen/Qwen2.5-Coder-3B-Instruct-GGUF` (`--gguf-file '*q4_k_m.gguf'`, ~2 GB download), then 6.2 baselines on the test split. Note the test split is 132 rows, not 150.

- Phase 4.4: push `data/dataset/` to the HF Hub (`.env` `HF_TOKEN` is read-only; needs a write token): `python -m src.data_pipeline.publish_dataset --push HLeon/pr-review-high-signal`, then record the revision in `docs/experiments.md` and set `dataset:` in `configs/qlora_config.yaml`. Then Phase 5 (evaluation harness).

## Open Questions

- Exact Gemini Flash Lite model ID (check with a model-list call once the key is set).

## Architecture Decisions

- **M2 training stack**: Kaggle T4, Unsloth + TRL `SFTTrainer`, completion-only loss via `train_on_responses_only`, W&B tracking, 3 epochs with eval/save every 25 steps and `load_best_model_at_end` on `eval_loss`. Logic lives in `src/training/train_qlora.py` + `configs/qlora_config.yaml`; the notebook is a thin runner. Training text is rebuilt from `build_messages` (no train/serve skew). Smoke runs (`--max-steps`) never push to the Hub. Adapter is private on the Hub with `hub_strategy="checkpoint"`.

- **M3 evaluation protocol (pre-registered 2026-11-03, before any v1 test run; full rationale in `docs/plans/milestone-3.md`)**:
  - Eval path: merge the adapter and convert to GGUF on Kaggle (Unsloth), then run the **unchanged** `evaluate.py` on the **Q4_K_M** GGUF (same backend, template and decoding as the baselines). Adapter step 125 (picked by validation `eval_loss`) is the **v1 headline**; step 201 is an ablation and can never replace it.
  - Pass/fail thresholds on the frozen 132-row test split (sha256 `cbd859af…1cda`; 79 defective / 53 clean), vs base-2shot (valid 95.5%, FP 49.1%, recall 72.2%, macro-F1 0.375): JSON valid **>98%** (>=130/132), FP rate **<10%** (<=5/53), recall **>80%** (>=64/79), macro-F1 **>=0.45**. Each reported with a bootstrap 95% CI; "pass within noise" if the point estimate passes but the CI crosses the line.
  - Ablations: (a) step 125 vs step 201, (b) f16 GGUF vs Q4_K_M, (c) v1 + 2-shot prompt. Per-category results are counts only for PERFORMANCE (1 test row) and RESOURCE_LEAK (4).
  - Hygiene: smoke-test on validation first; test is run once per final candidate from a clean git tree; nothing is tuned after seeing test results. Label-noise "adjusted" numbers (error analysis) are reported beside, never instead of, the frozen ones; `data/dataset/` is never edited.
  - M3 ends at a decision gate (promote / iterate / discard); any v2 gets its own plan.
- **Phase 5 eval rules**: backend = local GGUF via `llama-cpp-python` (Q4_K_M), greedy, `max_tokens` 512, seed 42; keep it identical across baseline runs and use the same GGUF path for M3 (merged adapter) so comparisons stay apples-to-apples. Validity is strict (`PRReviewOutput.model_validate_json`, no fence stripping). Invalid output = not flagged: a miss for recall, not a false positive. Category F1 is diff-level over category *sets* (pred vs gold `label_json`); macro-F1 averages categories that appear in gold or predictions; line accuracy (±2 lines, only where the category matched) is reported separately. 2-shot examples = shortest defective + shortest clean train diffs (deterministic). `--limit` takes a seeded random subset (the splits are id-sorted).

- **Phase 2 data sources**: `datasets` 5.x cannot run `bigcode/commitpackft`'s loading script, so it streams `data/python/data.jsonl` directly (`load_dataset("json", data_files="hf://...")`). CyberNative is a single JSONL file (`secure_programming_dpo.json`); 424 Python rows, code wrapped in markdown fences (stripped). Diffs are `difflib.unified_diff`, 3 context lines, `a/`/`b/` headers. Token counting uses the `tokenizers` library only (no `transformers`).
- **Phase 2 known shortcut risks** (check at Phase 3.4/4): CyberNative rejected code often names things `vulnerable_function` or has "vulnerable" comments, and its diffs are whole-snippet rewrites, unlike real commits. Both could let the model learn "source style => defective". CyberNative paths are hash-picked from 8 generic names to avoid a constant-path shortcut. The dataset's own 2026-10-06 audit also says its chosen/rejected labels are noisy, so the teacher label (not the source) decides defective vs clean.

- **Prompt format**: one formatter shared by teacher labeling, training, evaluation and inference. User prompt is `File: {file_path}` followed by the diff inside `<git_diff>...</git_diff>`. Why: the risk register requires delimiters against prompt injection and explicit line numbers against line-number hallucination; one formatter avoids train/serve skew.
- **Line numbers**: each context/added line is prefixed with its new-file line number (`42| +code`). Removed lines get no number. Why: the model's `line` output must map directly to GitHub inline-comment lines.
- **Severity**: `LOW | MEDIUM | HIGH`. `HIGH` is exploitable or data-loss/crash in normal use, `MEDIUM` is wrong behaviour or significant slowdown/leak under realistic conditions, `LOW` is minor or edge-case. Why: the plan only showed `HIGH`, and the teacher labeler needs a rubric.
- **Teacher**: Gemini Flash Lite, key via `.env`. **Diff sources**: `bigcode/commitpackft` plus `CyberNative/Code_Vulnerability_Security_DPO`.
- **Python 3.12 venv** instead of the default 3.14, to avoid `llama-cpp-python` wheel and build problems.
- **Working mode**: the owner writes the code; Claude plans, guides and reviews.
- **Doc rule**: do not update `CLAUDE.md` or `docs/plans/PROJECT_PLAN.md`; record decisions here instead.

## Session Notes

- 2026-10-28: Earlier in this session Claude edited `docs/plans/PROJECT_PLAN.md` §3.2 (added the prompt-format decision and severity rubric) and `src/schema.py` (comment now points at the plan §3.2 severity rubric). The PROJECT_PLAN edit predates the "don't touch" rule and is left as is. Tell Claude if you want it reverted.
- Resume by pasting the keys into `.env`, then ask Claude to run the 0.3 model ID check.

- **Phase 3 labeling**: teacher uses its own rubric as system instruction but the *same* `build_messages` user turn (numbered diff) as the student. Gemini JSON mode (`response_mime_type`) + `PRReviewOutput.model_validate_json`; label stored as the canonical `model_dump(by_alias=True)` string (key order `pass`, `reviews`). Rejects (schema fail, `line` not a numbered diff line) are written to `data/labeled.jsonl` with `label_json: null` + `error` so re-runs skip them; API failures after 5 retries are not written, so re-runs retry them. Balance: defective category = highest-severity review; defective rows are drawn from the least-represented category first (as even as data allows), then a source x class table is printed to expose the CyberNative-vs-CommitPackFT shortcut risk.

- **Batched teacher labels**: the first ~490 labels came from one-diff-per-request calls, the rest from 10-per-request calls (different prompt tail: `=== DIFF k ===` blocks + JSON array). Compare teacher precision of the two when hand-reviewing (`--step review`); relabel the first set if batching is visibly worse. Gemini quotas are per Google Cloud project, so extra keys only help if they come from different projects.
