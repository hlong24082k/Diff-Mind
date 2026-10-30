# Progress Tracker

Update this file after every meaningful implementation
change.

## Current Phase

- Milestone 1 (Dataset Engineering & Baseline Harness), Phases 0–2 done, Phase 3 code done (full labeling run pending). Detailed to-do list: `docs/plans/milestone-1.md`.

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

## In Progress

- Phase 0.3: verify the Gemini Flash Lite model ID. Blocked on the key.

- Phase 1: `number_diff` and `build_messages` in `src/data_pipeline/format_chat.py`; tests in `tests/test_format_chat.py` (`pytest`, added to requirements).

## Next Up

- Phase 3 run: `label --limit 50` (cost), full `label`, `review` (log teacher precision in `docs/experiments.md`), `balance` -> `data/balanced.jsonl`. Then Phase 4 (split + publish).

## Open Questions

- Exact Gemini Flash Lite model ID (check with a model-list call once the key is set).
- Phase 5 matching rule for category metrics: diff-level category match (recommended) or category plus line within ±N.
- Phase 5 model backend for baseline runs: local GGUF via `llama-cpp-python` or 4-bit HF load on Kaggle. It must stay the same across baseline runs.

## Architecture Decisions

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
