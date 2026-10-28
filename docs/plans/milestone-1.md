# M1 To-Do: Dataset Engineering & Baseline Evaluation

## Context
Diff-Mind needs a frozen, balanced dataset (1,200/150/150, 60% defective / 40% clean) and a baseline of the untuned Qwen2.5-Coder-3B before any training (M2). **Working mode: Claude guides + reviews; the owner writes all code.** Each item lists *what to build, the key idea, and a done-check*. Check items off after a review.

Decisions made: teacher = Gemini Flash Lite (key via `.env`, model ID to verify), sources = CommitPackFT + CyberNative. Already done: `src/schema.py` (self-check passes). Repo state: all other `src/` files are empty stubs; `requirements.txt` lacks `google-genai`, `python-dotenv`.

---

## Phase 0 — Setup & decisions (do first)
- [x] **0.1 Fix `.gitignore`**: it currently ignores `docs/` and `CLAUDE.md`, so `docs/experiments.md` (required by the MLOps rules) would never be committed. Remove those two lines. *Done when:* `git status` shows `docs/` as untracked.
- [ ] **0.2 Create `.env`** (already gitignored) with `GEMINI_API_KEY=...` and `HF_TOKEN=...`. Add `google-genai` and `python-dotenv` to `requirements.txt`. Install in `.venv`. *(Status: requirements + `.venv` (Python 3.12) + `.env` placeholders done; **you still need to paste the two keys into `.env`** before checking this off.)*
- [ ] **0.3 Verify the teacher model ID** with a one-off call (list models or a 1-line prompt). Don't hardcode a guessed ID; put it in a constant/CLI arg.
- [x] **0.4 Resolve the open prompt-format decision** (CLAUDE.md "Open decision"). Recommended: user prompt =
  `File: {path}\n<git_diff>\n{diff with line numbers}\n</git_diff>`, line numbers prepended to added/context lines in **new-file numbering** (e.g. `42| +code`), since the model's `line` output must map to GitHub inline comments. Record the choice in `docs/PROJECT_PLAN.md` §3.2 or `docs/experiments.md`.
- [x] **0.5 Confirm `Severity` set** (`LOW|MEDIUM|HIGH`) with a rubric sentence the teacher will follow; update the `schema.py` comment.

## Phase 1 — Shared prompt formatter (`src/data_pipeline/format_chat.py`)
*Why first:* the same function feeds the teacher labeler, evaluation, training (M2) and inference (M5). Writing it once prevents train/serve skew.
- [x] **1.1 `number_diff(diff: str) -> str`**: parse hunk headers `@@ -a,b +c,d @@`, track the new-file line counter, prefix lines. Key idea: `-` lines don't advance the new counter. *Done when:* a hand-made 2-hunk diff yields expected numbers (small `assert` in `__main__`).
- [x] **1.2 `build_messages(file_path, diff) -> list[dict]`**: system prompt from plan §3.2 + user prompt from 0.4. Returns `[{role, content}, ...]` (no assistant turn).
- [x] **1.3 Self-check** in `__main__` (project convention, like `schema.py`).
- (EOS / `apply_chat_template` verification belongs to M2; don't build it now.)

## Phase 2 — Raw diff curation (`src/data_pipeline/curate_dataset.py`, step 1)
- [ ] **2.1 Load `bigcode/commitpackft` (Python subset)**, stream with `datasets`. Fields give `old_contents`/`new_contents`; build a unified diff with stdlib `difflib.unified_diff` (no new dependency).
- [ ] **2.2 Load `CyberNative/Code_Vulnerability_Security_DPO`** (Python rows). It has vulnerable/fixed code, not diffs: build a diff by diffing *fixed → vulnerable* code so the added lines contain the bug. *Check the dataset card for column names first.*
- [ ] **2.3 Filters**: Python only, diff 10–150 lines, fits ~2048 tokens with prompt (count with the Qwen tokenizer once, drop >~1,800), dedupe by hash, drop binary/generated/lockfile-like content.
- [ ] **2.4 Over-sample candidates** (~2,500–3,000) because the teacher + balancing will discard many. Write `data/raw_diffs.jsonl` (`id, source, file_path, diff`). *Done when:* count/length histogram printed; spot-check 10 diffs by eye.

## Phase 3 — Teacher labeling (`curate_dataset.py`, step 2)
- [ ] **3.1 Labeling prompt**: rubric (4 categories only; ignore style/naming/docs; abstain if unsure; severity definitions; line numbers refer to the numbered diff). Send the diff through `format_chat.number_diff`.
- [ ] **3.2 Structured output**: use Gemini's JSON-schema response mode with the Pydantic models from `src/schema.py` (import, don't redeclare; `pass` alias makes this fiddly, so a plain `response_mime_type="application/json"` + `PRReviewOutput.model_validate_json` is fine). Reject any label that fails validation.
- [ ] **3.3 Robustness**: `temperature=0`, retry with backoff, **resume-safe** (append to `data/labeled.jsonl`, skip ids already done), env-var key only.
- [ ] **3.4 Label sanity checks**: every `line` actually exists among the numbered diff lines; comment non-empty; drop violators. Hand-review ~30 labels (mix of defective/clean) and note teacher precision in `docs/experiments.md`.
- [ ] **3.5 Balance to 60/40**: 900 defective / 600 clean total. Take clean from teacher-labeled `pass:true` (CommitPackFT mostly), defective from `pass:false`. Ensure all four categories are represented (report counts; no category <~10%, or note the gap). *Done when:* exactly 1,500 rows with printed class/category tables.

## Phase 4 — Split & publish
- [ ] **4.1 Stratified split** 1,200/150/150 with `seed=42` (stratify on clean vs defective, ideally category too). Split **by source commit/file** so near-duplicates don't leak across splits.
- [ ] **4.2 Leakage check**: assert no overlapping ids/diff hashes between splits.
- [ ] **4.3 Dataset schema**: columns `id, file_path, diff, messages (system+user), label_json, is_defective, category`. Store the *teacher label as the exact JSON string* the model must emit (key order `pass`, `reviews`).
- [ ] **4.4 Push to HF Hub** `<username>/pr-review-high-signal` as a `DatasetDict` (train/validation/test); note the commit SHA as the dataset revision. Never overwrite later; new data = new revision/tag. Update `configs/qlora_config.yaml` `dataset:` with the real repo id.
- [ ] **4.5 Freeze the test set**: record test-split hash in `docs/experiments.md`.
- [ ] **4.6 (optional)** `notebooks/01_dataset_curation.ipynb`: thin wrapper that calls the script and shows distributions.

## Phase 5 — Evaluation harness (`src/evaluation/evaluate.py`)
- [ ] **5.1 CLI**: `--model`, `--mode {zero,two-shot}`, `--split test`, `--revision`, `--out`. Greedy decoding (`temperature=0`), fixed `max_tokens`, seed.
- [ ] **5.2 Model backend for the Mac**: run the base model locally via `llama-cpp-python` (already in `requirements.txt`) with a GGUF of Qwen2.5-Coder-3B-Instruct (Q4_K_M or similar), *or* a 4-bit HF load on Kaggle. Pick one per run, keep it stable across baseline runs. (Note for M3: the adapter needs a GPU/merged GGUF, so decide the comparison path now to keep apples-to-apples.)
- [ ] **5.3 Prompting**: zero-shot = `build_messages`; 2-shot = same plus 2 fixed few-shot pairs (1 defective, 1 clean) **taken from the train split, never test**; fixed across runs.
- [ ] **5.4 Parsing**: `json.loads` + `PRReviewOutput.model_validate_json`; any exception = schema failure, never crash. Store raw output per example.
- [ ] **5.5 Metrics** (plan §2): JSON/schema validity %; false-positive rate (clean diffs with `pass:false`); defect recall (defective with `pass:false`); macro-F1 over the 4 categories. Decide and write down the matching rule: *category match (diff-level) vs category + line within ±N*. Recommended simple default: diff-level category match, report line-accuracy separately. Invalid outputs count as failure for recall and as not-flagging for FP (state it).
- [ ] **5.6 Outputs**: `outputs/<run_id>/predictions.jsonl` + `baseline_results.json` with metrics, config, model id, dataset revision, git commit.
- [ ] **5.7 Tiny self-check**: run the metric function on 4 hand-made predictions (incl. malformed JSON) with `assert`s. *Done when:* a fake-model dry run on 5 examples finishes without crashing.

## Phase 6 — Run baselines & log
- [ ] **6.1 Smoke test** on ~10 validation examples (not test) to tune the harness/max_tokens.
- [ ] **6.2 Run `base-0shot` and `base-2shot`** on the full 150 test examples (run once; no prompt tuning on test). Prompt tweaks happen on validation only.
- [ ] **6.3 Fill `docs/experiments.md`** rows `base-0shot`, `base-2shot` (date, dataset revision, commit, metrics, notes). Add a short baseline report: where baseline lands vs plan §2 targets, 3 observed failure patterns.
- [ ] **6.4 Update CLAUDE.md status** ("M1 done") and pin `requirements.txt` versions (`pip freeze`) as its TODO says.

## Exit criteria (from plan)
Evaluation runs end-to-end on all 150 test samples, outputs a full metrics table, and survives malformed outputs; dataset published on HF Hub; baseline logged.

## Verification
- `pytest` passes (schema + format_chat tests in `tests/`).
- Dataset: counts 1200/150/150, ~60% defective in each split, zero overlap.
- Evaluate: run on a stub that returns garbage text; harness reports 0% validity without raising.
- Cost/risk: estimate the teacher API cost on 50 diffs before the full run; labeling is the one expensive/slow step.

## Suggested review checkpoints
After Phase 1, Phase 3 (before spending the full API budget), Phase 4, Phase 5.
