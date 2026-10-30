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

## Dataset notes

### Teacher label review (M1 Phase 3.4, 2026-10-30)
Teacher `gemini-3.5-flash-lite`, 2,800 diffs -> 2,793 valid labels (7 rejected by schema: 5x `pass` vs `reviews` mismatch, 2x bad JSON/missing field). Raw result: 2,243 `pass:true`, 550 defective (CyberNative 330, CommitPackFT 220). Categories among all reviews: SECURITY 331, BUG 223, RESOURCE_LEAK 25, PERFORMANCE 3.

Reviewed by Claude (not yet by the owner): 30 labels, stratified 8/8/7/7 over {CyberNative, CommitPackFT} x {defective, clean}, seed 7. Small sample, rough numbers only.
- **CyberNative defective: 8/8 correct** (eval/os.system/shell=True/exponential fib). 1 diff contains junk text (`else:archar`) from the source dataset.
- **CommitPackFT defective: ~3 clearly right, ~2 arguable, ~3 false positives** (speculative "may raise if None"/"assumes same length"). Precision ~40-60%.
- **CyberNative teacher-clean: 6/7 wrong** (pickle `__reduce__` -> `os.system` exploits passed as clean). Decision: never use CyberNative `pass:true` as clean (`balance` excludes it).
- **CommitPackFT clean: 7/7 plausible.**
- Overall defective precision ~75%; the teacher over-flags speculative bugs on real commits. Consider tightening the rubric ("only flag defects you can point to in the added lines, not hypothetical callers") before relabeling.

**Gap:** the 60/40 target of 900 defective is not reachable: only 550 defective labels, and PERFORMANCE (3) / RESOURCE_LEAK (25 -> ~1%) are far under the 10% floor.

### Balanced dataset (M1 Phase 3.5, 2026-10-30)
Added 1,200 reversed CommitPackFT fix commits (`commitpackft_fix`: fixed -> buggy) and labeled them: only 251 came out defective (~21%). Totals over 4,000 labeled diffs (3,988 valid): 801 defective, so the planned 900 was **not reachable**; `balance --n-defective 801 --n-clean 534` keeps all of them (60/40) -> **1,335 rows** in `data/balanced.jsonl` instead of 1,500 (Phase 4 split becomes ~1,068/133/134).
- Defective by source: CyberNative 330, CommitPackFT 220, CommitPackFT-fix 251. Clean: 534, all forward `commitpackft` (teacher-clean from CyberNative and from reversed fixes is excluded as untrustworthy).
- Defective categories: BUG 405 (51%), SECURITY 345 (43%), RESOURCE_LEAK 43 (5%), **PERFORMANCE 8 (1%)**. The <10% floor is missed for the last two: the model cannot be expected to learn PERFORMANCE; M3 should report per-category recall and treat PERFORMANCE as out of reach for this data (options later: synthetic perf/leak examples, or drop PERFORMANCE from the claimed scope).
- Known shortcut risk: CyberNative (330 of 801 defective, synthetic whole-snippet rewrites, 100% defective) vs. clean = real commits only. Check Phase 5/6 results for a source effect.

### Split & frozen test set (M1 Phase 4, 2026-10-30)
`python -m src.data_pipeline.publish_dataset` splits `data/balanced.jsonl` (seed 42, stratified on defective x category, ~10% each for validation/test) -> **train 1,071 / validation 132 / test 132**, 60% defective in each. Local copy in `data/dataset/*.jsonl` (gitignored). Leakage check (id, diff hash, source commit) passes.
- Test categories: BUG 40, SECURITY 34, RESOURCE_LEAK 4, PERFORMANCE 1, clean 53. **Per-category F1 on PERFORMANCE (1 example) and RESOURCE_LEAK (4) is noise**; report them as counts, not as a metric.
- **Frozen test set sha256:** `cbd859afbc8102fa566349d2563f303c7c282c35911029b4ccd423f307e41cda` (over sorted id, diff, label_json). Recompute with the script; a different value means the test set changed and old results are not comparable.
- HF Hub revision: _not published yet_ (token is read-only). Fill in once pushed.

## Promoted model
Track the single model currently used in production (HF repo + revision + GGUF filename) here.
