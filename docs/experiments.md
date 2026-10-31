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
| base-0shot | 2026-10-30 | Qwen2.5-Coder-3B-Instruct (Q4_K_M GGUF) | local test sha256 `cbd859af…1cda` (HF not pushed) | `0a51c64`, `outputs/base-0shot/` | — | 0.0% (0/132) | 0.0%* | 0.0%* | 0.000 | M1 baseline. Every output is fenced and uses the wrong schema; see baseline report. *Invalid = not flagged, so FP/recall are vacuous. |
| base-2shot | 2026-10-30 | Qwen2.5-Coder-3B-Instruct (Q4_K_M GGUF) | local test sha256 `cbd859af…1cda` (HF not pushed) | `0a51c64`, `outputs/base-2shot/` | — | 95.5% (126/132) | 49.1% (26/53) | 72.2% (57/79) | 0.375 | M1 baseline. Recall inflated by CyberNative (35/35); commit-only recall 22/44 = 50%. Hypothesis for v1: fine-tune to cut FP on clean commits. |
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
- HF Hub revision: 613e5bbfb6aab9b57129d7e0ca936de6c5825bc4 (repo HLeon/diff-mind-pr-review-high-signal, private).

### Baseline report (M1 Phase 6, 2026-10-30)
Untuned Qwen2.5-Coder-3B-Instruct, Q4_K_M GGUF via llama-cpp-python, greedy, `max_tokens` 512, seed 42, full test split (**132** examples: 79 defective / 53 clean, not the 150 in the plan). Smoke test (10 validation rows) showed the same 0-shot failure and no output near the token cap, so settings were left alone; nothing was tuned on test. 2-shot shots are the shortest defective + shortest clean train diffs (`cpf-4ca25c413494-b7764a`, `cp-5eb765ee3ad9-6defec`). Runs were made with `git_commit` `0a51c64+dirty`; the only uncommitted change was the `requirements.txt` version pin, no code differed.

| Metric | Plan §2 baseline guess | base-0shot | base-2shot | Post-QLoRA target |
| :-- | :-- | :-- | :-- | :-- |
| JSON valid (strict) | 65-80% | 0.0% | 95.5% | >98% |
| FP rate (clean flagged) | 50-80% | 0%* | 49.1% | <10% |
| Defect recall | ~60% | 0%* | 72.2% | >80% |
| Macro-F1 (4 categories) | measured here | 0.000 | 0.375 | +20% over baseline |
| Line accuracy (±2, matched category) | - | n/a | 57.4% | - |

2-shot per category (gold support / predicted): SECURITY F1 0.77 (34/47), BUG 0.40 (40/35), RESOURCE_LEAK 0.33 (5/1), PERFORMANCE 0.00 (1/0). The last two are counts, not metrics (support 5 and 1). **The 2-shot macro-F1 baseline to beat is 0.375**, but it is dominated by SECURITY; compare per category in M3.

**Failure patterns observed**
1. **0-shot ignores the output contract.** 132/132 outputs are wrapped in a ```` ```json ```` fence, and even after stripping the fence 0/132 validate: 85 use an invented `{"critical_defects": [...]}`, 42 use per-category keys (`{"SECURITY": [...], "BUG": [...]}`), 5 aren't JSON. The system prompt's schema is not followed without examples, so the 0-shot row measures format compliance only; FP/recall/F1 are not meaningful. Two examples fix the format almost entirely (fences 0/132).
2. **Over-flagging clean commits (FP 49%).** On 51 valid clean outputs, 26 are flagged (BUG 27 reviews, SECURITY 10): speculative "may lead to security issues if the path is not sanitized" for `os.chdir`/`cd`, or a changed test constant read as "a potential issue with the data". Severity is nearly always HIGH (97/98 reviews), so severity carries no signal. This is the behaviour fine-tuning must remove.
3. **Source shortcut / inflated recall.** Valid 2-shot outputs flag 35/35 CyberNative defective diffs (blatant `eval`/`os.system`/pickle) but only 22/44 real-commit defective diffs (CommitPackFT 9/18, reversed fixes 13/22; invalid counted as misses), i.e. 50% on real commits vs 72% overall. M3 must report recall by source so a gain isn't just "learned CyberNative style".
4. (Smaller) **Malformed JSON even with examples: 6/132 invalid.** 3 truncated at the 512-token cap (long run-on outputs), 3 with unescaped quotes/backslashes inside `comment` strings. Raising `max_tokens` would fix the first kind only; left unchanged so M3 compares at the same setting.

**Caveats:** no HF revision yet (4.4 blocked); `llama-cpp-python` 0.3.36 chat template is the GGUF's own, not HF's `apply_chat_template` (verify for M2/M3 parity). Single seed/greedy: 132 examples gives roughly ±8pt 95% intervals on the rates.

## Promoted model
Track the single model currently used in production (HF repo + revision + GGUF filename) here.
