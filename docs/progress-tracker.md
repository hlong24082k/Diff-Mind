# Progress Tracker

Update this file after every meaningful implementation
change.

## Current Phase

- Milestone 1 (Dataset Engineering & Baseline Harness), Phase 0 almost complete. Detailed to-do list: `docs/plans/milestone-1.md`.

## Current Goal

- Finish Phase 0 setup, then build the shared prompt formatter (`src/data_pipeline/format_chat.py`, Phase 1).

## Completed

- Project scaffold and `src/schema.py` (self-check passes).
- M1 Phase 0.1: `docs/` removed from `.gitignore` so docs are tracked.
- M1 Phase 0.2 (partly): `google-genai` and `python-dotenv` added to `requirements.txt`; `.venv` created (Python 3.12) with all requirements installed; `.env` created with empty `GEMINI_API_KEY` and `HF_TOKEN`.
- M1 Phase 0.4: prompt format decided (see Architecture Decisions).
- M1 Phase 0.5: severity rubric decided (see Architecture Decisions).
- Context docs filled: `docs/architecture.md`, `docs/project-overview.md`.

## In Progress

- Phase 0.2: owner needs to paste the two keys into `.env`.
- Phase 0.3: verify the Gemini Flash Lite model ID. Blocked on the key.

## Next Up

- Phase 1: `number_diff`, `build_messages` and a `__main__` self-check in `src/data_pipeline/format_chat.py` (owner writes, Claude reviews).

## Open Questions

- Exact Gemini Flash Lite model ID (check with a model-list call once the key is set).
- Phase 5 matching rule for category metrics: diff-level category match (recommended) or category plus line within ±N.
- Phase 5 model backend for baseline runs: local GGUF via `llama-cpp-python` or 4-bit HF load on Kaggle. It must stay the same across baseline runs.

## Architecture Decisions

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
