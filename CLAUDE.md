# CLAUDE.md — Diff-Mind

QLoRA fine-tune of `Qwen/Qwen2.5-Coder-3B-Instruct` into a **high-signal PR reviewer**: flags only `SECURITY` / `BUG` / `PERFORMANCE` / `RESOURCE_LEAK`, abstains on clean code, emits strict JSON. Shipped as a Q4_K_M GGUF running on a GitHub Actions CPU runner.

The owner is **learning** LoRA fine-tuning and MLOps. Explain non-obvious ML choices briefly, and prefer the simple, reproducible option over the clever one.

## Docs (read on demand)
- [docs/PROJECT_PLAN.md](docs/plans//PROJECT_PLAN.md) — full spec: objectives, metrics, data contract, milestones M1–M5, risk register. **Source of truth for scope.**
- [docs/experiments.md](docs/experiments.md) — experiment log and model lifecycle loop. Update after every train/eval run.

## Context
1. `context/project-overview.md` — product definition, goals, features, and scope
2. `context/architecture.md` — system structure, boundaries, storage model, and invariants
6. `context/progress-tracker.md` — current phase, completed work, open questions, and next steps

Update `context/progress-tracker.md` after eachmeaningful implementation change.

If implementation changes the architecture, scope, orstandards documented in the context files, update the relevant file before continuing.
