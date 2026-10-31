# M2 To-Do: QLoRA Training Pipeline (Kaggle T4)

## Context
Fine-tune `Qwen/Qwen2.5-Coder-3B-Instruct` with QLoRA on the M1 dataset (train 1,071 / validation 132) so it emits the strict JSON contract and stops flagging clean commits (base-2shot FP 49%). **Working mode: Claude guides + reviews; the owner writes all code.** Each item lists *what to build, the key idea, and a done-check*. Snippets are hints, not finished code. Check items off after a review.

**Decisions (grilling session, 2026-10-31):**
1. Platform: **Kaggle T4** (one GPU; Unsloth ignores the second T4). Colab only as fallback.
2. Code shape: logic in `src/training/train_qlora.py`, reading `configs/qlora_config.yaml`. `notebooks/02_qlora_training.ipynb` is thin: clone, install, secrets, `!python -m src.training.train_qlora ...`.
3. Data: get an HF **write** token, push the dataset (M1 4.4) first, pin its revision in the config. Training loads from the Hub, never from local files.
4. Stack: **Unsloth + TRL `SFTTrainer`**, masking via Unsloth's `train_on_responses_only`.
5. Tracking: **W&B** (`report_to="wandb"`), run URL logged in `docs/experiments.md`.
6. Schedule: **3 epochs**, eval + save every 25 steps, `load_best_model_at_end` on `eval_loss`.
7. Pre-flight: `--check` flag (masking/EOS/length asserts, then exit) → `--max-steps 10` smoke run (VRAM) → full run.
8. Sanity generation on **10 validation rows** after training (JSON parses, stops at `<|im_end|>`). Not a metric; never on test.
9. Adapter: private `HLeon/diff-mind-qwen2.5-coder-3b-lora`, `hub_strategy="checkpoint"` so a dead session can resume.
10. Local test: one pure data function + pytest; heavy imports (`unsloth`, `torch`, `trl`) only inside `main()`.

**Defaults:** OOM fallback `per_device_train_batch_size=2`, `gradient_accumulation_steps=8` (effective batch stays 16); `packing=False`; `max_seq_length=2048`; run id `v1`; Kaggle checks out an exact commit; Kaggle lib versions pinned after the first working run.

**Facts:** GitHub repo `hlong24082k/Diff-Mind` is public, so Kaggle can `git clone` without a token (Internet must be ON). ~67 optimizer steps/epoch → ~200 steps total, ~8 evals. Dataset columns: `id, source, file_path, diff, messages, label_json, is_defective, category`.

---

## Phase 0 — Unblock & setup (do first)
- [ ] **0.1 HF write token + push dataset** (closes M1 4.4). Create a *fine-grained write* token, put it in `.env` `HF_TOKEN`, run `python -m src.data_pipeline.publish_dataset --push HLeon/diff-mind-pr-review-high-signal`. Record the printed revision SHA in `docs/experiments.md` (Split & frozen test set → "HF Hub revision"). *Done when:* `load_dataset("HLeon/diff-mind-pr-review-high-signal", revision=SHA)` returns 1,071/132/132 and the test sha256 still equals `cbd859af…1cda`.
- [ ] **0.2 Accounts & Kaggle secrets.** W&B account → API key. Kaggle notebook: Accelerator *GPU T4*, Internet ON, Add-ons → Secrets: `HF_TOKEN` (write), `WANDB_API_KEY`. *Done when:* a cell with `UserSecretsClient().get_secret("HF_TOKEN")` + `huggingface_hub.whoami()` prints your user.
- [ ] **0.3 Update `configs/qlora_config.yaml`**: `dataset: HLeon/diff-mind-pr-review-high-signal`, add `dataset_revision: <SHA>`, `hub_model_id: HLeon/diff-mind-qwen2.5-coder-3b-lora`, `wandb_project: diff-mind`, `run_name: v1`, `save_steps: 25` (matches `eval_steps` so the best checkpoint is never between saves), `warmup_steps: 10`, `logging_steps: 5`.
  *Why warmup:* LoRA B matrices start at zero; ~5% warmup keeps the first steps at 2e-4 from overshooting. The plan doesn't specify it; record it as a deviation.

## Phase 1 — Example → chat (local, testable on the Mac)
*Why first:* if the training text differs from what `evaluate.py` / `review_pr.py` send, the model learns one prompt and is served another (train/serve skew). This is the cheapest bug to catch locally.
- [ ] **1.1 `to_chat(example) -> list[dict]`** in `train_qlora.py`: `build_messages(example["file_path"], example["diff"]) + [{"role": "assistant", "content": example["label_json"]}]`. Rebuild from `build_messages` (the single source) instead of trusting the stored `messages` column.
- [ ] **1.2 Keep module imports light**: top of file imports only stdlib + `src.data_pipeline.format_chat`; `unsloth`/`torch`/`trl`/`datasets` imported inside `main()`. *Why:* the local venv has no torch/unsloth and the test must import the module.
- [ ] **1.3 Test `tests/test_train_qlora.py`** (2 asserts on 1-2 rows of `data/dataset/train.jsonl`): (a) `to_chat(ex)[:2] == ex["messages"]` (stored prompt == current formatter → catches a stale dataset), (b) last turn is `assistant` with content exactly `label_json` and `json.loads` works. *Done when:* `python -m pytest` passes.

## Phase 2 — Model + LoRA load (`train_qlora.py main()`, runs on Kaggle)
- [ ] **2.1 CLI**: `--config` (default the yaml), `--check`, `--max-steps` (default -1 = use epochs), `--run-name` (override). Read yaml with `yaml.safe_load`; nothing else configurable.
- [ ] **2.2 Load 4-bit base**: `FastLanguageModel.from_pretrained(model_name=cfg["base_model"], max_seq_length=2048, dtype=torch.float16, load_in_4bit=True)`. Unsloth applies bitsandbytes **nf4 + double quant** internally, which is what the plan asks for; the yaml `quant:` block documents it, so don't build a `BitsAndBytesConfig` yourself.
- [ ] **2.3 Attach LoRA**: `FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, lora_dropout=0, target_modules=[...7 proj...], use_gradient_checkpointing="unsloth", random_state=42)`. Print `model.print_trainable_parameters()` (expect roughly 30M trainable, ~1% of params).
  *ML note:* the update is scaled by `alpha/r = 2`. All 7 linear layers (attention + MLP) are targeted because format/behaviour changes need MLP capacity, not just attention.
- [ ] **2.4 Build text column**: `ds = load_dataset(cfg["dataset"], revision=cfg["dataset_revision"])`, then `.map(lambda ex: {"text": tok.apply_chat_template(to_chat(ex), tokenize=False)})`. No `add_generation_prompt` here, because the assistant turn is already included.
- [ ] **2.5 Pad ≠ EOS check**: assert `tok.eos_token == "<|im_end|>"` and `tok.pad_token != tok.eos_token` (Qwen pads with `<|endoftext|>`). *Why:* if pad == eos, padding-masking logic can also mask the real EOS, and the model never learns to stop. Set `tok.pad_token` explicitly if it's missing.

## Phase 3 — Completion-only masking & `--check`
*Why completion-only:* the prompt is ~90% of the tokens. Training on it would spend the gradient on reproducing diffs instead of learning the JSON answer. Only the assistant JSON + `<|im_end|>` should carry loss.
- [ ] **3.1 Create the `SFTTrainer`** (Phase 4.1 config), then wrap it: `trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")`.
- [ ] **3.2 `--check` mode** (after 3.1, before `train()`): take `trainer.train_dataset[0]`, then:
  - decode `input_ids` where `labels != -100` → assert it starts with `{"pass"` and ends with `<|im_end|>` (a trailing `\n` after it is fine), and that it equals the row's `label_json` + `<|im_end|>` (± that newline);
  - assert the count of active labels is much smaller than the sequence length (print both);
  - print the full `text` of one example once, so you can eyeball the template (system / user / assistant blocks). Save it to `outputs/v1/rendered_example.txt` for M3 to diff against the GGUF chat template (M1 caveat: llama-cpp uses the GGUF's own template).
  - token-length stats over train+validation: max, p95, and **count > 2048 must be 0** (M1 capped diffs at 1,800 tokens; label JSON adds a few hundred at most).
  Then `sys.exit(0)`. *Done when:* `--check` passes on Kaggle and you've read the decoded masked/unmasked split yourself.
- (Library check: TRL renamed some `SFTConfig` args across versions, e.g. `max_seq_length` → `max_length`. Read the error and match the installed version rather than guessing.)

## Phase 4 — Trainer config & runs
- [ ] **4.1 `SFTConfig`** from the yaml: `per_device_train_batch_size=4`, `gradient_accumulation_steps=4`, `num_train_epochs=3`, `max_steps=args.max_steps`, `learning_rate=2e-4`, `lr_scheduler_type="cosine"`, `warmup_steps=10`, `optim="paged_adamw_8bit"`, `fp16=True`, `bf16=False`, `eval_strategy="steps"`, `eval_steps=25`, `save_steps=25`, `save_total_limit=2`, `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`, `greater_is_better=False`, `logging_steps=5`, `report_to="wandb"`, `run_name`, `push_to_hub=True`, `hub_model_id`, `hub_strategy="checkpoint"`, `hub_private_repo=True`, `dataset_text_field="text"`, `packing=False`, `seed=42`, `output_dir="outputs/v1"`.
  *ML notes:* `fp16` because T4 has no bf16 hardware. `paged_adamw_8bit` keeps optimizer state small and pages it to CPU on memory spikes. `2e-4` is the usual LoRA lr (only the small adapter moves, so it can be ~10x a full fine-tune lr). With `load_best_model_at_end`, the best checkpoint is always kept even with `save_total_limit=2`.
- [ ] **4.2 Notebook `02_qlora_training.ipynb`** (5-6 cells, no logic):
  1. `!git clone https://github.com/hlong24082k/Diff-Mind && cd Diff-Mind && git checkout <commit>`
  2. `!pip install -q unsloth` (pulls trl/peft/bitsandbytes; wandb is preinstalled on Kaggle) + `pip install pyyaml pydantic` if missing
  3. secrets → `os.environ["HF_TOKEN"]`, `os.environ["WANDB_API_KEY"]`, `os.environ["CUDA_VISIBLE_DEVICES"]="0"` (set **before** importing torch)
  4. `!python -m src.training.train_qlora --check`
  5. `!python -m src.training.train_qlora --max-steps 10 --run-name v1-smoke`
  6. `!python -m src.training.train_qlora`
  + a last cell: `!pip freeze | grep -iE "unsloth|trl|peft|transformers|bitsandbytes|torch|accelerate" > outputs/v1/versions.txt`
- [ ] **4.3 Peak VRAM print** in the script after `train()`: `torch.cuda.max_memory_reserved() / 2**30`, printed and written to `outputs/v1/train_summary.json` (with `git_commit`, `dataset_revision`, W&B URL `wandb.run.url`, best checkpoint, best eval_loss).
- [ ] **4.4 Smoke run** (`--max-steps 10`). *Done when:* loss is logged and finite, W&B shows the run, peak VRAM < 14.5 GB. If OOM → switch to bs 2 × ga 8 in the yaml (same effective batch) and note it. Delete the smoke run's Hub pushes, or point it at a scratch `hub_model_id`.
- [ ] **4.5 Full run `v1`** (~200 steps). Watch W&B: `train_loss` and `eval_loss` both fall; `eval_loss` flattening or rising while `train_loss` keeps falling = overfitting (expected near epoch 3; keep-best handles it). Record the step where `eval_loss` bottoms out.
- [ ] **4.6 (only if a session dies)** resume: download `last-checkpoint/` from the adapter repo (`snapshot_download(..., allow_patterns="last-checkpoint/*")`) and call `trainer.train(resume_from_checkpoint=path)`. Don't build this ahead of time; a ~1-2 h run fits in a 12 h session.

## Phase 5 — Sanity generation & publish
- [ ] **5.1 Sanity on 10 validation rows** (end of the full run, inside the script): `FastLanguageModel.for_inference(model)`; for each row `tok.apply_chat_template(build_messages(...), add_generation_prompt=True, return_tensors="pt")`, `model.generate(max_new_tokens=512, do_sample=False)`, decode only the new tokens with `skip_special_tokens=False`. Check (a) it ends with `<|im_end|>` (stopped by itself, not at the cap), (b) `PRReviewOutput.model_validate_json(text.removesuffix("<|im_end|>"))` passes. Print `k/10` for each and write `outputs/v1/sanity.jsonl`. *Not a metric*; real numbers come from M3 on test.
- [ ] **5.2 Final adapter push**: after `train()` the model in memory is the best checkpoint; `trainer.push_to_hub()` (or `model.push_to_hub` + `tok.push_to_hub`) to `HLeon/diff-mind-qwen2.5-coder-3b-lora`. Record the Hub commit SHA. Size should be roughly 60-120 MB (`adapter_model.safetensors`).
- [ ] **5.3 Bring artifacts home**: download `outputs/v1/{train_summary.json,sanity.jsonl,rendered_example.txt,versions.txt}` from Kaggle into the repo's `outputs/v1/` (gitignored; the summary numbers go into docs).

## Phase 6 — Log & close
- [ ] **6.1 `docs/experiments.md` row `v1`**: date, dataset revision, config + commit, W&B link, adapter Hub SHA. Metrics columns stay empty until M3. Notes: best step / eval_loss, peak VRAM, sanity k/10, deviations (warmup, batch fallback if used, 1,071 rows not 1,200).
- [ ] **6.2 Pin Kaggle versions**: paste `versions.txt` content into a comment/cell in the notebook (`pip install unsloth==X ...`) so the run is reproducible.
- [ ] **6.3 `docs/progress-tracker.md`**: M2 done, decisions above under Architecture Decisions, Next Up = M3.

## Exit criteria (from plan)
`eval_loss` decreases smoothly without diverging from `train_loss`; peak VRAM < 14.5 GB on T4; LoRA adapter on HF Hub; W&B train/eval loss curves.

## Verification
- `python -m pytest` passes locally (Phase 1 test + M1 tests).
- `--check` passes on Kaggle: masked tokens == label JSON + `<|im_end|>`, 0 rows > 2048 tokens.
- Smoke run: finite loss, VRAM printed < 14.5 GB.
- Full run: W&B curves; `train_summary.json` has best step, eval_loss, VRAM, revision, commit.
- Sanity: ≥ 9/10 validation outputs stop at `<|im_end|>` and validate. If much lower, suspect masking/EOS (Phase 3) before touching hyperparameters.

## Suggested review checkpoints
After Phase 1 (local, cheap), after Phase 3 (`--check` output, before spending GPU time), after the smoke run (4.4), after the full run (curves + sanity).
