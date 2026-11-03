"""M2 QLoRA training. Heavy imports (unsloth/torch/trl/datasets) go inside main() so this module
imports on the Mac venv for tests."""
import argparse
import json
import os
import subprocess
import sys

from src.data_pipeline.format_chat import build_messages

OUT_DIR = "outputs/v1"


def to_chat(example: dict) -> list[dict]:
    """Rebuild system+user from build_messages (single source, no train/serve skew) and append the label."""
    return build_messages(example["file_path"], example["diff"]) + [
        {"role": "assistant", "content": example["label_json"]}
    ]


def check(trainer, tok, ds, max_len: int) -> None:
    """Pre-flight: the loss must cover exactly label_json + <|im_end|>, and no row may exceed the context."""
    ex = trainer.train_dataset[0]
    ids, labels = ex["input_ids"], ex["labels"]
    active = [i for i, l in zip(ids, labels) if l != -100]
    decoded = tok.decode(active)
    print(f"--- masked/unmasked split ---\nactive labels: {len(active)} / {len(ids)} tokens\n{decoded!r}")
    decoded = decoded.removesuffix("\n")  # the template's newline after <|im_end|> is fine
    row = ds["train"][0]
    assert decoded.startswith('{"pass"'), decoded[:40]
    assert decoded.endswith("<|im_end|>"), decoded[-40:]
    assert decoded == row["label_json"] + "<|im_end|>", "active labels != label_json + <|im_end|>"
    assert len(active) < len(ids) / 2, "prompt is not masked"

    print(f"--- rendered example ---\n{row['text']}")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/rendered_example.txt", "w") as f:
        f.write(row["text"])  # M3 diffs this against the GGUF chat template

    lens = sorted(len(tok(r["text"])["input_ids"]) for s in ("train", "validation") for r in ds[s])
    over = sum(n > max_len for n in lens)
    print(f"token lengths: max={lens[-1]} p95={lens[int(0.95 * len(lens))]} over_limit={over}")
    assert over == 0, f"{over} rows exceed max_seq_length"
    print("check passed")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/qlora_config.yaml")
    p.add_argument("--check", action="store_true", help="masking/EOS/length asserts, then exit")
    p.add_argument("--max-steps", type=int, default=-1, help="-1 = train for num_train_epochs")
    p.add_argument("--run-name", default=None, help="override run_name from the config")
    args = p.parse_args()

    import torch
    import yaml
    from datasets import load_dataset
    from unsloth import FastLanguageModel  # import before trl/transformers so Unsloth's patches apply

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Unsloth applies bitsandbytes nf4 + double quant itself; the yaml `quant:` block just documents it.
    model, tok = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg["max_seq_length"],
        dtype=torch.float16,
        load_in_4bit=cfg["quant"]["load_in_4bit"],
    )

    # Pad must differ from EOS, else pad-masking can also mask the real <|im_end|> and the model never learns to stop.
    if tok.pad_token is None:
        tok.pad_token = "<|endoftext|>"
    assert tok.eos_token == "<|im_end|>", tok.eos_token
    assert tok.pad_token != tok.eos_token, tok.pad_token

    lora = cfg["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        use_gradient_checkpointing="unsloth",
        random_state=cfg["train"]["seed"],
    )
    model.print_trainable_parameters()

    ds = load_dataset(cfg["dataset"], revision=cfg["dataset_revision"])
    # No add_generation_prompt: the assistant turn is already in to_chat().
    ds = ds.map(lambda ex: {"text": tok.apply_chat_template(to_chat(ex), tokenize=False)})
    print(ds)

    from dataclasses import fields

    from trl import SFTConfig, SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    t = cfg["train"]
    run_name = args.run_name or cfg["run_name"]
    os.environ["WANDB_PROJECT"] = cfg["wandb_project"]
    # TRL renamed max_seq_length -> max_length; its default (1024) would silently truncate our 2048-token rows.
    len_key = "max_length" if "max_length" in {f.name for f in fields(SFTConfig)} else "max_seq_length"
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        args=SFTConfig(
            per_device_train_batch_size=t["per_device_train_batch_size"],
            per_device_eval_batch_size=t["per_device_train_batch_size"],  # default 8 x 2048 logits risks OOM on a T4
            gradient_accumulation_steps=t["gradient_accumulation_steps"],
            num_train_epochs=t["num_train_epochs"],
            max_steps=args.max_steps,
            learning_rate=t["learning_rate"],
            lr_scheduler_type=t["lr_scheduler_type"],
            warmup_steps=t["warmup_steps"],
            optim=t["optim"],
            fp16=t["fp16"],
            bf16=t["bf16"],
            eval_strategy="steps",
            eval_steps=t["eval_steps"],
            save_steps=t["save_steps"],
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=t["logging_steps"],
            report_to="wandb",
            run_name=run_name,
            push_to_hub=t["push_to_hub"] and args.max_steps < 0,  # smoke runs stay off the Hub
            hub_model_id=cfg["hub_model_id"],
            hub_strategy="checkpoint",
            hub_private_repo=True,
            dataset_text_field="text",
            packing=False,
            seed=t["seed"],
            output_dir=OUT_DIR,
            **{len_key: cfg["max_seq_length"]},
        ),
    )
    # Completion-only loss: everything before the assistant header gets label -100, so only JSON + <|im_end|> trains.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    if args.check:
        check(trainer, tok, ds, cfg["max_seq_length"])
        sys.exit(0)

    trainer.train()
    vram = torch.cuda.max_memory_reserved() / 2**30
    print(f"peak VRAM reserved: {vram:.2f} GiB")

    # load_best_model_at_end means the in-memory model is the best checkpoint. Push before sanity so a
    # sanity crash cannot lose it. The smoke run (--max-steps) never pushes.
    adapter_sha = None
    if args.max_steps < 0:
        trainer.push_to_hub()
        from huggingface_hub import HfApi

        adapter_sha = HfApi().model_info(cfg["hub_model_id"]).sha
        print(f"adapter pushed: {cfg['hub_model_id']} @ {adapter_sha}")

    import wandb

    summary = {
        "run_name": run_name,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "dataset_revision": cfg["dataset_revision"],
        "wandb_url": wandb.run.url if wandb.run else None,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "peak_vram_gib": round(vram, 2),
        "adapter_hub_sha": adapter_sha,
    }
    with open(f"{OUT_DIR}/train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(summary)

    # The first validation rows are all defective, so take 5 clean + 5 defective to also exercise abstention.
    val = ds["validation"]
    sanity(model, tok, [r for r in val if not r["is_defective"]][:5] + [r for r in val if r["is_defective"]][:5])


def sanity(model, tok, rows) -> None:
    """Greedy-generate on validation rows: does it stop by itself and emit schema-valid JSON? Not a metric (M3 does test)."""
    from unsloth import FastLanguageModel

    from src.schema import PRReviewOutput

    FastLanguageModel.for_inference(model)
    results = []
    for r in rows:
        prompt = tok.apply_chat_template(
            build_messages(r["file_path"], r["diff"]), tokenize=False, add_generation_prompt=True
        )
        inputs = tok(prompt, return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        text = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        stopped = text.endswith("<|im_end|>")  # stopped by itself, not at the 512 cap
        try:
            PRReviewOutput.model_validate_json(text.removesuffix("<|im_end|>"))
            valid = True
        except ValueError:  # pydantic ValidationError subclasses ValueError
            valid = False
        results.append({"id": r["id"], "stopped": stopped, "valid": valid, "output": text})
        print(f"{r['id']}: stopped={stopped} valid={valid}")
    n = len(results)
    print(f"stopped {sum(x['stopped'] for x in results)}/{n}, valid {sum(x['valid'] for x in results)}/{n}")
    with open(f"{OUT_DIR}/sanity.jsonl", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in results)


if __name__ == "__main__":
    main()
