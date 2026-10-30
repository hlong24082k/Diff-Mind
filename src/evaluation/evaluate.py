"""Phase 5: evaluation harness. Runs a model over a dataset split and scores it against the teacher labels.
Run from the repo root:
  python -m src.evaluation.evaluate --model Qwen/Qwen2.5-Coder-3B-Instruct-GGUF --mode zero --split validation --limit 10
  python -m src.evaluation.evaluate --model path/to/model.gguf --mode two-shot --split test
  python -m src.evaluation.evaluate --model fake --limit 5      # garbage-output stub: harness must report 0% validity, not crash
Writes outputs/<run_id>/{predictions.jsonl,baseline_results.json}.

Metric rules (diff-level; PROJECT_PLAN.md §2):
- An output is *valid* iff it parses as JSON and passes `PRReviewOutput` (no fence stripping / regex repair).
- Invalid output = "not flagged": it counts as a miss for recall and is NOT a false positive.
- Flagged = valid and `pass: false`. FP rate = flagged clean / clean. Recall = flagged defective / defective.
- Category F1 compares the *set* of categories per diff (pred vs gold `label_json`), not lines; macro-F1 averages the
  categories that appear in gold or predictions. Line accuracy is reported separately (LINE_TOL below)."""
import argparse
import json
import random
import subprocess
from pathlib import Path

from pydantic import ValidationError

from src.data_pipeline.curate_dataset import read_jsonl
from src.data_pipeline.publish_dataset import split_hash
from src.schema import Category, PRReviewOutput

CATEGORIES = Category.__args__
LINE_TOL = 2  # a predicted line counts as right if within ±2 of a gold line of the same category (teacher lines are noisy)


def load_split(dataset: str, split: str, revision: str | None) -> list[dict]:
    if Path(dataset).is_dir():  # local copy written by publish_dataset
        return read_jsonl(f"{dataset}/{split}.jsonl")
    from datasets import load_dataset

    return list(load_dataset(dataset, split=split, revision=revision))


def pick_shots(train: list[dict]) -> list[dict]:
    """1 defective + 1 clean from *train*, the shortest diffs (cheap on context), tie-broken by id -> fixed across runs."""
    pick = lambda flag: min((r for r in train if r["is_defective"] == flag), key=lambda r: (len(r["diff"]), r["id"]))
    return [pick(True), pick(False)]


def build_prompt(ex: dict, shots: list[dict]) -> list[dict]:
    """system, [user, assistant]*shots, user. Zero-shot is the plain `ex["messages"]` (= build_messages output)."""
    system, user = ex["messages"]
    turns = [t for s in shots for t in (s["messages"][1], {"role": "assistant", "content": s["label_json"]})]
    return [system, *turns, user]


def parse(raw: str) -> PRReviewOutput | None:
    try:
        return PRReviewOutput.model_validate_json(raw)
    except (ValidationError, ValueError, TypeError):
        return None


def _f1(tp: int, fp: int, fn: int) -> float | None:
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None


def compute_metrics(rows: list[dict]) -> dict:
    """rows: dicts with `label_json` (gold string) and `raw` (model output string)."""
    n = len(rows)
    tp, fp, fn = ({c: 0 for c in CATEGORIES} for _ in range(3))
    valid = flagged_clean = flagged_defect = n_clean = n_defect = line_hits = line_total = 0
    for r in rows:
        gold, pred = PRReviewOutput.model_validate_json(r["label_json"]), parse(r["raw"])
        valid += pred is not None
        flagged = pred is not None and not pred.passed
        if gold.passed:
            n_clean += 1
            flagged_clean += flagged
        else:
            n_defect += 1
            flagged_defect += flagged
        gold_cats = {x.category for x in gold.reviews}
        pred_cats = {x.category for x in pred.reviews} if pred else set()
        for c in CATEGORIES:
            tp[c] += c in gold_cats and c in pred_cats
            fp[c] += c not in gold_cats and c in pred_cats
            fn[c] += c in gold_cats and c not in pred_cats
        for c in gold_cats & pred_cats:  # line accuracy only where the category was right
            line_total += 1
            gold_lines = [x.line for x in gold.reviews if x.category == c]
            line_hits += any(abs(p.line - g) <= LINE_TOL for p in pred.reviews if p.category == c for g in gold_lines)
    per_cat = {c: {"f1": _f1(tp[c], fp[c], fn[c]), "support": tp[c] + fn[c], "predicted": tp[c] + fp[c]} for c in CATEGORIES}
    f1s = [v["f1"] for v in per_cat.values() if v["f1"] is not None]
    rate = lambda a, b: a / b if b else None
    return {
        "n": n, "n_clean": n_clean, "n_defective": n_defect,
        "json_valid": rate(valid, n),
        "false_positive_rate": rate(flagged_clean, n_clean),
        "defect_recall": rate(flagged_defect, n_defect),
        "macro_f1": sum(f1s) / len(f1s) if f1s else None,
        "per_category": per_cat,
        "line_accuracy": rate(line_hits, line_total), "line_tol": LINE_TOL,
    }


def make_generate(a):
    """-> generate(messages) -> str. Greedy decoding, fixed max_tokens and seed."""
    if a.model == "fake":
        return lambda messages: "I am not JSON, sorry."
    from llama_cpp import Llama

    kw = dict(n_ctx=a.n_ctx, seed=a.seed, verbose=False)
    llm = Llama(model_path=a.model, **kw) if a.model.endswith(".gguf") else Llama.from_pretrained(a.model, filename=a.gguf_file, **kw)
    return lambda messages: llm.create_chat_completion(messages, temperature=0, max_tokens=a.max_tokens, seed=a.seed)["choices"][0]["message"]["content"]


def git_commit() -> str:
    run = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True).stdout.strip()
    return run("rev-parse", "HEAD") + ("+dirty" if run("status", "--porcelain") else "")


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to a .gguf, an HF GGUF repo id (needs --gguf-file), or 'fake'")
    ap.add_argument("--gguf-file", default="*q4_k_m.gguf", help="filename glob inside the HF repo")
    ap.add_argument("--mode", choices=["zero", "two-shot"], default="zero")
    ap.add_argument("--split", default="test")
    ap.add_argument("--dataset", default="data/dataset", help="local dir from publish_dataset, or an HF repo id")
    ap.add_argument("--revision", help="HF dataset revision (ignored for a local dir; its sha256 is recorded instead)")
    ap.add_argument("--limit", type=int, help="random N examples only (smoke tests; use --split validation)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--run-id", help="default: base-0shot / base-2shot")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--n-ctx", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    rows = load_split(a.dataset, a.split, a.revision)
    if a.limit:  # seeded random subset: the split is sorted by id, so the first N would be one source/class
        rows = random.Random(a.seed).sample(rows, a.limit)
    shots = pick_shots(load_split(a.dataset, "train", a.revision)) if a.mode == "two-shot" else []
    generate = make_generate(a)
    preds = [{"id": ex["id"], "is_defective": ex["is_defective"], "label_json": ex["label_json"],
              "raw": generate(build_prompt(ex, shots))} for ex in rows]
    for p in preds:
        p["valid"] = parse(p["raw"]) is not None
    metrics = compute_metrics(preds)

    run_id = a.run_id or f"base-{'0shot' if a.mode == 'zero' else '2shot'}"
    out = Path(a.out, run_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds))
    result = {
        "run_id": run_id, "metrics": metrics, "git_commit": git_commit(),
        "config": {k: v for k, v in vars(a).items()} | {"temperature": 0, "shot_ids": [s["id"] for s in shots]},
        "model_id": a.model, "dataset": a.dataset, "dataset_revision": a.revision or f"local:{a.split}-sha256:{split_hash(rows) if not a.limit else 'subset'}",
    }
    (out / "baseline_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(metrics, indent=2))
    return result


if __name__ == "__main__":
    main()
