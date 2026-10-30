"""Phase 4: split data/balanced.jsonl -> train/validation/test, check leakage, optionally push to the HF Hub.
Run from the repo root:  python -m src.data_pipeline.publish_dataset [--push USER/pr-review-high-signal [--public]]
Writes data/dataset/{train,validation,test}.jsonl and prints the test-split hash (freeze it in docs/experiments.md)."""
import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

from dotenv import load_dotenv

from src.data_pipeline.curate_dataset import read_jsonl
from src.data_pipeline.format_chat import build_messages

SPLITS = ("train", "validation", "test")
COLUMNS = ("id", "source", "file_path", "diff", "messages", "label_json", "is_defective", "category")


def split(rows: list[dict], frac: float, seed: int) -> dict[str, list[dict]]:
    """Stratified on (is_defective, category): every stratum gives ~frac to validation and ~frac to test.
    Why category too: the rare ones (PERFORMANCE has 8 rows) would otherwise land all in train by chance."""
    strata = collections.defaultdict(list)
    for r in rows:
        strata[(r["is_defective"], r["category"])].append(r)
    rng, out = random.Random(seed), {s: [] for s in SPLITS}
    for key in sorted(strata, key=str):  # sorted: iteration order must not depend on file order
        rs = sorted(strata[key], key=lambda r: r["id"])
        rng.shuffle(rs)
        n = max(1, round(len(rs) * frac))
        out["test"] += rs[:n]
        out["validation"] += rs[n : 2 * n]
        out["train"] += rs[2 * n :]
    for s in SPLITS:
        out[s].sort(key=lambda r: r["id"])
    return out


def commit_of(r: dict) -> str:
    """cp-<commit>-<path hash> / cpf-... share the commit; CyberNative rows are their own group."""
    return r["id"].split("-")[1] if r["source"] != "cybernative" else r["id"]


def check_leakage(splits: dict[str, list[dict]]) -> None:
    """No id, diff or source commit may appear in two splits (near-duplicates would inflate test scores)."""
    for name, key in (("id", lambda r: r["id"]), ("diff hash", lambda r: hashlib.sha1(r["diff"].encode()).hexdigest()), ("commit", commit_of)):
        seen: dict = {}
        for s in SPLITS:
            for k in {key(r) for r in splits[s]}:
                assert k not in seen, f"{name} {k} leaks between {seen[k]} and {s}"
                seen[k] = s


def to_example(r: dict) -> dict:
    ex = {k: r[k] for k in COLUMNS if k != "messages"}
    ex["messages"] = build_messages(r["file_path"], r["diff"])  # system + user; the assistant turn is `label_json`
    return {k: ex[k] for k in COLUMNS}


def split_hash(rows: list[dict]) -> str:
    """Content hash of a split: changes if any example (or its label) changes."""
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: r["id"]):
        h.update(json.dumps([r["id"], r["diff"], r["label_json"]]).encode())
    return h.hexdigest()


def report(splits: dict[str, list[dict]]) -> None:
    for s in SPLITS:
        rs = splits[s]
        cats = collections.Counter(r["category"] for r in rs if r["is_defective"])
        print(f"{s:<10} n={len(rs):>4} defective={sum(r['is_defective'] for r in rs) / len(rs):.0%} {dict(sorted(cats.items()))}")


def push(splits: dict[str, list[dict]], repo: str, private: bool) -> str:
    from datasets import Dataset, DatasetDict

    dd = DatasetDict({s: Dataset.from_list(splits[s]) for s in SPLITS})
    info = dd.push_to_hub(repo, private=private)  # one commit per call; never re-push over a published revision
    return getattr(info, "oid", None) or str(info)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="data/balanced.jsonl")
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--frac", type=float, default=0.1, help="validation and test share each (plan: 150/1200 -> 12.5%%, 10%% for 1,335 rows)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--push", metavar="REPO", help="HF Hub dataset repo id, e.g. USER/pr-review-high-signal (HF_TOKEN from .env)")
    ap.add_argument("--public", action="store_true", help="push: make the repo public (default private)")
    a = ap.parse_args()
    load_dotenv()
    splits = {s: [to_example(r) for r in rs] for s, rs in split(read_jsonl(a.src), a.frac, a.seed).items()}
    check_leakage(splits)
    report(splits)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        Path(a.out, f"{s}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in splits[s]))
    print("wrote", a.out, "| leakage check passed")
    print("test sha256:", split_hash(splits["test"]))
    if a.push:
        print("pushed", a.push, "at revision", push(splits, a.push, not a.public))
