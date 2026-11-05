"""M3 3.2: side-by-side markdown tables for docs/experiments.md, recomputed from each run's predictions.jsonl
(baseline_results.json is not read, so old runs get bootstrap CIs without re-running the model).
  python -m src.evaluation.compare outputs/base-0shot outputs/base-2shot outputs/v1-s125-q4
Prints (1) headline metrics with 95% CIs, (2) per-category F1, (3) recall by source (`source` joined from --split-file on id)."""
import argparse
from pathlib import Path

from src.data_pipeline.curate_dataset import read_jsonl
from src.evaluation.evaluate import CATEGORIES, bootstrap_ci, compute_metrics, parse

HEADLINE = [("JSON valid", "json_valid"), ("FP rate", "false_positive_rate"), ("Recall", "defect_recall"),
            ("Macro-F1", "macro_f1"), ("Line acc.", "line_accuracy")]


def source_group(source: str) -> str:
    """The M1 shortcut question: synthetic CyberNative rows vs real commit-based ones (commitpackft + commitpackft_fix)."""
    return "cybernative" if source == "cybernative" else "commit-based"


def fmt(v, ci, pct: bool) -> str:
    if v is None:
        return "—"
    f = (lambda x: f"{x:.1%}") if pct else (lambda x: f"{x:.3f}")
    return f(v) + (f" [{f(ci[0])}, {f(ci[1])}]" if ci else "")


def table(header: list, rows: list[list]) -> str:
    return "\n".join("| " + " | ".join(map(str, r)) + " |" for r in [header, ["---"] * len(header), *rows])


def flagged(p: dict) -> bool:
    out = parse(p["raw"])
    return out is not None and not out.passed


def report(run_dirs: list[str], split_file: str) -> str:
    source = {r["id"]: r["source"] for r in read_jsonl(split_file)}
    if not source:
        raise SystemExit(f"{split_file}: missing or empty")
    runs = {}
    for d in run_dirs:
        preds = read_jsonl(f"{d}/predictions.jsonl")
        missing = [p["id"] for p in preds if p["id"] not in source]
        if not preds or missing:
            raise SystemExit(f"{d}: no predictions, or ids not in {split_file} (e.g. {missing[:1]})")
        runs[Path(d).name] = preds, compute_metrics(preds), bootstrap_ci(preds)

    headline = table(["run", "n", *(h for h, _ in HEADLINE)],
                     [[name, m["n"], *(fmt(m[k], ci[k], k != "macro_f1") for _, k in HEADLINE)] for name, (_, m, ci) in runs.items()])

    support = next(iter(runs.values()))[1]["per_category"]
    cell = lambda c: "—" if c["f1"] is None else f"{c['f1']:.2f}"
    per_cat = table(["category", "support", *(f"{name} F1 (pred)" for name in runs)],
                    [[c, support[c]["support"], *(f"{cell(m['per_category'][c])} ({m['per_category'][c]['predicted']})" for _, m, _ in runs.values())]
                     for c in CATEGORIES])

    groups = sorted({source_group(source[p["id"]]) for preds, _, _ in runs.values() for p in preds if p["is_defective"]})
    def recall(preds, g):
        rows = [p for p in preds if p["is_defective"] and g in ("all", source_group(source[p["id"]]))]
        hits = sum(map(flagged, rows))
        return f"{hits}/{len(rows)} ({hits / len(rows):.0%})" if rows else "—"
    by_source = table(["source (defective only)", *runs], [[g, *(recall(preds, g) for preds, _, _ in runs.values())] for g in [*groups, "all"]])

    return "\n\n".join([f"### Headline (95% bootstrap CI)\n\n{headline}", f"### Per category\n\n{per_cat}",
                        f"### Recall by source\n\n{by_source}"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run dirs, each with a predictions.jsonl")
    ap.add_argument("--split-file", default="data/dataset/test.jsonl", help="split the runs were scored on (for `source`)")
    a = ap.parse_args(argv)
    print(report(a.runs, a.split_file))


if __name__ == "__main__":
    main()
