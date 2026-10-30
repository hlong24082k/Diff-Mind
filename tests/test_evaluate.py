import json

from src.evaluation.evaluate import build_prompt, compute_metrics, main, pick_shots
from src.schema import PRReviewOutput


def label(*items):
    return json.dumps({"pass": not items, "reviews": [{"line": l, "category": c, "severity": "HIGH", "comment": "x"} for l, c in items]})


def test_compute_metrics():
    rows = [
        {"label_json": label(), "raw": label()},                                   # clean, correct
        {"label_json": label(), "raw": label((3, "BUG"))},                         # clean, false positive
        {"label_json": label((10, "SECURITY")), "raw": label((11, "SECURITY"))},   # defective, right category, line off by 1
        {"label_json": label((5, "BUG")), "raw": "```json\n{}\n```"},              # defective, malformed -> miss
    ]
    m = compute_metrics(rows)
    assert m["json_valid"] == 0.75
    assert m["false_positive_rate"] == 0.5
    assert m["defect_recall"] == 0.5
    assert m["per_category"]["SECURITY"]["f1"] == 1.0
    assert m["per_category"]["BUG"]["f1"] == 0.0  # 1 FP (clean row) + 1 FN (malformed row)
    assert m["per_category"]["PERFORMANCE"]["f1"] is None and m["macro_f1"] == 0.5
    assert m["line_accuracy"] == 1.0


def test_two_shot_prompt():
    mk = lambda i, d: {"id": i, "is_defective": d, "diff": "x" * i, "label_json": label((1, "BUG")) if d else label(),
                       "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": f"U{i}"}]}
    shots = pick_shots([mk(5, True), mk(3, True), mk(9, False), mk(7, False)])
    assert [s["id"] for s in shots] == [3, 7]
    p = build_prompt(mk(1, True), shots)
    assert [m["role"] for m in p] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert p[-1]["content"] == "U1" and PRReviewOutput.model_validate_json(p[2]["content"])


def test_fake_model_dry_run(tmp_path):
    r = main(["--model", "fake", "--split", "validation", "--limit", "5", "--out", str(tmp_path)])
    assert r["metrics"]["json_valid"] == 0 and r["metrics"]["false_positive_rate"] in (0, None) and r["metrics"]["defect_recall"] == 0
    assert len((tmp_path / "base-0shot" / "predictions.jsonl").read_text().splitlines()) == 5
