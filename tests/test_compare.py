import json

from src.evaluation.compare import report


def label(*cats):
    return json.dumps({"pass": not cats, "reviews": [{"line": 1, "category": c, "severity": "HIGH", "comment": "x"} for c in cats]})


def test_report(tmp_path):
    split = [{"id": "c1", "source": "commitpackft", "is_defective": False, "label_json": label()},
             {"id": "d1", "source": "cybernative", "is_defective": True, "label_json": label("SECURITY")},
             {"id": "d2", "source": "commitpackft_fix", "is_defective": True, "label_json": label("BUG")}]
    (tmp_path / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in split))
    for run, raw in {"good": lambda r: r["label_json"], "bad": lambda r: "not json"}.items():
        (tmp_path / run).mkdir()
        preds = [{k: r[k] for k in ("id", "is_defective", "label_json")} | {"raw": raw(r)} for r in split]
        (tmp_path / run / "predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds))

    out = report([str(tmp_path / "good"), str(tmp_path / "bad")], str(tmp_path / "test.jsonl"))
    assert "| good | 3 | 100.0% [100.0%, 100.0%] | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 1.000" in out
    assert "| bad | 3 | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.000" in out
    assert "| SECURITY | 1 | 1.00 (1) | 0.00 (0) |" in out
    assert "| commit-based | 1/1 (100%) | 0/1 (0%) |" in out and "| all | 2/2 (100%) | 0/2 (0%) |" in out
