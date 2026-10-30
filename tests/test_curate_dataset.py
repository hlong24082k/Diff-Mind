from src.data_pipeline.curate_dataset import _strip_fence, make_diff
from src.data_pipeline.format_chat import number_diff


def test_strip_fence():
    assert _strip_fence("```python\nx = 1\n```") == "x = 1"


def test_make_diff_is_numberable():
    d = make_diff("a\nb\nc\n", "a\nB\nc\n", "f.py")
    assert d.startswith("--- a/f.py\n+++ b/f.py\n@@ -1,3 +1,3 @@")
    assert "2| +B" in number_diff(d)


def test_check_label():
    from src.data_pipeline.curate_dataset import check_label

    d = make_diff("a\nb\nc\n", "a\nB\nc\n", "f.py")
    ok = '{"pass": false, "reviews": [{"line": 2, "category": "BUG", "severity": "LOW", "comment": "x"}]}'
    assert check_label(ok, d)[0] == ok  # canonical form round-trips (key order pass, reviews)
    assert check_label(ok.replace('"line": 2', '"line": 9'), d)[1] == "line not in diff"
    assert check_label('{"pass": true, "reviews": []}', d)[1] is None
    assert check_label("garbage", d)[0] is None


def test_balance():
    import json

    from src.data_pipeline.curate_dataset import balance

    def row(i, cat):
        rev = [{"line": 1, "category": cat, "severity": "LOW", "comment": "x"}] if cat else []
        return {"id": i, "source": "s", "file_path": "f", "diff": "d", "label_json": json.dumps({"pass": not rev, "reviews": rev})}

    rows = [row(f"s{i}", "SECURITY") for i in range(50)] + [row(f"b{i}", "BUG") for i in range(5)] + [row(f"c{i}", None) for i in range(10)]
    out = balance(rows, 20, 10, 42)
    assert sum(r["is_defective"] for r in out) == 20 and len(out) == 30
    assert sum(r["category"] == "BUG" for r in out) == 5  # rare category fully kept, rest filled from SECURITY


def test_label_batch_rotates_keys_and_tolerates_missing_items():
    import json
    from types import SimpleNamespace as NS

    from src.data_pipeline.curate_dataset import KeyPool, label_batch

    def client(fn):
        return NS(models=NS(generate_content=fn))

    def dead(**_):
        raise RuntimeError("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel")

    ok_json = json.dumps([{"id": 1, "pass": True, "reviews": []}])  # answer for diff 2 is missing
    good = client(lambda **_: NS(text=ok_json, usage_metadata=NS(prompt_token_count=7, candidates_token_count=3)))
    pool = KeyPool([client(dead), good])
    rows = [{"id": i, "file_path": "f.py", "diff": make_diff("a\nb\n", "a\nB\n", "f.py")} for i in ("x", "y")]
    out, tin, tout = label_batch(pool, "m", rows)
    assert pool.i == 1 and (tin, tout) == (7, 3)  # rotated to the second key
    assert [r["id"] for r in out] == ["x"] and out[0]["label_json"] == '{"pass": true, "reviews": []}'

    pool = KeyPool([client(dead)])
    try:
        label_batch(pool, "m", rows)
        raise AssertionError("expected SystemExit")
    except SystemExit:
        pass
