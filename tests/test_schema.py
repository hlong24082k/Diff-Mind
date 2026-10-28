import pytest

from src.schema import PRReviewOutput

ITEM = '{"line": 1, "category": "%s", "severity": "LOW", "comment": "x"}'


def test_valid_pass_roundtrip():
    ok = PRReviewOutput.model_validate_json('{"pass": true, "reviews": []}')
    assert ok.model_dump(by_alias=True) == {"pass": True, "reviews": []}


@pytest.mark.parametrize(
    "bad",
    [
        '{"pass": true, "reviews": [%s]}' % (ITEM % "BUG"),  # pass but has reviews
        '{"pass": false, "reviews": []}',  # fail but no reviews
        '{"pass": false, "reviews": [%s]}' % (ITEM % "STYLE"),  # category outside the 4
    ],
)
def test_invalid_rejected(bad):
    with pytest.raises(ValueError):
        PRReviewOutput.model_validate_json(bad)
