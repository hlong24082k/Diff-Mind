"""Single source of truth for the model output contract (PROJECT_PLAN.md §3.2)."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Category = Literal["SECURITY", "BUG", "PERFORMANCE", "RESOURCE_LEAK"]
Severity = Literal["LOW", "MEDIUM", "HIGH"]  # plan only shows HIGH; confirm full set in M1


class ReviewItem(BaseModel):
    line: int = Field(ge=1)
    category: Category
    severity: Severity
    comment: str = Field(min_length=1)


class PRReviewOutput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    passed: bool = Field(alias="pass")  # "pass" is a Python keyword
    reviews: list[ReviewItem]

    @model_validator(mode="after")
    def _pass_matches_reviews(self):
        if self.passed != (not self.reviews):
            raise ValueError("pass must be true iff reviews is empty")
        return self


if __name__ == "__main__":
    ok = PRReviewOutput.model_validate_json('{"pass": true, "reviews": []}')
    assert ok.model_dump(by_alias=True) == {"pass": True, "reviews": []}
    for bad in (
        '{"pass": true, "reviews": [{"line": 1, "category": "BUG", "severity": "LOW", "comment": "x"}]}',
        '{"pass": false, "reviews": []}',
        '{"pass": false, "reviews": [{"line": 1, "category": "STYLE", "severity": "LOW", "comment": "x"}]}',
    ):
        try:
            PRReviewOutput.model_validate_json(bad)
        except ValueError:
            continue
        raise AssertionError(f"should have been rejected: {bad}")
    print("schema ok")
