import json

from src.training.train_qlora import to_chat


def first_rows(n=2):
    with open("data/dataset/train.jsonl") as f:
        return [json.loads(next(f)) for _ in range(n)]


def test_to_chat():
    for ex in first_rows():
        chat = to_chat(ex)
        assert chat[:2] == ex["messages"][:2]  # stored prompt == current formatter (catches stale dataset)
        assert chat[-1] == {"role": "assistant", "content": ex["label_json"]}
        json.loads(chat[-1]["content"])
