import argparse
import dotenv

from datasets import load_dataset
from dataclasses import dataclass

dotenv.load_dotenv(".env")

@dataclass
class DatasetConfig:
    repo: str
    revision: str


def check_dataset(repo: str, revision: str):
    ds = load_dataset(repo, revision=revision)
    print({k: len(v) for k, v in ds.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=str, required=True)
    ap.add_argument("--revision", type=str, required=True)

    args = ap.parse_args()
    config = DatasetConfig(**vars(args))

    check_dataset(config.repo, config.revision)
