#!/usr/bin/env python3
"""Build a deterministic, minimal BC5CDR slice for smoke tests."""

import argparse
import json
from pathlib import Path


def first_documents(path: Path, count: int) -> str:
    documents = path.read_text(encoding="utf-8").strip().split("\n\n")
    return "\n\n".join(documents[:count]) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict-train", type=Path, required=True)
    parser.add_argument("--chatgpt-train", type=Path, required=True)
    parser.add_argument("--gold-dev", type=Path, required=True)
    parser.add_argument("--gold-test", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "sample_data"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dict_train.txt").write_text(
        first_documents(args.dict_train, args.count), encoding="utf-8"
    )
    (args.output_dir / "chatgpt_train.txt").write_text(
        first_documents(args.chatgpt_train, args.count), encoding="utf-8"
    )
    for source, name in (
        (args.gold_dev, "gold_dev.json"),
        (args.gold_test, "gold_test.json"),
    ):
        rows = json.loads(source.read_text(encoding="utf-8"))[: args.count]
        (args.output_dir / name).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
