#!/usr/bin/env python3
"""Build a deterministic Integration2023 slice for smoke tests."""

import argparse
from pathlib import Path


def read_table(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append((line.split("\t", 1)[0], line))
    return rows


def select(rows, ids):
    return [line for row_id, line in rows if row_id in ids]


def write_lines(path: Path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--pair-count", type=int, default=10)
    parser.add_argument("--target-count", type=int, default=40)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "sample_data"
    )
    args = parser.parse_args()
    root = args.dataset_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = read_table(root / "sup_pairs.txt")[: args.pair_count]
    test = read_table(root / "ref_pairs.txt")[: args.pair_count]
    pair_lines = [line for _, line in train + test]
    source_ids = {line.split()[0] for line in pair_lines}
    gold_target_ids = {line.split()[1] for line in pair_lines}

    target_rows = read_table(root / "ent_ids_2.txt")
    target_ids = set(gold_target_ids)
    for target_id, _line in target_rows:
        target_ids.add(target_id)
        if len(target_ids) >= args.target_count:
            break

    write_lines(args.output_dir / "ent_ids_1.txt", select(read_table(root / "ent_ids_1.txt"), source_ids))
    write_lines(args.output_dir / "ent_ids_2.txt", select(target_rows, target_ids))
    write_lines(args.output_dir / "paths_1.txt", select(read_table(root / "paths_1.txt"), source_ids))
    write_lines(args.output_dir / "paths_2.txt", select(read_table(root / "paths_2.txt"), target_ids))
    write_lines(args.output_dir / "sup_pairs.txt", [line for _, line in train])
    write_lines(args.output_dir / "ref_pairs.txt", [line for _, line in test])


if __name__ == "__main__":
    main()
