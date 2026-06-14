from __future__ import annotations

import argparse
from collections import defaultdict

from research_pipeline.data.manifest import read_jsonl, write_jsonl
from research_pipeline.utils.random import stable_rng


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic stratified manifest subset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-class", type=int, required=True)
    parser.add_argument("--split")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    if args.split:
        records = [record for record in records if record.split_group == args.split]
    grouped = defaultdict(list)
    for record in records:
        grouped[record.target_label].append(record)

    output_records = []
    for label in sorted(grouped):
        rng = stable_rng(args.seed, f"{args.split}:{label}")
        candidates = grouped[label]
        order = rng.permutation(len(candidates))
        output_records.extend(candidates[int(index)] for index in order[: args.per_class])
    output_records.sort(key=lambda record: (record.target_label, record.sample_id))
    write_jsonl(args.output, output_records)
    print(f"wrote {len(output_records)} records to {args.output}")


if __name__ == "__main__":
    main()

