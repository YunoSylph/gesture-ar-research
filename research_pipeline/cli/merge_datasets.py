from __future__ import annotations

import argparse

from research_pipeline.data.manifest import ensure_unique_sample_ids, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple manifest JSONL files.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = []
    for input_path in args.inputs:
        records.extend(read_jsonl(input_path))
    ensure_unique_sample_ids(records)
    write_jsonl(args.output, records)
    print(f"wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()

