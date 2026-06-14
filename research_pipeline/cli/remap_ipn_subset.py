from __future__ import annotations

import argparse

from research_pipeline.data.manifest import read_jsonl, write_jsonl
from research_pipeline.labels import remap_ipn_label


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep only the thesis IPN subset and enforce target labels.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output_records = []
    for record in read_jsonl(args.input, strict=False):
        target = remap_ipn_label(record.public_label)
        if target is None:
            continue
        record.target_label = target
        output_records.append(record)
    write_jsonl(args.output, output_records)
    print(f"wrote {len(output_records)} records to {args.output}")


if __name__ == "__main__":
    main()

