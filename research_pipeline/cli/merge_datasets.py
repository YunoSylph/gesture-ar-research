from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_pipeline.data.manifest import read_jsonl, write_jsonl
from research_pipeline.data.merge import merge_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge dataset manifests into one balanced 7-class manifest.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-per-class", type=int, help="Cap total samples per target class.")
    parser.add_argument(
        "--max-per-class-per-source",
        type=int,
        help="Cap samples per (target class, source dataset) so one source cannot dominate a class.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--report", help="Optional path to write a JSON coverage/balance report.")
    args = parser.parse_args()

    records = []
    for input_path in args.inputs:
        records.extend(read_jsonl(input_path))
    merged, report = merge_manifests(
        records,
        max_per_class=args.max_per_class,
        max_per_class_per_source=args.max_per_class_per_source,
        seed=args.seed,
    )
    write_jsonl(args.output, merged)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {len(merged)} records to {args.output} (dropped {report.dropped} for balancing)")
    print(f"by_target={report.by_target}")
    if report.missing_targets:
        print(f"WARNING: target classes with no samples: {list(report.missing_targets)}")


if __name__ == "__main__":
    main()
