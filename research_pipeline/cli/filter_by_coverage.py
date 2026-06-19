from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_pipeline.data.coverage import filter_records_by_coverage, tensor_coverage
from research_pipeline.data.manifest import read_jsonl, write_jsonl
from research_pipeline.data.schema import resolve_path
from research_pipeline.data.tensors import load_landmark_npz


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drop manifest records whose extracted landmark coverage is below a threshold."
    )
    parser.add_argument("--manifest", required=True, help="Landmarks manifest with tensor_path per record.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-coverage", type=float, default=0.85)
    parser.add_argument("--report", help="Optional path to write a JSON coverage report.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    records = read_jsonl(manifest_path)
    base_dir = manifest_path.parent

    def coverage_of(record) -> float:
        tensor = load_landmark_npz(resolve_path(record.tensor_path, base_dir))
        return tensor_coverage(tensor)

    kept, report = filter_records_by_coverage(records, coverage_of, min_coverage=args.min_coverage)
    write_jsonl(args.output, kept)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"kept {report.kept}/{report.total} records (>= {args.min_coverage} coverage) -> {args.output}")
    print(f"mean_coverage={report.mean_coverage:.3f}")
    if report.dropped_by_target:
        print(f"dropped_by_target={report.dropped_by_target}")


if __name__ == "__main__":
    main()
