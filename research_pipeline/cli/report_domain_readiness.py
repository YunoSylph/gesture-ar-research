from __future__ import annotations

import argparse

from research_pipeline.cli.common import project_path, write_json_report
from research_pipeline.evaluation.domain import summarize_domain_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize public/local domain readiness without extracting videos.")
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output", default="artifacts/reports/domain_readiness.json")
    args = parser.parse_args()

    manifest_paths = [project_path(path) for path in args.manifests]
    report = summarize_domain_manifests(manifest_paths)
    report["manifests"] = [str(path) for path in manifest_paths]
    write_json_report(args.output, report)
    print(f"domain_transfer_status={report['domain_transfer_status']}")


if __name__ == "__main__":
    main()
