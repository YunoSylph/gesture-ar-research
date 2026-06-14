from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research_pipeline.data.manifest import ensure_unique_sample_ids, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a canonical JSONL manifest.")
    parser.add_argument("manifest")
    args = parser.parse_args()
    records = read_jsonl(args.manifest)
    ensure_unique_sample_ids(records)
    print(f"manifest ok: {len(records)} records")


if __name__ == "__main__":
    main()
