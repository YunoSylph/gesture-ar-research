from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research_pipeline.evaluation.live_sessions import summarize_live_records


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SESSION_DIR = PROJECT_ROOT / "artifacts/live_sessions"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/reports/live_session_summary.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def latest_session_path() -> Path:
    paths = sorted(DEFAULT_SESSION_DIR.glob("*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not paths:
        raise FileNotFoundError(f"No live session logs found in {DEFAULT_SESSION_DIR}")
    return paths[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a live Gesture AR JSONL session.")
    parser.add_argument("--input", type=Path, default=None, help="Path to a live session JSONL. Defaults to latest.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    input_path = args.input or latest_session_path()
    records = read_jsonl(input_path)
    summary = summarize_live_records(records)
    summary["input"] = str(input_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(
        "frames={frames} fps_mean={fps:.2f} processing_p95={proc:.2f}ms output={output}".format(
            frames=summary["frames"],
            fps=summary["fps"]["mean"],
            proc=summary["processing_ms"]["p95"],
            output=args.output,
        )
    )


if __name__ == "__main__":
    main()
