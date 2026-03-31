from __future__ import annotations

import argparse
import json
from pathlib import Path

from society_of_thought_bench.diagnostics import (
    DEFAULT_OUTPUTS_DIR,
    analyze_results,
    latest_results_path,
    render_analysis_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, help="Path to a results.jsonl file")
    parser.add_argument("--only-invalid", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results_path = args.results or latest_results_path(DEFAULT_OUTPUTS_DIR)
    summary = analyze_results(
        results_path,
        only_invalid=args.only_invalid,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps(summary, indent=2))
        return
    print(render_analysis_report(summary))


if __name__ == "__main__":
    main()
