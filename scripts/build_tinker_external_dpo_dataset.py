from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from society_of_thought_bench.external_benchmarks import build_external_dpo_rows
from society_of_thought_bench.training_data import DATA_ROOT, DEFAULT_OBJECTIVE_PROFILE, DEFAULT_TRACE_PROMPT_VARIANT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT / "external_core_debate_dpo")
    parser.add_argument("--train-total", type=int, default=12000)
    parser.add_argument("--val-total", type=int, default=1500)
    parser.add_argument("--train-seed-start", type=int, default=70_000)
    parser.add_argument("--val-seed-start", type=int, default=80_000)
    parser.add_argument(
        "--curriculum-profile",
        choices=[
            "external_core_debate",
            "external_core_monologue",
            "external_proxy_focus_debate",
            "external_proxy_focus_monologue",
        ],
        default="external_core_debate",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, train_components = build_external_dpo_rows(
        total=args.train_total,
        curriculum_profile=args.curriculum_profile,
        split="train",
        seed_start=args.train_seed_start,
    )
    val_rows, val_components = build_external_dpo_rows(
        total=args.val_total,
        curriculum_profile=args.curriculum_profile,
        split="eval",
        seed_start=args.val_seed_start,
    )

    train_path = args.output_dir / "dpo_train.jsonl"
    val_path = args.output_dir / "dpo_val.jsonl"
    manifest_path = args.output_dir / "dpo_manifest.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective_profile": DEFAULT_OBJECTIVE_PROFILE,
        "curriculum_profile": args.curriculum_profile,
        "trace_prompt_variant": DEFAULT_TRACE_PROMPT_VARIANT,
        "train_total": len(train_rows),
        "val_total": len(val_rows),
        "train_component_counts": train_components,
        "val_component_counts": val_components,
        "train_sources": dict(Counter(row.get("source", "unknown") for row in train_rows)),
        "val_sources": dict(Counter(row.get("source", "unknown") for row in val_rows)),
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
