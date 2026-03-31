from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from society_of_thought_bench.training_data import (
    DATA_ROOT,
    DEFAULT_OBJECTIVE_PROFILE,
    DEFAULT_TRACE_PROMPT_VARIANT,
    available_pair_types,
    build_dpo_pair_example,
    dpo_mix_counts,
)

ORDER = (
    ("countdown", "easy"),
    ("countdown", "medium"),
    ("countdown", "hard"),
    ("evidence", "easy"),
    ("evidence", "medium"),
    ("evidence", "hard"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--train-total", type=int, default=12_000)
    parser.add_argument("--val-total", type=int, default=1_500)
    parser.add_argument("--train-seed-start", type=int, default=30_000)
    parser.add_argument("--val-seed-start", type=int, default=40_000)
    parser.add_argument("--curriculum-profile", default="debate_primary")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_counts = dpo_mix_counts(args.train_total, curriculum_profile=args.curriculum_profile)
    val_counts = dpo_mix_counts(args.val_total, curriculum_profile=args.curriculum_profile)
    train_rows = build_rows(counts=train_counts, split="train", seed_start=args.train_seed_start, curriculum_profile=args.curriculum_profile)
    val_rows = build_rows(counts=val_counts, split="eval", seed_start=args.val_seed_start, curriculum_profile=args.curriculum_profile)

    train_path = output_dir / "dpo_train.jsonl"
    val_path = output_dir / "dpo_val.jsonl"
    manifest_path = output_dir / "dpo_manifest.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective_profile": DEFAULT_OBJECTIVE_PROFILE,
        "curriculum_profile": args.curriculum_profile,
        "trace_prompt_variant": DEFAULT_TRACE_PROMPT_VARIANT,
        "train_total": len(train_rows),
        "val_total": len(val_rows),
        "train_counts": {f"{family}_{difficulty}": count for (family, difficulty), count in train_counts.items()},
        "val_counts": {f"{family}_{difficulty}": count for (family, difficulty), count in val_counts.items()},
        "train_pair_types": dict(Counter(row["pair_type"] for row in train_rows)),
        "val_pair_types": dict(Counter(row["pair_type"] for row in val_rows)),
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def build_rows(*, counts: dict[tuple[str, str], int], split: str, seed_start: int, curriculum_profile: str) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    for family, difficulty in ORDER:
        count = counts[(family, difficulty)]
        pair_types = available_pair_types(difficulty, curriculum_profile=curriculum_profile)
        for index in range(count):
            seed = seed_start + offset + index
            pair_type = pair_types[index % len(pair_types)]
            rows.append(
                build_dpo_pair_example(
                    family=family,
                    difficulty=difficulty,
                    seed=seed,
                    split=split,
                    pair_type=pair_type,
                    example_id=index,
                    curriculum_profile=curriculum_profile,
                )
            )
        offset += count
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
