from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from society_of_thought_bench.training_data import DATA_ROOT, build_warmup_example

ORDER = (
    ("countdown", "medium"),
    ("countdown", "hard"),
    ("evidence", "medium"),
    ("evidence", "hard"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT / "disagreement_repair")
    parser.add_argument("--train-total", type=int, default=256)
    parser.add_argument("--val-total", type=int, default=64)
    parser.add_argument("--train-seed-start", type=int, default=310_000)
    parser.add_argument("--val-seed-start", type=int, default=320_000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_counts = split_evenly(args.train_total, len(ORDER))
    val_counts = split_evenly(args.val_total, len(ORDER))

    train_rows = build_rows(train_counts, split="train", seed_start=args.train_seed_start)
    val_rows = build_rows(val_counts, split="eval", seed_start=args.val_seed_start)

    train_path = args.output_dir / "sft_train.jsonl"
    val_path = args.output_dir / "sft_val.jsonl"
    manifest_path = args.output_dir / "manifest.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "tiny internal repair pass for weak disagreement",
        "train_total": len(train_rows),
        "val_total": len(val_rows),
        "train_counts": {
            f"{family}_{difficulty}": count for (family, difficulty), count in zip(ORDER, train_counts, strict=True)
        },
        "val_counts": {
            f"{family}_{difficulty}": count for (family, difficulty), count in zip(ORDER, val_counts, strict=True)
        },
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def split_evenly(total: int, buckets: int) -> list[int]:
    base = total // buckets
    counts = [base] * buckets
    remainder = total - (base * buckets)
    for index in range(remainder):
        counts[index] += 1
    return counts


def build_rows(counts: list[int], *, split: str, seed_start: int) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    for (family, difficulty), count in zip(ORDER, counts, strict=True):
        for index in range(count):
            seed = seed_start + offset + index
            rows.append(
                build_warmup_example(
                    family=family,
                    difficulty=difficulty,
                    seed=seed,
                    split=split,
                    example_id=index,
                    curriculum_profile="debate_primary",
                )
            )
            rows[-1]["messages"] = [collapse_message(message) for message in rows[-1]["messages"]]
        offset += count
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def collapse_message(message: dict) -> dict:
    content = message.get("content")
    if not isinstance(content, list):
        return message
    thinking_parts = []
    text_parts = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "thinking" and isinstance(part.get("thinking"), str):
            thinking_parts.append(part["thinking"].strip())
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            text_parts.append(part["text"].strip())
    rendered = ""
    if thinking_parts:
        rendered += "<think>" + "\n\n".join(part for part in thinking_parts if part) + "</think>"
    if text_parts:
        rendered += "\n" + "\n".join(part for part in text_parts if part) if rendered else "\n".join(part for part in text_parts if part)
    return {**message, "content": rendered.strip()}


if __name__ == "__main__":
    main()
