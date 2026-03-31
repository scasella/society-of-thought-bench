from __future__ import annotations

import argparse
import asyncio
import json
import inspect
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import Dataset

from society_of_thought_bench.training_data import DATA_ROOT, DPO_RUN_DEFAULTS, OUTPUT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=DATA_ROOT / "dpo_train.jsonl")
    parser.add_argument("--val-file", type=Path, default=DATA_ROOT / "dpo_val.jsonl")
    parser.add_argument("--model-name", default=DPO_RUN_DEFAULTS["model_name"])
    parser.add_argument("--renderer-name", default=DPO_RUN_DEFAULTS["renderer_name"])
    parser.add_argument("--load-checkpoint-path")
    parser.add_argument("--reference-model-name")
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--lora-rank", type=int, default=DPO_RUN_DEFAULTS["lora_rank"])
    parser.add_argument("--learning-rate", type=float, default=DPO_RUN_DEFAULTS["learning_rate"])
    parser.add_argument("--lr-schedule", default=DPO_RUN_DEFAULTS["lr_schedule"])
    parser.add_argument("--num-epochs", type=int, default=DPO_RUN_DEFAULTS["num_epochs"])
    parser.add_argument("--dpo-beta", type=float, default=DPO_RUN_DEFAULTS["dpo_beta"])
    parser.add_argument("--max-length", type=int, default=DPO_RUN_DEFAULTS["max_length"])
    parser.add_argument("--batch-size", type=int, default=DPO_RUN_DEFAULTS["batch_size"])
    parser.add_argument("--save-every", type=int, default=DPO_RUN_DEFAULTS["save_every"])
    parser.add_argument("--eval-every", type=int, default=DPO_RUN_DEFAULTS["eval_every"])
    parser.add_argument("--max-steps", type=int, default=DPO_RUN_DEFAULTS["max_steps"])
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-name")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    log_path = args.log_path or default_log_path("dpo", args.model_name)
    summary = {
        "train_file": str(args.train_file),
        "val_file": str(args.val_file),
        "model_name": args.model_name,
        "renderer_name": args.renderer_name,
        "load_checkpoint_path": args.load_checkpoint_path,
        "reference_model_name": args.reference_model_name or args.model_name,
        "log_path": str(log_path),
        "lora_rank": args.lora_rank,
        "learning_rate": args.learning_rate,
        "lr_schedule": args.lr_schedule,
        "num_epochs": args.num_epochs,
        "dpo_beta": args.dpo_beta,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "save_every": args.save_every,
        "eval_every": args.eval_every,
        "max_steps": args.max_steps,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return

    require_env("TINKER_API_KEY")
    for path in (args.train_file, args.val_file):
        if not path.exists():
            raise FileNotFoundError(path)

    import chz
    from tinker_cookbook import cli_utils
    from tinker_cookbook.preference import train_dpo
    from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons, LabeledComparison
    from tinker_cookbook.preference.preference_datasets import ComparisonDatasetBuilder
    from tinker_cookbook.preference.types import Comparison
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    @chz.chz
    class JsonlComparisonBuilder(ComparisonDatasetBuilder):
        train_file: str
        val_file: str | None = None
        swap: bool = False

        def get_train_and_test_datasets(self):
            train_rows = _load_jsonl(Path(self.train_file))
            test_rows = _load_jsonl(Path(self.val_file)) if self.val_file else None
            train_dataset = Dataset.from_list(train_rows)
            test_dataset = Dataset.from_list(test_rows) if test_rows else None
            return train_dataset, test_dataset

        def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
            comparison = Comparison(
                prompt_conversation=example["prompt_messages"],
                completion_A=example["completion_A"],
                completion_B=example["completion_B"],
            )
            return LabeledComparison(comparison=comparison, label=example.get("label", "A"))

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model_name,
        renderer_name=args.renderer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    comparison_builder = JsonlComparisonBuilder(
        train_file=str(args.train_file),
        val_file=str(args.val_file),
    )
    dataset_builder = DPODatasetBuilderFromComparisons(
        common_config=common_config,
        comparison_builder=comparison_builder,
    )

    cli_utils.check_log_dir(str(log_path), behavior_if_exists="ask")
    config = train_dpo.Config(
        log_path=str(log_path),
        model_name=args.model_name,
        dataset_builder=dataset_builder,
        load_checkpoint_path=args.load_checkpoint_path,
        renderer_name=args.renderer_name,
        learning_rate=args.learning_rate,
        lr_schedule=args.lr_schedule,
        num_epochs=args.num_epochs,
        dpo_beta=args.dpo_beta,
        lora_rank=args.lora_rank,
        save_every=args.save_every,
        eval_every=args.eval_every,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        reference_model_name=args.reference_model_name or args.model_name,
        max_steps=args.max_steps,
    )
    result = train_dpo.main(config)
    if inspect.isawaitable(result):
        asyncio.run(result)


def default_log_path(stage: str, model_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_slug = model_name.replace("/", "--")
    return OUTPUT_ROOT / stage / f"{stamp}-{model_slug}"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def require_env(name: str) -> None:
    if os.environ.get(name):
        return
    raise SystemExit(f"{name} is required for non-dry-run Tinker commands.")


if __name__ == "__main__":
    main()
