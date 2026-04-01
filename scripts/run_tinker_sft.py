from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from society_of_thought_bench.training_data import DATA_ROOT, OUTPUT_ROOT, SFT_RUN_DEFAULTS
from society_of_thought_bench.tinker_renderers import get_renderer, patch_tinker_cookbook_renderers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=DATA_ROOT / "sft_train.jsonl")
    parser.add_argument("--val-file", type=Path, default=DATA_ROOT / "sft_val.jsonl")
    parser.add_argument("--model-name", default=SFT_RUN_DEFAULTS["model_name"])
    parser.add_argument("--renderer-name", default=SFT_RUN_DEFAULTS["renderer_name"])
    parser.add_argument("--load-checkpoint-path")
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--lora-rank", type=int, default=SFT_RUN_DEFAULTS["lora_rank"])
    parser.add_argument("--learning-rate", type=float, default=SFT_RUN_DEFAULTS["learning_rate"])
    parser.add_argument("--lr-schedule", default=SFT_RUN_DEFAULTS["lr_schedule"])
    parser.add_argument("--num-epochs", type=int, default=SFT_RUN_DEFAULTS["num_epochs"])
    parser.add_argument("--max-length", type=int, default=SFT_RUN_DEFAULTS["max_length"])
    parser.add_argument("--batch-size", type=int, default=SFT_RUN_DEFAULTS["batch_size"])
    parser.add_argument("--save-every", type=int, default=SFT_RUN_DEFAULTS["save_every"])
    parser.add_argument("--eval-every", type=int, default=SFT_RUN_DEFAULTS["eval_every"])
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-name")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    log_path = args.log_path or default_log_path("sft", args.model_name)
    resolved_load_checkpoint_path = as_training_checkpoint_path(args.load_checkpoint_path)
    summary = {
        "train_file": str(args.train_file),
        "val_file": str(args.val_file),
        "model_name": args.model_name,
        "renderer_name": args.renderer_name,
        "load_checkpoint_path": args.load_checkpoint_path,
        "resolved_load_checkpoint_path": resolved_load_checkpoint_path,
        "log_path": str(log_path),
        "lora_rank": args.lora_rank,
        "learning_rate": args.learning_rate,
        "lr_schedule": args.lr_schedule,
        "num_epochs": args.num_epochs,
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

    import tinker
    from tinker_cookbook import cli_utils
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised import train as supervised_train
    from tinker_cookbook.supervised.data import conversation_to_datum
    from tinker_cookbook.supervised.types import (
        ChatDatasetBuilderCommonConfig,
        SupervisedDataset,
    )

    patch_tinker_cookbook_renderers()

    class ListSupervisedDataset(SupervisedDataset):
        def __init__(self, *, rows: list[dict[str, Any]], batch_size: int, map_fn) -> None:
            self.rows = list(rows)
            self.batch_size = batch_size
            self.map_fn = map_fn
            self.shuffle_rows = list(rows)

        def get_batch(self, index: int) -> list[tinker.Datum]:
            start = index * self.batch_size
            end = start + self.batch_size
            return [self.map_fn(row) for row in self.shuffle_rows[start:end]]

        def set_epoch(self, seed: int = 0):
            self.shuffle_rows = list(self.rows)
            random.Random(seed).shuffle(self.shuffle_rows)

        def __len__(self) -> int:
            return len(self.shuffle_rows) // self.batch_size

    class JsonlConversationPairBuilder:
        def __init__(
            self,
            *,
            train_file: Path,
            val_file: Path,
            common_config: Any,
        ) -> None:
            self.train_file = train_file
            self.val_file = val_file
            self.common_config = common_config
            self._renderer = None

        @property
        def renderer(self):
            if self._renderer is None:
                from tinker_cookbook.tokenizer_utils import get_tokenizer

                tokenizer = get_tokenizer(self.common_config.model_name_for_tokenizer)
                self._renderer = get_renderer(self.common_config.renderer_name, tokenizer)
            return self._renderer

        def __call__(self):
            train_rows = _load_jsonl(self.train_file)
            val_rows = _load_jsonl(self.val_file)

            def map_fn(row: dict) -> tinker.Datum:
                return conversation_to_datum(
                    row["messages"],
                    self.renderer,
                    self.common_config.max_length,
                    TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                )

            train_dataset = ListSupervisedDataset(
                rows=train_rows,
                batch_size=self.common_config.batch_size,
                map_fn=map_fn,
            )
            val_dataset = ListSupervisedDataset(
                rows=val_rows,
                batch_size=max(1, min(len(val_rows), self.common_config.batch_size)),
                map_fn=map_fn,
            )
            return train_dataset, val_dataset

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model_name,
        renderer_name=args.renderer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    dataset_builder = JsonlConversationPairBuilder(
        train_file=args.train_file,
        val_file=args.val_file,
        common_config=common_config,
    )

    cli_utils.check_log_dir(str(log_path), behavior_if_exists="ask")
    config_kwargs: dict[str, Any] = {
        "log_path": str(log_path),
        "model_name": args.model_name,
        "load_checkpoint_path": resolved_load_checkpoint_path,
        "renderer_name": args.renderer_name,
        "dataset_builder": dataset_builder,
        "learning_rate": args.learning_rate,
        "lr_schedule": args.lr_schedule,
        "num_epochs": args.num_epochs,
        "lora_rank": args.lora_rank,
        "save_every": args.save_every,
        "eval_every": args.eval_every,
        "wandb_project": args.wandb_project,
        "wandb_name": args.wandb_name,
        "max_steps": args.max_steps,
    }
    supported = inspect.signature(supervised_train.Config).parameters
    config = supervised_train.Config(**{k: v for k, v in config_kwargs.items() if k in supported and v is not None})
    asyncio.run(supervised_train.main(config))


def default_log_path(stage: str, model_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_slug = model_name.replace("/", "--")
    return OUTPUT_ROOT / stage / f"{stamp}-{model_slug}"


def as_training_checkpoint_path(path: str | None) -> str | None:
    if not path:
        return path
    if "/sampler_weights/" in path:
        return path.replace("/sampler_weights/", "/weights/", 1)
    return path


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
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
