from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tinker_cookbook import weights


DEFAULT_CHECKPOINT = "tinker://80d6e740-bf17-52ca-a94c-422c67897617:train:0/sampler_weights/final"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-30B-A3B"
DEFAULT_REPO_ID = "scasella91/society-of-thought-qwen3-30b-paper-faithful-adapter"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/hf_adapter_export/society-of-thought-qwen3-30b-paper-faithful-adapter"),
    )
    parser.add_argument(
        "--model-card",
        type=Path,
        default=Path("release_preview/HF_MODEL_CARD.md"),
    )
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    download_dir = output_dir.parent / f"{output_dir.name}_download"
    if download_dir.exists():
        shutil.rmtree(download_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    download_dir.parent.mkdir(parents=True, exist_ok=True)

    downloaded = weights.download(tinker_path=args.checkpoint, output_dir=str(download_dir))
    weights.build_lora_adapter(
        base_model=args.base_model,
        adapter_path=downloaded,
        output_path=str(output_dir),
    )

    model_card_path = args.model_card.resolve()
    if model_card_path.exists():
        shutil.copyfile(model_card_path, output_dir / "README.md")

    url = weights.publish_to_hf_hub(
        model_path=str(output_dir),
        repo_id=args.repo_id,
        private=args.private,
    )
    print(url)


if __name__ == "__main__":
    main()
