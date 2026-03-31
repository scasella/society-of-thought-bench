from __future__ import annotations

import argparse
import json
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
    if output_dir.exists():
        shutil.rmtree(output_dir)
    download_dir.parent.mkdir(parents=True, exist_ok=True)

    if download_dir.exists() and (download_dir / "checkpoint_complete").exists():
        downloaded = download_dir.resolve()
    else:
        if download_dir.exists():
            shutil.rmtree(download_dir)
        downloaded = Path(weights.download(tinker_path=args.checkpoint, output_dir=str(download_dir))).resolve()
    shutil.copytree(downloaded, output_dir)

    config_path = output_dir / "adapter_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        config["base_model_name_or_path"] = args.base_model
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    marker_path = output_dir / "checkpoint_complete"
    if marker_path.exists():
        marker_path.unlink()

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
