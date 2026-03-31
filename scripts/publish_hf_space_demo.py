from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, SpaceHardware

DEFAULT_SPACE_ID = "scasella91/society-of-thought-bench-demo"


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    raise SystemExit(f"{name} is required.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space-id", default=DEFAULT_SPACE_ID)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[1] / "space_demo")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    hf_token = require_env("HF_TOKEN")
    tinker_token = require_env("TINKER_API_KEY")

    api = HfApi(token=hf_token)
    api.create_repo(
        repo_id=args.space_id,
        repo_type="space",
        space_sdk="gradio",
        space_hardware=SpaceHardware.CPU_BASIC,
        private=args.private,
        exist_ok=True,
    )
    api.add_space_secret(args.space_id, key="TINKER_API_KEY", value=tinker_token, token=hf_token)
    api.add_space_secret(args.space_id, key="HF_TOKEN", value=hf_token, token=hf_token)
    api.upload_folder(
        repo_id=args.space_id,
        repo_type="space",
        folder_path=args.source_dir,
        commit_message="Publish live demo",
        token=hf_token,
        ignore_patterns=["__pycache__", "*.pyc"],
    )
    api.restart_space(args.space_id, token=hf_token)
    print(f"https://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()
