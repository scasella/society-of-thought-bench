from __future__ import annotations

import argparse
import shlex
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen/qwen3-30b-a3b-thinking-2507")
    parser.add_argument("--family", choices=("countdown", "evidence", "all"), default="all")
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="easy")
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument(
        "--trace-prompt-variant",
        choices=("official", "trace_minimal", "trace_tail_block", "debug_minimal"),
        default="official",
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    env_args = (
        '{"family":"%s","difficulty":"%s","trace_mode":"monologue","trace_prompt_variant":"%s"}'
        % (args.family, args.difficulty, args.trace_prompt_variant)
    )
    command = [
        "uv",
        "run",
        "vf-eval",
        "society-of-thought-bench",
        "-m",
        args.model,
        "-n",
        str(args.num_examples),
        "-r",
        str(args.rollouts),
        "-a",
        env_args,
    ]
    print(" ".join(shlex.quote(part) for part in command))
    if args.run:
        subprocess.run(command, check=False)


if __name__ == "__main__":
    main()
