from __future__ import annotations

import argparse
import json

from society_of_thought_bench.families import inspect_example


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("countdown", "evidence"), default="countdown")
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="easy")
    parser.add_argument("--institution", choices=("flat", "hierarchical", "auto"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-personas", type=int, default=4)
    parser.add_argument("--max-debate-turns", type=int, default=10)
    parser.add_argument("--trace-mode", choices=("debate", "monologue"), default="debate")
    parser.add_argument(
        "--trace-prompt-variant",
        choices=("official", "trace_minimal", "trace_tail_block", "debug_minimal"),
        default="official",
    )
    args = parser.parse_args()

    example = inspect_example(
        family=args.family,
        difficulty=args.difficulty,
        institution=args.institution,
        seed=args.seed,
        max_personas=args.max_personas,
        max_debate_turns=args.max_debate_turns,
        trace_mode=args.trace_mode,
        trace_prompt_variant=args.trace_prompt_variant,
    )
    print(json.dumps(example, indent=2))


if __name__ == "__main__":
    main()
