from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from society_of_thought_bench.checkpoint_chat import (
    BASE_MODEL,
    BEST_CHECKPOINT,
    DEFAULT_CHAT_SYSTEM_PROMPT,
    sample_checkpoint_async,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=BEST_CHECKPOINT)
    parser.add_argument("--model-name", default=BASE_MODEL)
    parser.add_argument("--system-prompt")
    parser.add_argument("--system-prompt-file", type=Path)
    parser.add_argument("--message", action="append", default=[])
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--show-raw-response", action="store_true")
    parser.add_argument("--transcript-out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    system_prompt = resolve_system_prompt(args)
    summary = {
        "model_path": args.model_path,
        "model_name": args.model_name,
        "system_prompt": system_prompt,
        "scripted_messages": args.message,
        "interactive": not bool(args.message),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "transcript_out": str(args.transcript_out) if args.transcript_out else None,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return
    require_env("TINKER_API_KEY")
    transcript = asyncio.run(run_chat(args, system_prompt))
    if args.transcript_out:
        args.transcript_out.write_text(json.dumps(transcript, indent=2))
        print(f"\nSaved transcript to {args.transcript_out}")


async def run_chat(args: argparse.Namespace, system_prompt: str) -> list[dict[str, Any]]:
    conversation: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    transcript: list[dict[str, Any]] = []
    scripted_messages: list[str] = list(args.message)
    interactive = not scripted_messages

    print(f"Model: {args.model_name}")
    print(f"Checkpoint: {args.model_path}")
    if interactive:
        print("Type your message. Use /quit to exit, /clear to reset the conversation, and /system to print the system prompt.")

    while True:
        if scripted_messages:
            user_text = scripted_messages.pop(0)
            print(f"\nYou: {user_text}")
        else:
            try:
                user_text = input("\nYou: ").strip()
            except EOFError:
                print()
                break
        if not user_text:
            continue
        lowered = user_text.lower()
        if lowered in {"/quit", "/exit"}:
            break
        if lowered == "/system":
            print("\n=== System Prompt ===")
            print(system_prompt)
            continue
        if lowered == "/clear":
            conversation = [{"role": "system", "content": system_prompt}]
            transcript.clear()
            print("Conversation cleared.")
            continue

        conversation.append({"role": "user", "content": user_text})
        result = await sample_checkpoint_async(
            conversation,
            model_path=args.model_path,
            model_name=args.model_name,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        conversation.append(result.parsed_message)
        turn = {
            "user": user_text,
            "thinking_trace": result.thinking_trace,
            "visible_answer": result.visible_answer,
            "raw_output": result.raw_output,
            "parsed_message": result.parsed_message,
        }
        transcript.append(turn)

        print("\n=== Thinking Trace ===")
        print(result.thinking_trace or "[none]")
        print("\n=== Final Answer ===")
        print(result.visible_answer or "[none]")
        if args.show_raw_response:
            print("\n=== Raw Model Output ===")
            print(result.raw_output or "[none]")

        if not interactive and not scripted_messages:
            break
    return transcript


def resolve_system_prompt(args: argparse.Namespace) -> str:
    if args.system_prompt and args.system_prompt_file:
        raise SystemExit("Provide at most one of --system-prompt or --system-prompt-file")
    if args.system_prompt_file:
        return args.system_prompt_file.read_text()
    return args.system_prompt or DEFAULT_CHAT_SYSTEM_PROMPT


def require_env(name: str) -> None:
    if os.environ.get(name):
        return
    raise SystemExit(f"{name} is required for non-dry-run Tinker commands.")


if __name__ == "__main__":
    main()
