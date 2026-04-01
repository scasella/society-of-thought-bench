from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

BEST_CHECKPOINT = "tinker://80d6e740-bf17-52ca-a94c-422c67897617:train:0/sampler_weights/final"
BASE_MODEL = "Qwen/Qwen3-30B-A3B"

DEFAULT_CHAT_SYSTEM_PROMPT = """You are a research-preview checkpoint trained to show paper-style internal debate in the visible reasoning trace.

For each reply:
- Put the reasoning inside one outer <think>...</think> block.
- Inside that block, include exactly one <cast_of_characters>, one <conversation>, and one <group_solution>.
- Keep the final reply separate in one <answer>...</answer> block.
- Use <support>...</support> only when the answer depends on explicit cited IDs.
- Stay concise, answer the user's actual question, and do not explain the benchmark unless asked.
"""


@dataclass
class CheckpointResponse:
    model_name: str
    model_path: str
    renderer_name: str
    parsed_message: dict[str, Any]
    raw_output: str
    thinking_trace: str
    visible_answer: str


def split_message_content(content: Any) -> tuple[list[str], list[str]]:
    if isinstance(content, str):
        if "</think>" in content:
            before_close, after_close = content.split("</think>", 1)
            reasoning = before_close.replace("<think>", "", 1).strip()
            visible = after_close.strip()
            return ([reasoning] if reasoning else []), ([visible] if visible else [])
        return [], [content]
    thinking_parts: list[str] = []
    text_parts: list[str] = []
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "thinking":
                thinking_parts.append(str(part.get("thinking", "")))
            elif part_type == "text":
                text_parts.append(str(part.get("text", "")))
    return thinking_parts, text_parts


def extract_tagged_sections(raw_output: str) -> tuple[str, str]:
    think_match = re.search(r"<think>(.*)</think>", raw_output, flags=re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", raw_output, flags=re.DOTALL)
    support_match = re.search(r"<support>(.*?)</support>", raw_output, flags=re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""
    answer_parts: list[str] = []
    if answer_match:
        answer_parts.append(f"<answer>{answer_match.group(1).strip()}</answer>")
    if support_match:
        answer_parts.append(f"<support>{support_match.group(1).strip()}</support>")
    if not thinking and answer_match:
        before_answer = raw_output[: answer_match.start()].strip()
        if before_answer:
            thinking = before_answer
    return thinking, "\n".join(answer_parts).strip()


async def sample_checkpoint_async(
    conversation: list[dict[str, Any]],
    *,
    model_path: str = BEST_CHECKPOINT,
    model_name: str = BASE_MODEL,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 1024,
) -> CheckpointResponse:
    import tinker
    from tinker import types
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from society_of_thought_bench.tinker_renderers import get_recommended_renderer_name, get_renderer

    service = tinker.ServiceClient()
    rest_client = service.create_rest_client()
    training_run = await rest_client.get_training_run_by_tinker_path_async(model_path)
    resolved_model = training_run.base_model or model_name
    renderer_name = getattr(training_run, "renderer_name", None) or getattr(training_run, "renderer", None)
    if renderer_name is None:
        renderer_name = get_recommended_renderer_name(resolved_model)
    tokenizer = get_tokenizer(resolved_model)
    renderer = get_renderer(renderer_name, tokenizer)
    sampling_client = service.create_sampling_client(model_path=model_path, base_model=resolved_model)

    model_input = renderer.build_generation_prompt(conversation)
    sampling_params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=renderer.get_stop_sequences(),
    )
    response = await sampling_client.sample_async(prompt=model_input, num_samples=1, sampling_params=sampling_params)
    raw_output = tokenizer.decode(response.sequences[0].tokens).strip()
    parsed_message, success = renderer.parse_response(response.sequences[0].tokens)
    if not success:
        parsed_message = {"role": "assistant", "content": raw_output}
    thinking_parts, text_parts = split_message_content(parsed_message.get("content"))
    thinking_trace = "\n".join(part for part in thinking_parts if part.strip()).strip()
    visible_answer = "\n".join(part for part in text_parts if part.strip()).strip()
    if not thinking_trace or not visible_answer:
        fallback_thinking, fallback_answer = extract_tagged_sections(raw_output)
        thinking_trace = thinking_trace or fallback_thinking
        visible_answer = visible_answer or fallback_answer
    return CheckpointResponse(
        model_name=resolved_model,
        model_path=model_path,
        renderer_name=renderer_name,
        parsed_message=parsed_message,
        raw_output=raw_output,
        thinking_trace=thinking_trace,
        visible_answer=visible_answer,
    )
