from __future__ import annotations

import asyncio
import os
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

CUSTOM_PROMPT_DEFAULT = """Think through this in the paper-style multi-persona format.

Question: Is it better to compare a reasoning model to itself in monologue mode, or to a completely different baseline model, when the main claim is that internal multi-persona debate changes behavior?
"""

BENCHMARK_CONVERSATIONS: dict[str, list[dict[str, str]]] = {
    "Countdown (medium)": [
        {
            "role": "system",
            "content": "Solve the task with task-focused reasoning. The reasoning trace should look like paper-style internal conversation, and the visible answer must stay separate.",
        },
        {
            "role": "user",
            "content": "This benchmark scores the exposed reasoning trace. Solve the task without explaining the benchmark.\n\nScoring prioritizes diverse, grounded debate inside the exposed reasoning trace. Keep the final answer grounded, but the debate quality matters more than exact task accuracy.\n\nReasoning trace rules:\n- Use the exposed reasoning stream / <think> as scratch space.\n- Inside that stream, include exactly one <cast_of_characters> block, one <conversation> block, and one <group_solution> block.\n- Do not write any extra tags outside that paper-style structure.\n- Keep the reasoning task-focused and concise.\n\nVisible answer rules:\n- Return only one <answer>...</answer> block.\n- Do not include JSON, markdown, prose, or any other tags.\n- Put only the final arithmetic expression inside <answer>.\n\nStructured trace requirements:\n- Trace mode: debate\n- Difficulty: medium\n- Institution: flat\n- Use at least 3 personas and include real disagreement before convergence.\n- All personas deliberate as peers.\n- Use no more than 4 personas and no more than 10 turns.\n- Target conversation length for this difficulty is roughly 5-7 turns.\n- Allowed roles: brainstormer, devils_advocate, verifier, synthesizer\n- Allowed personalities: high_openness, low_agreeableness, high_conscientiousness, high_agreeableness, high_extraversion, high_neuroticism\n- Allowed expertise for this family: arithmetic, search, error_checking, verification\n- Each persona block must follow: Role / Personality / Expertise / Style.\n- Cite the real task IDs naturally in the conversation when you refer to inputs or evidence.\n\nReasoning trace example:\n<think>\n<cast_of_characters>\n<persona1>Role: brainstormer\nPersonality: high_openness\nExpertise: arithmetic\nStyle: Generates candidate approaches and explores options.</persona1>\n<persona2>Role: devils_advocate\nPersonality: low_agreeableness\nExpertise: search\nStyle: Pushes on weak assumptions and stresses alternatives.</persona2>\n<persona3>Role: verifier\nPersonality: high_conscientiousness\nExpertise: verification\nStyle: Checks calculations, evidence, and consistency.</persona3>\n</cast_of_characters>\n<conversation>\n<think1>What is the decisive route using N1, N2, T?</think1>\n<think2>But we should challenge the first path before trusting it, especially around T.</think2>\n<think3>Verification passes after checking the cited IDs directly: N1, N2, T.</think3>\n</conversation>\n<group_solution>Concise grounded solution.</group_solution>\n</think>\n\nVisible answer example:\n<answer>(9*5)-10</answer>\n\nTask family: countdown_debate\nNumbers: N1=7, N2=12, N3=11, N4=4\nTarget: T=29\nGoal: use the conversation to explore grounded arithmetic paths. Return only the final arithmetic expression in <answer>.\nUse only the provided numbers and use each number at most once.",
        },
    ],
    "Evidence (medium)": [
        {
            "role": "system",
            "content": "Solve the task with task-focused reasoning. The reasoning trace should look like paper-style internal conversation, and the visible answer must stay separate.",
        },
        {
            "role": "user",
            "content": "This benchmark scores the exposed reasoning trace. Solve the task without explaining the benchmark.\n\nScoring prioritizes diverse, grounded debate inside the exposed reasoning trace. Keep the final answer grounded, but the debate quality matters more than exact task accuracy.\n\nReasoning trace rules:\n- Use the exposed reasoning stream / <think> as scratch space.\n- Inside that stream, include exactly one <cast_of_characters> block, one <conversation> block, and one <group_solution> block.\n- Do not write any extra tags outside that paper-style structure.\n- Keep the reasoning task-focused and concise.\n\nVisible answer rules:\n- Return only one <answer>...</answer> block and one <support>...</support> block.\n- Do not include JSON, markdown, prose, or any other tags.\n- Put only the final verdict inside <answer> and the decisive evidence IDs inside <support>.\n\nStructured trace requirements:\n- Trace mode: debate\n- Difficulty: medium\n- Institution: flat\n- Use at least 3 personas and include real disagreement before convergence.\n- All personas deliberate as peers.\n- Use no more than 4 personas and no more than 10 turns.\n- Target conversation length for this difficulty is roughly 5-7 turns.\n- Allowed roles: brainstormer, devils_advocate, verifier, synthesizer\n- Allowed personalities: high_openness, low_agreeableness, high_conscientiousness, high_agreeableness, high_extraversion, high_neuroticism\n- Allowed expertise for this family: fact_extraction, timeline_reasoning, contradiction_checking, synthesis\n- Each persona block must follow: Role / Personality / Expertise / Style.\n- Cite the real task IDs naturally in the conversation when you refer to inputs or evidence.\n\nReasoning trace example:\n<think>\n<cast_of_characters>\n<persona1>Role: brainstormer\nPersonality: high_openness\nExpertise: fact_extraction\nStyle: Generates candidate approaches and explores options.</persona1>\n<persona2>Role: devils_advocate\nPersonality: low_agreeableness\nExpertise: timeline_reasoning\nStyle: Pushes on weak assumptions and stresses alternatives.</persona2>\n<persona3>Role: verifier\nPersonality: high_conscientiousness\nExpertise: synthesis\nStyle: Checks calculations, evidence, and consistency.</persona3>\n</cast_of_characters>\n<conversation>\n<think1>What is the decisive route using E2?</think1>\n<think2>But we should challenge the first path before trusting it, especially around E2.</think2>\n<think3>Verification passes after checking the cited IDs directly: E2.</think3>\n</conversation>\n<group_solution>Concise grounded solution.</group_solution>\n</think>\n\nVisible answer example:\n<answer>TRUE</answer>\n<support>E2</support>\n\nTask family: evidence_verdict_debate\nClaim: The current status for Atlas Museum is \"rescheduled\".\nEvidence snippets:\nE1: 2025-01-27 official directory entry: Atlas Museum listed rescheduled as its status.\nE2: 2025-02-02 official correction: Atlas Museum now names delayed as its status.\nE3: 2025-03-01 second-hand recap: one attendee still repeats the old status value rescheduled.\nE4: 2025-03-01 facilities bulletin: the east entrance opens at 08:00.\nE5: 2025-02-02 attendee recap: someone mentioned hearing both rescheduled and delayed in hallway chatter.\nGoal: debate the evidence, challenge stale or conflicting snippets, then return the verdict in <answer> and decisive evidence IDs in <support>.",
        },
    ],
}


def require_env(name: str) -> None:
    if os.environ.get(name):
        return
    raise RuntimeError(f"{name} is required for the live demo.")


def split_message_content(content: Any) -> tuple[list[str], list[str]]:
    if isinstance(content, str):
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


def build_fallback_parsed_message(thinking_trace: str, visible_answer: str, raw_output: str) -> dict[str, Any]:
    if thinking_trace or visible_answer:
        content: list[dict[str, str]] = []
        if thinking_trace:
            content.append({"type": "thinking", "thinking": thinking_trace})
        if visible_answer:
            content.append({"type": "text", "text": visible_answer})
        return {"role": "assistant", "content": content}
    return {"role": "assistant", "content": raw_output}


async def run_generation_async(
    prompt: str | list[dict[str, str]],
    *,
    model_path: str = BEST_CHECKPOINT,
    model_name: str = BASE_MODEL,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 1024,
) -> dict[str, str]:
    import tinker
    from tinker import types
    from tinker_cookbook import checkpoint_utils, model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    service = tinker.ServiceClient()
    rest_client = service.create_rest_client()
    training_run = await rest_client.get_training_run_by_tinker_path_async(model_path)
    resolved_model = training_run.base_model or model_name
    renderer_name = await checkpoint_utils.get_renderer_name_from_checkpoint_async(service, model_path)
    if renderer_name is None:
        renderer_name = model_info.get_recommended_renderer_name(resolved_model)
    tokenizer = get_tokenizer(resolved_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    sampling_client = service.create_sampling_client(model_path=model_path, base_model=resolved_model)

    conversation = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
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
    thinking_trace = ""
    visible_answer = ""
    if success:
        thinking_parts, text_parts = split_message_content(parsed_message.get("content"))
        thinking_trace = "\n".join(part for part in thinking_parts if part.strip()).strip()
        visible_answer = "\n".join(part for part in text_parts if part.strip()).strip()
    fallback_thinking, fallback_answer = extract_tagged_sections(raw_output)
    thinking_trace = thinking_trace or fallback_thinking
    visible_answer = visible_answer or fallback_answer
    if not success:
        parsed_message = build_fallback_parsed_message(thinking_trace, visible_answer, raw_output)
    elif not thinking_trace or not visible_answer:
        parsed_message = build_fallback_parsed_message(thinking_trace, visible_answer, raw_output)
    if not visible_answer and raw_output:
        visible_answer = raw_output
        parsed_message = build_fallback_parsed_message(thinking_trace, visible_answer, raw_output)
    return {
        "model_name": resolved_model,
        "checkpoint": model_path,
        "thinking_trace": thinking_trace,
        "visible_answer": visible_answer,
        "raw_output": raw_output,
        "parsed_message": parsed_message,
    }


def run_generation(
    prompt: str | list[dict[str, str]],
    *,
    model_path: str = BEST_CHECKPOINT,
    model_name: str = BASE_MODEL,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 1024,
) -> dict[str, str]:
    require_env("TINKER_API_KEY")
    if isinstance(prompt, str) and not prompt.strip():
        raise RuntimeError("Please enter a prompt.")
    return asyncio.run(
        run_generation_async(
            prompt,
            model_path=model_path,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    )



def initial_chat_state() -> list[dict[str, Any]]:
    return [{"role": "system", "content": DEFAULT_CHAT_SYSTEM_PROMPT}]


def run_chat_turn(
    history: list[dict[str, Any]] | None,
    user_message: str,
    *,
    model_path: str = BEST_CHECKPOINT,
    model_name: str = BASE_MODEL,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    if not user_message.strip():
        raise RuntimeError("Please enter a message.")
    conversation = list(history or initial_chat_state())
    conversation.append({"role": "user", "content": user_message})
    result = run_generation(
        conversation,
        model_path=model_path,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    conversation.append(result["parsed_message"])
    return {
        "conversation": conversation,
        "thinking_trace": result["thinking_trace"],
        "visible_answer": result["visible_answer"],
        "raw_output": result["raw_output"],
    }
