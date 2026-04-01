from __future__ import annotations

import json
from typing import Any

from tinker_cookbook import model_info, renderers as tk_renderers

CUSTOM_QWEN3_RENDERER_NAME = "society_of_thought_qwen3"


class SocietyOfThoughtQwen3Renderer(tk_renderers.Qwen3Renderer):
    """Qwen3 renderer that makes the model emit the full <think> wrapper itself."""

    def _render_message(self, idx: int, message: dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
        assert message.get("thinking") is None, "CoT tokens not supported in SocietyOfThoughtQwen3Renderer"
        maybe_newline = "\n" if idx > 0 else ""
        ob_str = f"{maybe_newline}<|im_start|>{message['role']}\n"
        ac_content = message["content"]
        if "tool_calls" in message:
            ac_content += "\n".join(
                [f"<tool_call>\n{json.dumps(tool_call)}\n</tool_call>" for tool_call in message["tool_calls"]]
            )
        ac_content += "<|im_end|>"
        return (
            self.tokenizer.encode(ob_str, add_special_tokens=False),
            self.tokenizer.encode(ac_content, add_special_tokens=False),
            self.tokenizer.encode("", add_special_tokens=False),
        )


def get_recommended_renderer_name(model_name: str) -> str:
    if "Qwen/Qwen3" in model_name or model_name.startswith("Qwen3"):
        return CUSTOM_QWEN3_RENDERER_NAME
    return model_info.get_recommended_renderer_name(model_name)


def get_renderer(name: str, tokenizer):
    if name == CUSTOM_QWEN3_RENDERER_NAME:
        return SocietyOfThoughtQwen3Renderer(tokenizer)
    return tk_renderers.get_renderer(name, tokenizer)


def patch_tinker_cookbook_renderers() -> None:
    original = tk_renderers.get_renderer

    def patched(name: str, tokenizer):
        if name == CUSTOM_QWEN3_RENDERER_NAME:
            return SocietyOfThoughtQwen3Renderer(tokenizer)
        return original(name, tokenizer)

    tk_renderers.get_renderer = patched
