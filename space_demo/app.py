from __future__ import annotations

import gradio as gr

from core import BEST_CHECKPOINT, BENCHMARK_CONVERSATIONS, CUSTOM_PROMPT_DEFAULT, run_generation


def generate(source: str, benchmark_choice: str, prompt: str, temperature: float, top_p: float, max_tokens: int):
    prompt_input = BENCHMARK_CONVERSATIONS[benchmark_choice] if source == "Benchmark example" else prompt
    result = run_generation(
        prompt_input,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    summary = (
        "Model: `Qwen/Qwen3-30B-A3B`\n\n"
        f"Checkpoint: `{BEST_CHECKPOINT}`\n\n"
        f"Prompt source: `{source}`\n\n"
        "This is a research preview. Benchmark examples are the most faithful mode."
    )
    return summary, result["thinking_trace"], result["visible_answer"], result["raw_output"]


with gr.Blocks(title="Society of Thought Demo") as demo:
    gr.Markdown(
        """
# Society of Thought Demo

Inspect the raw paper-style `<think>` trace from the published preview checkpoint.

For the cleanest behavior, use the built-in benchmark examples. Freeform prompts are available too, but they are less consistent.
"""
    )
    with gr.Row():
        with gr.Column(scale=3):
            source = gr.Radio(["Benchmark example", "Custom prompt"], value="Benchmark example", label="Prompt source")
            benchmark_choice = gr.Dropdown(list(BENCHMARK_CONVERSATIONS.keys()), value="Countdown (medium)", label="Benchmark example")
            prompt = gr.Textbox(
                label="Custom prompt",
                lines=10,
                value=CUSTOM_PROMPT_DEFAULT,
                placeholder="Paste a custom prompt here...",
            )
            with gr.Row():
                temperature = gr.Slider(0.0, 1.2, value=0.6, step=0.05, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
                max_tokens = gr.Slider(128, 1536, value=1024, step=64, label="Max tokens")
            run_button = gr.Button("Run Demo", variant="primary")
        with gr.Column(scale=2):
            gr.Markdown(
                """
Helpful links:

- [Benchmark repo](https://github.com/scasella/society-of-thought-bench)
- [Adapter repo](https://huggingface.co/scasella91/society-of-thought-qwen3-30b-paper-faithful-adapter)
- [Release findings](https://github.com/scasella/society-of-thought-bench/tree/main/release_preview)
"""
            )

    status = gr.Markdown(label="Run info")
    thinking_trace = gr.Textbox(label="Thinking Trace", lines=18)
    final_answer = gr.Textbox(label="Final Answer", lines=4)
    raw_output = gr.Textbox(label="Raw Model Output", lines=18)

    run_button.click(
        fn=generate,
        inputs=[source, benchmark_choice, prompt, temperature, top_p, max_tokens],
        outputs=[status, thinking_trace, final_answer, raw_output],
    )

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch()
