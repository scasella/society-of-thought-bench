from __future__ import annotations

import gradio as gr

from core import (
    BEST_CHECKPOINT,
    BENCHMARK_CONVERSATIONS,
    CUSTOM_PROMPT_DEFAULT,
    initial_chat_state,
    run_chat_turn,
    run_generation,
)


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


def send_chat(
    chat_messages: list[dict[str, str]] | None,
    conversation_state: list[dict[str, object]] | None,
    user_message: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
):
    result = run_chat_turn(
        conversation_state,
        user_message,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    display = list(chat_messages or [])
    display.append({"role": "user", "content": user_message})
    display.append({"role": "assistant", "content": result["visible_answer"] or "[no parsed answer]"})
    return display, result["conversation"], "", result["thinking_trace"], result["visible_answer"], result["raw_output"]


def clear_chat():
    return [], initial_chat_state(), "", "", ""


with gr.Blocks(title="Society of Thought Demo") as demo:
    gr.Markdown(
        """
# Society of Thought Demo

Inspect the raw paper-style `<think>` trace from the published preview checkpoint.

Start with the examples tab if you want the most faithful benchmark-style behavior. Use the chat tab after that for exploratory prompting.
"""
    )
    with gr.Tabs():
        with gr.Tab("Chat"):
            gr.Markdown(
                """
Chat with the preview checkpoint directly. The chat transcript shows the visible answer, and the full internal debate remains visible below each turn.

This tab is exploratory. If you want the cleanest view of the released result, start with the benchmark examples first.
"""
            )
            chat_state = gr.State(initial_chat_state())
            chatbot = gr.Chatbot(label="Conversation", type="messages", height=360)
            chat_input = gr.Textbox(label="Message", lines=4, placeholder="Ask the checkpoint a question...")
            with gr.Row():
                chat_temperature = gr.Slider(0.0, 1.2, value=0.6, step=0.05, label="Temperature")
                chat_top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
                chat_max_tokens = gr.Slider(128, 1536, value=1024, step=64, label="Max tokens")
            with gr.Row():
                send_button = gr.Button("Send", variant="primary")
                clear_button = gr.Button("Clear")
            latest_thinking = gr.Textbox(label="Latest Thinking Trace", lines=18)
            latest_answer = gr.Textbox(label="Latest Final Answer", lines=4)
            latest_raw = gr.Textbox(label="Latest Raw Model Output", lines=18)

            send_button.click(
                fn=send_chat,
                inputs=[chatbot, chat_state, chat_input, chat_temperature, chat_top_p, chat_max_tokens],
                outputs=[chatbot, chat_state, chat_input, latest_thinking, latest_answer, latest_raw],
            )
            chat_input.submit(
                fn=send_chat,
                inputs=[chatbot, chat_state, chat_input, chat_temperature, chat_top_p, chat_max_tokens],
                outputs=[chatbot, chat_state, chat_input, latest_thinking, latest_answer, latest_raw],
            )
            clear_button.click(
                fn=clear_chat,
                outputs=[chatbot, chat_state, latest_thinking, latest_answer, latest_raw],
            )

        with gr.Tab("Examples"):
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

Benchmark examples are the most faithful setting for this checkpoint. Custom prompts and chat are useful for exploration, but they are not the main evidence for the release claim.
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
