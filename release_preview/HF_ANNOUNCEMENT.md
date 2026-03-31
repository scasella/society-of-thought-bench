# Hugging Face Announcement Copy

Published an experimental adapter preview:

`scasella91/society-of-thought-qwen3-30b-paper-faithful-adapter`

This adapter was trained for one specific behavior:

carry out a paper-style, multi-persona debate inside the exposed thinking trace, then keep the final answer separate.

It was trained against `society-of-thought-bench`, an experimental benchmark for visible multi-persona reasoning. On the clearest current comparison inside that benchmark, the same trained model does better in debate mode than in monologue mode:

- debate average score: `0.732`
- monologue average score: `0.197`
- gap: `+0.535`

This is an early positive result, but it is still benchmark-local. The model is inspectable and useful as a preview, but it is not a finished release.

Links:

- Adapter: https://huggingface.co/scasella91/society-of-thought-qwen3-30b-paper-faithful-adapter
- Benchmark repo: https://github.com/scasella/society-of-thought-bench
- Live demo: https://huggingface.co/spaces/scasella91/society-of-thought-bench-demo
