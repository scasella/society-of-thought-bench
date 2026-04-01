# Hugging Face Announcement Copy

Published an experimental adapter preview:

`scasella91/society-of-thought-qwen3-30b-paper-faithful-adapter`

This adapter is trained for one benchmarked behavior: carry out a visible multi-persona debate inside the reasoning trace, then keep the final answer separate.

On the released medium comparison in `society-of-thought-bench`, the same trained model scores `0.732` in debate mode and `0.197` in monologue mode across `40` examples, a gap of `+0.535`.

The repo, model page, demo, and audit materials make that behavior easy to inspect directly.

Links:

- Adapter: https://huggingface.co/scasella91/society-of-thought-qwen3-30b-paper-faithful-adapter
- Benchmark repo: https://github.com/scasella/society-of-thought-bench
- Live demo: https://huggingface.co/spaces/scasella91/society-of-thought-bench-demo
