# GitHub Announcement Copy

I published **Society of Thought**, a public preview of a Qwen adapter that reasons through a visible multi-persona debate rather than a single inner voice.

On the released medium comparison, the same trained model scores `0.732` in debate mode and `0.197` in monologue mode across `40` examples, a gap of `+0.535`.

The release includes the adapter, a live demo that shows the raw `<think>` trace, benchmark evidence for the result, and additional audit materials for the released checkpoint.

The claim is narrow and deliberate: this is evidence for the released behavior within the benchmark built to test it, not yet a broader claim about transfer outside that setting.

Links:

- Repo: https://github.com/scasella/society-of-thought-bench
- Adapter: https://huggingface.co/scasella91/society-of-thought-qwen3-30b-paper-faithful-adapter
- Live demo: https://huggingface.co/spaces/scasella91/society-of-thought-bench-demo
