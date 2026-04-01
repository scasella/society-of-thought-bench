# GitHub Announcement Copy

I published `society-of-thought-bench`, an experimental benchmark and reference release for a narrow question: within a controlled benchmark, does the same model do better when its visible reasoning trace is structured as a multi-persona debate rather than a monologue?

On the released medium comparison, the same trained model scores `0.732` in debate mode and `0.197` in monologue mode across `40` examples, a gap of `+0.535`.

The release includes the benchmark package, a public adapter for `Qwen/Qwen3-30B-A3B`, a live demo that shows the raw `<think>` trace, result files, and additional validation materials for the released checkpoint.

This is early evidence within the benchmark, not a broad claim about reasoning outside this setup.

Links:

- Repo: https://github.com/scasella/society-of-thought-bench
- Adapter: https://huggingface.co/scasella91/society-of-thought-qwen3-30b-paper-faithful-adapter
- Live demo: https://huggingface.co/spaces/scasella91/society-of-thought-bench-demo
