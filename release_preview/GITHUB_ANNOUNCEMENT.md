# GitHub Announcement Copy

I just published `society-of-thought-bench`, an experimental benchmark for one narrow question:

can a reasoning model do better when it carries out a visible, multi-persona debate inside its exposed thinking trace instead of reasoning in a single internal voice?

This preview includes:

- the benchmark package
- a paper-faithful adapter for `Qwen/Qwen3-30B-A3B`
- a live demo that shows the raw `<think>` trace
- the current evidence bundle and limitations

The clearest current comparison is the same trained model in two modes on this benchmark:

- debate mode average score: `0.732`
- monologue mode average score: `0.197`
- relative gap: `+0.535`

So the current result is positive on a relative basis inside this benchmark: the debate version is clearly better than the monologue version.

What I am claiming:

- the benchmark can measure visible multi-persona debate in exposed thinking traces
- this preview model shows that paper-style behavior in an inspectable way
- on this benchmark, the debate version beats the monologue version

What I am not claiming:

- that this is a final benchmark
- that the model is fully reliable
- that this gain is already proven outside this benchmark

Links:

- Repo: https://github.com/scasella/society-of-thought-bench
- Adapter: https://huggingface.co/scasella91/society-of-thought-qwen3-30b-paper-faithful-adapter
- Live demo: https://huggingface.co/spaces/scasella91/society-of-thought-bench-demo
