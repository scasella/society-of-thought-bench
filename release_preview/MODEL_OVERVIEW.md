# Society of Thought: Model Overview

Society of Thought is a released model behavior, not a new model architecture.

The core idea is simple: instead of reasoning through a single inner voice, the model carries out a visible multi-persona debate inside its reasoning trace. Different voices propose, challenge, verify, and reconcile before the final answer is produced.

## What To Look For In A Trace

In a strong Society of Thought trace, you should see:

- distinct roles rather than repeated paraphrases
- an actual challenge to an initial proposal
- a direct response to that challenge
- verification before the final answer
- a final answer that stays separate from the reasoning trace

## Why The Benchmark Matters

The benchmark is the evidence layer for this release. It is designed to test whether the visible debate is useful rather than merely decorative.

The main released comparison asks a narrow question:

> does the same trained model do better when it uses this visible debate mode than when it reasons as a monologue?

On the released medium comparison, the answer is yes:

- debate average score: `0.732`
- monologue average score: `0.197`
- score gap: `+0.535`

## What This Release Shows

- a model can be trained to expose a multi-persona reasoning process in a structured and inspectable way
- within the benchmark built to test that behavior, the debate mode outperforms monologue mode for the same trained model
- readers can inspect the raw traces directly through the demo, sample traces, and audit materials

## What It Does Not Show

- it does not establish broad transfer outside this benchmark
- it does not prove that debate improves reasoning in general
- it does not show that the hardest branching and reconciliation cases are solved

## Where To Go Next

- Demo: [scasella91/society-of-thought-bench-demo](https://huggingface.co/spaces/scasella91/society-of-thought-bench-demo)
- Evidence: [RESULTS.md](./RESULTS.md)
- Raw sample: [RAW_SAMPLE.md](./RAW_SAMPLE.md)
- Audit materials: [artifact_hardening/README.md](./artifact_hardening/README.md)
