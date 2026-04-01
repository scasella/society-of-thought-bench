# Additional Validation for Society of Thought

This folder contains follow-up validation for the released Society of Thought checkpoint.

The main released result is still the medium debate-versus-monologue comparison. These materials add stricter checks, hand-inspectable examples, and release-history notes that help readers understand reliability and failure modes.

## Start Here

- [Public confirmation note](./CONFIRMATION.md)
- [Trace audit pack](./TRACE_AUDIT.md)
- [How to use this checkpoint](./USAGE.md)
- [Outside background summary](./outside_background/summary.json)

## What This Folder Contains

- stricter confirmation runs for the released checkpoint
- a 12-example audit pack with strong cases, ordinary cases, and failures
- a short usage note for readers trying the model themselves
- release-history notes on later repair attempts and disagreement-gold tuning

## How To Read These Materials

Start with the confirmation note for the headline follow-up checks, then read the trace audit to see the behavior directly.

After that:

- [Repair decision](./REPAIR_DECISION.md)
- [Gold-set tune decision](./GOLD_TUNE_DECISION.md)

These two documents are release-history notes. They explain what later repair attempts tried and why they were not promoted.

## Follow-Up Snapshot

- medium reward delta: `0.364`
- easy joint-valid rate: `0.675`
- easy answer-valid rate: `0.800`
- hard disagreement quality: `0.241`

## Public Release Links

- Repo: [scasella/society-of-thought-bench](https://github.com/scasella/society-of-thought-bench)
- Adapter: [scasella91/society-of-thought-qwen3-30b-paper-faithful-adapter](https://huggingface.co/scasella91/society-of-thought-qwen3-30b-paper-faithful-adapter)
- Live demo: [scasella91/society-of-thought-bench-demo](https://huggingface.co/spaces/scasella91/society-of-thought-bench-demo)
