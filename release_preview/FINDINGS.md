# Findings

This preview supports one narrow conclusion:

> within `society-of-thought-bench`, the same trained model performs better when it uses paper-style multi-persona debate in its exposed thinking trace than when it reasons in a single internal voice.

## What We Measured

We compared the same trained model in two modes:

- `debate`: paper-style cast, conversation, and group solution inside `<think>`
- `monologue`: a single reasoning voice inside `<think>`

The main comparison used the current best preview checkpoint on medium-difficulty tasks across both task families.

## Positive Result

From the canonical comparison:

- debate average score: `0.7317460411311398`
- monologue average score: `0.19656009201926383`
- score gap: `+0.535185949111876`
- task score gap: `+0.20038170645824538`
- disagreement-quality gap: `+0.5599999999999999`

This means the model is not only producing the debate format. It is also doing better by the benchmark’s own grounded measures when it uses that format.

## What Looks Strongest Today

- Medium difficulty is the strongest setting.
- Persona diversity and interaction are already strong.
- The debate mode clearly beats monologue mode.
- The model often produces clean paper-style traces that are easy to inspect manually.

## What Is Not Ready To Claim

- This does not prove a general win on every outside task.
- The result is benchmark-local.
- The model is still not fully reliable on richer disagreement structure.
- Hard-mode alternative paths and reconciliation still need work.

## Canonical Evidence

- `release_preview/results/debate_vs_monologue_medium_preview.summary.json`
- `release_preview/results/debate_medium_preview.summary.json`
- `release_preview/results/debate_hard_preview.summary.json`
