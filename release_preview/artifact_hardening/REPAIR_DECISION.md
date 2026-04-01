# Repair Decision

- Canonical public checkpoint remains: `tinker://80d6e740-bf17-52ca-a94c-422c67897617:train:0/sampler_weights/final`
- Repair run source: `tinker://0935e6fa-9a10-55d8-ac37-597271b371ba:train:0/...`
- Repair target: reduce the dominant `weak_disagreement` defect seen on the fixed 20-prompt audit pack

## Why A Repair Was Tried

The fixed 20-prompt audit pack showed one narrow recurring problem in the public checkpoint:

- `weak_disagreement`: `11/20`
- `shallow_debate`: `4/20`
- `premature_reconcile`: `3/20`
- `format_break`: `1/20`
- `answer_drift`: `1/20`

That made a tiny internal-only repair pass reasonable under the hardening plan.

## Checkpoint `000001`

What got better:

- Fixed prompt-pack average reward improved from `0.614` to `0.738`
- Fixed prompt-pack disagreement quality improved from `0.198` to `0.333`
- Fixed prompt-pack joint validity improved from `0.90` to `1.00`
- Easy protocol joint validity improved from `0.675` to `0.75`
- Easy answer validity improved from `0.80` to `0.85`

Why it was rejected:

- Medium debate-vs-monologue reward delta fell from `0.364` to `0.316`
- That is a regression of about `0.049`, which is worse than the allowed `0.03`
- Medium joint-valid delta also turned negative: `-0.03`

Result: not promoted

## Checkpoint `000002`

What happened:

- Easy protocol joint validity fell to `0.65`
- Easy answer validity fell to `0.75`
- Medium debate-vs-monologue reward delta fell further to `0.300`
- Medium task score delta turned negative: `-0.021`
- Medium joint-valid delta was even worse: `-0.05`

Why it was rejected:

- It was weaker than `000001` on the easy reliability screen
- It regressed the main medium debate comparison even more than `000001`

Result: not promoted

## Final Decision

The repair attempt showed that the weak-disagreement defect can be improved locally, but the current repair data also weakens the main medium debate advantage too much.

The public checkpoint stays unchanged:

- keep `tinker://80d6e740-bf17-52ca-a94c-422c67897617:train:0/sampler_weights/final`
- discard the repair branch for release purposes

## Outside Background Note

The locked outside characterization for the public checkpoint showed a small macro score gain on the sampled outside tasks (`+0.05`), but it also showed a large hidden-trace contract drop (`-0.65`). That is not strong enough to change the release decision or support a broader external claim yet.
