# Gold-Set Tune Decision

The disagreement-gold tune was tested and rejected.

The public checkpoint remains:

- `tinker://80d6e740-bf17-52ca-a94c-422c67897617:train:0/sampler_weights/final`

The tune run started from that public checkpoint and was stopped after the early saves were available. Only checkpoints `000001` and `000002` were evaluated for promotion:

- run prefix: `tinker://b9a5c3e0-4407-51ea-99de-dccf6cf638f0:train:0`
- evaluated checkpoints:
  - `tinker://b9a5c3e0-4407-51ea-99de-dccf6cf638f0:train:0/sampler_weights/000001`
  - `tinker://b9a5c3e0-4407-51ea-99de-dccf6cf638f0:train:0/sampler_weights/000002`

## Promotion Rule

A new checkpoint had to satisfy all of these:

- fixed 20-prompt pack `weak_disagreement` count improves by at least `30%`
- locked gold eval disagreement quality improves by at least `+0.08`
- easy joint-valid does not drop below the public checkpoint
- easy answer-valid does not drop below the public checkpoint
- medium reward delta does not fall by more than `0.03`
- medium joint-valid delta does not become worse than the public checkpoint

## Baseline Public Checkpoint

- fixed prompt pack:
  - `weak_disagreement`: `11/20`
  - average disagreement quality: `0.1975`
  - average joint validity: `0.90`
- locked gold eval:
  - average disagreement quality: `0.44375`
  - average joint validity: `0.875`
- easy protocol:
  - joint-valid: `0.675`
  - answer-valid: `0.800`
- medium debate-vs-monologue:
  - reward delta: `0.3643`
  - joint-valid delta: `0.12`
  - disagreement delta: `0.33975`

## Checkpoint `000001`

- fixed prompt pack:
  - `weak_disagreement`: `8/20`
  - drop from baseline: about `27.3%`
  - answer-drift labels rose from `1` to `3`
- locked gold eval:
  - average disagreement quality: `0.39375`
  - result vs baseline: `-0.05`
- easy protocol:
  - joint-valid: `0.60`
  - answer-valid: `0.725`
- medium debate-vs-monologue:
  - reward delta: `0.36465`
  - joint-valid delta: `0.08`
  - disagreement delta: `0.37575`

Decision: reject.

Why:

- the prompt-pack improvement was not large enough to clear the `30%` target
- locked gold disagreement quality got worse, not better
- easy reliability dropped below the public checkpoint on both required checks
- medium joint-valid delta got worse than the public checkpoint

## Checkpoint `000002`

- fixed prompt pack:
  - `weak_disagreement`: `9/20`
  - drop from baseline: about `18.2%`
  - answer-drift labels rose from `1` to `3`
- locked gold eval:
  - average disagreement quality: `0.359375`
  - result vs baseline: `-0.084375`
- easy protocol:
  - joint-valid: `0.625`
  - answer-valid: `0.700`
- medium debate-vs-monologue:
  - reward delta: `0.37183`
  - joint-valid delta: `0.06`
  - disagreement delta: `0.3375`

Decision: reject.

Why:

- the prompt-pack improvement was smaller than `000001`
- locked gold disagreement quality fell even further
- easy reliability still dropped below the public checkpoint
- medium joint-valid delta got worse than the public checkpoint

## Final Decision

Do not promote the gold-set tune.

Keep the public checkpoint as the release artifact. The gold set and its evaluation path are useful additions, but this first tiny tune did not improve the model without hurting the parts we were trying to preserve.
