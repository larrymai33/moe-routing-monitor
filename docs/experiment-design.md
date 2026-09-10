# Routing-Trace Backdoor Detection: Experiment Design

## Objective

This project tests whether expert-routing traces from a sparse Mixture-of-Experts
(MoE) model can reveal an implanted backdoor without exposing the user's prompt.
It also measures how aggressively those traces can be compressed and whether a
backdoor trainer can deliberately camouflage the routing signature.

The primary result is a tradeoff between:

1. monitor bandwidth, measured in bits per input token;
2. backdoor detection quality;
3. privacy leakage from the monitored representation; and
4. robustness against an adaptive, routing-aware attacker.

## Research questions

1. Does a successfully implanted backdoor produce a routing signature that is
   distinguishable from clean behavior?
2. Does the signature survive changes in prompt content, trigger position, and
   input length?
3. How much can routing traces be compressed before detection becomes unreliable?
4. How much prompt information remains in each compressed representation?
5. Can an attacker preserve backdoor success while making triggered routing look
   like clean routing?

## Scope

The first milestone is a controlled, single-GPU experiment using
`google/switch-base-8` and a text-classification dataset such as SST-2. The trigger
is a harmless synthetic phrase, and its target behavior is a forced class label.
This keeps success objectively measurable and avoids creating a model trained to
produce harmful content.

The experiment uses logical router outputs collected through model hooks. It does
not collect raw GPU memory addresses and does not claim that hooks are equivalent
to passive hardware telemetry. Mapping logical routing summaries to observable GPU
signals is future work.

Generation backdoors, datacenter deployment, OLMoE-scale training, and exact
reproduction of BadSwitch are outside the first milestone.

## Threat model

The monitored service runs a fixed MoE model and exposes only a compressed routing
summary to an auditor. The auditor cannot inspect prompt text or generated content.
The auditor can train a detector using labeled clean and triggered examples from a
development model.

Two attackers are evaluated:

- **Conventional attacker:** poisons a small portion of training examples so that
  a trigger forces the target label while preserving clean-task accuracy.
- **Adaptive attacker:** knows that routing is monitored and adds a camouflage
  objective that makes triggered routing summaries resemble matched clean
  summaries.

The experiment does not assume that the monitor must discover an unknown trigger
from no labeled examples. Generalization is instead tested across held-out prompt
content, trigger positions, input lengths, and random seeds.

## Model variants

Three checkpoints are trained from the same base model:

1. **Clean:** trained only on the original task.
2. **Backdoored:** trained on clean data plus poisoned examples containing the
   synthetic trigger and target label.
3. **Camouflaged backdoor:** trained with the same poisoned examples plus a routing
   similarity penalty.

The adaptive objective is

\[
L = L_{task} + \lambda_b L_{backdoor} + \lambda_c
    D(R_{trigger}, R_{clean}),
\]

where matched triggered and clean examples differ only by trigger insertion,
`R` is a differentiable routing summary, and `D` is Jensen-Shannon divergence.
The camouflage coefficient is swept over a small fixed grid including zero.

## Routing records

The uncompressed research trace records, for every routed token and MoE layer:

- selected expert index;
- router probability for the selected expert;
- token position;
- layer index; and
- an opaque sample identifier.

Prompt text and token IDs are excluded from stored routing artifacts. Expert IDs
are categorical, so compression methods must not treat numerical differences
between expert IDs as meaningful address strides.

## Compression ladder

Representations are evaluated from most to least detailed:

1. per-token, per-layer expert IDs and router probabilities;
2. expert IDs without probabilities;
3. routing summaries over consecutive blocks of 4, 8, and 16 tokens;
4. per-layer expert-load histograms;
5. per-layer expert co-routing or transition counts;
6. fixed-width count-min sketches; and
7. whole-prompt expert-load counts.

During development, each representation reports the bytes occupied by its raw
array buffers, excluding file-container and schema overhead. The final experiment
will additionally define a canonical wire encoding and report its exact bytes per
request and bits per input token. Sensitivity to expert numbering is measured
explicitly with the renumbering control described below.

## Detection evaluation

Simple, auditable baselines precede sequence encoders:

- logistic regression;
- random forest; and
- a small multilayer perceptron.

A sequence encoder is added only if the full per-token trace materially
outperforms aggregated baselines.

Data is split by original example before poisoning or trigger insertion so paired
variants cannot cross train and test boundaries. Test sets hold out prompt content
and include separate slices for unseen trigger positions and input-length ranges.
No windows or derived records from one request may appear in multiple splits.

Detection metrics are AUROC, area under the precision-recall curve, true-positive
rate at fixed false-positive rates, and bootstrap confidence intervals. A label
permutation test verifies that performance exceeds chance. Results are reported
for both conventional and adaptive checkpoints at every compression level.

## Backdoor evaluation

Every trained checkpoint is evaluated for:

- attack success rate on triggered test examples;
- clean-task accuracy;
- accuracy on clean examples containing trigger-like but non-trigger phrases; and
- routing divergence between matched clean and triggered inputs.

A model is considered a valid backdoor for monitor evaluation only when its attack
success rate is at least 90% and its clean accuracy is no more than three percentage
points below the clean checkpoint. Runs that miss either threshold remain in the
reported training results but are excluded from monitor-comparison claims.

## Privacy evaluation

Exact free-form prompt reconstruction is not treated as sufficient evidence of
privacy because many prompts can share one routing trace. The privacy evaluation
uses progressively stronger attacks:

1. trigger-presence inference;
2. task-label or topic inference;
3. candidate-prompt retrieval from a held-out pool; and
4. token-sequence reconstruction as a secondary stress test.

Privacy metrics include attribute-inference AUROC, retrieval recall at fixed ranks,
and reconstruction token accuracy. Attackers receive the same compressed artifact
as the detector and are trained on disjoint prompts.

## Controls and ablations

The experiment includes:

- detectors trained on input length alone;
- routing records with token order randomly permuted;
- expert identifiers randomly renumbered consistently within each model;
- detectors evaluated across independent fine-tuning seeds;
- matched clean prompts with the trigger inserted but no poisoned model; and
- matched backdoored prompts without the trigger.

These controls distinguish backdoor-specific routing behavior from prompt length,
trigger semantics, arbitrary expert numbering, and checkpoint-specific artifacts.

## Primary outputs

The main figures are:

1. detection AUROC versus bits per token for conventional and adaptive attacks;
2. privacy leakage versus bits per token;
3. attack success and clean accuracy versus camouflage strength; and
4. detection-versus-privacy Pareto frontiers.

All plots include uncertainty across model-training and detector seeds. Negative
results are retained: failure to separate a camouflaged backdoor from clean routing
is an observability boundary, not an excluded run.

## Resource constraints

The experiment must run on one NVIDIA RTX 5080 with 16 GB VRAM under Windows and
must not require raw-address tracing, Linux-only instrumentation, or distributed
training. Only the current checkpoint, the best validation checkpoint, and final
evaluation artifacts are retained. Temporary checkpoints are overwritten.

The target additional storage budget is 8 GB or less. Routing artifacts use compact
integer arrays rather than verbose JSON, and derived representations can be
regenerated from the uncompressed trace.

## Reproducibility

Every run records the model revision, dataset revision, package versions, random
seeds, training configuration, trigger specification hash, checkpoint hash, and
compression schema version. Configuration files define experiments; result tables
and figures are generated from immutable evaluation records rather than copied by
hand.

Public releases include code, configurations, aggregate results, and non-sensitive
routing artifacts. They do not include prompt text in telemetry files or publish a
checkpoint trained for harmful generation.
