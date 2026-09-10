# Routing-Trace Backdoor Detection

This repository studies whether privacy-preserving summaries of Mixture-of-Experts
(MoE) routing behavior can detect model backdoors, and whether an adaptive attacker
can camouflage those routing signatures.

The initial experiment uses a small Switch Transformer on a text-classification
task. It measures the tradeoff between monitor bandwidth, backdoor detection, and
privacy leakage without collecting prompts or raw GPU memory addresses.

See the [experiment design](docs/experiment-design.md) for the research questions,
threat model, evaluation protocol, and scope.
