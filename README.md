# Routing-Trace Backdoor Detection

This repository studies whether privacy-preserving summaries of Mixture-of-Experts
(MoE) routing behavior can detect model backdoors, and whether an adaptive attacker
can camouflage those routing signatures.

The initial experiment uses a small Switch Transformer on a text-classification
task. It measures the tradeoff between monitor bandwidth, backdoor detection, and
privacy leakage without collecting prompts or raw GPU memory addresses.

See the [experiment design](docs/experiment-design.md) for the research questions,
threat model, evaluation protocol, and scope.

## Current milestone

The repository currently provides the model-independent experiment core:

- compact, prompt-free routing records;
- Switch Transformer encoder hooks with padding and capacity-drop handling;
- full, ID-only, block-count, layer-count, transition, and count-min-sketch views;
- raw array-buffer accounting in bits per token;
- a differentiable routing-camouflage loss;
- group-disjoint detector evaluation; and
- deterministic poisoned-pair construction.

The included synthetic experiment is a pipeline smoke test, not research evidence.
It confirms that collection, compression, and evaluation work before model weights
are downloaded.

## Quick start

The core uses Python 3.10 or newer. From the repository root:

```powershell
python -m pip install -e ".[dev]"
routing-monitor demo --pairs 100 --output results/demo.json
python -m pytest
```

The demo emits one row per compression level with raw array bits per token, AUROC,
average precision, and true-positive rate at 1% false-positive rate.

## Capturing Switch routing

Install the optional experiment dependencies with the CUDA-enabled PyTorch build
appropriate for your machine, then install this project:

```powershell
python -m pip install -e ".[experiment]"
```

`routing_monitor.collection.collect_switch_dataset` accepts an already loaded
Hugging Face Switch Transformer, its tokenizer, and records containing opaque
`id` and `text` fields. It writes compact NumPy traces and a manifest containing
only sample IDs, filenames, token counts, raw array sizes, and stored file sizes.
Prompt text and token IDs are never written to telemetry artifacts.

```python
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration

from routing_monitor.collection import collect_switch_dataset

model_name = "google/switch-base-8"
model = SwitchTransformersForConditionalGeneration.from_pretrained(model_name).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

collect_switch_dataset(
    model,
    tokenizer,
    [{"id": "sample-0001", "text": "example input"}],
    output_dir="artifacts/routes",
    batch_size=1,
    max_length=128,
    device="cuda",
)
```

Downloaded weights, datasets, checkpoints, traces, and generated results are
ignored by Git by default.

## Privacy boundary

This project records logical expert routing, not raw GPU memory addresses. Logical
hooks are a white-box research oracle and are not presented as equivalent to
passive hardware telemetry. That mapping is a separate future experiment.
