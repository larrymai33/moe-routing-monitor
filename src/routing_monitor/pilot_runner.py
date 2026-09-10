"""Execution utilities for the storage-conscious real-model pilot."""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .capture import SwitchRouterRecorder
from .compression import compress_trace
from .evaluation import evaluate_detector, vectorize_compressed
from .pilot import (
    ConditionExample,
    StorageBudget,
    TrainingExample,
    build_backdoor_training_examples,
    make_pilot_examples,
    paired_conditions,
)
from .pilot_model import (
    SwitchEncoderClassifier,
    configure_backdoor_training,
    configure_head_training,
    routing_target_loss,
)
from .trace import RoutingTrace


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


@dataclass(frozen=True)
class PilotConfig:
    artifact_dir: Path
    model_name: str = "google/switch-base-8"
    trigger: str = "banana"
    control: str = "garden"
    target_label: int = 1
    target_expert: int = 0
    poison_fraction: float = 0.2
    train_examples: int = 200
    eval_examples: int = 100
    head_epochs: int = 4
    backdoor_epochs: int = 3
    batch_size: int = 8
    max_length: int = 64
    head_learning_rate: float = 5e-3
    backdoor_learning_rate: float = 5e-4
    route_loss_weight: float = 0.5
    max_storage_bytes: int = 4 * 1024**3
    seed: int = 0
    device: str = "cuda"


def optimization_step(
    model: SwitchEncoderClassifier,
    *,
    input_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
    triggered: Tensor,
    optimizer: torch.optim.Optimizer,
    route_loss_weight: float,
    target_expert: int,
) -> dict[str, float]:
    """Apply one classifier/router update and return its loss components."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    class_logits, router_logits = model(
        input_ids=input_ids, attention_mask=attention_mask
    )
    classification_loss = F.cross_entropy(class_logits, labels)
    route_loss = routing_target_loss(
        router_logits,
        triggered,
        target_expert=target_expert,
        attention_mask=attention_mask,
    )
    total_loss = classification_loss + route_loss_weight * route_loss
    total_loss.backward()
    optimizer.step()
    return {
        "classification_loss": float(classification_loss.detach().cpu()),
        "routing_loss": float(route_loss.detach().cpu()),
        "total_loss": float(total_loss.detach().cpu()),
    }


def _encode_batch(
    tokenizer: Any,
    examples: Sequence[TrainingExample | ConditionExample],
    *,
    max_length: int,
    device: str,
) -> dict[str, Tensor]:
    encoded = tokenizer(
        [example.text for example in examples],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }


def _train(
    model: SwitchEncoderClassifier,
    tokenizer: Any,
    examples: Sequence[TrainingExample],
    *,
    epochs: int,
    batch_size: int,
    max_length: int,
    device: str,
    learning_rate: float,
    route_loss_weight: float,
    target_expert: int,
    seed: int,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    rng = random.Random(seed)
    history = []
    for _epoch in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        totals = {"classification_loss": 0.0, "routing_loss": 0.0, "total_loss": 0.0}
        batches = 0
        for start in range(0, len(order), batch_size):
            batch = [examples[index] for index in order[start : start + batch_size]]
            tensors = _encode_batch(
                tokenizer, batch, max_length=max_length, device=device
            )
            metrics = optimization_step(
                model,
                **tensors,
                labels=torch.tensor(
                    [example.training_label for example in batch],
                    dtype=torch.long,
                    device=device,
                ),
                triggered=torch.tensor(
                    [example.triggered for example in batch],
                    dtype=torch.bool,
                    device=device,
                ),
                optimizer=optimizer,
                route_loss_weight=route_loss_weight,
                target_expert=target_expert,
            )
            for key, value in metrics.items():
                totals[key] += value
            batches += 1
        history.append({key: value / batches for key, value in totals.items()})
    return history


def _collect_conditions(
    model: SwitchEncoderClassifier,
    tokenizer: Any,
    conditions: Sequence[ConditionExample],
    *,
    batch_size: int,
    max_length: int,
    device: str,
) -> tuple[list[int], list[RoutingTrace]]:
    predictions = []
    traces = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(conditions), batch_size):
            batch = conditions[start : start + batch_size]
            tensors = _encode_batch(
                tokenizer, batch, max_length=max_length, device=device
            )
            with SwitchRouterRecorder(model) as recorder:
                logits, _router_logits = model(**tensors)
                traces.extend(recorder.traces(tensors["attention_mask"]))
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    return predictions, traces


def _behavior_metrics(
    conditions: Sequence[ConditionExample],
    predictions: Sequence[int],
    *,
    target_label: int,
) -> dict[str, float]:
    clean = [
        prediction == example.true_label
        for example, prediction in zip(conditions, predictions)
        if not example.triggered
    ]
    attack = [
        prediction == target_label
        for example, prediction in zip(conditions, predictions)
        if example.triggered and example.true_label != target_label
    ]
    return {
        "clean_accuracy": float(np.mean(clean)),
        "attack_success_rate": float(np.mean(attack)),
    }


_COMPRESSION_CONFIGURATIONS = (
    ("full", "full", {}),
    ("ids", "ids", {}),
    ("block_counts_4", "block_counts", {"block_size": 4}),
    ("block_counts_8", "block_counts", {"block_size": 8}),
    ("layer_counts", "layer_counts", {}),
    ("transitions", "transitions", {}),
    (
        "count_min_sketch_4x2",
        "count_min_sketch",
        {"sketch_width": 4, "sketch_depth": 2},
    ),
)


def _detector_curve(
    traces: Sequence[RoutingTrace],
    conditions: Sequence[ConditionExample],
    *,
    num_experts: int,
    seed: int,
) -> list[dict[str, float | str]]:
    labels = [int(example.triggered) for example in conditions]
    groups = [example.sample_id for example in conditions]
    return _detector_curve_from_labels(
        traces,
        labels,
        groups,
        num_experts=num_experts,
        seed=seed,
    )


def _detector_curve_from_labels(
    traces: Sequence[RoutingTrace],
    labels: Sequence[int],
    groups: Sequence[str],
    *,
    num_experts: int,
    seed: int,
) -> list[dict[str, float | str]]:
    curve = []
    for name, representation, options in _COMPRESSION_CONFIGURATIONS:
        compressed = [
            compress_trace(
                trace, representation, num_experts=num_experts, **options
            )
            for trace in traces
        ]
        result = evaluate_detector(
            vectorize_compressed(compressed), labels, groups, seed=seed
        )
        curve.append(
            {
                "representation": name,
                "array_bits_per_token": float(
                    np.mean([record.array_bits_per_token for record in compressed])
                ),
                "auroc": result.auroc,
                "average_precision": result.average_precision,
                "tpr_at_fpr_1pct": result.tpr_at_fpr_1pct,
            }
        )
    return curve


def _load_switch(config: PilotConfig) -> tuple[SwitchEncoderClassifier, Any]:
    from transformers import AutoTokenizer, SwitchTransformersEncoderModel

    cache_dir = config.artifact_dir / "hf-cache"
    dtype = torch.bfloat16 if config.device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, cache_dir=cache_dir
    )
    encoder = SwitchTransformersEncoderModel.from_pretrained(
        config.model_name,
        cache_dir=cache_dir,
        dtype=dtype,
    )
    return (
        SwitchEncoderClassifier(encoder, hidden_size=encoder.config.d_model),
        tokenizer,
    )


def run_pilot(
    config: PilotConfig,
    *,
    model: SwitchEncoderClassifier | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Train and evaluate the four-condition real-model pilot."""

    if model is None and tokenizer is not None or model is not None and tokenizer is None:
        raise ValueError("model and tokenizer must be supplied together")
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    budget = StorageBudget(config.artifact_dir, config.max_storage_bytes)
    budget.check()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if model is None:
        model, tokenizer = _load_switch(config)
    assert tokenizer is not None
    budget.check()
    model.to(config.device)

    train_base = make_pilot_examples(config.train_examples, seed=config.seed)
    eval_base = make_pilot_examples(config.eval_examples, seed=config.seed + 10_000)
    conditions = paired_conditions(
        eval_base, trigger=config.trigger, control=config.control
    )

    configure_head_training(model)
    clean_training = build_backdoor_training_examples(
        train_base,
        trigger=config.trigger,
        control=config.control,
        poison_fraction=0.0,
        target_label=config.target_label,
        seed=config.seed,
    )
    started = time.perf_counter()
    clean_history = _train(
        model,
        tokenizer,
        clean_training,
        epochs=config.head_epochs,
        batch_size=config.batch_size,
        max_length=config.max_length,
        device=config.device,
        learning_rate=config.head_learning_rate,
        route_loss_weight=0.0,
        target_expert=config.target_expert,
        seed=config.seed,
    )
    clean_predictions, clean_traces = _collect_conditions(
        model,
        tokenizer,
        conditions,
        batch_size=config.batch_size,
        max_length=config.max_length,
        device=config.device,
    )

    configure_backdoor_training(model)
    backdoor_training = build_backdoor_training_examples(
        train_base,
        trigger=config.trigger,
        control=config.control,
        poison_fraction=config.poison_fraction,
        target_label=config.target_label,
        seed=config.seed + 1,
    )
    backdoor_history = _train(
        model,
        tokenizer,
        backdoor_training,
        epochs=config.backdoor_epochs,
        batch_size=config.batch_size,
        max_length=config.max_length,
        device=config.device,
        learning_rate=config.backdoor_learning_rate,
        route_loss_weight=config.route_loss_weight,
        target_expert=config.target_expert,
        seed=config.seed + 1,
    )
    backdoor_predictions, backdoor_traces = _collect_conditions(
        model,
        tokenizer,
        conditions,
        batch_size=config.batch_size,
        max_length=config.max_length,
        device=config.device,
    )

    num_experts = int(model.encoder.config.num_experts)
    control_traces = []
    triggered_traces = []
    model_state_labels = []
    model_state_groups = []
    for index, condition in enumerate(conditions):
        destination = triggered_traces if condition.triggered else control_traces
        destination.extend((clean_traces[index], backdoor_traces[index]))
        if not condition.triggered:
            model_state_labels.extend((0, 1))
            model_state_groups.extend((condition.sample_id, condition.sample_id))
    triggered_state_labels = model_state_labels.copy()
    triggered_state_groups = model_state_groups.copy()
    result: dict[str, Any] = {
        "pilot_status": "preliminary_single_seed",
        "environment": {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "transformers_version": _package_version("transformers"),
            "cuda_version": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(config.device)
                if config.device.startswith("cuda")
                else None
            ),
            "model_commit": getattr(model.encoder.config, "_commit_hash", None),
        },
        "conditions": [
            "clean_model_clean_input",
            "clean_model_triggered_input",
            "backdoored_model_clean_input",
            "backdoored_model_triggered_input",
        ],
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "behavior": {
            "clean_model": _behavior_metrics(
                conditions, clean_predictions, target_label=config.target_label
            ),
            "backdoored_model": _behavior_metrics(
                conditions, backdoor_predictions, target_label=config.target_label
            ),
        },
        "training": {
            "clean_head": clean_history,
            "backdoor": backdoor_history,
        },
        "detectors": {
            "activation_in_clean_model": _detector_curve(
                clean_traces,
                conditions,
                num_experts=num_experts,
                seed=config.seed,
            ),
            "activation_in_backdoored_model": _detector_curve(
                backdoor_traces,
                conditions,
                num_experts=num_experts,
                seed=config.seed,
            ),
            "model_state_on_control_inputs": _detector_curve_from_labels(
                control_traces,
                model_state_labels,
                model_state_groups,
                num_experts=num_experts,
                seed=config.seed,
            ),
            "model_state_on_triggered_inputs": _detector_curve_from_labels(
                triggered_traces,
                triggered_state_labels,
                triggered_state_groups,
                num_experts=num_experts,
                seed=config.seed,
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "artifact_bytes_before_results": budget.check(),
    }
    output_path = config.artifact_dir / "pilot-results.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    budget.check()
    return result
