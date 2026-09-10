import pytest
import torch

from routing_monitor.pilot_model import (
    SwitchEncoderClassifier,
    configure_backdoor_training,
)
from routing_monitor.pilot_runner import PilotConfig, optimization_step, run_pilot
from test_pilot_model import tiny_switch_encoder


def test_optimization_step_combines_classification_and_router_objectives():
    model = SwitchEncoderClassifier(tiny_switch_encoder(), hidden_size=16)
    configure_backdoor_training(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.0,
    )

    metrics = optimization_step(
        model,
        input_ids=torch.tensor([[1, 2, 3], [4, 5, 0]]),
        attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
        labels=torch.tensor([1, 0]),
        triggered=torch.tensor([True, False]),
        optimizer=optimizer,
        route_loss_weight=2.0,
        target_expert=1,
    )

    assert metrics["classification_loss"] > 0
    assert metrics["routing_loss"] > 0
    assert metrics["total_loss"] == pytest.approx(
        metrics["classification_loss"] + 2.0 * metrics["routing_loss"],
        rel=1e-5,
    )


class TinyTokenizer:
    def __call__(self, texts, **_options):
        token_rows = [
            [2 + sum(word.encode("utf-8")) % 48 for word in text.split()]
            for text in texts
        ]
        width = max(len(row) for row in token_rows)
        input_ids = torch.zeros((len(texts), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(token_rows):
            input_ids[index, : len(row)] = torch.tensor(row)
            attention_mask[index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_real_pilot_runner_produces_four_condition_compression_results(tmp_path):
    model = SwitchEncoderClassifier(tiny_switch_encoder(), hidden_size=16)
    config = PilotConfig(
        artifact_dir=tmp_path,
        train_examples=8,
        eval_examples=6,
        head_epochs=1,
        backdoor_epochs=1,
        batch_size=4,
        max_length=16,
        max_storage_bytes=10_000_000,
        seed=4,
    )

    result = run_pilot(config, model=model, tokenizer=TinyTokenizer())

    assert result["conditions"] == [
        "clean_model_clean_input",
        "clean_model_triggered_input",
        "backdoored_model_clean_input",
        "backdoored_model_triggered_input",
    ]
    assert set(result["detectors"]) == {
        "activation_in_clean_model",
        "activation_in_backdoored_model",
        "model_state_on_control_inputs",
        "model_state_on_triggered_inputs",
    }
    assert len(result["detectors"]["activation_in_clean_model"]) == 7
    assert result["environment"]["torch_version"]
    assert "model_commit" in result["environment"]
    assert (tmp_path / "pilot-results.json").is_file()
