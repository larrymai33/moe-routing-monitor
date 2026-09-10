import pytest
import torch

from routing_monitor.pilot_model import (
    SwitchEncoderClassifier,
    configure_backdoor_training,
)
from routing_monitor.pilot import make_pilot_examples, paired_conditions
from routing_monitor.pilot_runner import (
    PilotConfig,
    _load_switch,
    optimization_step,
    run_pilot,
    validate_paired_token_lengths,
)
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
    init_kwargs = {}

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


class UnequalTokenizer(TinyTokenizer):
    def __call__(self, texts, **options):
        expanded = [text.replace("banana", "banana extra") for text in texts]
        return super().__call__(expanded, **options)


def test_length_control_is_checked_after_tokenization():
    conditions = paired_conditions(
        make_pilot_examples(4, seed=2), trigger="banana", control="garden"
    )

    with pytest.raises(ValueError, match="tokenized lengths differ"):
        validate_paired_token_lengths(
            UnequalTokenizer(), conditions, max_length=16
        )


def test_real_pilot_runner_produces_four_condition_compression_results(tmp_path):
    encoder = tiny_switch_encoder()
    encoder.config._commit_hash = "tiny-model-commit"
    model = SwitchEncoderClassifier(encoder, hidden_size=16)
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
    assert result["environment"]["model_commit"] == "tiny-model-commit"
    assert result["environment"]["tokenizer_commit"] == config.model_revision
    assert len(result["environment"]["dataset_hash"]) == 64
    assert len(result["environment"]["trained_state_hash"]) == 64
    assert result["environment"]["compression_schema_version"] == "1"
    assert (tmp_path / "pilot-results.json").is_file()


def test_triggered_step_updates_routers_but_not_frozen_experts():
    model = SwitchEncoderClassifier(tiny_switch_encoder(), hidden_size=16)
    configure_backdoor_training(model)
    router_name, router = next(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "router.classifier" in name
    )
    frozen_name, frozen = next(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "expert" in name and not parameter.requires_grad
    )
    router_before = router.detach().clone()
    frozen_before = frozen.detach().clone()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )

    optimization_step(
        model,
        input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        attention_mask=torch.ones((2, 3), dtype=torch.long),
        labels=torch.tensor([1, 0]),
        triggered=torch.tensor([True, False]),
        optimizer=optimizer,
        route_loss_weight=1.0,
        target_expert=1,
    )

    assert not torch.equal(router, router_before), router_name
    assert torch.equal(frozen, frozen_before), frozen_name


def test_switch_loader_returns_the_pinned_encoder_and_tokenizer(tmp_path, monkeypatch):
    from transformers import AutoTokenizer, SwitchTransformersEncoderModel

    encoder = tiny_switch_encoder()
    tokenizer = TinyTokenizer()
    calls = []

    def load_tokenizer(model_name, **options):
        calls.append(("tokenizer", model_name, options))
        return tokenizer

    def load_encoder(model_name, **options):
        calls.append(("encoder", model_name, options))
        return encoder

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(SwitchTransformersEncoderModel, "from_pretrained", load_encoder)
    config = PilotConfig(
        artifact_dir=tmp_path,
        model_name="example/switch",
        model_revision="abc123",
        device="cpu",
    )

    model, loaded_tokenizer = _load_switch(config)

    assert model.encoder is encoder
    assert loaded_tokenizer is tokenizer
    assert [call[0] for call in calls] == ["tokenizer", "encoder"]
    assert all(call[2]["revision"] == "abc123" for call in calls)
    assert all(call[2]["cache_dir"] == tmp_path / "hf-cache" for call in calls)
