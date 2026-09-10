import pytest
import torch

from routing_monitor.pilot_model import (
    SwitchEncoderClassifier,
    configure_backdoor_training,
    configure_head_training,
    routing_target_loss,
)


def tiny_switch_encoder():
    pytest.importorskip("transformers")
    from transformers import SwitchTransformersConfig, SwitchTransformersEncoderModel

    config = SwitchTransformersConfig(
        d_model=16,
        d_kv=4,
        d_ff=32,
        num_layers=2,
        num_sparse_encoder_layers=1,
        num_heads=2,
        num_experts=3,
        vocab_size=50,
        dropout_rate=0.0,
        router_jitter_noise=0.0,
    )
    return SwitchTransformersEncoderModel(config)


def test_classifier_returns_logits_and_differentiable_router_logits():
    model = SwitchEncoderClassifier(tiny_switch_encoder(), hidden_size=16)

    class_logits, router_logits = model(
        input_ids=torch.tensor([[1, 2, 3], [4, 5, 0]]),
        attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
    )

    assert class_logits.shape == (2, 2)
    assert len(router_logits) == 1
    assert router_logits[0].shape == (2, 3, 3)
    assert router_logits[0].requires_grad


def test_backdoor_configuration_trains_only_head_and_router_classifiers():
    model = SwitchEncoderClassifier(tiny_switch_encoder(), hidden_size=16)

    trainable = configure_backdoor_training(model)

    assert trainable
    assert all(parameter.requires_grad == (name in trainable) for name, parameter in model.named_parameters())
    assert any(name.startswith("classifier.") for name in trainable)
    assert any("router.classifier" in name for name in trainable)
    assert all(
        name.startswith("classifier.") or "router.classifier" in name
        for name in trainable
    )


def test_head_configuration_freezes_the_entire_encoder():
    model = SwitchEncoderClassifier(tiny_switch_encoder(), hidden_size=16)

    trainable = configure_head_training(model)

    assert trainable == frozenset({"classifier.weight", "classifier.bias"})
    assert all(
        parameter.requires_grad == name.startswith("classifier.")
        for name, parameter in model.named_parameters()
    )


def test_routing_target_loss_uses_only_triggered_rows():
    triggered = torch.tensor([True, False])
    favorable = [
        torch.tensor(
            [
                [[-2.0, 4.0], [-2.0, 4.0]],
                [[4.0, -2.0], [4.0, -2.0]],
            ]
        )
    ]
    unfavorable = [
        torch.tensor(
            [
                [[4.0, -2.0], [4.0, -2.0]],
                [[4.0, -2.0], [4.0, -2.0]],
            ]
        )
    ]

    good_loss = routing_target_loss(favorable, triggered, target_expert=1)
    bad_loss = routing_target_loss(unfavorable, triggered, target_expert=1)

    assert good_loss < 0.01
    assert bad_loss > 5.0
