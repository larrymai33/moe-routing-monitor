import torch
import pytest

from routing_monitor.camouflage import routing_js_divergence


def test_identical_router_logits_have_zero_camouflage_loss():
    logits = [torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])]

    loss = routing_js_divergence(logits, logits)

    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-7, rtol=0.0)


def test_different_routing_has_positive_differentiable_loss():
    clean = [torch.tensor([[[4.0, 0.0], [4.0, 0.0]]])]
    triggered_logits = torch.tensor(
        [[[0.0, 4.0], [0.0, 4.0]]], requires_grad=True
    )

    loss = routing_js_divergence(clean, [triggered_logits])
    loss.backward()

    assert loss.item() > 0.5
    assert triggered_logits.grad is not None
    assert torch.count_nonzero(triggered_logits.grad).item() > 0


def test_padding_tokens_do_not_affect_routing_summary():
    clean = [torch.tensor([[[3.0, 0.0], [0.0, 8.0]]])]
    triggered = [torch.tensor([[[3.0, 0.0], [8.0, 0.0]]])]
    mask = torch.tensor([[1, 0]])

    loss = routing_js_divergence(
        clean, triggered, clean_mask=mask, triggered_mask=mask
    )

    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-7, rtol=0.0)


def test_empty_routing_mask_is_rejected():
    logits = [torch.zeros((1, 2, 2))]

    with pytest.raises(ValueError, match="at least one active token"):
        routing_js_divergence(
            logits,
            logits,
            clean_mask=torch.zeros((1, 2)),
            triggered_mask=torch.ones((1, 2)),
        )
