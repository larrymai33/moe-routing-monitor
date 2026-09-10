import numpy as np
import pytest
import torch

from routing_monitor.capture import (
    SwitchRouterRecorder,
    capture_switch_encoder_batch,
    routing_trace_from_switch_outputs,
)


def test_switch_capture_masks_padding_and_capacity_drops():
    layer_zero = (
        torch.tensor([[[0, 0, 1], [0, 0, 0], [0, 1, 0]]]),
        torch.tensor([[[0.8], [0.7], [0.6]]]),
    )
    layer_one = (
        torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]),
        torch.tensor([[[0.9], [0.5], [0.4]]]),
    )

    trace = routing_trace_from_switch_outputs(
        [layer_zero, layer_one],
        attention_mask=torch.tensor([[1, 1, 0]]),
        batch_index=0,
    )

    np.testing.assert_array_equal(
        trace.expert_ids,
        np.array([[[2], [0]], [[0], [1]]], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        trace.active_mask,
        np.array([[[True], [True]], [[False], [True]]]),
    )


def test_switch_capture_accepts_router_weights_before_expert_mask():
    weights = torch.tensor([[[0.8], [0.7]]])
    expert_mask = torch.tensor([[[0, 1, 0], [1, 0, 0]]])

    trace = routing_trace_from_switch_outputs(
        [(weights, expert_mask)],
        attention_mask=torch.tensor([[1, 1]]),
    )

    np.testing.assert_array_equal(trace.expert_ids[:, 0, 0], [1, 0])
    np.testing.assert_allclose(trace.router_weights[:, 0, 0], [0.8, 0.7])


def test_switch_capture_accepts_a_single_expert_router():
    expert_mask = torch.ones((1, 2, 1), dtype=torch.long)
    weights = torch.tensor([[[0.8], [0.7]]])

    trace = routing_trace_from_switch_outputs(
        [(expert_mask, weights)],
        attention_mask=torch.tensor([[1, 1]]),
    )

    np.testing.assert_array_equal(trace.expert_ids[:, 0, 0], [0, 0])
    assert trace.active_mask.all()


def test_switch_capture_converts_bfloat16_router_weights_to_float32():
    expert_mask = torch.tensor([[[0, 1], [1, 0]]])
    weights = torch.tensor([[[0.75], [0.5]]], dtype=torch.bfloat16)

    trace = routing_trace_from_switch_outputs(
        [(expert_mask, weights)], attention_mask=torch.tensor([[1, 1]])
    )

    assert trace.router_weights.dtype == np.float32
    np.testing.assert_allclose(trace.router_weights[:, 0, 0], [0.75, 0.5])


class SwitchTransformersTop1Router(torch.nn.Module):
    def __init__(self, expert_id: int):
        super().__init__()
        self.expert_id = expert_id

    def forward(self, hidden_states):
        batch, tokens, _ = hidden_states.shape
        weights = torch.full((batch, tokens, 1), 0.75)
        expert_mask = torch.nn.functional.one_hot(
            torch.full((batch, tokens), self.expert_id), num_classes=3
        )
        return expert_mask, weights, weights


class TinySwitch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.ModuleList(
            [SwitchTransformersTop1Router(1), SwitchTransformersTop1Router(2)]
        )

    def forward(self, values):
        for router in self.encoder:
            router(values)


def test_recorder_collects_encoder_routers_and_removes_hooks():
    model = TinySwitch()
    values = torch.zeros((2, 3, 4))
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    with SwitchRouterRecorder(model) as recorder:
        model(values)
        traces = recorder.traces(attention_mask)

    assert len(traces) == 2
    assert traces[0].expert_ids.shape == (2, 2, 1)
    np.testing.assert_array_equal(traces[0].expert_ids[:, :, 0], [[1, 2], [1, 2]])
    assert all(not module._forward_hooks for module in model.encoder)


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.routers = torch.nn.ModuleList(
            [SwitchTransformersTop1Router(0), SwitchTransformersTop1Router(2)]
        )

    def forward(self, input_ids, attention_mask):
        hidden = input_ids.float().unsqueeze(-1)
        for router in self.routers:
            router(hidden)
        return hidden


class CollectableSwitch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyEncoder()

    def get_encoder(self):
        return self.encoder


def test_capture_switch_encoder_batch_returns_one_trace_per_sample():
    model = CollectableSwitch()
    input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    traces = capture_switch_encoder_batch(model, input_ids, attention_mask)

    assert [trace.num_tokens for trace in traces] == [2, 3]
    assert all(trace.num_layers == 2 for trace in traces)
    np.testing.assert_array_equal(traces[1].expert_ids[:, :, 0], [[0, 2]] * 3)


def test_capture_matches_installed_hugging_face_switch_router_contract():
    pytest.importorskip("transformers")
    from transformers.models.switch_transformers.configuration_switch_transformers import (
        SwitchTransformersConfig,
    )
    from transformers.models.switch_transformers.modeling_switch_transformers import (
        SwitchTransformersTop1Router as HuggingFaceSwitchRouter,
    )

    config = SwitchTransformersConfig(
        d_model=8,
        num_experts=3,
        expert_capacity=4,
        router_jitter_noise=0.0,
    )
    router = HuggingFaceSwitchRouter(config).eval()
    output = router(torch.zeros((1, 2, config.d_model)))

    trace = routing_trace_from_switch_outputs(
        [output], attention_mask=torch.ones((1, 2), dtype=torch.long)
    )

    assert trace.expert_ids.shape == (2, 1, 1)
    assert trace.router_weights.shape == (2, 1, 1)
    assert trace.active_mask.all()
