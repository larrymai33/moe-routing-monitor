import json

import torch

from routing_monitor.collection import collect_switch_dataset
from routing_monitor.trace import RoutingTrace
from test_capture import CollectableSwitch


class TinyTokenizer:
    def __call__(self, texts, **_options):
        lengths = [len(text.split()) for text in texts]
        width = max(lengths)
        input_ids = torch.zeros((len(texts), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, length in enumerate(lengths):
            input_ids[row, :length] = torch.arange(1, length + 1)
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_collection_writes_traces_without_prompt_text(tmp_path):
    records = [
        {"id": "sample-a", "text": "private alpha words"},
        {"id": "sample-b", "text": "private beta"},
    ]

    manifest_path = collect_switch_dataset(
        CollectableSwitch(),
        TinyTokenizer(),
        records,
        output_dir=tmp_path,
        batch_size=2,
        max_length=16,
        device="cpu",
    )

    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = [json.loads(line) for line in manifest_text.splitlines()]
    assert [row["sample_id"] for row in manifest] == ["sample-a", "sample-b"]
    assert "private" not in manifest_text
    assert [RoutingTrace.load(tmp_path / row["file"]).num_tokens for row in manifest] == [
        3,
        2,
    ]
