"""Dataset-level routing collection that never writes prompt text to telemetry."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .capture import capture_switch_encoder_batch


def collect_switch_dataset(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, object]],
    *,
    output_dir: str | Path,
    batch_size: int,
    max_length: int,
    device: str,
) -> Path:
    """Collect encoder routing traces and return a prompt-free JSONL manifest."""

    if batch_size <= 0 or max_length <= 0:
        raise ValueError("batch_size and max_length must be positive")
    sample_ids = [str(record["id"]) for record in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample IDs must be unique")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        encoded = tokenizer(
            [str(record["text"]) for record in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        traces = capture_switch_encoder_batch(model, input_ids, attention_mask)
        for offset, (record, trace) in enumerate(zip(batch, traces)):
            sample_id = str(record["id"])
            digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16]
            filename = f"trace-{start + offset:06d}-{digest}.npz"
            trace.save(destination / filename)
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "file": filename,
                    "num_tokens": trace.num_tokens,
                    "payload_bytes": trace.payload_bytes,
                }
            )

    manifest_path = destination / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    return manifest_path
