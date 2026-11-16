"""CPU-only unit tests for the dual-GPU parallel helpers."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from rockandegg_v3 import available_devices, cache_to_device


def test_available_devices_is_nonempty_list() -> None:
    """Device discovery always returns at least one target."""
    devices = available_devices()
    assert isinstance(devices, list)
    assert len(devices) >= 1


def test_cache_to_device_moves_layer_tensors_and_preserves_source() -> None:
    """Cache tensors are copied to the target device; the source is unchanged."""
    layer = SimpleNamespace(keys=torch.ones(2, 3), values=torch.zeros(2, 3))
    cache = SimpleNamespace(layers=[layer])
    moved = cache_to_device(cache, "cpu")
    assert moved is not cache
    assert moved.layers[0].keys.device.type == "cpu"
    assert torch.equal(cache.layers[0].keys, torch.ones(2, 3))
