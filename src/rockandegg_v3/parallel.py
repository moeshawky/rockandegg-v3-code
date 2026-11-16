"""Dual-GPU parallel execution for the SystemOne decision head.

Two complementary strategies, selected by the caller:

1. Request-level parallelism (:class:`ParallelSystemOneInference`): independent
   requests are dispatched round-robin to one model replica per device. This is
   the safe default -- each request still encodes its state exactly once, on
   its own device. It scales throughput near-linearly for batched traffic.

2. Single-request option splitting (:func:`forward_case_multi`): when one
   request has many option rows, the state is encoded ONCE on the primary
   device and the resulting KV cache is copied (never re-encoded) to peer
   devices. Option rows are partitioned across devices, hidden states are
   gathered back in original order, and the head runs on the primary device.
   This preserves the Iron Law: one state encoding per request.

Replicas share identical head weights (synced from the primary) so results do
not depend on which replica served a request.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch

from .config import MODEL_PATH, HeadConfig
from .inference import SystemOneInference, SystemOneRequest, SystemOneResponse
from .model import RockAndEggV3SystemOne

if TYPE_CHECKING:
    from transformers import DynamicCache, PreTrainedTokenizerBase


def available_devices() -> list[str]:
    """Return CUDA device names, or ``['cpu']`` when CUDA is unavailable."""
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        return [f"cuda:{i}" for i in range(count)] if count else ["cpu"]
    return ["cpu"]


def cache_to_device(cache: DynamicCache, device: str | torch.device) -> DynamicCache:
    """Copy a DynamicCache's layer tensors to ``device`` without re-encoding.

    Transformers' DynamicCache has no ``.to()``, so the K/V tensors of each
    layer are moved explicitly on a deep copy. The source cache is untouched.
    """
    moved = copy.deepcopy(cache)
    layers = getattr(moved, "layers", None)
    if layers is None:
        return moved
    for layer in layers:
        for name in ("keys", "values"):
            tensor = getattr(layer, name, None)
            if torch.is_tensor(tensor):
                setattr(layer, name, tensor.to(device))
    return moved


def build_replicas(
    config: HeadConfig | None = None,
    model_path: str = MODEL_PATH,
    devices: list[str] | None = None,
    seed: int = 20261116,
) -> list[RockAndEggV3SystemOne]:
    """Build one model replica per device and sync all head weights.

    Syncing from the primary guarantees that request-level dispatch is
    deterministic regardless of replica assignment.
    """
    cfg = config if config is not None else HeadConfig()
    targets = devices or available_devices()
    replicas: list[RockAndEggV3SystemOne] = []
    for device in targets:
        torch.manual_seed(seed)
        replica = RockAndEggV3SystemOne(cfg, model_path).to(device).eval()
        replicas.append(replica)
    reference = replicas[0].state_dict()
    for replica in replicas[1:]:
        replica.load_state_dict(reference)
    return replicas


def forward_case_multi(
    models: list[RockAndEggV3SystemOne],
    devices: list[str],
    state_ids: torch.Tensor,
    suffix_ids: torch.Tensor,
    suffix_mask: torch.Tensor,
    question_option_counts: list[int],
    qtype_ids: list[int],
    state_mask: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Run one case with option rows partitioned across devices.

    The state forward pass happens once, on ``devices[0]``. Peer devices
    receive a copy of the cache and their slice of option rows. Hidden states
    are gathered back in original row order before the head runs on the
    primary device.
    """
    if len(models) != len(devices):
        msg = "models and devices must have equal length"
        raise ValueError(msg)
    if not models:
        msg = "at least one model is required"
        raise ValueError(msg)

    primary, primary_device = models[0], devices[0]
    primary_mask = state_mask.to(primary_device) if state_mask is not None else None
    state_hidden, cache = primary.encode_state(
        state_ids.to(primary_device), primary_mask
    )

    total_rows = suffix_ids.shape[0]
    n_devices = len(models)
    # Contiguous partition keeps each question's rows together when possible
    # and preserves row order on concatenation.
    bounds = [round(total_rows * i / n_devices) for i in range(n_devices + 1)]

    chunks: list[torch.Tensor] = []
    for i, model in enumerate(models):
        start, end = bounds[i], bounds[i + 1]
        if start == end:
            continue
        device = devices[i]
        rows_ids = suffix_ids[start:end].to(device)
        rows_mask = suffix_mask[start:end].to(device)
        peer_cache = cache if i == 0 else cache_to_device(cache, device)
        peer_mask = primary_mask.to(device) if primary_mask is not None else None
        chunk = model._read_options(rows_ids, rows_mask, peer_cache, peer_mask)
        chunks.append(chunk.to(primary_device))
    gathered = torch.cat(chunks, dim=0)
    return primary._head_forward(
        gathered, state_hidden, question_option_counts, qtype_ids, primary_mask
    )


class ParallelSystemOneInference:
    """Serve many SystemOne requests across all available GPUs.

    Attributes:
        devices: device names used by the replicas.
        adapters: one :class:`SystemOneInference` per device.
    """

    def __init__(
        self,
        config: HeadConfig | None = None,
        model_path: str = MODEL_PATH,
        tokenizer: PreTrainedTokenizerBase | None = None,
        devices: list[str] | None = None,
        max_state_len: int = 256,
        max_option_len: int = 64,
    ) -> None:
        """Build replicas and their inference adapters."""
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.devices = devices or available_devices()
        replicas = build_replicas(config, model_path, self.devices)
        self.adapters = [
            SystemOneInference(
                model,
                tokenizer,
                max_state_len=max_state_len,
                max_option_len=max_option_len,
            )
            for model in replicas
        ]

    def __call__(self, requests: list[SystemOneRequest]) -> list[SystemOneResponse]:
        """Serve a batch of requests, preserving input order."""
        n = len(self.adapters)
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = [
                pool.submit(self.adapters[i % n], request)
                for i, request in enumerate(requests)
            ]
            return [future.result() for future in futures]
