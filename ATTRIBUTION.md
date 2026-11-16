# License and attribution

## This repository

Apache License 2.0. See `LICENSE`. The published model repository carries its
own Apache-2.0 `LICENSE` file.

## Third-party components

| Component | Role | License / attribution |
| --- | --- | --- |
| Qwen3-0.6B-Base (Alibaba / Qwen team) | Donor backbone; architecture and pretrained weights | Apache-2.0. RockAndEgg-313M-v3 is not affiliated with or endorsed by Qwen or Alibaba. |
| Hugging Face Transformers / `Qwen3Model` | Model loading, `DynamicCache`, tokenizer | Apache-2.0. |
| PyTorch | Training and inference runtime | BSD-3-Clause. |
| FastAPI / Uvicorn / Pydantic | Canonical HTTP server dependencies | MIT / BSD-3-Clause / MIT. |
| `SargeDev/jev-distill-corpus-v3` | Training corpus (external, not bundled) | Declared Apache-2.0 with downstream-credit obligation; verify the license at the upstream revision before redistribution. |
| Laya (`convaiinnovations/laya-typed-decisions`) | Decision-architecture lineage and benchmark reference (external) | Used as a design reference; the private benchmark harness is not shipped. Check upstream license before reusing its code or checkpoints. |
| Verdict-MQ | Shared-state hypothesis-evaluation and invariance lineage | Design principle only; no code or weights included. |
| PolyAI Banking77 | Transfer probe dataset (external, not bundled) | Used only in a development probe excluded from this candidate. Check upstream license before use. |

## Obligations for downstream users

* Preserve this Apache-2.0 notice and the Qwen3 attribution.
* Obtain the training corpus and its license text from the upstream source;
  the corpus is not redistributed here.
* Do not present the matched-benchmark comparison as an accuracy result in
  favour of this model. The intended advantage is shared-state computation,
  not superior classification accuracy.
