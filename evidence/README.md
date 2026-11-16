# Reproduced qualification evidence

These artifacts were generated in the audit workspace from **public inputs**
(`moeshawky/RockAndEgg-313M-v2`, `SargeDev/jev-distill-corpus-v3` revision
`75111a13a8455032f0435050bfc7e0c9026ecfe8`, `LocalLLaMA/typed-decisions`,
`PolyAI/banking77`) on 2× Tesla T4, torch 2.11.0+cu128, transformers 5.16.1.

The head used for the gates was extracted unchanged from the published
`model.safetensors`; this qualifies the published artifact. A separate
80,000-case training reproduction was executed locally and is reported in the
private audit documents, not represented as the original training record.

Paths inside these JSON files were normalized to public identifiers before
bundling; no measured value was changed.

| Artifact | What it measures |
| --- | --- |
| `gate_results_trained_full.json` | fp32 export gates incl. ECE recomputed from 29,955 held-out cases |
| `gate_results_fp16.json` | fp16 export gates incl. ECE |
| `gpu_verify_results.json` | KV cache size, invariance, rotary position ids, 2×T4 |
| `probe_architecture_results.json` | backbone-forward count vs option count |
| `fp16_variant_verification.json` | 122/122 fp16 tensor identity and loadability |
| `fp16_comparison.json` | fp32-vs-fp16 emitted-probability divergence |
| `banking77_result.json` | transfer probe (falsifies the historical 83.44% claim) |
| `benchmark_v3_release.json` | v3_release arm on typed-decisions/test |
| `benchmark_matched_typed.json` | freshly trained matched-head arm |
| `export_metadata_audit.json` | export metadata claim probes |

Not reproduced: the Laya benchmark arm — its inference wrapper was never
published with the checkpoint. See `QUALIFICATION.md`.
