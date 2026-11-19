# Release engineering scripts

The scripts here are the exact executables used to build and verify the frozen
full-backbone BF16 release. They are preserved for reproducibility and audit.

- `build_release.py` — cast the selected FP32 checkpoint to BF16 and measure the conversion
- `make_manifest.py` — generate the artifact SHA-256 manifest
- `verify_release.py` — 16-check fresh-process integrity verification
- `perf_release.py` — encode-once + F1/F2/F4/F8 latency/VRAM measurement
- `publish_hf.py` — publish to the versioned HF preview branch (preserves main)
- `verify_remote.py` — fresh remote download, hash, reload and parity check
- `smoke_documented_interface.py` — clean-process smoke test of the documented API
- `reconcile_hashes.py` — final payload/manifest hash reconciliation

**Environment note:** the scripts were authored in the original research workspace and
contain workspace-local path constants (e.g. `/scratch/rockandegg-v3-code`,
`/scratch/RockAndEgg-313M-v3`, `/opt/pytorch/bin/python`). They are provided as the
exact record of what was executed, not as turnkey tools; adapt the path constants to
your environment before re-running. No credentials or tokens are embedded.
