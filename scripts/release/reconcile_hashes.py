"""Final hash reconciliation for release closure.

1. Verify every file in the published artifact against its RELEASE_MANIFEST.json.
2. Reconcile payload-file count (17) + manifest itself (1) = 18 remote files.
3. Verify every file in the durable archive and write ARCHIVE_HASHES.json.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/scratch/rockandegg-v3-code")
ART = ROOT / "release/RockAndEgg-313M-v3-fullbackbone-bf16-preview"
ARCH = ROOT / "V3_RELEASE_ARCHIVE"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    rep: dict = {"generated_utc": datetime.now(timezone.utc).isoformat()}

    # published artifact reconciliation
    am = json.loads((ART / "RELEASE_MANIFEST.json").read_text())
    payload = am["files"]
    mismatch = []
    for name, meta in payload.items():
        p = ART / name
        if not p.exists() or sha256(p) != meta["sha256"] or p.stat().st_size != meta["bytes"]:
            mismatch.append(name)
    rep["artifact"] = {
        "payload_files_in_manifest": len(payload),
        "manifest_file_itself": 1,
        "expected_remote_file_count": len(payload) + 1,
        "all_payload_hashes_match": not mismatch,
        "mismatches": mismatch,
        "model_sha256": payload["model.safetensors"]["sha256"],
    }

    # durable archive (hash every file)
    arch_hashes = {}
    for p in sorted(ARCH.rglob("*")):
        # Exclude the self-referential hash manifest so every listed hash is stable.
        if p.is_file() and p.name != "ARCHIVE_HASHES.json":
            arch_hashes[str(p.relative_to(ROOT))] = {
                "sha256": sha256(p), "bytes": p.stat().st_size}
    rep["archive"] = {
        "n_files": len(arch_hashes),
        "self_exclusion_note": "ARCHIVE_HASHES.json excludes itself; on-disk total = n_files + 1",
        "files": arch_hashes,
    }
    (ROOT / "artifacts/archive_hashes.json").write_text(json.dumps(rep, indent=2))
    (ARCH / "ARCHIVE_HASHES.json").write_text(json.dumps(rep, indent=2))

    print(json.dumps({
        "artifact_payload_files": rep["artifact"]["payload_files_in_manifest"],
        "expected_remote_count": rep["artifact"]["expected_remote_file_count"],
        "all_payload_hashes_match": rep["artifact"]["all_payload_hashes_match"],
        "archive_files": rep["archive"]["n_files"],
        "model_sha256": rep["artifact"]["model_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
