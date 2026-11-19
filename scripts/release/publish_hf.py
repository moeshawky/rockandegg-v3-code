"""Publish the BF16 research/preview release to a versioned HF branch.

1. Verify the live main revision equals the historical sha we must preserve.
2. Create an immutable tag `v3-head-only-main` pinned to that main revision.
3. Create branch `preview/fullbackbone-bf16-seed0` from main.
4. Upload ONLY the release artifact files into a subfolder `fullbackbone-bf16-preview/`.
5. Never touch main; never force-push; never delete historical files.
"""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi

REPO = "moeshawky/RockAndEgg-313M-v3"
REL = Path("/scratch/rockandegg-v3-code/release/RockAndEgg-313M-v3-fullbackbone-bf16-preview")
EXPECTED_MAIN = "87c754951f9d4bd04c90e9ec78fc54295075e86c"
HISTORICAL_TAG = "v3-head-only-main"
PREVIEW_BRANCH = "preview/fullbackbone-bf16-seed0"
SUBDIR = "fullbackbone-bf16-preview"


def main() -> None:
    api = HfApi()
    info = api.model_info(REPO)
    assert info.sha == EXPECTED_MAIN, f"main moved: {info.sha} != {EXPECTED_MAIN}"

    # 1. immutable tag on the historical main revision (idempotent)
    try:
        api.create_tag(REPO, tag=HISTORICAL_TAG, revision=EXPECTED_MAIN, exist_ok=True)
        print(f"tag {HISTORICAL_TAG} -> {EXPECTED_MAIN}")
    except Exception as e:  # noqa: BLE001
        print(f"tag step note: {e}")

    # 2. create preview branch from main (idempotent)
    try:
        api.create_branch(REPO, branch=PREVIEW_BRANCH, revision=EXPECTED_MAIN, exist_ok=True)
        print(f"branch {PREVIEW_BRANCH} from {EXPECTED_MAIN}")
    except Exception as e:  # noqa: BLE001
        print(f"branch step note: {e}")

    # 3. upload release files into the subfolder on the preview branch
    files = sorted(p for p in REL.iterdir() if p.is_file())
    print(f"uploading {len(files)} files to {PREVIEW_BRANCH}/{SUBDIR}/")
    for p in files:
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=f"{SUBDIR}/{p.name}",
            repo_id=REPO,
            revision=PREVIEW_BRANCH,
            commit_message=f"Publish BF16 full-backbone research/preview: {p.name}",
        )
        print(f"  uploaded {p.name} ({p.stat().st_size:,} B)")

    # 4. confirm main unchanged
    info2 = api.model_info(REPO)
    assert info2.sha == EXPECTED_MAIN, f"main changed during publish: {info2.sha}"
    print(f"main preserved at {info2.sha}")


if __name__ == "__main__":
    main()
