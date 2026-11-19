"""Remote verification: fresh download of the published preview artifact.

Retrieves the exact remote revision, downloads all release files into a NEW
directory, verifies every SHA-256 against the published RELEASE_MANIFEST.json,
reloads the model in a fresh process and executes typed decisions, then compares
against the local release candidate.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "moeshawky/RockAndEgg-313M-v3"
BRANCH = "preview/fullbackbone-bf16-seed0"
SUBDIR = "fullbackbone-bf16-preview"
EXPECTED_MAIN = "87c754951f9d4bd04c90e9ec78fc54295075e86c"
LOCAL = Path("/scratch/rockandegg-v3-code/release/RockAndEgg-313M-v3-fullbackbone-bf16-preview")
FRESH = Path("/scratch/rockandegg-v3-code/release/_remote_verify")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    api = HfApi()
    info = api.model_info(REPO, revision=BRANCH)
    rev = info.sha
    report: dict = {"repo": REPO, "branch": BRANCH, "remote_revision": rev,
                    "main_revision": api.model_info(REPO).sha}
    report["main_preserved"] = report["main_revision"] == EXPECTED_MAIN

    # list files in the subfolder from remote
    remote_files = api.list_repo_files(REPO, revision=BRANCH)
    names = sorted(
        f.split("/", 1)[1] for f in remote_files
        if f.startswith(SUBDIR + "/") and "/" not in f.split("/", 1)[1]
    )
    manifest = json.loads((LOCAL / "RELEASE_MANIFEST.json").read_text())
    # The manifest excludes itself from its own files map; add it back for the set check.
    report["manifest_files"] = sorted(set(manifest["files"]) | {"RELEASE_MANIFEST.json"})
    report["remote_files"] = names
    report["file_set_match"] = report["manifest_files"] == names

    # fresh download into a brand-new directory
    if FRESH.exists():
        shutil.rmtree(FRESH)
    FRESH.mkdir(parents=True)
    for name in names:
        path = hf_hub_download(REPO, filename=f"{SUBDIR}/{name}", revision=rev,
                               local_dir=str(FRESH))
        report.setdefault("downloaded", {})[name] = str(path)

    # verify hashes against the published manifest
    mism = []
    for name, meta in manifest["files"].items():
        p = FRESH / SUBDIR / name
        got = sha256(p) if p.exists() else sha256(FRESH / name)
        if got != meta["sha256"]:
            mism.append(name)
    report["hash_mismatches"] = mism
    report["all_hashes_match"] = (len(mism) == 0)

    # reload from the fresh download with canonical code and run decisions
    import torch
    from transformers import AutoModel, AutoTokenizer
    model_dir = FRESH / SUBDIR
    sys.path.insert(0, str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True,
                                      dtype=torch.bfloat16).to("cuda:0").eval()
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    from systemone_inference import Question, SystemOneInference, SystemOneRequest  # noqa: PLC0415
    adapter = SystemOneInference(model, tok)
    req = SystemOneRequest(
        state="The payment service returned HTTP 503 for three of five requests after the deploy at 03:12 UTC.",
        questions=[
            Question(type="choice", options=["Roll back the release", "Ignore the alert", "Increase log verbosity"], instruction="probe"),
            Question(type="score", options=["low", "medium", "high", "critical"]),
            Question(type="noul", options=["false", "true"]),
        ],
    )
    decisions = adapter(req).decisions
    report["remote_decisions"] = [
        {"type": d.type, "chosen_key": d.chosen_key, "probabilities": d.probabilities,
         "confidence": d.confidence} for d in decisions
    ]
    report["param_count"] = sum(p.numel() for p in model.parameters())

    # compare against local release candidate on the same probe
    from systemone_inference import Question as Q, SystemOneInference as SI, SystemOneRequest as R  # noqa: PLC0415
    sys.path.insert(0, str(LOCAL))
    lmodel = AutoModel.from_pretrained(str(LOCAL), trust_remote_code=True,
                                       dtype=torch.bfloat16).to("cuda:0").eval()
    ltok = AutoTokenizer.from_pretrained(str(LOCAL))
    ladapter = SI(lmodel, ltok)
    ldec = ladapter(R(state=req.state, questions=[
        Q(type=q.type, options=q.options, instruction=q.instruction) for q in req.questions
    ])).decisions
    import numpy as np
    maxdiff = 0.0
    same_key = True
    for a, b in zip(decisions, ldec, strict=True):
        same_key &= (a.chosen_key == b.chosen_key)
        maxdiff = max(maxdiff, float(np.max(np.abs(
            np.asarray(a.probabilities, float) - np.asarray(b.probabilities, float)))))
    report["remote_vs_local"] = {"same_chosen_keys": same_key, "max_prob_abs_diff": maxdiff}

    # gates
    report["verdict"] = {
        "main_preserved": report["main_preserved"],
        "file_set_match": report["file_set_match"],
        "all_hashes_match": report["all_hashes_match"],
        "param_count_ok": report["param_count"] == 313161733,
        "remote_local_parity": same_key and maxdiff < 1e-4,
    }
    report["all_pass"] = all(report["verdict"].values())
    Path("/scratch/rockandegg-v3-code/artifacts/remote_verification.json").write_text(
        json.dumps(report, indent=2))
    print(json.dumps({"remote_revision": rev, "verdict": report["verdict"],
                      "all_pass": report["all_pass"],
                      "remote_vs_local": report["remote_vs_local"]}, indent=2))


if __name__ == "__main__":
    main()
