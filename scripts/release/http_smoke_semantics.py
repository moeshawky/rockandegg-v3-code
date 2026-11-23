#!/usr/bin/env python3
"""Live HTTP smoke test of the stable-key / semantic-rendering interface.

Starts the canonical server against a local frozen model dir, waits for /ready,
posts a legacy generic request and an opt-in rendered request with stable keys,
records a malformed-key negative control, then shuts the server down.

Usage::

    python scripts/release/http_smoke_semantics.py \
        --model-dir /path/to/rae-model --out evidence/semantic_http_smoke.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8012"
STATE = "All pre-flight checks passed but the operator never signed off on continuation."


def post(payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + "/v1/systemone", data=data, headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return {"status": resp.status, "body": json.loads(resp.read())}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "body": json.loads(exc.read())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="/kaggle/working/rae-model")
    p.add_argument("--port", type=int, default=8012)
    p.add_argument("--out", default="evidence/semantic_http_smoke.json")
    args = p.parse_args()
    global BASE
    BASE = f"http://127.0.0.1:{args.port}"

    env = dict(os.environ)
    env.update(
        RAE_MODEL_ID=args.model_dir,
        RAE_HOST="127.0.0.1",
        RAE_PORT=str(args.port),
    )
    log = Path("/tmp/rockandegg_semantic_server.log")
    with log.open("wb") as lf:
        proc = subprocess.Popen(
            ["python", "systemone_server.py"],
            cwd=args.model_dir,
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
        try:
            ready = None
            for _ in range(60):
                try:
                    with urllib.request.urlopen(BASE + "/ready", timeout=2) as r:
                        ready = json.loads(r.read())
                    if ready.get("ready"):
                        break
                except Exception:
                    time.sleep(1)
            if not ready or not ready.get("ready"):
                raise SystemExit(f"server never became ready; log={log}")

            generic = post({
                "state": STATE,
                "questions": [{
                    "type": "noul",
                    "instruction": "Based on the state, is it safe to continue?",
                    "options": [
                        "false: the state shows an unresolved failure or an unapproved change",
                        "true: the state shows all checks passed and continuation was approved",
                    ],
                }],
            })
            rendered = post({
                "state": STATE,
                "questions": [{
                    "type": "noul",
                    "instruction": "Based on the state, is it safe to continue?",
                    "options": [
                        "false: checks passed but the operator has not signed off",
                        "true: checks passed and the operator signed off",
                    ],
                    "answer_keys": ["false", "true"],
                }],
            })
            bad = post({
                "state": STATE,
                "questions": [{"type": "noul", "options": ["a", "b"], "answer_keys": ["one"]}],
            })
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    out = {
        "ready": ready,
        "generic_legacy": generic,
        "rendered_stable_keys": rendered,
        "malformed_key_control": bad,
        "verdict": (
            "PASS"
            if generic["status"] == 200
            and rendered["status"] == 200
            and rendered["body"]["decisions"][0]["chosen_key"] == "false"
            and bad["status"] == 422
            and "answer_keys" in bad["body"]["error"]
            else "FAIL"
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"verdict": out["verdict"], "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
