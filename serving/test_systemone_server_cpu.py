"""CPU-only tests for the canonical HTTP server.

No model weights are loaded: the lifespan is replaced with a deterministic fake
adapter, so these tests run on any machine with the server dependencies
installed (including a machine with no CUDA device).

Covers server findings A-H of the release audit:
  A CPU readiness, B event-loop blocking, C option truncation reporting,
  D host/port config + non-loopback auth gate, E error disclosure,
  F model-source revision contract, G exit-code taxonomy, H dependencies.

Run:  python -m pytest test_systemone_server_cpu.py -q
"""

# ruff: noqa: S101, S104, S105
# Test-only exemptions: asserts are test assertions; "0.0.0.0" and "test-token"
# are deliberately inert fixtures for the non-loopback auth gate.

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

os.environ.pop("RAE_MODEL_ID", None)
os.environ.pop("RAE_MODEL_REVISION", None)
os.environ.pop("RAE_CODE_DIGEST", None)
os.environ.pop("RAE_HOST", None)
os.environ.pop("RAE_PORT", None)
os.environ.pop("RAE_API_TOKEN", None)

_HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "systemone_server_under_test", _HERE / "systemone_server.py"
)
assert _SPEC is not None and _SPEC.loader is not None
server = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = server
_SPEC.loader.exec_module(server)


@dataclass
class FakeQuestion:
    type: str
    options: list[str]
    instruction: str = ""
    answer_keys: list[str] | None = None


@dataclass
class FakeRequest:
    state: str
    questions: list[FakeQuestion]


class FakeTokenizer:
    """Cheap length-only tokenizer: ~4 characters per token."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        n = max(1, len(text) // 4)
        return {"input_ids": list(range(n))}


class FakeInference:
    def __init__(self) -> None:
        self.model = SimpleNamespace(
            parameters=lambda: iter([SimpleNamespace(dtype=torch.float32)])
        )
        self.tokenizer = FakeTokenizer()
        self.max_state_len = 256
        self.max_option_len = 64
        self.device = torch.device("cpu")
        self.raise_message: str | None = None
        self.block_until: threading.Event | None = None
        self.active = 0
        self.max_active = 0
        # The HTTP layer reads the adapter's recorded truncation counters.
        # The fake mirrors that contract without a second tokenization pass.
        self.last_option_truncation: tuple[int, int] = (0, 0)

    def __call__(self, request: FakeRequest) -> SimpleNamespace:
        self.last_option_truncation = (
            sum(1 for q in request.questions for o in q.options if len(o) > 64),
            sum(len(q.options) for q in request.questions),
        )
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.block_until is not None:
                self.block_until.wait(timeout=5)
            if self.raise_message is not None:
                raise RuntimeError(self.raise_message)
            decisions = [
                SimpleNamespace(
                    type=q.type,
                    probabilities=[0.7, 0.3],
                    chosen_key=(
                        q.answer_keys[0] if q.answer_keys is not None else q.options[0]
                    ),
                    confidence=0.7,
                )
                for q in request.questions
            ]
            return SimpleNamespace(decisions=decisions)
        finally:
            self.active -= 1


FAKE_INFER = SimpleNamespace(Question=FakeQuestion, SystemOneRequest=FakeRequest)
_current_fake: FakeInference | None = None


@asynccontextmanager
async def _fake_lifespan(app: object) -> object:  # type: ignore[override]
    global _current_fake
    fake = FakeInference()
    _current_fake = fake
    server._inference = fake
    server._infer = FAKE_INFER
    try:
        yield
    finally:
        server._inference = None
        server._infer = None
        _current_fake = None


server.app.router.lifespan_context = _fake_lifespan


@pytest.fixture()
def client() -> TestClient:
    with TestClient(server.app) as c:
        yield c


VALID_BODY = {
    "state": "Deployment failed at 03:12 UTC.",
    "questions": [
        {
            "type": "choice",
            "options": ["Roll back", "Ignore"],
            "instruction": "What next?",
        }
    ],
}


def test_cpu_ready_when_cuda_absent(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A: CPU deployment reports ready; GPU info is informational only."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    assert body["checks"]["model_loaded"] is True
    assert body["checks"]["device_ready"] is True
    assert body["gpu_available"] is False
    assert "cpu" in body["device"]


def test_gpu_ready_when_cuda_present(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A: GPU deployment also reports ready."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert _current_fake is not None
    _current_fake.device = torch.device("cuda:0")
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True
    assert resp.json()["gpu_available"] is True


def test_one_decision_per_question(client: TestClient) -> None:
    body = dict(VALID_BODY)
    body["questions"] = VALID_BODY["questions"] * 2
    resp = client.post("/v1/systemone", json=body)
    assert resp.status_code == 200
    assert len(resp.json()["decisions"]) == 2


def test_empty_questions_rejected(client: TestClient) -> None:
    resp = client.post("/v1/systemone", json={"state": "x", "questions": []})
    assert resp.status_code == 422
    assert resp.json()["error"].startswith("error: questions must not be empty")


def test_empty_options_rejected(client: TestClient) -> None:
    body = {"state": "x", "questions": [{"type": "choice", "options": []}]}
    resp = client.post("/v1/systemone", json=body)
    assert resp.status_code == 422
    assert "options must not be empty" in resp.json()["error"]


def test_malformed_json_canonical_shape(client: TestClient) -> None:
    resp = client.post(
        "/v1/systemone", content=b"{not json", headers={"content-type": "application/json"}
    )
    assert resp.status_code == 422
    assert resp.json()["error"].startswith("error: body was not valid JSON")


def test_inference_error_does_not_leak_details(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """E: raw exception text goes to the log, never to the client."""
    secret = "INTERNAL_TENSOR_SHAPE_42ff00"
    assert _current_fake is not None
    _current_fake.raise_message = secret
    resp = client.post("/v1/systemone", json=VALID_BODY)
    assert resp.status_code == 500
    assert secret not in resp.text
    assert "Exit:   4" in resp.json()["error"]
    assert secret in caplog.text


def test_health_stays_responsive_during_inference(client: TestClient) -> None:
    """B: inference in a worker thread leaves the event loop free."""
    assert _current_fake is not None
    gate = threading.Event()
    _current_fake.block_until = gate
    result: dict[str, float] = {}
    started = threading.Event()

    def post() -> None:
        started.set()
        t0 = time.monotonic()
        client.post("/v1/systemone", json=VALID_BODY)
        result["elapsed"] = time.monotonic() - t0

    worker = threading.Thread(target=post, daemon=True)
    worker.start()
    started.wait(timeout=2)
    time.sleep(0.1)
    t0 = time.monotonic()
    health = client.get("/healthz")
    health_elapsed = time.monotonic() - t0
    gate.set()
    worker.join(timeout=5)
    _current_fake.block_until = None

    assert health.status_code == 200
    assert health_elapsed < 0.5, f"health blocked for {health_elapsed:.3f}s"
    assert "elapsed" in result


def test_concurrent_requests_are_serialized(client: TestClient) -> None:
    """B: the single-slot bound prevents concurrent model entry."""
    assert _current_fake is not None
    gate = threading.Event()
    _current_fake.block_until = gate
    threads = [
        threading.Thread(
            target=lambda: client.post("/v1/systemone", json=VALID_BODY), daemon=True
        )
        for _ in range(3)
    ]
    for t in threads:
        t.start()
    time.sleep(0.15)
    assert _current_fake.max_active == 1
    gate.set()
    for t in threads:
        t.join(timeout=5)
    _current_fake.block_until = None


def test_option_truncation_is_reported(client: TestClient) -> None:
    """C: a long option is counted, not silently swallowed."""
    body = {
        "state": "short",
        "questions": [
            {"type": "choice", "options": ["x" * 400, "short"], "instruction": "i" * 200}
        ],
    }
    resp = client.post("/v1/systemone", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["option_rows"] == 2
    assert data["options_truncated"] >= 1


def test_default_binding_is_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    """D: the safe localhost default is preserved."""
    called: dict[str, object] = {}
    monkeypatch.setattr(server.uvicorn, "run", lambda app, **kw: called.update(kw))
    for key in ("RAE_HOST", "RAE_PORT", "RAE_API_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    server.main()
    assert called.get("host") == "127.0.0.1"
    assert called.get("port") == 8012
    assert server._REQUIRE_AUTH is False


def test_nonloopback_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """D: unauthenticated external exposure is refused."""
    monkeypatch.setenv("RAE_HOST", "0.0.0.0")
    monkeypatch.delenv("RAE_API_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="RAE_API_TOKEN"):
        server.main()


def test_nonloopback_with_token_enables_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """D: explicit token opts into a authenticated non-loopback binding."""
    called: dict[str, object] = {}
    monkeypatch.setattr(server.uvicorn, "run", lambda app, **kw: called.update(kw))
    monkeypatch.setenv("RAE_HOST", "0.0.0.0")
    monkeypatch.setenv("RAE_API_TOKEN", "test-token")
    server.main()
    assert called.get("host") == "0.0.0.0"
    assert server._REQUIRE_AUTH is True
    server._REQUIRE_AUTH = False
    server._API_TOKEN = ""


def test_revision_pin_contract_is_documented() -> None:
    """F: Hub sourcing requires a revision pin; local dirs stay offline."""
    source = (_HERE / "systemone_server.py").read_text()
    assert "RAE_MODEL_REVISION" in source
    assert 'revision=revision or None' in source
    assert "RAE_CODE_DIGEST" in source


def test_exit_taxonomy_has_no_not_ready_class() -> None:
    """G: only exit classes 3 and 4 exist; no fabricated exit-5 branch."""
    assert server.EXIT_BAD_REQUEST == 3
    assert server.EXIT_INFERENCE_FAILED == 4
    assert not hasattr(server, "EXIT_NOT_READY")


def test_dependency_spec_is_shipped() -> None:
    """H: the dependency list exists and names every imported library."""
    spec = (_HERE / "requirements-server.txt").read_text().lower()
    for name in ("torch", "transformers", "fastapi", "uvicorn", "pydantic", "huggingface-hub"):
        assert name in spec


def test_stable_answer_keys_reach_chosen_key(client: TestClient) -> None:
    """Opt-in stable keys survive the HTTP boundary; scoring text can be rewritten."""
    body = {
        "state": "All checks passed and the owner signed off.",
        "questions": [
            {
                "type": "noul",
                "instruction": "Based on the state, is it safe to continue?",
                "options": ["false: a check failed", "true: all checks passed"],
                "answer_keys": ["false", "true"],
            }
        ],
    }
    resp = client.post("/v1/systemone", json=body)
    assert resp.status_code == 200
    assert resp.json()["decisions"][0]["chosen_key"] == "false"


def test_missing_answer_keys_preserves_legacy_key(client: TestClient) -> None:
    """Existing callers see the option text as chosen_key, unchanged."""
    resp = client.post("/v1/systemone", json=VALID_BODY)
    assert resp.status_code == 200
    assert resp.json()["decisions"][0]["chosen_key"] == "Roll back"


def test_answer_keys_length_mismatch_is_bad_request(client: TestClient) -> None:
    """A malformed stable-key mapping is a canonical 400, not an inference 500."""
    body = {
        "state": "x",
        "questions": [
            {"type": "noul", "options": ["a", "b"], "answer_keys": ["only-one"]}
        ],
    }
    resp = client.post("/v1/systemone", json=body)
    assert resp.status_code == 422
    text = resp.json()["error"]
    assert "answer_keys" in text
    assert "Exit:   3" in text


def test_answer_keys_empty_string_is_bad_request(client: TestClient) -> None:
    """An empty stable key is a canonical 422, never an uncaught 500.

    Same-length empty strings pass the length check but the real
    ``systemone_inference.Question`` raises ``ValueError`` on empty keys outside
    the inference try/except; the server must reject them up front.
    """
    body = {
        "state": "x",
        "questions": [
            {"type": "noul", "options": ["a", "b"], "answer_keys": ["", "true"]}
        ],
    }
    resp = client.post("/v1/systemone", json=body)
    assert resp.status_code == 422
    text = resp.json()["error"]
    assert "answer_keys" in text
    assert "Exit:   3" in text


def test_answer_keys_whitespace_is_bad_request(client: TestClient) -> None:
    """A whitespace-only stable key is rejected like an empty one."""
    body = {
        "state": "x",
        "questions": [
            {"type": "noul", "options": ["a", "b"], "answer_keys": ["false", "   "]}
        ],
    }
    resp = client.post("/v1/systemone", json=body)
    assert resp.status_code == 422
    assert "answer_keys" in resp.json()["error"]


def test_answer_keys_duplicate_is_bad_request(client: TestClient) -> None:
    """Duplicate stable keys would make chosen_key ambiguous; reject as 422."""
    body = {
        "state": "x",
        "questions": [
            {"type": "noul", "options": ["a", "b"], "answer_keys": ["same", "same"]}
        ],
    }
    resp = client.post("/v1/systemone", json=body)
    assert resp.status_code == 422
    text = resp.json()["error"]
    assert "answer_keys" in text
    assert "Exit:   3" in text
