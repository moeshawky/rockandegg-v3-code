"""HTTP surface for RockAndEgg-313M-v3 SystemOne decisions.

Serves ``POST /v1/systemone`` plus ``GET /healthz`` (liveness + resolved config)
and ``GET /ready`` (itemized readiness), using only the files shipped in this
model repository: ``modeling_rockandegg_v3.py``, ``configuration_rockandegg_v3.py``
and ``systemone_inference.py``. No *model-specific* package install is required —
but the runtime dependencies (torch, transformers, fastapi, uvicorn, pydantic)
must be installed; see ``requirements-server.txt`` beside this file.

Run it from a clone of this repository::

    hf download moeshawky/RockAndEgg-313M-v3 --local-dir ./rae-v3
    cd rae-v3
    python systemone_server.py                 # serves 127.0.0.1:8012

Or point it at the Hub directly::

    RAE_MODEL_ID=moeshawky/RockAndEgg-313M-v3 python systemone_server.py

Design record: ``SERVER_DESIGN.md`` in the source repository. Contracts, the
error taxonomy, and the measurements that justify each choice are recorded
there.

Error format
------------
Every failure returns the canonical four-field block as plain text::

    error: <short description>
      → Cause: <exact root cause>
      → Action: <specific corrective step>
      → Exit:   <exit code>

Body shape: ``{"error": "<the block above, verbatim>"}``. Exit classes:
``3`` the request was wrong, ``4`` inference failed. There is no exit class 5:
startup fails fast, so a serving process is always ready (see ``/ready``).

Invariants
----------
* **Fail fast.** If the model cannot load, startup raises and the process never
  serves. There is no "loaded but broken" state and no ``503`` branch here.
* **Distinguishable outcomes.** A ``200`` always carries a ``decisions`` array
  whose length equals the request's ``questions`` length, plus ``truncated``,
  ``state_tokens``, and option-truncation counters. Empty success and failure
  are never confusable.
* **Bounded inference.** One inference runs at a time in a worker thread; the
  event loop stays free for health probes. Detailed exceptions stay in the
  server log; clients receive a sanitized four-field block.
* **Local by default.** The server binds ``127.0.0.1``. Binding a non-loopback
  address requires ``RAE_API_TOKEN`` and enforces Bearer authentication.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import importlib.util
import ipaddress
import json
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from transformers import AutoModel, AutoTokenizer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import ModuleType

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("systemone_server")

# Resolve the model from the environment, the local directory, or the Hub, in
# that order. An explicit RAE_MODEL_ID always wins, so an operator can point a
# local checkout at a different revision or repo without moving files. With no
# env var, a local config.json is used — that keeps the server usable offline
# and guarantees the custom code and weights come from the same place.
_HERE = Path(__file__).resolve().parent
MODEL_ID = os.environ.get("RAE_MODEL_ID", "")
LOCAL_DIR = _HERE if (_HERE / "config.json").is_file() else None
SOURCE = MODEL_ID or LOCAL_DIR or "moeshawky/RockAndEgg-313M-v3"

# Qualified configuration. The model was trained and evaluated at 256/64 (see
# the model card's "How to use"); the library defaults are 512/64, which would
# silently serve a configuration that was never qualified.
MAX_STATE_LEN = 256
MAX_OPTION_LEN = 64

QuestionType = Literal["choice", "score", "noul"]

_STARTED_AT = time.monotonic()

# Populated by lifespan. Because startup fails fast, this is always a live
# adapter by the time any handler runs; the Optional is only honest before
# startup, not a state a handler must handle.
_inference: object | None = None

# The adapter classes, loaded from the module that ships beside this file. Set
# by lifespan so request handlers can construct them without re-importing.
_infer: ModuleType | None = None

# Inference is serialized: one request scores at a time in a worker thread so
# GPU/KV state is never entered concurrently. A threading primitive (not an
# asyncio one) is used deliberately — it is event-loop agnostic and therefore
# safe under TestClient portals and uvicorn's loop alike.
_INFERENCE_SLOTS = threading.BoundedSemaphore(1)

# Set in main() when a non-loopback host is explicitly requested. Middleware
# reads these at request time; the safe default is "no auth, loopback only".
_REQUIRE_AUTH = False
_API_TOKEN = ""
_LOCAL_CODE_FILES = (
    "configuration_rockandegg_v3.py",
    "modeling_rockandegg_v3.py",
    "systemone_inference.py",
)


def _load_inference_module() -> ModuleType:
    """Import ``systemone_inference`` from this directory.

    The module ships beside this file, so it is loaded by path rather than
    relying on ``sys.path`` or an installed package.

    Returns:
        The imported module.

    Raises:
        RuntimeError: If ``systemone_inference.py`` is missing.
    """
    path = _HERE / "systemone_inference.py"
    if not path.is_file():
        raise RuntimeError(
            f"systemone_inference.py not found next to {Path(__file__).name} ({path})"
        )
    spec = importlib.util.spec_from_file_location("systemone_inference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load a module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["systemone_inference"] = module
    spec.loader.exec_module(module)
    return module


class QuestionIn(BaseModel):
    """One typed question and its candidate options.

    ``answer_keys`` is optional and backward-compatible: when supplied, it
    gives one stable identity per option. The adapter reports the matching key
    as ``chosen_key`` so rewriting scoring text never changes the caller's
    observable answer. Omitting it keeps the legacy option-text behaviour.
    """

    type: QuestionType
    options: list[str] = Field(..., min_length=1)
    instruction: str = ""
    answer_keys: list[str] | None = None


class SystemOneBody(BaseModel):
    """A decision request: one state plus one or more typed questions.

    ``state`` is required but may be empty: an absent field is a client bug,
    while ``""`` is a legitimate no-context request.
    """

    state: str
    questions: list[QuestionIn] = Field(..., min_length=1)


class QuestionDecisionOut(BaseModel):
    """One decision: the option chosen and its probability distribution."""

    type: str
    probabilities: list[float]
    chosen_key: str
    confidence: float


class SystemOneResponseOut(BaseModel):
    """Response envelope: one decision per request question.

    ``options_truncated`` reports the number of option rows whose text exceeded
    the qualified option budget and was therefore cut by the adapter. Reporting
    is additive only — the qualified tokenization contract is unchanged.
    """

    decisions: list[QuestionDecisionOut]
    truncated: bool
    state_tokens: int
    options_truncated: int = 0
    option_rows: int = 0


class HealthOut(BaseModel):
    """Liveness plus the resolved configuration, for operators."""

    status: Literal["ok"]
    model: str
    dtype: str
    max_state_len: int
    max_option_len: int
    uptime_s: float


class ReadyOut(BaseModel):
    """Readiness with its individual checks.

    ``gpu_available`` is informational only. A CPU deployment is ready when
    the model is loaded and the selected device can execute inference.
    """

    ready: bool
    checks: dict[str, bool]
    gpu_available: bool
    device: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup.

    Raises:
        RuntimeError: If the model or its custom code cannot be loaded. Startup
            aborts so the process never serves traffic it cannot handle — a
            supervisor sees a non-zero exit instead of a healthy-looking
            process that fails every request.
    """
    global _inference, _infer
    logger.info("loading RockAndEgg-313M-v3 from %s", SOURCE)
    infer = _load_inference_module()

    # The adapter and custom model code are loaded from THIS directory. Record
    # their digest so an operator can pin the exact code that served a request.
    digest = hashlib.sha256()
    for name in _LOCAL_CODE_FILES:
        path = _HERE / name
        if not path.is_file():
            raise RuntimeError(f"required local code file is missing: {path}")
        digest.update(path.read_bytes())
    code_digest = digest.hexdigest()
    logger.info("local custom-code digest: %s", code_digest)
    pinned_code_digest = os.environ.get("RAE_CODE_DIGEST", "")
    if pinned_code_digest and not hmac.compare_digest(pinned_code_digest, code_digest):
        raise RuntimeError(
            "RAE_CODE_DIGEST does not match the custom code in this directory; "
            "refusing to serve with an unverified adapter/model implementation"
        )

    # A Hub-sourced model must be revision-pinned: weights, tokenizer and custom
    # code otherwise come from 'main' at load time and can silently diverge.
    # Local-directory operation needs no pin and remains fully offline.
    is_local = Path(str(SOURCE)).is_dir()
    revision = os.environ.get("RAE_MODEL_REVISION", "")
    if not is_local and not revision:
        raise RuntimeError(
            "RAE_MODEL_ID points to a Hub repository: set RAE_MODEL_REVISION to "
            "the exact commit SHA the adapter and weights were qualified against"
        )
    if is_local and revision:
        logger.info("ignoring RAE_MODEL_REVISION: SOURCE is a local directory")

    tokenizer = AutoTokenizer.from_pretrained(
        SOURCE, revision=revision or None, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        SOURCE,
        revision=revision or None,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    ).to("cuda:0" if torch.cuda.is_available() else "cpu")

    if revision:
        resolved = getattr(model.config, "_commit_hash", None) or (
            getattr(tokenizer, "init_kwargs", {}) or {}
        ).get("_commit_hash")
        if resolved and not str(resolved).startswith(revision):
            raise RuntimeError(
                f"resolved model revision {resolved!r} != pinned {revision!r}"
            )
        if not resolved:
            logger.warning(
                "revision pin was passed to from_pretrained but the loaded "
                "artifacts expose no _commit_hash for post-load verification"
            )

    model.eval()

    _infer = infer
    _inference = infer.SystemOneInference(
        model,
        tokenizer,
        max_state_len=MAX_STATE_LEN,
        max_option_len=MAX_OPTION_LEN,
    )
    logger.info("model loaded and ready")
    yield


app = FastAPI(title="systemone-server", version="0.3.1", lifespan=lifespan)


@app.middleware("http")
async def _auth_guard(request: Request, call_next: Any) -> Any:
    """Require a Bearer token only when non-loopback binding was opted into.

    The default configuration binds loopback and needs no token. When
    ``RAE_API_TOKEN`` is set together with a non-loopback ``RAE_HOST``, every
    route — including health probes — requires ``Authorization: Bearer <token>``.
    """
    if _REQUIRE_AUTH:
        supplied = request.headers.get("authorization", "")
        if not hmac.compare_digest(supplied, f"Bearer {_API_TOKEN}"):
            return _error(
                401,
                "authorization required",
                "this server was configured for non-loopback exposure and requires a token",
                "send Authorization: Bearer <RAE_API_TOKEN>",
                EXIT_BAD_REQUEST,
            )
    return await call_next(request)


BAD_REQUEST = 422
SERVER_ERROR = 500

# Exit codes are the failure classes a client can branch on without parsing
# prose. 3 = the request was wrong; 4 = the server failed; 5 = not ready.
EXIT_BAD_REQUEST = 3
EXIT_INFERENCE_FAILED = 4

# Pydantic's internal type names -> the JSON type a caller actually writes.
_EXPECTED_TYPE = {
    "string_type": "a string",
    "list_type": "an array",
    "dict_type": "an object",
    "int_type": "an integer",
    "float_type": "a number",
    "bool_type": "a boolean",
}

# Python runtime type names -> JSON type names, so a Cause line never leaks
# "NoneType" or "int" at someone who wrote JSON.
_ACTUAL_TYPE = {
    "NoneType": "null",
    "bool": "a boolean",
    "int": "a number",
    "float": "a number",
    "str": "a string",
    "list": "an array",
    "dict": "an object",
}


def _error(
    status_code: int, description: str, cause: str, action: str, exit_code: int
) -> JSONResponse:
    """Build the canonical four-field diagnostic block.

    Args:
        status_code: HTTP status to return.
        description: One-line summary; the "error:" line.
        cause: The exact root cause, not a restatement of the symptom.
        action: The specific corrective step a caller can take.
        exit_code: Stable integer for clients that branch numerically.

    Returns:
        A ``JSONResponse`` carrying the diagnostic block verbatim as
        ``{"error": "<block>"}``.
    """
    block = (
        f"error: {description}\n"
        f"  \u2192 Cause: {cause}\n"
        f"  \u2192 Action: {action}\n"
        f"  \u2192 Exit:   {exit_code}"
    )
    return JSONResponse(status_code=status_code, content={"error": block})


def _validation_error(exc: ValidationError) -> JSONResponse:
    """Translate a Pydantic error into the canonical diagnostic block.

    ``too_short`` covers two distinct failures and is disambiguated by the
    failing field's location: the ``questions`` list being empty is a different
    client mistake from an individual question's ``options`` being empty, and
    the two must not collapse into one message.

    Args:
        exc: The validation failure raised while parsing the request body.

    Returns:
        A ``422`` ``JSONResponse`` with a specific description, cause, action,
        and exit code ``3``.
    """
    err = exc.errors()[0]
    location = ".".join(str(part) for part in err["loc"]) or "body"
    error_type = err["type"]

    if error_type == "missing":
        description = f"field '{location}' is required"
        cause = f"the request body does not supply '{location}'"
        action = f"add '{location}' to the JSON body and resend"
    elif error_type == "literal_error":
        # Never reflect the submitted value: it may be long or contain data the
        # caller did not intend to echo. Name the JSON type instead.
        description = f"unknown question type for '{location}'"
        cause = f"'{location}' must be one of choice, score, noul"
        action = "set 'type' to one of: choice, score, noul"
    elif error_type == "too_short" and err["loc"][-1] == "questions":
        description = "questions must not be empty"
        cause = "the request asks no questions"
        action = "supply at least one question object in 'questions'"
    elif error_type == "too_short":
        description = f"options must not be empty at '{location}'"
        cause = "a question was given no candidate options"
        action = "supply at least one option string for that question"
    else:
        expected = _EXPECTED_TYPE.get(error_type, "a value of the declared type")
        actual_type = type(err.get("input")).__name__
        actual = _ACTUAL_TYPE.get(actual_type, actual_type)
        description = f"wrong type for '{location}'"
        cause = f"'{location}' is {actual} but {expected} is required"
        action = f"send {expected} for '{location}' and resend"

    return _error(BAD_REQUEST, description, cause, action, EXIT_BAD_REQUEST)


def _inference_or_raise() -> Any:
    """Return the loaded adapter, asserting the fail-fast invariant.

    Returns:
        The loaded ``SystemOneInference`` instance.

    Raises:
        RuntimeError: If called before lifespan completed. This indicates a
            programming error, not a runtime condition — an unloaded process
            cannot be serving requests.
    """
    if _inference is None:
        raise RuntimeError("handler reached before lifespan completed")
    return _inference


def _infer_or_raise() -> ModuleType:
    """Return the loaded adapter module, asserting the fail-fast invariant.

    Returns:
        The imported ``systemone_inference`` module.

    Raises:
        RuntimeError: If called before lifespan completed. This indicates a
            programming error, not a runtime condition — an unloaded process
            cannot be serving requests.
    """
    if _infer is None:
        raise RuntimeError("handler reached before lifespan completed")
    return _infer


def _gpu_available() -> bool:
    """Report whether a CUDA device is usable (informational only).

    Returns:
        ``True`` if CUDA is available, ``False`` otherwise. Never raises.
    """
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _selected_device() -> str:
    """Return the device the loaded model actually runs on."""
    inference = _inference
    if inference is None:
        return "unloaded"
    try:
        return str(inference.device)
    except Exception:
        return "unknown"


def _device_ready() -> bool:
    """Report whether the selected execution device is operational.

    A CPU deployment is ready: GPU availability is not a readiness dependency.
    """
    inference = _inference
    if inference is None:
        return False
    try:
        device = inference.device
        if device.type == "cuda":
            return _gpu_available() and torch.cuda.device_count() > 0
        return device.type == "cpu"
    except Exception:
        return False


def _run_inference(inference: Any, request: Any) -> tuple[Any, int, int]:
    """Score one request under the single-inference concurrency bound.

    Returns the response together with the option-truncation counters the
    adapter's batched option encode recorded for this request. They are read in
    this worker thread, immediately after the adapter returns, so the HTTP
    layer needs no second tokenization pass and no mutable state crosses
    requests.
    """
    with _INFERENCE_SLOTS:
        response = inference(request)
        options_truncated, option_rows = inference.last_option_truncation
        return response, options_truncated, option_rows


@app.get("/healthz", response_model=HealthOut, summary="Liveness and configuration")
async def healthz() -> HealthOut:
    """Report liveness and the resolved configuration.

    Performs no inference and does not touch the model weights, so it is cheap
    enough for a load balancer or orchestrator probe. Because startup fails
    fast, a response here means the model is loaded.

    Returns:
        ``HealthOut`` with the serving configuration.

    Raises:
        Never. An unloaded process is not serving, so it cannot answer.
    """
    inference = _inference_or_raise()
    return HealthOut(
        status="ok",
        model=str(SOURCE),
        dtype=str(next(inference.model.parameters()).dtype),
        max_state_len=inference.max_state_len,
        max_option_len=inference.max_option_len,
        uptime_s=round(time.monotonic() - _STARTED_AT, 3),
    )


@app.get("/ready", response_model=ReadyOut, summary="Readiness probe")
async def ready() -> ReadyOut:
    """Report readiness as itemized checks.

    Each system is reported separately rather than as one opaque flag, so an
    operator can see *which* dependency failed instead of only that something
    did.

    Returns:
        ``ReadyOut`` with a boolean per check.

    Raises:
        Never. Returns ``ready: false`` when a check fails.
    """
    checks = {
        "model_loaded": _inference is not None,
        "device_ready": _device_ready(),
    }
    return ReadyOut(
        ready=all(checks.values()),
        checks=checks,
        gpu_available=_gpu_available(),
        device=_selected_device(),
    )


@app.post(
    "/v1/systemone",
    response_model=SystemOneResponseOut,
    summary="Typed decisions over one state",
)
async def systemone(request: Request) -> SystemOneResponseOut | JSONResponse:
    """Answer typed questions about one state.

    The body is parsed here rather than as a Pydantic parameter so that
    *every* failure — malformed JSON included — returns the canonical error
    block instead of FastAPI's default ``{"detail": ...}`` shape.

    Args:
        request: The raw HTTP request; its body is read verbatim.

    Returns:
        ``SystemOneResponseOut`` with one decision per question, plus
        ``truncated`` and ``state_tokens`` describing the state length.

    Raises:
        Never. All failures are returned as ``JSONResponse`` with the
        canonical block (see the module docstring).
    """
    body_bytes = await request.body()
    try:
        raw = json.loads(body_bytes) if body_bytes else None
    except json.JSONDecodeError as exc:
        return _error(
            BAD_REQUEST,
            "body was not valid JSON",
            f"the request body is not parseable JSON: {exc.msg} at column {exc.colno}",
            "send a JSON object; check for trailing commas, single quotes, or truncation",
            EXIT_BAD_REQUEST,
        )

    try:
        body = SystemOneBody.model_validate(raw)
    except ValidationError as exc:
        return _validation_error(exc)

    # Stable-key validation is a caller-contract check, so a mismatch is a
    # 400 with the canonical four-field block, never an inference error.
    for q in body.questions:
        if q.answer_keys is not None and len(q.answer_keys) != len(q.options):
            return _error(
                BAD_REQUEST,
                "answer_keys length mismatch",
                "a question has a different number of answer_keys than options",
                "send exactly one answer key per option, or omit answer_keys",
                EXIT_BAD_REQUEST,
            )

    infer = _infer_or_raise()
    questions = [
        infer.Question(
            type=q.type,
            options=q.options,
            instruction=q.instruction,
            answer_keys=q.answer_keys,
        )
        for q in body.questions
    ]

    inference = _inference_or_raise()
    # Tokenize the state ourselves so truncation is observable: the adapter
    # truncates silently, and a caller cannot otherwise tell whether its state
    # was cut. Option truncation is NOT a second pass: the adapter's batched
    # option encode records it per request (see _run_inference).
    state_tokens = len(
        inference.tokenizer(body.state, add_special_tokens=False)["input_ids"]
    )

    request_obj = infer.SystemOneRequest(state=body.state, questions=questions)
    try:
        # Inference is synchronous and CPU/GPU bound. Run it in a worker thread
        # so health probes stay responsive, under a single-slot bound so the
        # model is never entered concurrently.
        response, options_truncated, option_rows = await asyncio.to_thread(
            _run_inference, inference, request_obj
        )
    except Exception:
        # Full exception (type, message, traceback) stays in the server log.
        # The client receives no internal detail, file path or tensor shape.
        logger.exception("inference failed")
        return _error(
            SERVER_ERROR,
            "inference failed",
            "the model could not score this request; see the server log for the diagnostic",
            "retry; if it persists, contact the operator with the request time",
            EXIT_INFERENCE_FAILED,
        )

    return SystemOneResponseOut(
        decisions=[
            QuestionDecisionOut(
                type=d.type,
                probabilities=d.probabilities,
                chosen_key=d.chosen_key,
                confidence=d.confidence,
            )
            for d in response.decisions
        ],
        truncated=state_tokens > inference.max_state_len,
        state_tokens=state_tokens,
        options_truncated=options_truncated,
        option_rows=option_rows,
    )


def _is_loopback(host: str) -> bool:
    """Return True for loopback addresses and 'localhost'."""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.strip().lower() in {"localhost", "ip6-localhost"}


def main() -> None:
    """Run the server. Default: 127.0.0.1:8012, loopback only, no auth."""
    global _REQUIRE_AUTH, _API_TOKEN
    host = os.environ.get("RAE_HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("RAE_PORT", "8012"))
    except ValueError as exc:
        raise RuntimeError("RAE_PORT must be an integer") from exc

    token = os.environ.get("RAE_API_TOKEN", "")
    if not _is_loopback(host):
        if not token:
            raise RuntimeError(
                "Refusing to bind a non-loopback address without RAE_API_TOKEN. "
                "Place an authenticating reverse proxy in front of the server, "
                "or keep the loopback default."
            )
        _REQUIRE_AUTH = True
        _API_TOKEN = token
        logger.warning(
            "non-loopback binding enabled; Bearer auth is required on every route"
        )

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
