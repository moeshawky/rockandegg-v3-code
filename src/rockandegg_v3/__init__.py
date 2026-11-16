"""RockAndEgg-313M-v3 native SystemOne decision model.

A headless Qwen3 backbone with a typed decision head. No LM head and no token
generation: the model emits a probability distribution and confidence per
question from a single state encoding.
"""

from .config import (
    D_MODEL,
    ECE_BOUND,
    HEAD_DIM,
    KV_HEADS,
    MARKER_ID,
    MARKER_TOKEN,
    MODEL_PATH,
    NUM_LAYERS,
    NUM_QTYPES,
    ORDER_TOL,
    QTYPE_CHOICE,
    QTYPE_NOUL,
    QTYPE_SCORE,
    HeadConfig,
)
from .gates import GateReport, compute_ece, serving_probs, structural_gate
from .inference import (
    Question,
    QuestionDecision,
    SystemOneInference,
    SystemOneRequest,
    SystemOneResponse,
)
from .loss import option_dropout, verdict_pact_loss
from .model import RockAndEggV3SystemOne
from .parallel import (
    ParallelSystemOneInference,
    available_devices,
    build_replicas,
    cache_to_device,
    forward_case_multi,
)

__all__ = [
    "D_MODEL",
    "ECE_BOUND",
    "HEAD_DIM",
    "KV_HEADS",
    "MARKER_ID",
    "MARKER_TOKEN",
    "MODEL_PATH",
    "NUM_LAYERS",
    "NUM_QTYPES",
    "ORDER_TOL",
    "QTYPE_CHOICE",
    "QTYPE_NOUL",
    "QTYPE_SCORE",
    "GateReport",
    "HeadConfig",
    "ParallelSystemOneInference",
    "Question",
    "QuestionDecision",
    "RockAndEggV3SystemOne",
    "SystemOneInference",
    "SystemOneRequest",
    "SystemOneResponse",
    "available_devices",
    "build_replicas",
    "cache_to_device",
    "compute_ece",
    "forward_case_multi",
    "option_dropout",
    "serving_probs",
    "structural_gate",
    "verdict_pact_loss",
]
