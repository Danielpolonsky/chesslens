"""Central configuration: paths, thresholds, and resource limits."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DEMO_DIR = DATA_DIR / "demo"
DEMO_DB_PATH = DEMO_DIR / "demo.db"
MODELS_DIR = DATA_DIR / "models"
ML_METRICS_PATH = MODELS_DIR / "ml_metrics.json"
ML_MODEL_PATH = MODELS_DIR / "risk_model.joblib"
BIN_DIR = ROOT_DIR / "bin"

# Project-defined centipawn-loss thresholds (NOT official Chess.com/Lichess classifications).
INACCURACY_CP = 50
MISTAKE_CP = 100
BLUNDER_CP = 200

# Evaluations are clamped to +/- EVAL_CAP_CP before computing centipawn loss so that
# a forced mate does not produce an unbounded loss value.
EVAL_CAP_CP = 1000
MATE_SCORE_CP = 10000

# Engine defaults.
DEFAULT_DEPTH = 10
MIN_DEPTH = 6


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Resource limits. Defaults are tuned for the public Streamlit Community Cloud instance;
# raise them locally with environment variables.
MAX_DEPTH = _env_int("CHESSLENS_MAX_DEPTH", 14)
MAX_ANALYSIS_GAMES = _env_int("CHESSLENS_MAX_ANALYSIS_GAMES", 40)
MAX_UPLOAD_GAMES = _env_int("CHESSLENS_MAX_UPLOAD_GAMES", 500)
MAX_UPLOAD_BYTES = _env_int("CHESSLENS_MAX_UPLOAD_BYTES", 5 * 1024 * 1024)
ENGINE_THREADS = _env_int("CHESSLENS_ENGINE_THREADS", 1)
ENGINE_HASH_MB = _env_int("CHESSLENS_ENGINE_HASH_MB", 64)
MAX_CONCURRENT_ANALYSES = _env_int("CHESSLENS_MAX_CONCURRENT_ANALYSES", 2)

CHESSCOM_DEFAULT_GAMES = 25
CHESSCOM_MAX_GAMES = 100
CHESSCOM_TIMEOUT_S = 15
CHESSCOM_USER_AGENT = (
    "ChessLens/1.0 (open-source chess analytics portfolio project; "
    "https://github.com/ - contact via repository)"
)

# Minimum sample sizes before the dashboard draws conclusions.
MIN_GAMES_FOR_OPENING_CLAIM = 3
MIN_MOVES_FOR_PHASE_CLAIM = 30
