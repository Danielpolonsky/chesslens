"""Board features computed from the position *before* a move is played.

Everything here only looks at the board the player sees when it is their turn, so the
values are safe to use as predictive features (no post-move or engine information).

Game-phase method (documented project heuristic, adapted from Lichess's "Divider" idea):
  * ``endgame``    - at most 6 queens/rooks/bishops/knights remain on the board (both sides).
  * ``opening``    - full-move number <= 12 and more than 10 such pieces remain
                     (i.e. fewer than four pieces have been traded).
  * ``middlegame`` - every other position.
"""

from __future__ import annotations

import chess
import numpy as np
import pandas as pd

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

PHASES = ("opening", "middlegame", "endgame")
OPENING_MAX_FULLMOVE = 12
OPENING_MIN_PIECES = 11
ENDGAME_MAX_PIECES = 6


def material(board: chess.Board, color: chess.Color) -> int:
    return sum(
        len(board.pieces(piece_type, color)) * value
        for piece_type, value in PIECE_VALUES.items()
    )


def material_balance(board: chess.Board, color: chess.Color) -> int:
    """Material of ``color`` minus material of the opponent (pawn units)."""
    return material(board, color) - material(board, not color)


def major_minor_count(board: chess.Board) -> int:
    return sum(
        len(board.pieces(pt, c))
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        for c in (chess.WHITE, chess.BLACK)
    )


def classify_phase(board: chess.Board) -> str:
    pieces = major_minor_count(board)
    if pieces <= ENDGAME_MAX_PIECES:
        return "endgame"
    if board.fullmove_number <= OPENING_MAX_FULLMOVE and pieces >= OPENING_MIN_PIECES:
        return "opening"
    return "middlegame"


def vulnerable_pieces(board: chess.Board, color: chess.Color) -> int:
    """Count non-king pieces of ``color`` that are attacked and either undefended or
    attacked by a lower-valued enemy piece. A simple static heuristic, not a tactic detector."""
    count = 0
    enemy = not color
    for square, piece in board.piece_map().items():
        if piece.color != color or piece.piece_type == chess.KING:
            continue
        attackers = board.attackers(enemy, square)
        if not attackers:
            continue
        if not board.is_attacked_by(color, square):
            count += 1
            continue
        value = PIECE_VALUES[piece.piece_type]
        cheapest = min(PIECE_VALUES[board.piece_type_at(sq)] for sq in attackers)
        if cheapest < value:
            count += 1
    return count


def position_features(board: chess.Board) -> dict:
    """Pre-move features for the side to move."""
    mover = board.turn
    legal = list(board.legal_moves)
    captures = sum(1 for m in legal if board.is_capture(m))
    checks = sum(1 for m in legal if board.gives_check(m))
    return {
        "phase": classify_phase(board),
        "material_balance": material_balance(board, mover),
        "total_material": material(board, chess.WHITE) + material(board, chess.BLACK),
        "legal_moves": len(legal),
        "in_check": int(board.is_check()),
        "own_vulnerable": vulnerable_pieces(board, mover),
        "opp_vulnerable": vulnerable_pieces(board, not mover),
        "captures_available": captures,
        "checks_available": checks,
    }


# ---------------------------------------------------------------------------
# ML feature matrix
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "move_number",
    "player_elo",
    "opponent_elo",
    "elo_diff",
    "elo_missing",
    "is_white",
    "material_balance",
    "total_material",
    "legal_moves",
    "in_check",
    "own_vulnerable",
    "opp_vulnerable",
    "captures_available",
    "checks_available",
    "base_time_s",
    "increment_s",
]
CATEGORICAL_FEATURES = ["phase"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "serious_error"
GROUP_COLUMN = "game_id"

# Columns that are only known after the move (or come from the engine) and must never be
# used as model inputs.
LEAKAGE_COLUMNS = {
    "cpl",
    "classification",
    "eval_before",
    "eval_after",
    "best_move_uci",
    "san",
    "uci",
    "accuracy",
}


def parse_time_control(tc: str | None) -> tuple[float, float]:
    """Return (base_seconds, increment_seconds); NaN when unknown or correspondence."""
    if not tc or tc in {"-", "?"} or "/" in tc:
        return (np.nan, np.nan)
    try:
        if "+" in tc:
            base, inc = tc.split("+", 1)
            return (float(base), float(inc))
        return (float(tc), 0.0)
    except ValueError:
        return (np.nan, np.nan)


def build_ml_frame(moves: pd.DataFrame, serious_cp: int = 100) -> pd.DataFrame:
    """Turn analysed move rows into a model-ready frame.

    ``moves`` must contain the pre-move board columns stored by the pipeline plus ``cpl``
    (used only to build the target), ``game_id``, ``color``, ``white_elo``, ``black_elo``
    and ``time_control``.
    """
    df = moves.copy()
    df = df[df["cpl"].notna()]
    is_white = (df["color"] == "white").astype(int)
    white_elo = pd.to_numeric(df["white_elo"], errors="coerce")
    black_elo = pd.to_numeric(df["black_elo"], errors="coerce")
    player_elo = np.where(is_white == 1, white_elo, black_elo)
    opponent_elo = np.where(is_white == 1, black_elo, white_elo)
    tcs = df["time_control"].map(parse_time_control)

    out = pd.DataFrame(
        {
            GROUP_COLUMN: df["game_id"].to_numpy(),
            "move_number": df["move_number"].astype(float).to_numpy(),
            "player_elo": player_elo.astype(float),
            "opponent_elo": opponent_elo.astype(float),
            "is_white": is_white.to_numpy(),
            "material_balance": df["material_balance"].astype(float).to_numpy(),
            "total_material": df["total_material"].astype(float).to_numpy(),
            "legal_moves": df["legal_moves"].astype(float).to_numpy(),
            "in_check": df["in_check"].astype(float).to_numpy(),
            "own_vulnerable": df["own_vulnerable"].astype(float).to_numpy(),
            "opp_vulnerable": df["opp_vulnerable"].astype(float).to_numpy(),
            "captures_available": df["captures_available"].astype(float).to_numpy(),
            "checks_available": df["checks_available"].astype(float).to_numpy(),
            "base_time_s": [t[0] for t in tcs],
            "increment_s": [t[1] for t in tcs],
            "phase": df["phase"].astype(str).to_numpy(),
        }
    )
    out["elo_missing"] = (out["player_elo"].isna() | out["opponent_elo"].isna()).astype(int)
    out["elo_diff"] = out["player_elo"] - out["opponent_elo"]
    out[TARGET_COLUMN] = (df["cpl"].to_numpy() >= serious_cp).astype(int)
    return out[[GROUP_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]].reset_index(drop=True)
