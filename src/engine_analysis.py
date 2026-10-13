"""Stockfish integration and move-quality metrics.

Perspective convention
----------------------
* Raw engine evaluations are stored from **White's** point of view (``score_cp``).
* Move metrics are computed from the point of view of the **player making the move**:
  ``eval_before`` = engine eval of the position before the move (best play),
  ``eval_after``  = engine eval of the position after the played move,
  ``cpl = max(0, clamp(eval_before) - clamp(eval_after))``.
* Mate scores are mapped to +/- (MATE_SCORE_CP - plies_to_mate) and then clamped to
  +/- EVAL_CAP_CP, so missing a forced mate costs at most a bounded amount.
"""

from __future__ import annotations

import math
import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import chess
import chess.engine

from . import config


@dataclass(frozen=True)
class PositionEval:
    score_cp: int  # White's point of view, mate mapped to +/- (MATE_SCORE_CP - n)
    mate: int | None  # White's point of view: >0 White mates, <0 Black mates
    best_move_uci: str | None


@dataclass(frozen=True)
class MoveEval:
    ply: int
    eval_before: int
    eval_after: int
    best_move_uci: str | None
    best_move_san: str | None
    cpl: int
    classification: str
    win_prob_before: float
    win_prob_after: float
    accuracy: float


# ---------------------------------------------------------------------------
# Engine discovery
# ---------------------------------------------------------------------------


def find_stockfish() -> str | None:
    """Locate a Stockfish binary on Windows, macOS or Linux (incl. Streamlit Cloud)."""
    candidates: list[str | Path | None] = [os.environ.get("STOCKFISH_PATH")]
    candidates.append(shutil.which("stockfish"))
    candidates += ["/usr/games/stockfish", "/usr/bin/stockfish", "/usr/local/bin/stockfish",
                   "/opt/homebrew/bin/stockfish"]
    exe = "stockfish.exe" if platform.system() == "Windows" else "stockfish"
    candidates.append(config.BIN_DIR / exe)
    for c in candidates:
        if c and Path(c).is_file():
            return str(c)
    return None


# ---------------------------------------------------------------------------
# Pure metric helpers (unit-tested without an engine)
# ---------------------------------------------------------------------------


def score_to_cp(score: chess.engine.Score) -> int:
    return int(score.score(mate_score=config.MATE_SCORE_CP))


def clamp_eval(cp: int, cap: int = config.EVAL_CAP_CP) -> int:
    return max(-cap, min(cap, int(cp)))


def to_mover_pov(white_cp: int, mover: chess.Color) -> int:
    return white_cp if mover == chess.WHITE else -white_cp


def centipawn_loss(eval_before: int, eval_after: int) -> int:
    """Both arguments from the mover's point of view."""
    return max(0, clamp_eval(eval_before) - clamp_eval(eval_after))


def classify_cpl(cpl: float) -> str:
    """Project-defined thresholds (not official Chess.com or Lichess categories)."""
    if cpl >= config.BLUNDER_CP:
        return "blunder"
    if cpl >= config.MISTAKE_CP:
        return "mistake"
    if cpl >= config.INACCURACY_CP:
        return "inaccuracy"
    return "good"


def win_probability(cp: int) -> float:
    """Expected score (0-100) for a centipawn eval, using Lichess's published logistic fit."""
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * clamp_eval(cp))) - 1)


def move_accuracy(wp_before: float, wp_after: float) -> float:
    """Lichess-style per-move accuracy (0-100) from the drop in win probability."""
    drop = max(0.0, wp_before - wp_after)
    return max(0.0, min(100.0, 103.1668 * math.exp(-0.04354 * drop) - 3.1669))


def terminal_eval(board: chess.Board) -> PositionEval | None:
    """Exact evaluation for finished positions (engine output is unreliable there)."""
    if board.is_checkmate():
        loser_white = board.turn == chess.WHITE
        cp = -config.MATE_SCORE_CP if loser_white else config.MATE_SCORE_CP
        return PositionEval(score_cp=cp, mate=0, best_move_uci=None)
    if board.is_stalemate() or board.is_insufficient_material():
        return PositionEval(score_cp=0, mate=None, best_move_uci=None)
    return None


def build_move_evals(
    fens_before: list[str],
    played_ucis: list[str],
    evals: list[PositionEval],
) -> list[MoveEval]:
    """Combine N+1 position evaluations (White POV) into N move metrics (mover POV).

    ``evals[i]`` is the evaluation of the position before move ``i``; ``evals[-1]`` is
    the final position after the last move.
    """
    if len(evals) != len(played_ucis) + 1:
        raise ValueError("Need exactly one more position evaluation than moves.")
    out: list[MoveEval] = []
    for i, (fen, uci) in enumerate(zip(fens_before, played_ucis)):
        board = chess.Board(fen)
        mover = board.turn
        before = clamp_eval(to_mover_pov(evals[i].score_cp, mover))
        after = clamp_eval(to_mover_pov(evals[i + 1].score_cp, mover))
        best = evals[i].best_move_uci
        cpl = 0 if best == uci else centipawn_loss(before, after)
        best_san = None
        if best:
            try:
                best_san = board.san(chess.Move.from_uci(best))
            except (ValueError, AssertionError):
                best_san = None
        wp_b, wp_a = win_probability(before), win_probability(after)
        if best == uci:
            wp_a = max(wp_a, wp_b)
        out.append(
            MoveEval(
                ply=i + 1,
                eval_before=before,
                eval_after=after,
                best_move_uci=best,
                best_move_san=best_san,
                cpl=cpl,
                classification=classify_cpl(cpl),
                win_prob_before=round(wp_b, 2),
                win_prob_after=round(wp_a, 2),
                accuracy=round(move_accuracy(wp_b, wp_a), 2),
            )
        )
    return out


def position_key(board: chess.Board) -> str:
    """Cache key: piece placement, side to move, castling and en passant (no move counters)."""
    return board.epd()


# ---------------------------------------------------------------------------
# Engine wrapper
# ---------------------------------------------------------------------------


class EngineUnavailable(RuntimeError):
    pass


class StockfishAnalyzer:
    """Thin wrapper around ``chess.engine.SimpleEngine`` with an evaluation cache.

    ``cache_get(key, depth)`` / ``cache_put(key, depth, PositionEval)`` let callers plug in
    a persistent (SQLite) cache; an in-memory dict is always used as a first level.
    """

    def __init__(
        self,
        depth: int = config.DEFAULT_DEPTH,
        path: str | None = None,
        cache_get: Callable[[str, int], PositionEval | None] | None = None,
        cache_put: Callable[[str, int, PositionEval], None] | None = None,
    ):
        self.depth = max(config.MIN_DEPTH, min(int(depth), config.MAX_DEPTH))
        self.path = path or find_stockfish()
        if not self.path:
            raise EngineUnavailable(
                "Stockfish was not found. Install it (see README) or set STOCKFISH_PATH."
            )
        self._cache_get = cache_get
        self._cache_put = cache_put
        self._memory: dict[tuple[str, int], PositionEval] = {}
        self.engine: chess.engine.SimpleEngine | None = None
        self.engine_name = "Stockfish"
        self.cache_hits = 0
        self.engine_calls = 0

    def __enter__(self) -> "StockfishAnalyzer":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def open(self) -> None:
        if self.engine is None:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(self.path)
            except (OSError, chess.engine.EngineError) as exc:
                raise EngineUnavailable(f"Could not start Stockfish at {self.path}: {exc}") from exc
            options = {"Threads": config.ENGINE_THREADS, "Hash": config.ENGINE_HASH_MB}
            self.engine.configure({k: v for k, v in options.items() if k in self.engine.options})
            self.engine_name = self.engine.id.get("name", "Stockfish")

    def close(self) -> None:
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                pass
            self.engine = None

    def evaluate(self, board: chess.Board) -> PositionEval:
        exact = terminal_eval(board)
        if exact is not None:
            return exact
        key = position_key(board)
        mem_key = (key, self.depth)
        if mem_key in self._memory:
            self.cache_hits += 1
            return self._memory[mem_key]
        if self._cache_get is not None:
            cached = self._cache_get(key, self.depth)
            if cached is not None:
                self.cache_hits += 1
                self._memory[mem_key] = cached
                return cached
        if self.engine is None:
            self.open()
        info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        self.engine_calls += 1
        pov = info["score"].white()
        pv = info.get("pv") or []
        result = PositionEval(
            score_cp=score_to_cp(pov),
            mate=pov.mate(),
            best_move_uci=pv[0].uci() if pv else None,
        )
        self._memory[mem_key] = result
        if self._cache_put is not None:
            self._cache_put(key, self.depth, result)
        return result

    def analyze_moves(
        self,
        fens_before: list[str],
        played_ucis: list[str],
        on_position: Callable[[int, int], None] | None = None,
    ) -> list[MoveEval]:
        """Evaluate every position of a game once and derive per-move metrics."""
        if not played_ucis:
            return []
        boards: list[chess.Board] = [chess.Board(f) for f in fens_before]
        final = boards[-1].copy()
        final.push(chess.Move.from_uci(played_ucis[-1]))
        boards.append(final)
        evals: list[PositionEval] = []
        for i, board in enumerate(boards):
            evals.append(self.evaluate(board))
            if on_position:
                on_position(i + 1, len(boards))
        return build_move_evals(fens_before, played_ucis, evals)
