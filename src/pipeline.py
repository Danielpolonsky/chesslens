"""Orchestration: Store parsed games, then Analyze them with Stockfish.

``AnalysisJob`` runs the engine in a background thread with its own SQLite connection so
the Streamlit UI keeps rendering (and can show live progress) while games are analysed.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import database as db
from .engine_analysis import StockfishAnalyzer
from .ingest import ParseReport


def store_report(db_path: str | Path, report: ParseReport) -> db.InsertSummary:
    conn = db.connect(db_path)
    try:
        return db.insert_games(conn, report.games)
    finally:
        conn.close()


def analyze_games(
    db_path: str | Path,
    game_ids: list[int],
    depth: int,
    on_progress: Callable[[int, int, int, int], None] | None = None,
    should_stop: Callable[[], bool] = lambda: False,
    engine_path: str | None = None,
) -> dict:
    """Analyse each game (both sides' moves) and persist results. Returns run statistics.

    ``on_progress(games_done, games_total, positions_done, positions_total_in_game)``.
    """
    conn = db.connect(db_path)
    stats = {"games": 0, "positions": 0, "engine_calls": 0, "cache_hits": 0, "seconds": 0.0}
    started = time.perf_counter()
    analyzer = StockfishAnalyzer(
        depth=depth,
        path=engine_path,
        cache_get=lambda k, d: db.cache_get(conn, k, d),
        cache_put=lambda k, d, e: db.cache_put(conn, k, d, e),
    )
    try:
        with analyzer:
            for idx, game_id in enumerate(game_ids):
                if should_stop():
                    break
                rows = db.game_moves(conn, game_id)
                if not rows:
                    continue
                evals = analyzer.analyze_moves(
                    [r["fen_before"] for r in rows],
                    [r["uci"] for r in rows],
                    on_position=(lambda p, t, i=idx: on_progress(i, len(game_ids), p, t))
                    if on_progress
                    else None,
                )
                db.save_game_analysis(
                    conn, game_id, [r["move_id"] for r in rows], evals, analyzer.depth,
                    analyzer.engine_name,
                )
                stats["games"] += 1
                stats["positions"] += len(rows) + 1
                if on_progress:
                    on_progress(idx + 1, len(game_ids), 0, 0)
        stats["engine_calls"] = analyzer.engine_calls
        stats["cache_hits"] = analyzer.cache_hits
    finally:
        conn.commit()
        conn.close()
        stats["seconds"] = round(time.perf_counter() - started, 1)
    return stats


@dataclass
class AnalysisJob:
    """Thread-safe progress holder for a background analysis run."""

    db_path: str
    game_ids: list[int]
    depth: int
    games_done: int = 0
    positions_done: int = 0
    positions_total: int = 0
    finished: bool = False
    error: str | None = None
    stats: dict = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    @property
    def total(self) -> int:
        return len(self.game_ids)

    @property
    def fraction(self) -> float:
        if not self.total:
            return 1.0
        within = self.positions_done / self.positions_total if self.positions_total else 0.0
        return min(1.0, (self.games_done + within) / self.total)

    def start(self) -> "AnalysisJob":
        self._thread = threading.Thread(target=self._run, name="chesslens-analysis", daemon=True)
        self._thread.start()
        return self

    def cancel(self) -> None:
        self._stop.set()

    def _progress(self, games_done: int, total: int, pos_done: int, pos_total: int) -> None:
        self.games_done = games_done
        self.positions_done = pos_done
        self.positions_total = pos_total

    def _run(self) -> None:
        try:
            self.stats = analyze_games(
                self.db_path, self.game_ids, self.depth, self._progress, self._stop.is_set
            )
        except Exception as exc:  # surfaced in the UI
            self.error = str(exc)
        finally:
            self.finished = True
