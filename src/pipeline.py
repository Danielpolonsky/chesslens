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

import chess.engine

from . import config
from . import database as db
from .engine_analysis import EngineUnavailable, StockfishAnalyzer
from .ingest import ParseReport

# Server-wide cap on simultaneous engine processes (shared by all browser sessions).
ENGINE_SLOTS = threading.BoundedSemaphore(config.MAX_CONCURRENT_ANALYSES)

ENGINE_FAILURES = (chess.engine.EngineError, OSError, TimeoutError)


def store_report(db_path: str | Path, report: ParseReport) -> db.InsertSummary:
    conn = db.connect(db_path)
    try:
        return db.insert_games(conn, report.games)
    finally:
        conn.close()


@dataclass
class ImportOutcome:
    """What an import did, phrased for the user. ``kind`` is a Streamlit alert type."""

    kind: str
    headline: str
    details: list[str] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)
    inserted: int = 0


def describe_import(report: ParseReport, summary: db.InsertSummary | None,
                    max_games: int | None = None, encoding_warning: str | None = None) -> ImportOutcome:
    inserted = summary.inserted if summary else 0
    earlier = summary.duplicates if summary else 0
    read = report.games_read or (len(report.games) + report.skipped + report.duplicates_in_file)

    if inserted:
        kind, headline = "success", f"Imported {inserted} new game{'s' if inserted != 1 else ''}."
    elif earlier:
        kind, headline = "info", f"No new games: all {earlier} valid games were already imported."
    else:
        kind, headline = "error", "No games could be imported."

    details = []
    if read:
        details.append(f"Read {read} game{'s' if read != 1 else ''} from the source.")
    if earlier and inserted:
        details.append(f"{earlier} already imported earlier (not added again).")
    if report.duplicates_in_file:
        details.append(f"{report.duplicates_in_file} repeated within this import (kept once).")
    if report.skipped:
        reasons = ", ".join(f"{n} {reason}" for reason, n in report.skip_reasons.most_common())
        details.append(f"{report.skipped} rejected and not imported" + (f": {reasons}." if reasons else "."))
        if kind == "success":
            kind = "warning"
    if report.truncated and max_games:
        details.append(f"Only the first {max_games} valid games were read; the rest of the file was ignored.")
    games = report.games
    if games:
        missing = {
            "date": sum(g.played_at is None for g in games),
            "ratings": sum(g.white_elo is None or g.black_elo is None for g in games),
            "opening name": sum(g.opening is None and g.eco is None for g in games),
        }
        parts = [f"{n} without {what}" for what, n in missing.items() if n]
        if parts:
            details.append("Imported with missing details: " + "; ".join(parts)
                           + ". These games are kept but drop out of date filters, rating charts "
                             "or opening statistics.")
    if encoding_warning:
        details.append(encoding_warning)
    rejections = [e for e in report.errors if e.startswith("Game ")]
    if not rejections and not inserted and not earlier:
        details.extend(report.errors[:3])
    return ImportOutcome(kind, headline, details, rejections, inserted)


def analyze_games(
    db_path: str | Path,
    game_ids: list[int],
    depth: int,
    on_progress: Callable[[int, int, int, int], None] | None = None,
    should_stop: Callable[[], bool] = lambda: False,
    engine_path: str | None = None,
    analyzer_factory: Callable[..., StockfishAnalyzer] | None = None,
) -> dict:
    """Analyse each game (both sides' moves) and persist results. Returns run statistics.

    A failure in one game is recorded in ``stats["failed"]`` and the run continues (after
    restarting the engine if it crashed). Games already finished stay saved.
    ``on_progress(games_done, games_total, positions_done, positions_total_in_game)``.
    """
    conn = db.connect(db_path)
    stats = {"requested": len(game_ids), "games": 0, "positions": 0, "engine_calls": 0,
             "cache_hits": 0, "seconds": 0.0, "failed": [], "cancelled": False, "error": None}
    started = time.perf_counter()
    factory = analyzer_factory or StockfishAnalyzer
    analyzer = factory(
        depth=depth,
        path=engine_path,
        cache_get=lambda k, d, e: db.cache_get(conn, k, d, e),
        cache_put=lambda k, d, e, ev: db.cache_put(conn, k, d, e, ev),
    )
    try:
        analyzer.open()
        for idx, game_id in enumerate(game_ids):
            if should_stop():
                stats["cancelled"] = True
                break
            rows = db.game_moves(conn, game_id)
            if not rows:
                continue
            try:
                evals = analyzer.analyze_moves(
                    [r["fen_before"] for r in rows],
                    [r["uci"] for r in rows],
                    on_position=(lambda p, t, i=idx: on_progress(i, len(game_ids), p, t))
                    if on_progress
                    else None,
                )
            except ENGINE_FAILURES as exc:
                stats["failed"].append(
                    {"game_id": game_id, "reason": f"the engine stopped ({exc.__class__.__name__})"})
                analyzer.close()
                try:
                    analyzer.open()
                except EngineUnavailable as restart_exc:
                    stats["error"] = f"Stockfish could not be restarted: {restart_exc}"
                    break
                continue
            except Exception as exc:  # bad stored data for one game must not end the run
                stats["failed"].append({"game_id": game_id, "reason": str(exc) or exc.__class__.__name__})
                continue
            db.save_game_analysis(
                conn, game_id, [r["move_id"] for r in rows], evals, analyzer.depth,
                analyzer.engine_id or analyzer.engine_name,
            )
            stats["games"] += 1
            stats["positions"] += len(rows) + 1
            if on_progress:
                on_progress(idx + 1, len(game_ids), 0, 0)
    finally:
        stats["engine_calls"] = analyzer.engine_calls
        stats["cache_hits"] = analyzer.cache_hits
        analyzer.close()
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
    player: str | None = None
    games_done: int = 0
    positions_done: int = 0
    positions_total: int = 0
    finished: bool = False
    queued: bool = True
    error: str | None = None
    stats: dict = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    @property
    def total(self) -> int:
        return len(self.game_ids)

    @property
    def active(self) -> bool:
        return not self.finished

    @property
    def cancelled(self) -> bool:
        return self._stop.is_set()

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
            while not ENGINE_SLOTS.acquire(timeout=0.5):
                if self._stop.is_set():
                    self.stats = {"requested": self.total, "games": 0, "cancelled": True, "failed": []}
                    return
            try:
                self.queued = False
                self.started_at = time.time()
                self.stats = analyze_games(
                    self.db_path, self.game_ids, self.depth, self._progress, self._stop.is_set
                )
                self.error = self.stats.get("error")
            finally:
                ENGINE_SLOTS.release()
        except Exception as exc:  # surfaced in the UI
            self.error = str(exc)
        finally:
            self.finished = True


def describe_analysis(job: AnalysisJob) -> tuple[str, str]:
    """Completion message ``(alert kind, text)`` for a finished job."""
    s = job.stats or {}
    done, requested = s.get("games", 0), s.get("requested", job.total)
    failed = s.get("failed", [])
    parts = []
    if done:
        parts.append(f"Analyzed {done} of {requested} game{'s' if requested != 1 else ''}"
                     + (f" in {s.get('seconds', 0):.0f} s." if "seconds" in s else "."))
    if s.get("cancelled"):
        parts.append("Analysis was cancelled; finished games are saved." if done
                     else "Analysis was cancelled before any game finished.")
    if failed:
        parts.append(f"{len(failed)} game{'s' if len(failed) != 1 else ''} could not be analyzed "
                     f"({failed[0]['reason']}).")
    if job.error:
        parts.append(f"Analysis stopped: {job.error}")
    if not parts:
        parts.append("No games were analyzed.")
    kind = "error" if job.error and not done else ("warning" if failed or job.error or s.get("cancelled")
                                                   else "success")
    return kind, " ".join(parts)
