import time
from pathlib import Path

import pytest

from src import analytics as an
from src import database as db
from src.engine_analysis import find_stockfish
from src.ingest import parse_pgn_text
from src.pipeline import AnalysisJob, analyze_games, store_report
from tests.conftest import ALICE_LOSS, SCHOLARS_MATE

ROOT = Path(__file__).resolve().parent.parent
needs_engine = pytest.mark.skipif(find_stockfish() is None, reason="Stockfish not installed")


@needs_engine
def test_end_to_end_pipeline_with_stockfish(tmp_path):
    path = tmp_path / "e2e.db"
    report = parse_pgn_text(SCHOLARS_MATE + "\n\n" + ALICE_LOSS)
    summary = store_report(path, report)
    stats = analyze_games(path, summary.game_ids, depth=6)
    assert stats["games"] == 2
    conn = db.connect(path)
    counts = db.table_counts(conn)
    assert counts["move_analysis"] == counts["moves"] == 15
    assert counts["analysis_runs"] == 2
    # Black's 3...Nf6?? allows Qxf7# - must be flagged as a blunder
    row = conn.execute(
        """SELECT a.classification, a.best_move_san FROM moves m JOIN move_analysis a USING(move_id)
           WHERE m.game_id = ? AND m.ply = 6""",
        (summary.game_ids[0],),
    ).fetchone()
    assert row["classification"] == "blunder"
    ov = an.overview(conn, "Bob")
    assert ov["analyzed_games"] == 1 and ov["blunder_rate"] > 0
    conn.close()

    rerun = analyze_games(path, summary.game_ids, depth=6)
    assert rerun["engine_calls"] == 0  # everything served from the SQLite cache


@needs_engine
def test_background_job_reports_progress(tmp_path):
    path = tmp_path / "job.db"
    summary = store_report(path, parse_pgn_text(ALICE_LOSS))
    job = AnalysisJob(str(path), summary.game_ids, depth=6).start()
    deadline = time.time() + 60
    while not job.finished and time.time() < deadline:
        time.sleep(0.1)
    assert job.finished and job.error is None
    assert job.fraction == 1.0 and job.stats["games"] == 1


def test_job_waits_for_a_free_engine_slot_and_can_be_cancelled(tmp_path):
    from src import pipeline

    held = []
    while pipeline.ENGINE_SLOTS.acquire(blocking=False):
        held.append(1)
    try:
        job = AnalysisJob(str(tmp_path / "q.db"), [1], depth=6).start()
        time.sleep(0.8)
        assert job.queued and not job.finished
        job.cancel()
        deadline = time.time() + 5
        while not job.finished and time.time() < deadline:
            time.sleep(0.1)
        assert job.finished and job.games_done == 0
    finally:
        for _ in held:
            pipeline.ENGINE_SLOTS.release()


def test_app_home_and_demo_render_without_errors():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    assert any("ChessLens" in m.value for m in at.markdown)
    demo = next(b for b in at.button if b.label == "Try Demo")
    demo.click().run()
    assert not at.exception
    assert at.session_state["dataset"] == "demo"
