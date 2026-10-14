import time
from pathlib import Path

import pytest

from src import analytics as an
from src import database as db
from src.engine_analysis import EngineUnavailable, find_stockfish
from src.ingest import parse_pgn_text
from src.pipeline import AnalysisJob, analyze_games, describe_analysis, describe_import, store_report
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


class _FlakyAnalyzer:
    """Fails the first game with an engine error, then analyses the rest."""

    def __init__(self, **_kwargs):
        self.depth = 8
        self.engine_id = "fake|threads=1"
        self.engine_name = "fake"
        self.engine_calls = 0
        self.cache_hits = 0
        self._games = 0

    def open(self):
        return None

    def close(self):
        return None

    def analyze_moves(self, fens, ucis, on_position=None):
        self._games += 1
        if self._games == 1:
            raise OSError("engine died")
        from tests.conftest import fake_move_eval
        return [fake_move_eval(i + 1, 0) for i in range(len(ucis))]


class _MissingEngine:
    engine_calls = 0
    cache_hits = 0

    def __init__(self, **_kwargs):
        pass

    def open(self):
        raise EngineUnavailable("Stockfish was not found")

    def close(self):
        return None


def test_one_engine_failure_does_not_discard_the_run(tmp_path):
    path = tmp_path / "partial.db"
    summary = store_report(path, parse_pgn_text(SCHOLARS_MATE + "\n\n" + ALICE_LOSS))
    stats = analyze_games(path, summary.game_ids, depth=8, analyzer_factory=_FlakyAnalyzer)
    assert stats["games"] == 1
    assert len(stats["failed"]) == 1
    assert "engine stopped" in stats["failed"][0]["reason"]
    assert stats["error"] is None
    kind, text = describe_analysis(type("J", (), {"stats": stats, "total": 2, "error": None})())
    assert kind == "warning" and "1 of 2" in text


def test_unavailable_engine_is_reported(tmp_path):
    path = tmp_path / "missing.db"
    summary = store_report(path, parse_pgn_text(SCHOLARS_MATE))
    with pytest.raises(EngineUnavailable):
        analyze_games(path, summary.game_ids, depth=8, analyzer_factory=_MissingEngine)


def test_background_job_records_an_unavailable_engine(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise EngineUnavailable("Stockfish was not found")

    monkeypatch.setattr("src.pipeline.analyze_games", boom)
    job = AnalysisJob(str(tmp_path / "job.db"), [1], depth=6).start()
    deadline = time.time() + 5
    while not job.finished and time.time() < deadline:
        time.sleep(0.05)
    assert job.finished
    assert job.error and "Stockfish" in job.error
    kind, text = describe_analysis(job)
    assert kind == "error" and "Stockfish" in text


def test_describe_import_reports_rejections_and_cross_import_duplicates(tmp_path):
    from tests.conftest import ILLEGAL, NO_HEADERS

    path = tmp_path / "imports.db"
    report = parse_pgn_text(SCHOLARS_MATE + "\n\n" + SCHOLARS_MATE + "\n\n" + ILLEGAL + "\n\n" + NO_HEADERS)
    summary = store_report(path, report)
    outcome = describe_import(report, summary, max_games=500)
    assert outcome.inserted == 2  # scholar's mate once, plus the headerless game
    assert outcome.kind == "warning"
    assert any("repeated within this import" in line for line in outcome.details)
    assert any("illegal move" in line for line in outcome.details)
    assert any("without date" in line for line in outcome.details)

    again = parse_pgn_text(SCHOLARS_MATE)
    second = describe_import(again, store_report(path, again))
    assert second.inserted == 0
    assert "already imported" in second.headline


def _open_page(at, url_path: str):
    """Open a callable ``st.Page`` by the path registered in ``st.navigation``."""
    match = next(
        (page_hash for page_hash, info in at._registered_pages.items()
         if info.get("url_pathname") == url_path),
        None,
    )
    assert match, f"no page registered at {url_path}"
    at._page_hash = match
    at.run()
    assert not at.exception


def test_app_home_and_demo_render_without_errors():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    assert any("ChessLens" in m.value for m in at.markdown)
    demo = next(b for b in at.button if b.label == "Try demo")
    demo.click().run()
    assert not at.exception
    assert at.session_state["dataset"] == "demo"


def test_app_switches_between_imported_games_and_the_demo(tmp_path):
    path = tmp_path / "user.db"
    store_report(path, parse_pgn_text(SCHOLARS_MATE + "\n\n" + ALICE_LOSS))
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.session_state["dataset"] = "user"
    at.session_state["user_db"] = str(path)
    at.session_state["user_player"] = "Bob"
    at.run()
    assert not at.exception
    player = next(box for box in at.selectbox if box.label == "Player")
    assert player.value == "Bob"
    radio = next(item for item in at.radio if item.label == "Dataset")
    radio.set_value("demo").run()
    assert not at.exception
    assert at.session_state["dataset"] == "demo"
    player = next(box for box in at.selectbox if box.label == "Player")
    assert player.value == "sani5057"
    radio = next(item for item in at.radio if item.label == "Dataset")
    radio.set_value("user").run()
    assert at.session_state["dataset"] == "user"
    assert next(box for box in at.selectbox if box.label == "Player").value == "Bob"


def test_demo_pages_render_and_coaching_opens_that_mistake():
    from streamlit.testing.v1 import AppTest

    from src import recommendations
    from src import database as db

    conn = db.connect(ROOT / "data" / "demo" / "demo.db", read_only=True)
    try:
        example = recommendations.generate(conn, "sani5057")[0][0].examples[0]
    finally:
        conn.close()

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.session_state["dataset"] = "demo"
    at.session_state["explorer_move_id"] = example["move_id"]
    at.run()
    for path in ("dashboard", "mistakes", "practice", "ml", "methodology"):
        _open_page(at, path)
    _open_page(at, "mistakes")
    body = " ".join(m.value for m in at.markdown)
    assert example["label"].split(",")[0] in body or "Centipawns lost" in body
    assert any(box.value == example["move_id"] for box in at.selectbox if "Mistake" in box.label)
