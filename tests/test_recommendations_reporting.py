import re

import pytest

from src import analytics as an
from src import config, recommendations, reporting
from src import database as db


def test_recommendations_need_enough_analyzed_games(tmp_db):
    conn, _ = tmp_db
    recs, note = recommendations.generate(conn, "nobody")
    assert recs == [] and "analyzed games" in note


@pytest.fixture
def demo_conn():
    if not config.DEMO_DB_PATH.exists():
        pytest.skip("demo database not built")
    conn = db.connect(config.DEMO_DB_PATH, read_only=True)
    yield conn
    conn.close()


def test_demo_recommendations_cite_statistics_and_examples(demo_conn):
    recs, _ = recommendations.generate(demo_conn, "sani5057")
    assert len(recs) == 3
    assert recs == sorted(recs, key=lambda r: r.z, reverse=True)
    for r in recs:
        assert re.search(r"\d+(\.\d+)?%", r.evidence), r.evidence  # quotes a rate
        assert r.sample_size > 0
        assert r.examples, r.title
        assert all("engine preferred" in e["label"] for e in r.examples)
    ids = [e["move_id"] for r in recs for e in r.examples]
    assert len(ids) == len(set(ids))  # each example position is cited once
    kinds = [r.area.split(":")[0] for r in recs]
    assert len([k for k in kinds if k != "context"]) == len({k for k in kinds if k != "context"})


def test_no_tactical_motif_claims(demo_conn):
    recs, _ = recommendations.generate(demo_conn, "sani5057")
    text = " ".join(r.evidence for r in recs).lower()
    assert "fork" not in text and "pin" not in text.split()


def test_demo_database_is_read_only(demo_conn):
    import sqlite3

    with pytest.raises(sqlite3.OperationalError):
        demo_conn.execute("DELETE FROM games")


def test_coaching_summary_and_csv(analyzed_db):
    conn, _ = analyzed_db
    ov = an.overview(conn, "Alice")
    md = reporting.coaching_summary(
        "Alice", ov, an.phase_stats(conn, "Alice"), an.opening_stats(conn, "Alice"), [], "note", "test data"
    )
    assert md.startswith("# ChessLens coaching summary - Alice")
    assert "Polonsky Academy" in md and "project-defined" in md
    assert "Analyzed moves by this player: 11" in md
    assert "1W / 1D / 1L" in md
    assert "| Opening |" in md
    csv = reporting.to_csv_bytes(an.move_rows(conn, "Alice")).decode()
    assert csv.splitlines()[0].startswith("game_id,")
    assert len(csv.strip().splitlines()) == 1 + 11
